#!/usr/bin/env python3
"""Train a direction head alongside the binary interaction head.

DESIGN -- why the symmetry conflict does not arise here
  The binary label ("do these two taxa interact?") is symmetric; 5,591 reversal rows
  in v4_species_train train the encoder toward argument-order insensitivity. Rather
  than fight that, the encoder input is `mark_canon`: segment A is the two taxa sorted
  alphabetically and segment B is the passage with the alphabetically-first taxon in
  @...@ and the second in #...#. Swapping the stored (s1, s2) therefore produces a
  BYTE-IDENTICAL input, so the binary head is order-invariant by construction rather
  than by training, and there is no gradient to isolate.

  The direction head predicts "is the @ taxon the subject of the relation?" -- a
  question about the passage, not about storage order. Decoding back to FORWARD /
  REVERSE happens outside the model by comparing the @ taxon to the stored species1.
  That is the correct equivariance: the model's answer is invariant under swap, the
  F/R label flips. Measured: P(@ is subject) = 0.507 over the training set, so
  canonical marking also removes the 98.9% leftmost-taxon positional prior for free.

  The head is antisymmetric by construction: s = g(hA,hB,.) - g(hB,hA,.), so
  P(@ subject) = 1 - P(# subject) exactly, at every point in training.

  Relation polarity enters ONLY at the head, as a 2-valued embedding. The encoder
  never sees the relation string, so the relation-form shortcut that made earlier
  attempts degenerate is unavailable to it, and the binary input is unchanged.
"""
import argparse, json, os, sys, time, math
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(Path(__file__).parent))
import xenc_format as X

AT_ID, HASH_ID = 36, 7          # '@' and '#' in the BiomedBERT uncased vocab
ENC = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"


# ------------------------------------------------------------------ data
class BinaryDS(Dataset):
    def __init__(self, df, tok, max_len=256):
        self.a, self.b = X.build_many("mark_canon", df.source_species, df.interaction_type,
                                      df.target_species, df.text.astype(str))
        self.y = df.label.to_numpy(dtype="float32"); self.tok, self.ml = tok, max_len
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.a[i], self.b[i], self.y[i]
    def collate(self, batch):
        a, b, y = zip(*batch)
        e = self.tok(list(a), list(b), truncation="only_second", max_length=self.ml,
                     padding=True, return_tensors="pt")
        e["labels_bin"] = torch.tensor(y)
        return e


class DirectionDS(Dataset):
    """label = 1 if the @ (alphabetically first) taxon is the subject of the relation.

    Rows with directed == 0 are relations the polarity lexicon calls symmetric (interacts
    with, co-occurs with, cross-feeding). They carry no subject, so their direction label is
    masked out of the direction loss and they supervise the directedness branch instead.
    """
    def __init__(self, df, tok, max_len=256):
        s1, s2 = df.s1.astype(str).tolist(), df.s2.astype(str).tolist()
        self.a, self.b = X.build_many("mark_canon", s1, df.rel.astype(str), s2, df.passage.astype(str))
        at_is_s1 = np.array([x.lower() <= y.lower() for x, y in zip(s1, s2)])
        subj_is_s1 = (df.subject_is.to_numpy() == 1)
        self.y = (at_is_s1 == subj_is_s1).astype("float32")
        self.d = df.directed.to_numpy().astype("float32")
        # 0 patient-side, 1 agent-side, 2 symmetric
        pol = df.pol.to_numpy()
        self.pol = np.where(self.d == 0, 2, np.where(pol > 0, 1, 0)).astype("int64")
        self.w = np.where(df.quality.to_numpy() == "BOTH", 1.0, 0.5).astype("float32")
        self.tok, self.ml = tok, max_len
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return self.a[i], self.b[i], self.y[i], self.pol[i], self.w[i], self.d[i]
    def collate(self, batch):
        a, b, y, p, w, d = zip(*batch)
        e = self.tok(list(a), list(b), truncation="only_second", max_length=self.ml,
                     padding=True, return_tensors="pt")
        e["labels_dir"] = torch.tensor(y); e["pol"] = torch.tensor(p)
        e["w"] = torch.tensor(w); e["directed"] = torch.tensor(d)
        return e


# ------------------------------------------------------------------ model
def span_pool(hidden, ids, marker):
    """Mean-pool the tokens strictly between the first two `marker` tokens of each row."""
    B, T, H = hidden.shape
    m = (ids == marker)
    out = hidden.new_zeros(B, H); ok = hidden.new_zeros(B)
    idx = [torch.nonzero(m[i], as_tuple=False).flatten() for i in range(B)]
    for i, p in enumerate(idx):
        if p.numel() >= 2:
            lo, hi = int(p[0]) + 1, int(p[1])
            if hi > lo:
                out[i] = hidden[i, lo:hi].mean(0); ok[i] = 1.0
        elif p.numel() == 1:                       # marker pair truncated away
            lo = int(p[0]) + 1
            if lo < T:
                out[i] = hidden[i, lo:min(lo + 8, T)].mean(0); ok[i] = 1.0
    return out, ok


class DirectionHead(nn.Module):
    """Two scores with deliberately opposite symmetry.

        direction    s = g(hA,hB,.) - g(hB,hA,.)     exactly ANTIsymmetric
        directedness u = h(hA,hB,.) + h(hB,hA,.)     exactly SYMMETRIC

    The pairing is the whole point. "Which taxon is the subject" MUST flip when the two
    arguments are swapped; "does this relation have a direction at all" must NOT. Both are
    algebraic properties of the head, true at every parameter setting, before and during
    training.

    A single unconstrained softmax over {rightward, leftward, undirected} -- the standard
    formulation -- guarantees neither: swapping the arguments can move probability mass
    between rightward and leftward AND into or out of undirected, so the model can hold two
    mutually inconsistent beliefs about the same pair.

    Relation polarity enters as a 3-valued embedding: patient-side, agent-side, symmetric.
    """
    def __init__(self, hid=768, pol_dim=16, inner=256):
        super().__init__()
        self.pol_emb = nn.Embedding(3, pol_dim)          # 0 patient, 1 agent, 2 symmetric
        self.g = nn.Sequential(nn.Linear(3 * hid + pol_dim, inner), nn.GELU(),
                               nn.Dropout(0.1), nn.Linear(inner, 1))
        self.h = nn.Sequential(nn.Linear(3 * hid + pol_dim, inner), nn.GELU(),
                               nn.Dropout(0.1), nn.Linear(inner, 1))

    def forward(self, hA, hB, cls, pol):
        pe = self.pol_emb(pol)
        ab = torch.cat([hA, hB, cls, pe], -1)
        ba = torch.cat([hB, hA, cls, pe], -1)
        s_dir = (self.g(ab) - self.g(ba)).squeeze(-1)    # antisymmetric
        s_und = (self.h(ab) + self.h(ba)).squeeze(-1)    # symmetric
        return s_dir, s_und

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class Student(nn.Module):
    def __init__(self, enc=ENC, detach_dir=False):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(enc, num_labels=2)
        self.dir = DirectionHead(self.bert.config.hidden_size)
        self.detach_dir = detach_dir
    def encode(self, input_ids, attention_mask, token_type_ids=None):
        kw = dict(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        if token_type_ids is not None: kw["token_type_ids"] = token_type_ids
        o = self.bert(**kw)
        return o.logits, o.hidden_states[-1]
    def forward(self, input_ids, attention_mask, token_type_ids=None, pol=None):
        logits, hid = self.encode(input_ids, attention_mask, token_type_ids)
        h = hid.detach() if self.detach_dir else hid
        hA, okA = span_pool(h, input_ids, AT_ID)
        hB, okB = span_pool(h, input_ids, HASH_ID)
        if pol is None:
            return logits, None, None, (okA * okB)
        s_dir, s_und = self.dir(hA, hB, h[:, 0], pol)
        return logits, s_dir, s_und, (okA * okB)


# ------------------------------------------------------------------ train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir-data", default=str(REPO / "data/training/distill/direction_train_v3.parquet"))
    ap.add_argument("--bin-data", default=str(REPO / "data/training/distill/v4_species_train.csv"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.5, help="direction loss weight; 0 = binary-only control")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--detach-dir", action="store_true", help="frozen-trunk probe: no direction gradient into the encoder")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="pair-grouped held-out fraction of the binary data, used ONLY to pick "
                         "the operating point. 0 reproduces the old no-holdout behaviour.")
    ap.add_argument("--precision-floor", type=float, default=0.90,
                    help="record the lowest threshold on dev whose precision is at least this; "
                         "this is the operating point the precision-first policy asks for.")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(ENC)

    bdf = pd.read_csv(a.bin_data)
    vdf = None
    if a.val_frac > 0:
        # group on the unordered taxon pair so no pair spans train and dev
        pk = [" || ".join(sorted([str(x).lower(), str(y).lower()]))
              for x, y in zip(bdf.source_species, bdf.target_species)]
        bdf = bdf.assign(_pk=pk)
        pairs = bdf._pk.drop_duplicates().sample(frac=1.0, random_state=a.seed).tolist()
        devp = set(pairs[:int(len(pairs) * a.val_frac)])
        vdf = bdf[bdf._pk.isin(devp)].reset_index(drop=True)
        bdf = bdf[~bdf._pk.isin(devp)].reset_index(drop=True)
        print(f"binary split: train {len(bdf)} / dev {len(vdf)} "
              f"({len(devp)} dev pairs, pair-grouped)", flush=True)
    bds = BinaryDS(bdf, tok, a.max_len)
    bdl = DataLoader(bds, batch_size=a.bs, shuffle=True, collate_fn=bds.collate,
                     num_workers=4, drop_last=True)

    ddl = None
    if a.alpha > 0:
        ddf = pd.read_parquet(a.dir_data)
        ddf = ddf[ddf.split == "train"].reset_index(drop=True)
        print(f"direction rows {len(ddf):,}  directed {int(ddf.directed.sum()):,} "
              f"undirected {int((ddf.directed==0).sum()):,}", flush=True)
        dds = DirectionDS(ddf, tok, a.max_len)
        ddl = DataLoader(dds, batch_size=a.bs, shuffle=True, collate_fn=dds.collate,
                         num_workers=4, drop_last=True)
        print(f"direction train rows {len(dds)}  P(@ subject)={dds.y.mean():.4f}")
    print(f"binary train rows {len(bds)}  pos rate {bdf.label.mean():.4f}")

    m = Student(detach_dir=a.detach_dir).to(dev)
    print(f"direction head params: {m.dir.n_params():,}  "
          f"(+{100*m.dir.n_params()/sum(p.numel() for p in m.bert.parameters()):.4f}% of encoder)")

    steps = len(bdl) * a.epochs
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler("cuda")
    bce = nn.BCEWithLogitsLoss(reduction="none")
    ce = nn.CrossEntropyLoss()

    t0 = time.time()
    for ep in range(a.epochs):
        m.train(); dit = iter(ddl) if ddl else None
        run_b = run_d = nb = 0
        for i, batch in enumerate(bdl):
            batch = {k: v.to(dev) for k, v in batch.items()}
            yb = batch.pop("labels_bin")
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _, _, _ = m(batch["input_ids"], batch["attention_mask"],
                                    batch.get("token_type_ids"))
                loss_b = ce(logits.float(), yb.long())
                loss = loss_b; ld = torch.tensor(0.0)
                if dit is not None:
                    try: db = next(dit)
                    except StopIteration:
                        dit = iter(ddl); db = next(dit)
                    db = {k: v.to(dev) for k, v in db.items()}
                    yd, pol, w = db.pop("labels_dir"), db.pop("pol"), db.pop("w")
                    dd = db.pop("directed")
                    _, s_dir, s_und, ok = m(db["input_ids"], db["attention_mask"],
                                            db.get("token_type_ids"), pol=pol)
                    base = (ok > 0).float() * w
                    # direction loss only on rows that HAVE a direction
                    msk = base * dd
                    ld = (bce(s_dir.float(), yd) * msk).sum() / msk.sum().clamp(min=1)
                    # directedness loss on every row
                    lu = (bce(s_und.float(), dd) * base).sum() / base.sum().clamp(min=1)
                    ld = ld + lu
                    loss = loss + a.alpha * ld
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            run_b += float(loss_b); run_d += float(ld); nb += 1
            if i % 200 == 0:
                print(f"ep{ep} {i}/{len(bdl)} bin {run_b/nb:.4f} dir {run_d/nb:.4f} "
                      f"{time.time()-t0:.0f}s", flush=True)
                run_b = run_d = nb = 0

    # ---- operating point, chosen on the held-out dev split and never on any test set
    thr_f1, thr_prec, dev_auprc = 0.5, None, None
    if vdf is not None and len(vdf):
        from sklearn.metrics import average_precision_score as _ap, precision_score as _p, \
            recall_score as _r, f1_score as _f1
        m.eval()
        vds = BinaryDS(vdf, tok, a.max_len)
        vdl = DataLoader(vds, batch_size=64, shuffle=False, collate_fn=vds.collate, num_workers=2)
        P, Y = [], []
        with torch.no_grad():
            for b in vdl:
                yb = b.pop("labels_bin"); b = {k: v.to(dev) for k, v in b.items()}
                lg, _, _, _ = m(b["input_ids"], b["attention_mask"], b.get("token_type_ids"))
                P.extend(torch.softmax(lg.float(), -1)[:, 1].cpu().numpy()); Y.extend(yb.numpy())
        P, Y = np.array(P), np.array(Y).astype(int)
        dev_auprc = float(_ap(Y, P))
        grid = np.arange(0.01, 1.00, 0.01)
        thr_f1 = float(max(((_f1(Y, (P >= t).astype(int), zero_division=0), float(t)) for t in grid))[1])
        ok = [float(t) for t in grid if _p(Y, (P >= t).astype(int), zero_division=0) >= a.precision_floor]
        thr_prec = min(ok) if ok else None
        print(f"dev AUPRC {dev_auprc:.4f} | thr@maxF1 {thr_f1:.2f} | "
              f"thr@P>={a.precision_floor:.2f} {thr_prec}", flush=True)
        m.train()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    m.bert.save_pretrained(out); tok.save_pretrained(out)
    torch.save(m.dir.state_dict(), out / "direction_head.pt")
    json.dump({"input_format": "mark_canon", "encoder": ENC, "head_version": "v3_directedness", "max_len": a.max_len,
               "epochs": a.epochs, "lr": a.lr, "seed": a.seed, "alpha": a.alpha,
               "detach_dir": a.detach_dir,
               "threshold_dev": thr_f1,
               "threshold_precision_floor": thr_prec,
               "precision_floor": a.precision_floor,
               "dev_auprc": dev_auprc,
               "val_frac": a.val_frac,
               "direction_head_params": m.dir.n_params(),
               "bin_data": a.bin_data, "dir_data": a.dir_data if a.alpha > 0 else None},
              open(out / "student_config.json", "w"), indent=2)
    print("saved", out, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
