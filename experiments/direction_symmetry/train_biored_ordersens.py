#!/usr/bin/env python3
"""Train OUR direction architecture on BioRED, to compare on their benchmark.

Identical to experiments/multitask/train_direction_v3.py in every structural respect:
  * canonical input   -- the two entities sorted alphabetically and marked @first@ / #second#
                         in the passage, so swapping them gives a byte-identical input
  * antisymmetric direction score   s = g(A,B) - g(B,A)
  * symmetric directedness score    u = h(A,B) + h(B,A)

One adaptation: the head's categorical side takes BioRED's relation TYPE where the species model
took relation polarity. Biomedical relations have no agent/patient polarity lexicon; the slot is
the same, the vocabulary differs.

No binary interaction head: every BioRED row here already has a relation, so that task does not
exist on this data.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

ENC = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
AT_ID, HASH_ID = 36, 7
RELS = ["Association", "Bind", "Comparison", "Conversion", "Cotreatment",
        "Drug_Interaction", "Negative_Correlation", "Positive_Correlation"]
R2I = {r: i for i, r in enumerate(RELS)}


def mark(passage, a, b):
    """@a@ ... #b# in place, first occurrence, case-insensitive."""
    import re
    out = passage
    for ent, o, c in ((a, "@", "@"), (b, "#", "#")):
        m = re.search(re.escape(ent), out, re.I)
        if m:
            out = out[:m.start()] + o + " " + m.group(0) + " " + c + out[m.end():]
    return out


def build(df):
    """ORDER-SENSITIVE arm: entities stay in stored order and the markers denote ROLE
    (@ = the designated source, # = the target), which is standard practice. Exchanging the
    two arguments therefore changes the encoder input, exactly as it does for BioREDirect."""
    s1 = df.s1.astype(str).tolist(); s2 = df.s2.astype(str).tolist()
    A, B = [], []
    for x, y, p in zip(s1, s2, df.passage.astype(str)):
        A.append(f"{x} [SEP] {y}")
        B.append(mark(p, x, y))
    return A, B, np.ones(len(s1), dtype=bool)


class DS(Dataset):
    def __init__(self, df, tok, max_len=256):
        self.a, self.b, at = build(df)
        subj_is_s1 = (df.subject_is.to_numpy() == 1)
        self.y = (at == subj_is_s1).astype("float32")      # is the @ entity the subject?
        self.d = df.directed.to_numpy().astype("float32")
        self.r = np.array([R2I.get(r, 0) for r in df.rel], dtype="int64")
        self.tok, self.ml = tok, max_len
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.a[i], self.b[i], self.y[i], self.r[i], self.d[i]
    def collate(self, batch):
        a, b, y, r, d = zip(*batch)
        e = self.tok(list(a), list(b), truncation="only_second", max_length=self.ml,
                     padding=True, return_tensors="pt")
        e["y"] = torch.tensor(y); e["r"] = torch.tensor(r); e["d"] = torch.tensor(d)
        return e


def span_pool(h, ids, marker):
    B, T, H = h.shape
    out = h.new_zeros(B, H); ok = h.new_zeros(B)
    for i in range(B):
        p = torch.nonzero(ids[i] == marker, as_tuple=False).flatten()
        if p.numel() >= 2 and int(p[1]) > int(p[0]) + 1:
            out[i] = h[i, int(p[0]) + 1:int(p[1])].mean(0); ok[i] = 1.0
        elif p.numel() >= 1:
            lo = int(p[0]) + 1
            if lo < T: out[i] = h[i, lo:min(lo + 8, T)].mean(0); ok[i] = 1.0
    return out, ok


class Head(nn.Module):
    """UNCONSTRAINED arm: one 3-way classifier over {rightward, leftward, undirected}.

    This is the BioREDirect formulation. Nothing ties P(rightward) to P(leftward) under an
    argument swap, so the model has to learn the equivariance from data if it learns it at all.
    """
    def __init__(self, hid=768, rel_dim=16, inner=256):
        super().__init__()
        self.rel_emb = nn.Embedding(len(RELS), rel_dim)
        self.g = nn.Sequential(nn.Linear(3*hid+rel_dim, inner), nn.GELU(), nn.Dropout(0.1), nn.Linear(inner, 3))
    def forward(self, hA, hB, cls, r):
        lg = self.g(torch.cat([hA, hB, cls, self.rel_emb(r)], -1))
        s_dir = lg[:, 0] - lg[:, 1]                                   # rightward vs leftward
        s_und = torch.logsumexp(lg[:, :2], dim=-1) - lg[:, 2]         # directed vs undirected
        return s_dir, s_und


class Model(nn.Module):
    def __init__(self, enc=ENC):
        super().__init__()
        self.bert = AutoModel.from_pretrained(enc)
        self.head = Head(self.bert.config.hidden_size)
    def forward(self, input_ids, attention_mask, token_type_ids=None, r=None):
        kw = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None: kw["token_type_ids"] = token_type_ids
        h = self.bert(**kw).last_hidden_state
        hA, okA = span_pool(h, input_ids, AT_ID)
        hB, okB = span_pool(h, input_ids, HASH_ID)
        sd, su = self.head(hA, hB, h[:, 0], r)
        return sd, su, okA * okB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(ENC)

    tr = pd.read_csv("biored_train_ours.csv")
    ds = DS(tr, tok); dl = DataLoader(ds, batch_size=a.bs, shuffle=True, collate_fn=ds.collate,
                                      num_workers=4, drop_last=True)
    m = Model().to(dev)
    print(f"train {len(ds):,}  directed {int(ds.d.sum()):,}  P(@ subject)={ds.y.mean():.4f}", flush=True)
    steps = len(dl)*a.epochs
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.06*steps), steps)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    scaler = torch.amp.GradScaler("cuda")
    for ep in range(a.epochs):
        m.train(); run = nb = 0
        for i, b in enumerate(dl):
            b = {k: v.to(dev) for k, v in b.items()}
            y, r, d = b.pop("y"), b.pop("r"), b.pop("d")
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                sd, su, ok = m(b["input_ids"], b["attention_mask"], b.get("token_type_ids"), r=r)
                base = (ok > 0).float()
                ld = (bce(sd.float(), y)*base*d).sum()/ (base*d).sum().clamp(min=1)
                lu = (bce(su.float(), d)*base).sum()/ base.sum().clamp(min=1)
                loss = ld + lu
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            run += float(loss); nb += 1
            if i % 50 == 0:
                print(f"  ep{ep} {i}/{len(dl)} loss {run/max(nb,1):.4f}", flush=True); run = nb = 0
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), out/"model.pt")
    json.dump({"encoder": ENC, "epochs": a.epochs, "lr": a.lr, "seed": a.seed,
               "rels": RELS, "arch": "canonical input + antisymmetric direction + symmetric directedness"},
              open(out/"config.json", "w"), indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
