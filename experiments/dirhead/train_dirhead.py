#!/usr/bin/env python3
"""Train the joint binary + direction cross-encoder.

  segment A = the two taxa sorted alphabetically   (so the binary task is order-invariant
              at the input rather than having to learn the invariance from 5,591 reversal
              rows -- that is the whole answer to the symmetry conflict)
  segment B = the passage with the two mentions marked

  binary head     : symmetric features only  ([CLS], hA+hB, hA*hB)
  direction head  : odd under swapping the two spans, by construction

Direction supervision comes from experiments/dirhead/direction_pool.parquet.  The split is
TAXON-DISJOINT: distinct taxon names are hashed into ten buckets, the test rows use only
bucket-0 taxa and the dev rows only bucket-1 taxa, so no taxon the model was trained on
appears at test time.  That is deliberate -- the distant labels are a taxon-role lookup, and
a taxon-disjoint test is the only way to see whether the head has learned anything else.
All taxa and all sentences that appear in the human direction gold are excluded from
training outright.
"""
import argparse, json, hashlib, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from sklearn.metrics import average_precision_score, f1_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO / "scripts"))
import dirlib, data as D
from model import DirModel

BUCKETS = 10
HOLD = {0, 1, 2}          # taxa reserved for evaluation


def bucket(name):
    return int(hashlib.md5(str(name).lower().strip().encode()).hexdigest(), 16) % BUCKETS


def direction_frame(source, max_fr, max_sym, seed, split="taxon"):
    p = pd.read_parquet(HERE / "direction_pool.parquet")
    g = dirlib.load_gold()
    gold_tax = set()
    for _, r in g.iterrows():
        gold_tax.add(str(r.species1).lower().strip()); gold_tax.add(str(r.species2).lower().strip())
    gold_sent = set(str(s) for s in g.sentence)
    n0 = len(p)
    p = p[~p.s1.str.lower().str.strip().isin(gold_tax) & ~p.s2.str.lower().str.strip().isin(gold_tax)]
    p = p[~p.passage.astype(str).isin(gold_sent)]
    print(f"direction pool {n0} -> {len(p)} after removing gold taxa/sentences", flush=True)

    if source == "consensus":
        # only the rows where the taxon-role rule and the passage-syntax rule agree.
        # measured on a 40-row adjudication this teacher is ~0.83 accurate against ~0.76
        # for the rule alone, at 46% of the rule's volume.
        m = (p.rule_role_canon != 0) & (p.rule_role_canon == p.syn_canon)
        fr = p[m].copy(); fr["y_dir"] = fr.rule_role_canon - 1
    else:
        col = {"rule": "rule_role_canon", "syn": "syn_canon", "globi": "globi_canon"}[source]
        fr = p[p[col] != 0].copy()
        fr["y_dir"] = fr[col] - 1
    sym = p[p.sym].copy()
    sym["y_dir"] = 2
    rng = np.random.RandomState(seed)
    if max_fr and len(fr) > max_fr:
        fr = fr.iloc[rng.choice(len(fr), max_fr, replace=False)]
    if max_sym and len(sym) > max_sym:
        sym = sym.iloc[rng.choice(len(sym), max_sym, replace=False)]
    d = pd.concat([fr, sym], ignore_index=True)
    d["b1"] = [bucket(x) for x in d.s1]
    d["b2"] = [bucket(x) for x in d.s2]
    d = d.rename(columns={"passage": "text"})
    d["y_bin"] = -100
    if split == "taxon":
        ev = d[d.b1.isin(HOLD) & d.b2.isin(HOLD)]
        tr = d[~d.b1.isin(HOLD) & ~d.b2.isin(HOLD)]
    else:                                     # random split: the memorisation control
        h = np.array([bucket("R" + str(t)) for t in d.triplet_key])
        ev, tr = d[np.isin(h, list(HOLD))], d[~np.isin(h, list(HOLD))]
    half = np.array([bucket("H" + str(t)) for t in ev.triplet_key]) < 5
    # a second held-out slice whose taxa the model HAS seen: the difference between this
    # and the taxon-disjoint slice is exactly how much of the head is name memorisation
    seen = np.array([bucket("S" + str(t)) for t in tr.doc_id]) == 0
    return (tr[~seen].reset_index(drop=True), ev[half].reset_index(drop=True),
            ev[~half].reset_index(drop=True), tr[seen].reset_index(drop=True))


def binary_frame(path, seed, val_frac=0.1):
    df = pd.read_csv(REPO / path)
    df = df.rename(columns={"source_species": "s1", "target_species": "s2",
                            "interaction_type": "rel", "label": "y_bin"})
    df["y_dir"] = -100
    df["_pk"] = [tuple(sorted([str(x).lower(), str(y).lower()])) for x, y in zip(df.s1, df.s2)]
    rng = np.random.RandomState(seed)
    pairs = df._pk.unique(); rng.shuffle(pairs)
    devp = set(pairs[:int(len(pairs) * val_frac)])
    return df[~df._pk.isin(devp)].reset_index(drop=True), df[df._pk.isin(devp)].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    ap.add_argument("--binary-data", default="data/training/distill/v4_species_train.csv")
    ap.add_argument("--dir-source", default="rule", choices=["rule", "syn", "globi", "consensus"])
    ap.add_argument("--max-fr", type=int, default=0)
    ap.add_argument("--max-sym", type=int, default=30000)
    ap.add_argument("--split", default="taxon", choices=["taxon", "random"],
                    help="taxon = no taxon shared between train and eval (the real test); "
                         "random = ordinary split (the name-memorisation control)")
    ap.add_argument("--alpha", type=float, default=1.0, help="weight on the direction loss")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--no-binary", action="store_true", help="direction only (ablation)")
    ap.add_argument("--no-direction", action="store_true", help="binary only (control arm)")
    ap.add_argument("--names-only", action="store_true", help="replace the passage by the two names")
    ap.add_argument("--freeze-trunk", action="store_true")
    ap.add_argument("--no-augment", action="store_true", help="turn off marker-swap augmentation")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.set_float32_matmul_precision("high"); set_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.encoder)

    frames_tr, frames_de = [], []
    if not a.no_binary:
        btr, bde = binary_frame(a.binary_data, a.seed)
        print(f"binary  train {len(btr)} dev {len(bde)}", flush=True)
        frames_tr.append(btr); frames_de.append(bde)
    if not a.no_direction:
        dtr, dde, dte, dseen = direction_frame(a.dir_source, a.max_fr, a.max_sym, a.seed, a.split)
        print(f"direction train {len(dtr)} dev {len(dde)} test {len(dte)}  "
              f"(taxon-disjoint) source={a.dir_source}", flush=True)
        frames_tr.append(dtr); frames_de.append(dde)
        (REPO / a.out).mkdir(parents=True, exist_ok=True)
        dte.to_parquet(REPO / a.out / "dir_test.parquet")
        dde.to_parquet(REPO / a.out / "dir_dev.parquet")
        dseen.to_parquet(REPO / a.out / "dir_test_seen.parquet")
        print(f"  held-out-but-seen-taxa slice: {len(dseen)}", flush=True)
    cols = ["s1", "s2", "rel", "text", "y_bin", "y_dir"]
    tr = pd.concat([f[cols] for f in frames_tr], ignore_index=True)
    de = pd.concat([f[cols] for f in frames_de], ignore_index=True)
    print(f"TOTAL train {len(tr)} dev {len(de)}", flush=True)

    model = DirModel(a.encoder).to(dev)
    if a.freeze_trunk:
        for p in model.enc.parameters():
            p.requires_grad = False
    cf = D.collate(tok)
    tl = DataLoader(D.Joint(tr, tok, a.max_len, augment=not a.no_augment,
                            names_only=a.names_only, seed=a.seed),
                    batch_size=a.batch_size, shuffle=True, num_workers=4, collate_fn=cf)
    vl = DataLoader(D.Joint(de, tok, a.max_len, augment=False, names_only=a.names_only),
                    batch_size=a.batch_size * 2, num_workers=4, collate_fn=cf)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * len(tl) * a.epochs), len(tl) * a.epochs)
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
    scaler = torch.amp.GradScaler("cuda")

    best, best_state, hist = -1e9, None, []
    for ep in range(a.epochs):
        model.train(); t0 = time.time()
        for b in tl:
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            yb, yd = b.pop("y_bin"), b.pop("y_dir"); b.pop("ok")
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                bl, dl = model(**b)
                loss = 0.0
                if (yb != -100).any():
                    loss = loss + ce(bl.float(), yb)
                if (yd != -100).any():
                    loss = loss + a.alpha * ce(dl.float(), yd)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        model.eval(); PB, YB, PD, YD = [], [], [], []
        with torch.no_grad():
            for b in vl:
                b = {k: v.to(dev) for k, v in b.items()}
                yb, yd = b.pop("y_bin"), b.pop("y_dir"); b.pop("ok")
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    bl, dl = model(**b)
                m = yb != -100
                if m.any():
                    PB.extend(torch.softmax(bl[m].float(), -1)[:, 1].cpu().numpy()); YB.extend(yb[m].cpu().numpy())
                m = yd != -100
                if m.any():
                    PD.extend(dl[m].float().argmax(-1).cpu().numpy()); YD.extend(yd[m].cpu().numpy())
        auprc = float(average_precision_score(YB, PB)) if YB else float("nan")
        dacc = float(np.mean(np.array(PD) == np.array(YD))) if YD else float("nan")
        obj = (0 if np.isnan(auprc) else auprc) + (0 if np.isnan(dacc) else dacc)
        hist.append(dict(epoch=ep + 1, dev_auprc=auprc, dev_dir_acc=dacc))
        print(f"  ep{ep+1} dev AUPRC {auprc:.4f}  dev dir-acc {dacc:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if obj > best:
            best, best_state = obj, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    out = REPO / a.out; out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "pytorch_model.bin")
    tok.save_pretrained(out)
    t = 0.5
    if YB:
        grid = np.arange(0.01, 1.0, 0.01)
        t = float(grid[int(np.argmax([f1_score(YB, (np.array(PB) >= g).astype(int),
                                               zero_division=0) for g in grid]))])
    json.dump(dict(vars(a) | {"threshold_dev": t, "input_format": "mark_canon",
                              "history": hist}), open(out / "student_config.json", "w"), indent=2)
    print(f"saved {out}  dev_t={t:.3f}", flush=True)


if __name__ == "__main__":
    main()
