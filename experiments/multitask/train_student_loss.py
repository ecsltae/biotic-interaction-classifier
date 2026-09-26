#!/usr/bin/env python3
"""Loss-shaping variants of the cross-encoder student.

Identical to train_student.py in data handling, split, schedule and threshold
derivation. The only difference is the training objective, selected with --loss:

  ce                 plain cross-entropy (reproduces train_student.py)
  weighted           class-weighted CE, weight on the positive class = --pos-weight
  focal              focal loss, gamma = --gamma, alpha = --alpha (weight on class 1)
  ls                 symmetric label smoothing, eps = --eps
  ls_neg             ASYMMETRIC label smoothing: smoothing applied to NEGATIVE
                     targets only. Motivated by the measured 34.5% species-level
                     flip rate among teacher NO rows -- the noise is one-sided, so
                     the smoothing should be too. eps_neg = --eps
  gce                generalised cross-entropy (Zhang & Sabuncu), q = --q; a
                     noise-robust loss that interpolates CE (q->0) and MAE (q=1)

Selection metric is dev AUPRC on a pair-grouped held-out split, exactly as in the
baseline trainer, so the objective is the only moving part.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup, set_seed)
from sklearn.metrics import average_precision_score, f1_score

REPO = Path(__file__).resolve().parents[2]


class DS(Dataset):
    def __init__(self, df, tok, max_len=256):
        self.q = [f"{a} [SEP] {r} [SEP] {b}" for a, r, b in
                  zip(df.source_species, df.interaction_type, df.target_species)]
        self.p = df.text.astype(str).tolist()
        self.y = df.label.astype(int).tolist()
        self.tok, self.ml = tok, max_len

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        e = self.tok(self.q[i], self.p[i], truncation="only_second", max_length=self.ml,
                     padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in e.items()} | {"labels": torch.tensor(self.y[i])}


def compute_loss(logits, y, a):
    """logits (B,2), y (B,) int64.

    --pw-mult, when != 1.0, multiplies the per-sample loss of POSITIVE examples by
    that factor on top of whatever base loss is selected. It lets a positive-weight
    arm be combined with a negative-smoothing arm (e.g. ls_neg + pw-mult), which the
    `weighted` loss on its own cannot express.
    """
    pw = getattr(a, "pw_mult", 1.0)

    def reduce(per_sample):
        if pw == 1.0:
            return per_sample.mean()
        w = torch.where(y == 1, float(pw), 1.0).to(per_sample.dtype)
        return (per_sample * w).sum() / w.sum()

    if a.loss == "ce":
        if pw != 1.0:
            return reduce(F.cross_entropy(logits, y, reduction="none"))
        return F.cross_entropy(logits, y)

    if a.loss == "weighted":
        w = torch.tensor([1.0, a.pos_weight], device=logits.device, dtype=logits.dtype)
        return F.cross_entropy(logits, y, weight=w)

    if a.loss == "focal":
        logp = F.log_softmax(logits, -1)
        lp = logp.gather(1, y.unsqueeze(1)).squeeze(1)          # log p_t
        pt = lp.exp()
        alpha = torch.where(y == 1, a.alpha, 1.0 - a.alpha).to(logits.dtype)
        return reduce(-alpha * (1.0 - pt).pow(a.gamma) * lp)

    if a.loss in ("ls", "ls_neg"):
        # soft target on the TRUE class; the rest goes to the other class
        eps = torch.full_like(y, 0.0, dtype=logits.dtype).fill_(a.eps)
        if a.loss == "ls_neg":
            eps = torch.where(y == 0, eps, torch.zeros_like(eps))  # smooth negatives only
        logp = F.log_softmax(logits, -1)
        tgt = torch.zeros_like(logp)
        tgt.scatter_(1, y.unsqueeze(1), 1.0)
        tgt = tgt * (1.0 - eps).unsqueeze(1) + (1.0 - tgt) * eps.unsqueeze(1)
        return reduce(-(tgt * logp).sum(1))

    if a.loss == "gce":
        p = F.softmax(logits, -1).gather(1, y.unsqueeze(1)).squeeze(1).clamp_min(1e-7)
        return reduce((1.0 - p.pow(a.q)) / a.q)

    raise ValueError(a.loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/training/distill/v3_combined_train.csv")
    ap.add_argument("--encoder", default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--loss", default="ce",
                    choices=["ce", "weighted", "focal", "ls", "ls_neg", "gce"])
    ap.add_argument("--pos-weight", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--q", type=float, default=0.7)
    ap.add_argument("--pw-mult", type=float, default=1.0,
                    help="extra multiplier on the per-sample loss of positives; "
                         "combines with any base loss. 1.0 = off.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.set_float32_matmul_precision("high")
    set_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(REPO / a.data)
    # group split on taxon pair so a pair never spans train and dev (identical to baseline)
    df["_pk"] = [tuple(sorted([str(x).lower(), str(y).lower()]))
                 for x, y in zip(df.source_species, df.target_species)]
    rng = np.random.RandomState(a.seed)
    pairs = df._pk.unique(); rng.shuffle(pairs)
    devp = set(pairs[:int(len(pairs) * a.val_frac)])
    tr, va = df[~df._pk.isin(devp)], df[df._pk.isin(devp)]
    print(f"train {len(tr)} / dev {len(va)}  (pair-grouped split, {len(devp)} dev pairs)", flush=True)
    print(f"loss={a.loss} pos_weight={a.pos_weight} gamma={a.gamma} alpha={a.alpha} "
          f"eps={a.eps} q={a.q} pw_mult={a.pw_mult} seed={a.seed}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.encoder)
    model = AutoModelForSequenceClassification.from_pretrained(a.encoder, num_labels=2).to(dev)
    tl = DataLoader(DS(tr, tok), batch_size=a.batch_size, shuffle=True, num_workers=2)
    vl = DataLoader(DS(va, tok), batch_size=a.batch_size * 2, num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * len(tl) * a.epochs), len(tl) * a.epochs)

    best, best_state, best_PY, hist = -1, None, None, []
    for ep in range(a.epochs):
        model.train(); t0 = time.time()
        for b in tl:
            y = b.pop("labels").to(dev)
            b = {k: v.to(dev) for k, v in b.items()}
            loss = compute_loss(model(**b).logits.float(), y, a)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
        model.eval(); P, Y = [], []
        with torch.no_grad():
            for b in vl:
                y = b.pop("labels"); b = {k: v.to(dev) for k, v in b.items()}
                P.extend(torch.softmax(model(**b).logits.float(), -1)[:, 1].cpu().numpy())
                Y.extend(y.numpy())
        ap_ = average_precision_score(Y, P)
        hist.append({"epoch": ep + 1, "dev_auprc": float(ap_)})
        print(f"  ep{ep+1} dev AUPRC {ap_:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if ap_ > best:
            best = ap_
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_PY = (np.array(P), np.array(Y))
    model.load_state_dict(best_state)
    out = REPO / a.out; out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out); tok.save_pretrained(out)
    P, Y = best_PY
    grid = np.arange(0.01, 1.0, 0.01)
    t = float(grid[int(np.argmax([f1_score(Y, (P >= g).astype(int), zero_division=0) for g in grid]))])
    (out / "student_config.json").write_text(json.dumps({
        "threshold_dev": t, "best_dev_auprc": best, "seed": a.seed, "epochs": a.epochs,
        "data": a.data, "train_pos_rate": float(df.label.mean()),
        "loss": a.loss, "pos_weight": a.pos_weight, "gamma": a.gamma, "alpha": a.alpha,
        "eps": a.eps, "q": a.q, "pw_mult": a.pw_mult, "input_format": "triple",
        "git": subprocess.run(["git", "describe", "--always", "--dirty"], cwd=REPO,
                              capture_output=True, text=True).stdout.strip(),
        "history": hist}, indent=2))
    np.save(out / "dev_scores.npy", np.stack([P, Y]))
    print(f"saved {out}  dev_t={t:.3f}  dev AUPRC {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
