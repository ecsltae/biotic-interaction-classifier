#!/usr/bin/env python3
"""V2 multi-task trainer.

Differs from train.py (frozen as V1) in the following ways, each traceable to a
finding in docs/AUDIT_2026-08-19.md:

  A7   --seed, transformers.set_seed, and a seeded DataLoader generator. V1 had
       none, and two identical V1 commands scored 0.8743 vs 0.8346.
  B10  KD loss puts the teacher in LOGIT space before applying temperature. V1
       divided only the student by T, which sharpens the teacher instead of
       softening both. Loss is now summed over examples and divided by the batch
       size once, rather than divided by the number of active branches.
  B11  Checkpoint selection on validation AUPRC. V1 selected on cls_f1 at a fixed
       0.5, whose value ranged 0.379-0.410 across all configs and correlated with
       EP-relax F1 at Spearman 0.027 (p=0.92).
  B1   The training temperature is recorded in the checkpoint so inference can
       divide logits by it. V1 trained at T=2 and read at T=1 in all 17 places.
  B2   Closed-form prior-shift threshold and a dev-derived threshold are both
       computed and stored with the checkpoint. No threshold is ever taken from
       a test set.
  B4   --pair-conditioning encodes (query, sentence) as a segment pair so the
       model is asked about a specific taxon pair rather than about the sentence.
  B12  TF32 matmul, measured at 3.5x on this A100 with zero label flips.
  C5   A missing soft-label file is fatal, not a silent fallback to hard CE.
  C6   Full provenance: git sha, seed, resolved data path + sha256, row counts,
       train positive rate, soft-label coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import set_seed

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from model import MultiTaskBiomedBERT          # noqa: E402
from data import load_multitask_splits          # noqa: E402
from eval.core import threshold_from_prior      # noqa: E402

ENCODER = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
REPO = Path(__file__).resolve().parents[2]


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.run(["git", "describe", "--always", "--dirty"], cwd=REPO,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def code_provenance() -> dict:
    """File-level hashes of the code that actually ran.

    A git sha does not identify the executed code when the files defining
    training are untracked or modified — which was true of train_v2.py,
    sweep_v2.py and data.py for the first 48-run sweep.
    """
    out = {}
    for rel in ("experiments/multitask/train_v2.py", "experiments/multitask/sweep_v2.py",
                "experiments/multitask/data.py", "experiments/multitask/model.py",
                "src/eval/core.py"):
        f = REPO / rel
        out[rel] = sha256(f)[:16] if f.exists() else None
    try:
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()
        out["_working_tree_dirty_paths"] = len([l for l in dirty.splitlines() if l.strip()])
    except Exception:
        pass
    return out


# ── loss ──────────────────────────────────────────────────────────────────

def cls_loss_fn(logits, hard_labels, soft_labels, temperature, kd_mode="fixed"):
    """Soft KD where a teacher probability exists, hard CE otherwise.

    kd_mode="fixed"  : teacher converted to logits, both sides divided by T.
    kd_mode="v1_bug" : V1 behaviour, kept so the bug's effect is measurable
                       rather than assumed.
    """
    has_soft = soft_labels >= 0
    total = torch.tensor(0.0, device=logits.device)
    n_ex = 0

    if has_soft.any():
        sl = soft_labels[has_soft].clamp(1e-6, 1 - 1e-6)
        if kd_mode == "v1_bug":
            teacher = torch.stack([1 - sl, sl], dim=1)
        else:
            # p -> logit -> temperature-scaled softmax, the correct KD target
            t_logit = torch.stack([torch.zeros_like(sl), torch.log(sl / (1 - sl))], dim=1)
            teacher = F.softmax(t_logit / temperature, dim=-1)
        student = F.log_softmax(logits[has_soft] / temperature, dim=-1)
        kl = F.kl_div(student, teacher, reduction="sum")
        total = total + (temperature ** 2) * kl
        n_ex += int(has_soft.sum())

    if (~has_soft).any():
        ce = F.cross_entropy(logits[~has_soft], hard_labels[~has_soft], reduction="sum")
        total = total + ce
        n_ex += int((~has_soft).sum())

    return total / max(n_ex, 1)


# ── evaluation on the dev split ───────────────────────────────────────────

@torch.no_grad()
def dev_probs(model, loader, device):
    model.eval()
    probs, labels, ner_p, ner_t = [], [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                    ner_labels=batch["ner_labels"])
        probs.extend(torch.softmax(out["cls_logits"].float(), -1)[:, 1].cpu().numpy().tolist())
        labels.extend(batch["cls_label"].cpu().numpy().tolist())
        nl, np_ = batch["ner_labels"], out["ner_logits"].argmax(-1)
        m = nl != -100
        ner_p.extend(np_[m].cpu().numpy().tolist())
        ner_t.extend(nl[m].cpu().numpy().tolist())
    return np.array(probs), np.array(labels), np.array(ner_p), np.array(ner_t)


def dev_metrics(probs, labels, ner_p, ner_t, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    return {
        "cls_auprc": float(average_precision_score(labels, probs)) if len(set(labels)) > 1 else 0.0,
        "cls_f1": float(f1_score(labels, preds, zero_division=0)),
        "cls_prec": float(precision_score(labels, preds, zero_division=0)),
        "cls_rec": float(recall_score(labels, preds, zero_division=0)),
        "ner_f1": float(f1_score(ner_t, ner_p, average="macro", zero_division=0)) if len(ner_t) else 0.0,
    }


# ── training ──────────────────────────────────────────────────────────────

def train(args):
    torch.set_float32_matmul_precision("high")          # B12
    set_seed(args.seed)                                  # A7
    g = torch.Generator(); g.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # C5: fail hard on a missing soft-label file
    soft_path = None
    if args.soft_labels and args.soft_labels.lower() != "none":
        p = Path(args.soft_labels)
        if not p.is_absolute():
            p = REPO / args.soft_labels
        if not p.exists():
            raise FileNotFoundError(
                f"Soft labels not found: {args.soft_labels} (resolved {p}).\n"
                f"V1 silently fell back to hard CE here, a ~0.05 F1 change in the objective.\n"
                f"Pass --soft-labels none to train on hard labels deliberately.")
        soft_path = str(p)

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = REPO / args.data
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    train_ds, val_ds = load_multitask_splits(
        str(data_path), args.encoder, args.ner_scheme,
        val_frac=args.val_frac, seed=args.split_seed,
        max_length=args.max_length, soft_labels_path=soft_path,
        **({"pair_conditioning": True} if args.pair_conditioning else {}),
    )
    print(f"  Train {len(train_ds)}   Dev {len(val_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, generator=g, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=2)

    model = MultiTaskBiomedBERT(args.encoder, args.ner_scheme, args.alpha).to(device)

    steps = len(train_loader) * (args.epochs + args.pretrain_ner_epochs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)

    history, best_score, best_state, best_epoch = [], -1.0, None, -1
    sel_key = args.select_on

    def run_epoch(ner_only: bool):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        token_type_ids=batch["token_type_ids"],
                        ner_labels=batch["ner_labels"])
            n_loss = out["ner_loss"] if out["ner_loss"] is not None else torch.tensor(0.0, device=device)
            if ner_only:
                loss = n_loss
            else:
                c_loss = cls_loss_fn(out["cls_logits"], batch["cls_label"],
                                     batch["soft_label"], args.temperature, args.kd_mode)
                loss = args.alpha * c_loss + (1 - args.alpha) * n_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

    for ep in range(args.pretrain_ner_epochs):
        t0 = time.time(); run_epoch(ner_only=True)
        p, l, npd, ntd = dev_probs(model, val_loader, device)
        m = dev_metrics(p, l, npd, ntd)
        print(f"  ner-pretrain {ep+1}/{args.pretrain_ner_epochs}  ner_F1={m['ner_f1']:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    for ep in range(args.epochs):
        t0 = time.time(); run_epoch(ner_only=False)
        p, l, npd, ntd = dev_probs(model, val_loader, device)
        m = dev_metrics(p, l, npd, ntd)
        m["epoch"] = ep + 1
        history.append(m)
        print(f"  ep {ep+1}/{args.epochs}  AUPRC={m['cls_auprc']:.4f}  F1@.5={m['cls_f1']:.4f}  "
              f"ner_F1={m['ner_f1']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if m[sel_key] > best_score:                       # B11
            best_score, best_epoch = m[sel_key], ep + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # ── thresholds, never from test ───────────────────────────────────────
    p, l, npd, ntd = dev_probs(model, val_loader, device)
    import pandas as pd
    df_all = pd.read_csv(data_path)
    lab_col = "label" if "label" in df_all.columns else "hard_label"
    train_pos_rate = float(df_all[lab_col].mean())

    grid = np.arange(0.01, 1.00, 0.01)
    dev_t = float(grid[int(np.argmax([f1_score(l, (p >= t).astype(int), zero_division=0) for t in grid]))])
    prior_t = threshold_from_prior(train_pos_rate, 0.5)

    model.save(str(Path(args.output_dir)))

    # B1/B2/C6: everything inference needs, stored with the checkpoint
    cfg_path = Path(args.output_dir) / "multitask_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    cfg.update({
        "temperature": args.temperature,
        "kd_mode": args.kd_mode,
        "threshold_dev": dev_t,
        "threshold_prior_at_0.5": prior_t,
        "train_pos_rate": train_pos_rate,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "select_on": sel_key,
        "pair_conditioning": bool(args.pair_conditioning),
        "data_path": str(data_path),
        "data_sha256": sha256(data_path),
        "data_rows": int(len(df_all)),
        "soft_labels_path": soft_path,
        "soft_labels_sha256": sha256(soft_path) if soft_path else None,
        "git_sha": git_sha(),
        "code_provenance": code_provenance(),
        "trained_by": "train_v2.py",
    })
    cfg_path.write_text(json.dumps(cfg, indent=2))

    rd = Path(args.results_dir); rd.mkdir(parents=True, exist_ok=True)
    summary = {**cfg, "encoder": args.encoder, "ner_scheme": args.ner_scheme,
               "alpha": args.alpha, "epochs": args.epochs,
               "pretrain_ner_epochs": args.pretrain_ner_epochs,
               "batch_size": args.batch_size, "lr": args.lr,
               "best_epoch": best_epoch, f"best_dev_{sel_key}": best_score,
               "dev_metrics_at_best": dev_metrics(p, l, npd, ntd),
               "history": history}
    (rd / "train_summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(rd / "dev_probs.npz", probs=p, labels=l)
    print(f"  saved -> {args.output_dir}   dev_t={dev_t:.3f}  prior_t={prior_t:.3f}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--encoder", default=ENCODER)
    ap.add_argument("--ner-scheme", default="full_typed", choices=["basic", "typed", "full", "full_typed"])
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--pretrain-ner-epochs", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--soft-labels", default="data/training/distillation_soft_labels.csv")
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--kd-mode", default="fixed", choices=["fixed", "v1_bug"])
    ap.add_argument("--select-on", default="cls_auprc", choices=["cls_auprc", "cls_f1"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--pair-conditioning", action="store_true")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--results-dir", required=True)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
