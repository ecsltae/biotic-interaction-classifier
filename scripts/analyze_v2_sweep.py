#!/usr/bin/env python3
"""Score every V2 sweep checkpoint and analyse the grid.

Selection discipline: configs are ranked on the DEV split (AUPRC) and on the
test set's threshold-free AUPRC. The decision threshold always comes from dev
or from the closed-form prior — never from the set being reported.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np, pandas as pd, torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "multitask"))
from eval.core import (load_benchmark, metrics_at, majority_baseline, bootstrap_ci,  # noqa
                       paired_bootstrap_delta, mcnemar, threshold_from_prior)
sys.path.insert(0, str(REPO / "scripts"))
from score_model import load, predict, BENCHES  # noqa


def score_all(sweep_root: Path, res_root: Path, out_json: Path, device):
    benches = {}
    for bn, spec in BENCHES.items():
        spec = dict(spec); path = spec.pop("path")
        benches[bn] = load_benchmark(bn, path, **spec)

    rows, probs_store = [], {}
    dirs = sorted(d for d in sweep_root.iterdir() if (d / "multitask_config.json").exists())
    print(f"scoring {len(dirs)} checkpoints", flush=True)
    for i, d in enumerate(dirs, 1):
        cfg = json.loads((d / "multitask_config.json").read_text())
        summ_p = res_root / d.name / "train_summary.json"
        if not summ_p.exists():
            continue
        summ = json.loads(summ_p.read_text())
        model, tok, _ = load(d, device)
        rec = {"name": d.name,
               "regime": d.name[0],
               "kd_mode": cfg.get("kd_mode"), "alpha": summ.get("alpha"),
               "ner_scheme": summ.get("ner_scheme"),
               "pretrain": summ.get("pretrain_ner_epochs"),
               "pair": bool(cfg.get("pair_conditioning")),
               "seed": cfg.get("seed"),
               "dev_auprc": summ.get("best_dev_cls_auprc"),
               "threshold_dev": cfg.get("threshold_dev"),
               "train_pos_rate": cfg.get("train_pos_rate")}
        for bn, b in benches.items():
            pairs = None
            if cfg.get("pair_conditioning"):
                spec = dict(BENCHES[bn]); p = spec.pop("path")
                df = pd.read_csv(p, sep=spec.get("sep", ","), encoding=spec.get("encoding", "utf-8"))
                c1 = next((c for c in ("species1", "species1_term", "source_species") if c in df.columns), None)
                c2 = next((c for c in ("species2", "species2_term", "target_species") if c in df.columns), None)
                if c1 and c2:
                    pairs = [f"{str(x).strip()} [SEP] {str(y).strip()}" for x, y in zip(df[c1], df[c2])]
            # B1: read at the temperature the model was trained at
            pr = predict(model, tok, b.texts, device,
                         temperature=float(cfg.get("temperature", 1.0)), pairs=pairs)
            probs_store[f"{d.name}|{bn}"] = pr.tolist()
            t_dev = float(cfg.get("threshold_dev", 0.5))
            t_prior = threshold_from_prior(float(cfg.get("train_pos_rate", 0.5)), b.prevalence)
            m_dev = metrics_at(pr, b.labels, t_dev)
            m_pri = metrics_at(pr, b.labels, t_prior)
            rec[f"{bn}_auprc"] = m_dev["auprc"]
            rec[f"{bn}_auc"] = m_dev["auc"]
            rec[f"{bn}_f1_devt"] = m_dev["f1"]
            rec[f"{bn}_f1_priort"] = m_pri["f1"]
            rec[f"{bn}_t_prior"] = t_prior
        rows.append(rec)
        del model; torch.cuda.empty_cache()
        if i % 6 == 0:
            print(f"  {i}/{len(dirs)}", flush=True)

    df = pd.DataFrame(rows)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_json.with_suffix(".csv"), index=False)
    np.savez_compressed(out_json.with_suffix(".probs.npz"),
                        **{k: np.array(v) for k, v in probs_store.items()})
    print(f"saved -> {out_json.with_suffix('.csv')}")
    return df


def main_effects(df: pd.DataFrame, metric="test500_auprc"):
    """Marginal effect of each axis, within regime, with spread across the
    other cells. With one seed this is a screen, not a significance test."""
    out = []
    for regime, g in df.groupby("regime"):
        axes = ["kd_mode", "alpha", "ner_scheme", "pretrain"] if regime == "S" \
            else ["pair", "alpha", "ner_scheme", "pretrain"]
        for ax in axes:
            for lvl, gg in g.groupby(ax):
                out.append({"regime": regime, "axis": ax, "level": lvl,
                            "n": len(gg), f"mean_{metric}": gg[metric].mean(),
                            f"sd_{metric}": gg[metric].std(),
                            f"max_{metric}": gg[metric].max()})
    return pd.DataFrame(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-root", default="models/v2_sweep")
    ap.add_argument("--res-root", default="results/v2_sweep")
    ap.add_argument("--out", default="results/v2/sweep_scores.json")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    df = score_all(REPO / a.sweep_root, REPO / a.res_root, REPO / a.out, device)
    pd.set_option("display.width", 200)
    print("\n=== MAIN EFFECTS (test500 AUPRC) ===")
    print(main_effects(df).to_string(index=False))
    print("\n=== TOP 12 by test500 AUPRC ===")
    print(df.nlargest(12, "test500_auprc")[
        ["name", "dev_auprc", "test500_auprc", "test500_f1_devt", "ep_relax_auprc", "ep_relax_f1_devt"]
    ].to_string(index=False))
