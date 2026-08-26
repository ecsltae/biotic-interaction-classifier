#!/usr/bin/env python3
"""Score the V1-trained arm of the architecture x dataset factorial.

V1's save() writes only encoder_name/ner_scheme/alpha/ner_labels, so the
threshold comes from train_summary.json's best_threshold (V1's own val sweep).
V1 trains at T=2 and reads at T=1, so probabilities are read at T=1 here --
that IS V1's behaviour and the comparison must preserve it.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/"src")); sys.path.insert(0, str(REPO/"experiments"/"multitask"))
sys.path.insert(0, str(REPO/"scripts"))
from eval.core import load_benchmark, metrics_at            # noqa
from score_model import load, predict, BENCHES               # noqa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")

benches = {}
for bn in ("test299", "test500", "ep_relax"):
    spec = dict(BENCHES[bn]); p = spec.pop("path")
    benches[bn] = load_benchmark(bn, p, **spec)

rows, store = [], {}
for d in sorted((REPO/"models/v1_arm").iterdir()):
    if not (d/"multitask_config.json").exists():
        continue
    summ_p = REPO/"results/v1_arm"/d.name/"train_summary.json"
    if not summ_p.exists():
        continue
    summ = json.loads(summ_p.read_text())
    thr = float(summ.get("best_threshold", 0.5))
    model, tok, _ = load(d, device)
    rec = {"name": d.name, "dataset": d.name.split("_")[1], "repeat": d.name.split("_")[-1],
           "arch": "V1", "threshold": thr,
           "best_val_cls_f1": summ.get("best_val_cls_f1")}
    for bn, b in benches.items():
        pr = predict(model, tok, b.texts, device, temperature=1.0)   # V1 reads at T=1
        store[f"{d.name}|{bn}"] = pr.tolist()
        m = metrics_at(pr, b.labels, thr)
        rec[f"{bn}_auprc"] = m["auprc"]; rec[f"{bn}_f1"] = m["f1"]; rec[f"{bn}_auc"] = m["auc"]
    rows.append(rec); del model; torch.cuda.empty_cache()
    print(f"  {d.name}: t={thr:.2f} test299 AUPRC {rec['test299_auprc']:.4f} F1 {rec['test299_f1']:.4f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(REPO/"results/v2/v1_arm_scores.csv", index=False)
np.savez_compressed(REPO/"results/v2/v1_arm_scores.probs.npz", **{k: np.array(v) for k, v in store.items()})
print("\n=== V1 ARM: dataset means (3 repeats each, NO seed control -- spread is V1's own variance) ===")
print(df.groupby("dataset").agg(n=("test299_auprc","size"),
      auprc=("test299_auprc","mean"), auprc_sd=("test299_auprc","std"),
      f1=("test299_f1","mean"), f1_sd=("test299_f1","std"),
      val_f1=("best_val_cls_f1","mean")).round(4).to_string())
