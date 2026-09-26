#!/usr/bin/env python3
"""Score every checkpoint in models/encoder_sweep against the unified test set.

Reports two operating points per encoder:
  * dev threshold  -- the one recorded in student_config.json, chosen on the held-out
                      dev split. This is the honest deployable number.
  * oracle best F1 -- threshold swept on the reporting set. An upper bound only;
                      it is NOT an achievable operating point and is labelled as such.
"""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_unified import evaluate, score, model_format          # noqa: E402
from sklearn.metrics import f1_score, precision_score, recall_score  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SWEEP = REPO / "models/encoder_sweep"


def dev_threshold(md):
    cfg = Path(md) / "student_config.json"
    return json.loads(cfg.read_text()).get("threshold_dev") if cfg.exists() else None


def main():
    ckpts = sorted(p for p in SWEEP.iterdir() if (p / "student_config.json").exists())
    # models/species_v4/triple_s1 is the BiomedBERT incumbent trained on exactly this
    # recipe (same data, format, epochs, lr, seed 1), so it is the like-for-like control.
    ctrl = REPO / "models/species_v4/triple_s1"
    if (ctrl / "student_config.json").exists():
        ckpts.append(ctrl)
    if not ckpts:
        print("no finished checkpoints yet"); return
    rows = []
    for md in ckpts:
        o, S, d = evaluate([md], md.name)
        y = d.label.to_numpy()
        t = dev_threshold(md)
        pred = (S >= t).astype(int)
        cfg = json.loads((md / "student_config.json").read_text())
        rows.append({
            "ckpt": md.name if md.parent == SWEEP else "biomedbert(CONTROL)",
            "encoder": cfg.get("encoder", "?"),
            "dev_auprc": cfg.get("best_dev_auprc"),
            "test_auprc": o["auprc"],
            "dev_thr": t,
            "P@dev": float(precision_score(y, pred, zero_division=0)),
            "R@dev": float(recall_score(y, pred, zero_division=0)),
            "F1@dev": float(f1_score(y, pred, zero_division=0)),
            "oracle_f1": o["best_f1"],
            "oracle_thr": o["best_thr"],
            "n": o["n"],
        })
        np.save(SWEEP / f"{md.name if md.parent == SWEEP else 'CONTROL_biomedbert'}.scores.npy", S)
        print(f"  scored {md.name}", flush=True)

    rows.sort(key=lambda r: -r["test_auprc"])
    hdr = f'{"checkpoint":26} {"testAUPRC":>9} {"F1@dev":>7} {"P@dev":>7} {"R@dev":>7} {"oracleF1":>8} {"devthr":>6}'
    print("\n" + hdr); print("-" * len(hdr))
    for r in rows:
        print(f'{r["ckpt"]:26} {r["test_auprc"]:9.4f} {r["F1@dev"]:7.4f} '
              f'{r["P@dev"]:7.4f} {r["R@dev"]:7.4f} {r["oracle_f1"]:8.4f} {r["dev_thr"]:6.2f}')
    print("\noracleF1 is a labelled upper bound (threshold swept on the reporting set), "
          "not an achievable operating point.")
    (SWEEP / "sweep_results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
