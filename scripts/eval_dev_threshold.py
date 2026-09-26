#!/usr/bin/env python3
"""Evaluate at the DEV-derived threshold, not the test-set-optimal one.

The published baselines quote "best F1", which is an oracle threshold picked on the
test set. For a claim about beating V1 in deployment the honest operating point is the
one each model chose on its own dev split (student_config.json -> threshold_dev).
"""
import json, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score
REPO = Path('/home/egaillac/MetaP/classifier'); sys.path.insert(0, str(REPO/'scripts'))
from eval_unified import evaluate, mcnemar

for spec in sys.argv[1:]:
    name, dirs = spec.split('=')[0], spec.split('=')[1].split(',')
    mds = [Path(d) for d in dirs]
    thrs = []
    for d in mds:
        cfg = REPO/d/'student_config.json'
        if cfg.exists():
            thrs.append(json.loads(cfg.read_text()).get('threshold_dev'))
    t = float(np.mean([x for x in thrs if x is not None])) if thrs else 0.5
    o, S, dd = evaluate([REPO/d for d in dirs], name)
    y = dd.label.to_numpy()
    hv = dd.v1.notna().to_numpy()
    v1 = dd.v1.fillna(0).to_numpy().astype(int)
    pred = (S >= t).astype(int)
    p, n01, n10 = mcnemar(v1[hv], pred[hv], y[hv])
    print(json.dumps({
      'model': name, 'dev_threshold': round(t, 3),
      'all440': {'f1': round(f1_score(y, pred, zero_division=0), 4),
                 'precision': round(precision_score(y, pred, zero_division=0), 4),
                 'recall': round(recall_score(y, pred, zero_division=0), 4),
                 'auprc': round(o['auprc'], 4)},
      'on_v1_150': {'f1': round(f1_score(y[hv], pred[hv], zero_division=0), 4),
                    'precision': round(precision_score(y[hv], pred[hv], zero_division=0), 4),
                    'recall': round(recall_score(y[hv], pred[hv], zero_division=0), 4),
                    'mcnemar_p': round(p, 4), 'k_model_fixes': n01, 'm_new_errors': n10},
      'v1_reference_on_150': {'f1': 0.8636, 'precision': 0.8352, 'recall': 0.8941},
    }))
