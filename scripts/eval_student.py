#!/usr/bin/env python3
"""Evaluate the distilled cross-encoder against HUMAN gold on both benchmarks.
Threshold comes from the student's own dev split, never from these sets."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
from scipy.stats import binomtest
REPO=Path(__file__).resolve().parents[1]

@torch.no_grad()
def score(mdir, q, p, dev, bs=64):
    tok=AutoTokenizer.from_pretrained(mdir, local_files_only=True)
    m=AutoModelForSequenceClassification.from_pretrained(mdir, local_files_only=True).to(dev).eval()
    out=[]
    for i in range(0,len(q),bs):
        e=tok(q[i:i+bs], p[i:i+bs], truncation="only_second", max_length=256,
              padding=True, return_tensors="pt").to(dev)
        out.extend(torch.softmax(m(**e).logits.float(),-1)[:,1].cpu().numpy())
    del m; torch.cuda.empty_cache()
    return np.array(out)

dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")

b=pd.read_csv(REPO/'data/evaluation/biotx_retrieval_eval_100.csv',sep=';',encoding='utf-8-sig')
qb=[f"{a} [SEP] {r} [SEP] {c}" for a,r,c in zip(b.species1_term,b.interaction_term,b.species2_term)]
yb=b.triples_ok_full.to_numpy(); base=yb.mean()

t=pd.read_csv(REPO/'data/evaluation/test500_paired.csv')
qt=[f"{a} [SEP] interacts with [SEP] {c}" for a,c in zip(t.species1,t.species2)]
yt=t.label.to_numpy()

res={}
for s in (1,2,3):
    md=REPO/f"models/student/xenc_s{s}"
    cfg=json.loads((md/"student_config.json").read_text()); thr=cfg["threshold_dev"]
    pb=score(md,qb,b.sentence.astype(str).tolist(),dev)
    pt=score(md,qt,t.sentence.astype(str).tolist(),dev)
    accb=int((pb>=thr).sum()); tpb=int(((pb>=thr)&(yb==1)).sum())
    res[s]=dict(thr=thr,
        biotx_prec=tpb/max(accb,1), biotx_acc=accb,
        biotx_rej=int(((pb<thr)&(yb==0)).sum()),
        biotx_p=binomtest(tpb,accb,base,alternative='greater').pvalue if accb else 1.0,
        biotx_auprc=average_precision_score(yb,pb),
        t299_f1=f1_score(yt,(pt>=thr).astype(int)), t299_auprc=average_precision_score(yt,pt))
    r=res[s]
    print(f"  seed {s} (t={thr:.2f}): Biotx100 accepts {accb}, prec {r['biotx_prec']:.4f}, "
          f"rej {r['biotx_rej']}/35, p={r['biotx_p']:.4f}, AUPRC {r['biotx_auprc']:.4f} | "
          f"test299 F1 {r['t299_f1']:.4f} AUPRC {r['t299_auprc']:.4f}", flush=True)

d=pd.DataFrame(res).T
print(f"\n=== DISTILLED STUDENT, mean of 3 seeds ===")
print(f"  Biotx100  precision {d.biotx_prec.mean():.4f} (sd {d.biotx_prec.std():.4f})  "
      f"rejects {d.biotx_rej.mean():.1f}/35   AUPRC {d.biotx_auprc.mean():.4f}")
print(f"  test299   F1 {d.t299_f1.mean():.4f} (sd {d.t299_f1.std():.4f})   AUPRC {d.t299_auprc.mean():.4f}")
print(f"\n  reference: accept-all prec 0.6500 | deployed filter 0.7143 (9/35, p=0.119)")
print(f"             LLM teacher 0.9492 (32/35) | per-component oracle ceiling 0.8025")
print(f"             test299: template baseline F1 0.8213 | LLM teacher 0.8860")
(REPO/'results/v2/student_eval.json').write_text(json.dumps(res,indent=2,default=float))
