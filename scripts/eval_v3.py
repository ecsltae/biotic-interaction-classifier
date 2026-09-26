#!/usr/bin/env python3
"""Evaluate V3 against V2 on all three benchmarks, including the new recall set."""

# ---------------------------------------------------------------------------
# SUPERSEDED -- do not publish numbers from this script.
#
# It carries three defects that were fixed elsewhere but never fixed here, and
# it writes result artifacts, so a casual re-run silently republishes bad numbers.
#
#   1. Threshold leak. threshold_from_prior(tpr, y.mean()) takes the target prior
#      from the REPORTING set's own gold labels, on all three benchmarks. The
#      decision boundary is therefore fitted on the data it then scores.
#   2. Wrong label column. biotx100 is scored against `triples_ok_full`, the
#      full-triple judgement. The benchmark semantics are SPECIES-LEVEL
#      (`triples_ok_species`): does this pair interact, regardless of whether the
#      stored relation term is the right one. These disagree on a large fraction
#      of rows.
#   3. Canonical/surface mismatch. Queries are built from *_term (canonical names)
#      while the students were trained on *_form (surface names). They differ on
#      34-59 of 100 biotx100 rows, and the canonical name is absent from the
#      sentence in roughly a third.
#
# Use scripts/eval_unified.py (canonical harness) and
# scripts/compare_vs_v1_full.py (V1 comparison) instead.
# ---------------------------------------------------------------------------
import sys as _sys
if "--i-know-this-is-superseded" not in _sys.argv:
    _sys.exit(
        "scripts/eval_v3.py is superseded and its numbers are not trustworthy.\n"
        "  threshold fitted on the reporting set; biotx100 scored against\n"
        "  triples_ok_full instead of triples_ok_species; queries built from\n"
        "  canonical *_term while training used surface *_form.\n"
        "Use scripts/eval_unified.py, or pass --i-know-this-is-superseded to override."
    )

import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
from scipy.stats import binomtest
REPO=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(REPO/"src"))
from eval.core import threshold_from_prior
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.set_float32_matmul_precision("high")

def score(md,q,p,bs=64):
    tok=AutoTokenizer.from_pretrained(md,local_files_only=True)
    m=AutoModelForSequenceClassification.from_pretrained(md,local_files_only=True).to(dev).eval()
    P=[]
    with torch.no_grad():
        for i in range(0,len(q),bs):
            e=tok(q[i:i+bs],p[i:i+bs],truncation="only_second",max_length=256,padding=True,return_tensors="pt").to(dev)
            P.extend(torch.softmax(m(**e).logits.float(),-1)[:,1].cpu().numpy())
    del m; torch.cuda.empty_cache(); return np.array(P)

# benchmarks
b=pd.read_csv(REPO/'data/evaluation/biotx_retrieval_eval_100.csv',sep=';',encoding='utf-8-sig')
qb=[f"{a} [SEP] {r} [SEP] {c}" for a,r,c in zip(b.species1_term,b.interaction_term,b.species2_term)]
yb=b.triples_ok_full.to_numpy()
t=pd.read_csv(REPO/'data/evaluation/test299_with_relation.csv'); t=t[t.relation.notna()].reset_index(drop=True)
qt=[f"{a} [SEP] {r} [SEP] {c}" for a,r,c in zip(t.species1,t.relation,t.species2)]
yt=t.label.to_numpy()
rj=pd.read_csv(REPO/'data/evaluation/biotx_rejected_50_testset.csv')
c1=next(c for c in ('species1_form','species1') if c in rj.columns)
c2=next(c for c in ('species2_form','species2') if c in rj.columns)
cr=next(c for c in ('interaction_form','interaction') if c in rj.columns)
qr=[f"{a} [SEP] {r} [SEP] {c}" for a,r,c in zip(rj[c1],rj[cr],rj[c2])]
yr=rj.gold_label.to_numpy()

out={}
for tag,root in [("V2","models/student"),("V3","models/student_v3")]:
    accs=[]
    for s in (1,2,3):
        md=REPO/root/f"xenc_s{s}"
        if not md.exists(): continue
        cfg=json.loads((md/"student_config.json").read_text()); tpr=cfg["train_pos_rate"]
        pb=score(md,qb,b.sentence.astype(str).tolist())
        pt=score(md,qt,t.sentence.astype(str).tolist())
        pr=score(md,qr,rj.sentence.astype(str).tolist())
        thb=threshold_from_prior(tpr,float(yb.mean()))
        tht=threshold_from_prior(tpr,float(yt.mean()))
        thr=threshold_from_prior(tpr,float(yr.mean()))
        acc=int((pb>=thb).sum()); tp=int(((pb>=thb)&(yb==1)).sum())
        accs.append(dict(
            biotx_prec=tp/max(acc,1), biotx_acc=acc, biotx_rej=int(((pb<thb)&(yb==0)).sum()),
            biotx_auprc=average_precision_score(yb,pb),
            t299_f1=f1_score(yt,(pt>=tht).astype(int)), t299_auprc=average_precision_score(yt,pt),
            rej_recall=recall_score(yr,(pr>=thr).astype(int)), rej_auprc=average_precision_score(yr,pr),
            rej_acc=float(((pr>=thr).astype(int)==yr).mean())))
    d=pd.DataFrame(accs); out[tag]=d
    print(f"\n=== {tag} (n={len(d)} seeds) ===")
    print(f"  Biotx100 : precision {d.biotx_prec.mean():.4f} (sd {d.biotx_prec.std():.4f})  "
          f"accepts {d.biotx_acc.mean():.1f}  rejects {d.biotx_rej.mean():.1f}/35  AUPRC {d.biotx_auprc.mean():.4f}")
    print(f"  Test299  : F1 {d.t299_f1.mean():.4f} (sd {d.t299_f1.std():.4f})  AUPRC {d.t299_auprc.mean():.4f}")
    print(f"  Reject50 : recall-on-11-pos {d.rej_recall.mean():.4f}  acc {d.rej_acc.mean():.4f}  AUPRC {d.rej_auprc.mean():.4f}")
print("\n=== V3 - V2 ===")
for k in ['biotx_prec','biotx_auprc','t299_f1','t299_auprc','rej_recall','rej_auprc']:
    print(f"  {k:14} {out['V3'][k].mean()-out['V2'][k].mean():+.4f}")
json.dump({k:v.mean().to_dict() for k,v in out.items()},open(REPO/'results/v2/v3_eval.json','w'),indent=2)
