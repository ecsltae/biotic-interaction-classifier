import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
REPO=Path('/home/egaillac/MetaP/classifier')
torch.set_num_threads(8)
dev=torch.device('cpu')
OUT=Path('/tmp/claude-1001/-home-egaillac-MetaP/367f8693-1735-41aa-941a-a7af5136b38d/scratchpad')

def score(md,q,p,bs=32):
    tok=AutoTokenizer.from_pretrained(md,local_files_only=True)
    m=AutoModelForSequenceClassification.from_pretrained(md,local_files_only=True).to(dev).eval()
    P=[]
    with torch.no_grad():
        for i in range(0,len(q),bs):
            e=tok(q[i:i+bs],p[i:i+bs],truncation='only_second',max_length=256,padding=True,return_tensors='pt')
            P.extend(torch.softmax(m(**e).logits.float(),-1)[:,1].numpy())
    del m; return np.array(P)

def norm(s): return " ".join(str(s).split()).strip()
def first(x):
    if isinstance(x,(list,np.ndarray)): return str(x[0]).strip() if len(x) else ""
    return str(x).strip() if pd.notna(x) else ""

MD=REPO/'models/student_v3/xenc_s1'
MD2=REPO/'models/student/xenc_s1'

# Biotx100
b=pd.read_csv(REPO/'data/evaluation/biotx_retrieval_eval_100.csv',sep=';',encoding='utf-8-sig')
qb=[f"{a} [SEP] {r} [SEP] {c}" for a,r,c in zip(b.species1_term,b.interaction_term,b.species2_term)]
b['p_v3']=score(MD,qb,b.sentence.astype(str).tolist())
b['p_v2']=score(MD2,qb,b.sentence.astype(str).tolist())
b.to_csv(OUT/'biotx100_scored.csv',index=False)
print('biotx100 done')

# reject 50
import ast
rj=pd.read_csv(REPO/'data/evaluation/biotx_rejected_50_testset.csv')
def parse(v):
    try:
        x=ast.literal_eval(v)
        return str(x[0]) if isinstance(x,list) and x else str(v)
    except Exception: return str(v)
rj['s1']=rj.species1_form.map(parse); rj['s2']=rj.species2_form.map(parse); rj['ifm']=rj.interaction_form.map(parse)
qr=[f"{a} [SEP] {r} [SEP] {c}" for a,r,c in zip(rj.species1,rj.interaction,rj.species2)]
rj['p_v3']=score(MD,qr,rj.sentence.astype(str).tolist())
rj['p_v2']=score(MD2,qr,rj.sentence.astype(str).tolist())
rj.to_csv(OUT/'reject50_scored.csv',index=False)
print('reject50 done')

# pool sample 4000 for deployment prior / score distribution
pool=pd.read_parquet(REPO/'data/training/distill/med25_pool.parquet')
for c in ('species1_form','species2_form','interaction_form'): pool[c]=pool[c].map(first)
s=pool.sample(4000,random_state=7).reset_index(drop=True)
qs=[f"{a} [SEP] {r} [SEP] {c}" for a,r,c in zip(s.species1_form,s.interaction_form,s.species2_form)]
s['p_v3']=score(MD,qs,s.passage.astype(str).tolist())
s[['triplet_key','doc_id','field','species1_form','species2_form','interaction_form','passage','p_v3']].to_csv(OUT/'pool_sample_scored.csv',index=False)
print('pool sample done')
