import ast, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
REPO=Path("/home/egaillac/MetaP/classifier")
OUT=Path("/tmp/claude-1001/-home-egaillac-MetaP/367f8693-1735-41aa-941a-a7af5136b38d/scratchpad")
torch.set_num_threads(8)
def score(md,q,p,bs=32):
    tok=AutoTokenizer.from_pretrained(md,local_files_only=True)
    m=AutoModelForSequenceClassification.from_pretrained(md,local_files_only=True).eval()
    P=[]
    with torch.no_grad():
        for i in range(0,len(q),bs):
            e=tok(q[i:i+bs],p[i:i+bs],truncation="only_second",max_length=256,padding=True,return_tensors="pt")
            P.extend(torch.softmax(m(**e).logits.float(),-1)[:,1].numpy())
    del m; return np.array(P,dtype=float)
def first(x):
    s=str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            v=ast.literal_eval(s)
            if isinstance(v,(list,tuple)) and v: return str(v[0]).strip()
        except Exception: pass
    return s.split("|")[0].strip()

# reject50: first-element form query + reversed query
r=pd.read_csv(OUT/"reject50_audit.csv")
s1=[first(x) for x in r.species1_form]; s2=[first(x) for x in r.species2_form]; iv=[first(x) for x in r.interaction_form]
qf =[f"{a} [SEP] {b} [SEP] {c}" for a,b,c in zip(s1,iv,s2)]
qrv=[f"{c} [SEP] {b} [SEP] {a}" for a,b,c in zip(s1,iv,s2)]
sr=r.sentence.astype(str).tolist()
# biotx100: first-form + reversed (both canonical and form)
b=pd.read_csv(OUT/"biotx100_audit.csv")
B1=[first(x) for x in b.species1_form]; B2=[first(x) for x in b.species2_form]; BI=[first(x) for x in b.interaction_form]
qbf =[f"{x} [SEP] {y} [SEP] {z}" for x,y,z in zip(B1,BI,B2)]
qbrv=[f"{z} [SEP] {y} [SEP] {x}" for x,y,z in zip(B1,BI,B2)]
qbcrv=[f"{z} [SEP] {y} [SEP] {x}" for x,y,z in zip(b.species1_term,b.interaction_term,b.species2_term)]
sb=b.sentence.astype(str).tolist()
for s in (1,2,3):
    md=REPO/f"models/student_v3/xenc_s{s}"
    r[f"pf1_s{s}"]=score(md,qf,sr); r[f"prev_s{s}"]=score(md,qrv,sr)
    b[f"pf1_s{s}"]=score(md,qbf,sb); b[f"prev_s{s}"]=score(md,qbrv,sb); b[f"pcrev_s{s}"]=score(md,qbcrv,sb)
    print("seed",s,"done",flush=True)
r["pf1"]=r[[f"pf1_s{s}" for s in (1,2,3)]].mean(axis=1); r["prev"]=r[[f"prev_s{s}" for s in (1,2,3)]].mean(axis=1)
b["pf1"]=b[[f"pf1_s{s}" for s in (1,2,3)]].mean(axis=1); b["prev"]=b[[f"prev_s{s}" for s in (1,2,3)]].mean(axis=1)
b["pcrev"]=b[[f"pcrev_s{s}" for s in (1,2,3)]].mean(axis=1)
r.to_csv(OUT/"reject50_audit.csv",index=False); b.to_csv(OUT/"biotx100_audit.csv",index=False)
print("ok")
