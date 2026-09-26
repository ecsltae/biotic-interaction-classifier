import json
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
def firstform(x):
    s=str(x)
    return s.split("|")[0].strip()
b=pd.read_csv(OUT/"biotx100_scored.csv")
qf=[f"{firstform(a)} [SEP] {firstform(r)} [SEP] {firstform(c)}"
    for a,r,c in zip(b.species1_form,b.interaction_form,b.species2_form)]
print(qf[:5])
for tag,root in (("V3","models/student_v3"),("V2","models/student")):
    for s in (1,2,3):
        md=REPO/root/f"xenc_s{s}"
        b[f"pform_{tag}_s{s}"]=score(md,qf,b.sentence.astype(str).tolist())
        print(tag,s,"done",flush=True)
b.to_csv(OUT/"biotx100_scored.csv",index=False)
print("ok")
