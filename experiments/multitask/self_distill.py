#!/usr/bin/env python3
"""Self-distillation and compression for CPU deployment.

Two questions:
  (a) SELF-DISTILL, same size: does training a fresh 12-layer student on V3's
      soft outputs over unlabelled pool data help? Self-distillation often
      improves calibration even with no new labels, which is this model's
      documented weakness (per-source score compression).
  (b) COMPRESS to 6 layers: the deployment target is CPU. Halving depth should
      roughly double CPU throughput; the question is what it costs.
"""
import argparse, json, subprocess, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, AutoConfig,
                          get_linear_schedule_with_warmup, set_seed)
from sklearn.metrics import average_precision_score
REPO=Path(__file__).resolve().parents[2]

class DS(Dataset):
    def __init__(self, df, tok, soft=None, ml=256):
        self.q=[f"{a} [SEP] {r} [SEP] {b}" for a,r,b in
                zip(df.source_species, df.interaction_type, df.target_species)]
        self.p=df.text.astype(str).tolist(); self.y=df.label.astype(int).tolist()
        self.s=soft; self.tok=tok; self.ml=ml
    def __len__(self): return len(self.y)
    def __getitem__(self,i):
        e=self.tok(self.q[i],self.p[i],truncation="only_second",max_length=self.ml,
                   padding="max_length",return_tensors="pt")
        o={k:v.squeeze(0) for k,v in e.items()}
        o["labels"]=torch.tensor(self.y[i])
        o["soft"]=torch.tensor(self.s[i] if self.s is not None else -1.0, dtype=torch.float)
        return o

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--teacher", default="models/student_v3/xenc_s1")
    ap.add_argument("--data", default="data/training/distill/v3_combined_train.csv")
    ap.add_argument("--layers", type=int, default=12, help="6 = compressed")
    ap.add_argument("--alpha", type=float, default=0.5, help="weight on KD vs hard CE")
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a=ap.parse_args()
    torch.set_float32_matmul_precision("high"); set_seed(a.seed)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ENC="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    tok=AutoTokenizer.from_pretrained(ENC)
    df=pd.read_csv(REPO/a.data)
    df["_pk"]=[tuple(sorted([str(x).lower(),str(y).lower()])) for x,y in zip(df.source_species,df.target_species)]
    rng=np.random.RandomState(a.seed); pairs=df._pk.unique(); rng.shuffle(pairs)
    devp=set(pairs[:int(len(pairs)*0.1)])
    tr,va=df[~df._pk.isin(devp)],df[df._pk.isin(devp)]

    # ---- teacher soft targets over the SAME data (self-distillation) ----
    th=AutoModelForSequenceClassification.from_pretrained(REPO/a.teacher,local_files_only=True).to(dev).eval()
    def soft_for(d):
        q=[f"{x} [SEP] {r} [SEP] {y}" for x,r,y in zip(d.source_species,d.interaction_type,d.target_species)]
        p=d.text.astype(str).tolist(); out=[]
        with torch.no_grad():
            for i in range(0,len(q),128):
                e=tok(q[i:i+128],p[i:i+128],truncation="only_second",max_length=256,padding=True,return_tensors="pt").to(dev)
                out.extend(torch.softmax(th(**e).logits.float(),-1)[:,1].cpu().numpy())
        return np.array(out)
    t0=time.time(); s_tr=soft_for(tr); s_va=soft_for(va)
    print(f"teacher soft targets in {time.time()-t0:.0f}s  (mean {s_tr.mean():.3f})",flush=True)
    del th; torch.cuda.empty_cache()

    cfg=AutoConfig.from_pretrained(ENC,num_labels=2)
    if a.layers!=cfg.num_hidden_layers:
        full=AutoModelForSequenceClassification.from_pretrained(ENC,num_labels=2)
        cfg.num_hidden_layers=a.layers
        model=AutoModelForSequenceClassification.from_config(cfg)
        # initialise from evenly-spaced teacher layers
        src=full.bert.encoder.layer; step=len(src)/a.layers
        for i in range(a.layers):
            model.bert.encoder.layer[i].load_state_dict(src[int(i*step)].state_dict())
        model.bert.embeddings.load_state_dict(full.bert.embeddings.state_dict())
        del full
        print(f"compressed to {a.layers} layers ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)",flush=True)
    else:
        model=AutoModelForSequenceClassification.from_pretrained(ENC,num_labels=2)
        print(f"same-size self-distillation ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)",flush=True)
    model.to(dev)

    tl=DataLoader(DS(tr,tok,s_tr),batch_size=a.batch_size,shuffle=True,num_workers=2)
    vl=DataLoader(DS(va,tok,s_va),batch_size=a.batch_size*2,num_workers=2)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=0.01)
    sch=get_linear_schedule_with_warmup(opt,int(0.1*len(tl)*a.epochs),len(tl)*a.epochs)
    best,best_state=-1,None
    for ep in range(a.epochs):
        model.train(); t0=time.time()
        for b in tl:
            soft=b.pop("soft").to(dev); y=b.pop("labels").to(dev)
            b={k:v.to(dev) for k,v in b.items()}
            lg=model(**b).logits
            ce=F.cross_entropy(lg,y)
            sl=soft.clamp(1e-6,1-1e-6)
            tlog=torch.stack([torch.zeros_like(sl),torch.log(sl/(1-sl))],1)
            kd=F.kl_div(F.log_softmax(lg/a.temperature,-1),
                        F.softmax(tlog/a.temperature,-1),reduction="batchmean")*a.temperature**2
            (a.alpha*kd+(1-a.alpha)*ce).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); sch.step(); opt.zero_grad()
        model.eval(); P,Y=[],[]
        with torch.no_grad():
            for b in vl:
                b.pop("soft"); y=b.pop("labels"); b={k:v.to(dev) for k,v in b.items()}
                P.extend(torch.softmax(model(**b).logits.float(),-1)[:,1].cpu().numpy()); Y.extend(y.numpy())
        ap_=average_precision_score(Y,P); print(f"  ep{ep+1} dev AUPRC {ap_:.4f} ({time.time()-t0:.0f}s)",flush=True)
        if ap_>best: best,best_state=ap_,{k:v.cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    out=REPO/a.out; out.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(out); tok.save_pretrained(out)
    from sklearn.metrics import f1_score
    grid=np.arange(0.01,1.0,0.01)
    t=float(grid[int(np.argmax([f1_score(Y,(np.array(P)>=g).astype(int),zero_division=0) for g in grid]))])
    (out/"student_config.json").write_text(json.dumps({
        "threshold_dev":t,"best_dev_auprc":best,"seed":a.seed,"layers":a.layers,
        "teacher":a.teacher,"alpha":a.alpha,"temperature":a.temperature,
        "train_pos_rate":float(df.label.mean()),
        "params_M":sum(p.numel() for p in model.parameters())/1e6},indent=2))
    print(f"saved {out}  dev_t={t:.3f}  AUPRC {best:.4f}",flush=True)

if __name__=="__main__": main()
