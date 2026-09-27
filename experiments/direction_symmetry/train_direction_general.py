#!/usr/bin/env python3
"""Direction models over any (s1, s2, rel, passage, subject_is, directed) corpus.

Two design axes, one flag each, so a 2x2 is one loop:

  --input  canonical : the two entities sorted alphabetically, marked @first@ / #second#.
                       Exchanging the arguments gives a BYTE-IDENTICAL encoder input.
           text      : stored order, markers denote role. Standard practice; the input
                       carries the argument order.
  --head   symmetric : s = g(A,B) - g(B,A) exactly antisymmetric (which is the subject)
                       u = h(A,B) + h(B,A) exactly symmetric     (is it directed at all)
           uncon     : one 3-way classifier, no symmetry guarantee. The BioREDirect shape.

The relation TYPE enters only at the head, as a small categorical embedding; it never reaches
the encoder, so direction cannot re-enter through the relation surface form.
"""
import argparse, json, re
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

ENC = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
AT_ID, HASH_ID = 36, 7


def mark(passage, a, b):
    out = passage
    for ent, o, c in ((a, "@", "@"), (b, "#", "#")):
        m = re.search(re.escape(ent), out, re.I)
        if m:
            out = out[:m.start()] + o + " " + m.group(0) + " " + c + out[m.end():]
    return out


class DS(Dataset):
    def __init__(self, df, tok, rel2i, order="canonical", max_len=256):
        s1 = df.s1.astype(str).tolist(); s2 = df.s2.astype(str).tolist()
        A, B, at = [], [], []
        for x, y, p in zip(s1, s2, df.passage.astype(str)):
            if order == "canonical":
                f, s = (x, y) if x.lower() <= y.lower() else (y, x)
                at.append(x.lower() <= y.lower())
            else:
                f, s = x, y
                at.append(True)
            A.append(f"{f} [SEP] {s}"); B.append(mark(p, f, s))
        self.a, self.b = A, B
        at = np.array(at)
        self.y = (at == (df.subject_is.to_numpy() == 1)).astype("float32")
        self.d = df.directed.to_numpy().astype("float32")
        self.r = np.array([rel2i.get(r, 0) for r in df.rel], dtype="int64")
        self.tok, self.ml = tok, max_len
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.a[i], self.b[i], self.y[i], self.r[i], self.d[i]
    def collate(self, batch):
        a, b, y, r, d = zip(*batch)
        e = self.tok(list(a), list(b), truncation="only_second", max_length=self.ml,
                     padding=True, return_tensors="pt")
        e["y"] = torch.tensor(y); e["r"] = torch.tensor(r); e["d"] = torch.tensor(d)
        return e


def span_pool(h, ids, marker):
    B, T, H = h.shape
    out = h.new_zeros(B, H); ok = h.new_zeros(B)
    for i in range(B):
        p = torch.nonzero(ids[i] == marker, as_tuple=False).flatten()
        if p.numel() >= 2 and int(p[1]) > int(p[0]) + 1:
            out[i] = h[i, int(p[0]) + 1:int(p[1])].mean(0); ok[i] = 1.0
        elif p.numel() >= 1:
            lo = int(p[0]) + 1
            if lo < T: out[i] = h[i, lo:min(lo + 8, T)].mean(0); ok[i] = 1.0
    return out, ok


class SymHead(nn.Module):
    def __init__(self, nrel, hid=768, rd=16, inner=256):
        super().__init__()
        self.rel = nn.Embedding(nrel, rd)
        self.g = nn.Sequential(nn.Linear(3*hid+rd, inner), nn.GELU(), nn.Dropout(0.1), nn.Linear(inner, 1))
        self.h = nn.Sequential(nn.Linear(3*hid+rd, inner), nn.GELU(), nn.Dropout(0.1), nn.Linear(inner, 1))
    def forward(self, hA, hB, cls, r):
        e = self.rel(r)
        ab = torch.cat([hA, hB, cls, e], -1); ba = torch.cat([hB, hA, cls, e], -1)
        return (self.g(ab)-self.g(ba)).squeeze(-1), (self.h(ab)+self.h(ba)).squeeze(-1)


class UnconHead(nn.Module):
    def __init__(self, nrel, hid=768, rd=16, inner=256):
        super().__init__()
        self.rel = nn.Embedding(nrel, rd)
        self.g = nn.Sequential(nn.Linear(3*hid+rd, inner), nn.GELU(), nn.Dropout(0.1), nn.Linear(inner, 3))
    def forward(self, hA, hB, cls, r):
        lg = self.g(torch.cat([hA, hB, cls, self.rel(r)], -1))
        return lg[:, 0]-lg[:, 1], torch.logsumexp(lg[:, :2], -1)-lg[:, 2]


class Model(nn.Module):
    def __init__(self, nrel, head="symmetric", enc=ENC):
        super().__init__()
        self.bert = AutoModel.from_pretrained(enc)
        H = SymHead if head == "symmetric" else UnconHead
        self.head = H(nrel, self.bert.config.hidden_size)
    def forward(self, input_ids, attention_mask, token_type_ids=None, r=None):
        kw = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None: kw["token_type_ids"] = token_type_ids
        h = self.bert(**kw).last_hidden_state
        hA, okA = span_pool(h, input_ids, AT_ID)
        hB, okB = span_pool(h, input_ids, HASH_ID)
        sd, su = self.head(hA, hB, h[:, 0], r)
        return sd, su, okA*okB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--input", default="canonical", choices=["canonical", "text"])
    ap.add_argument("--head", default="symmetric", choices=["symmetric", "uncon"])
    ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(ENC)
    tr = pd.read_csv(a.train)
    rels = sorted(tr.rel.astype(str).unique()); rel2i = {r: i for i, r in enumerate(rels)}
    ds = DS(tr, tok, rel2i, order=a.input)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, collate_fn=ds.collate, num_workers=4, drop_last=True)
    m = Model(len(rels), head=a.head).to(dev)
    print(f"{a.out}: train {len(ds):,}  directed {int(ds.d.sum()):,}  "
          f"P(@ subj)={ds.y.mean():.4f}  rels={len(rels)}", flush=True)
    steps = len(dl)*a.epochs
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.06*steps), steps)
    bce = nn.BCEWithLogitsLoss(reduction="none"); scaler = torch.amp.GradScaler("cuda")
    for ep in range(a.epochs):
        m.train(); run = nb = 0
        for i, b in enumerate(dl):
            b = {k: v.to(dev) for k, v in b.items()}
            y, r, d = b.pop("y"), b.pop("r"), b.pop("d")
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                sd, su, ok = m(b["input_ids"], b["attention_mask"], b.get("token_type_ids"), r=r)
                base = (ok > 0).float()
                ld = (bce(sd.float(), y)*base*d).sum()/(base*d).sum().clamp(min=1)
                lu = (bce(su.float(), d)*base).sum()/base.sum().clamp(min=1)
                loss = ld + lu
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            run += float(loss); nb += 1
            if i % 100 == 0:
                print(f"  ep{ep} {i}/{len(dl)} loss {run/max(nb,1):.4f}", flush=True); run = nb = 0
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), out/"model.pt")
    json.dump(dict(encoder=ENC, input=a.input, head=a.head, epochs=a.epochs, lr=a.lr,
                   seed=a.seed, rels=rels, train=a.train), open(out/"config.json", "w"), indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
