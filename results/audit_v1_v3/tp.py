import sys, re, ast, numpy as np, pandas as pd
sys.path.insert(0,"/home/egaillac/MetaP/classifier/src")
from data.interaction_lexicon import score_sentence
OUT="/tmp/claude-1001/-home-egaillac-MetaP/367f8693-1735-41aa-941a-a7af5136b38d/scratchpad/"
b=pd.read_csv(OUT+"biotx100_audit.csv"); y=b.triples_ok_full.to_numpy()
def first(x):
    s=str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            v=ast.literal_eval(s)
            if isinstance(v,(list,tuple)) and v: return str(v[0]).strip()
        except Exception: pass
    return s.split("|")[0].strip()
lex=[score_sentence(str(s)) for s in b.sentence]
b["lex_hit"]=[x[0] for x in lex]; b["lex_strength"]=[x[1] for x in lex]
b["lex_n"]=[len(x[2]) for x in lex]
print("=== interaction lexicon (106 patterns) on Biotx100 ===")
print(pd.crosstab(b.triples_ok_full,b.lex_hit))
tp=int(((b.lex_hit)&(y==1)).sum()); fp=int(((b.lex_hit)&(y==0)).sum())
fn=int(((~b.lex_hit)&(y==1)).sum()); tn=int(((~b.lex_hit)&(y==0)).sum())
pr=tp/(tp+fp); rc=tp/(tp+fn)
print(f"lexicon-alone: TP{tp} FP{fp} FN{fn} TN{tn} P={pr:.4f} R={rc:.4f} F1={2*pr*rc/(pr+rc):.4f}  (all-accept P=0.65 F1=0.7879)")
# strength threshold sweep
for t in (0.0,0.5,0.7,0.85):
    pred=(b.lex_strength>=t)&(b.lex_hit)
    tp=int((pred&(y==1)).sum()); fp=int((pred&(y==0)).sum()); fn=int((~pred&(y==1)).sum())
    print(f"  strength>={t}: P={tp/max(tp+fp,1):.3f} R={tp/max(tp+fn,1):.3f} n_accept={int(pred.sum())}")
# --- adjacency / same clause ---
def spans(sent, cands):
    s=sent.lower()
    best=None
    for c in cands:
        c=str(c).strip().lower()
        if not c: continue
        i=s.find(c)
        if i>=0 and (best is None or i<best[0]): best=(i,i+len(c),c)
        else:
            # try first token (genus) or last token
            for tokn in (c.split()[0], c.split()[-1]):
                if len(tokn)<4: continue
                j=s.find(tokn)
                if j>=0 and (best is None or j<best[0]): best=(j,j+len(tokn),tokn)
    return best
rows=[]
for _,x in b.iterrows():
    s=str(x.sentence)
    a=spans(s,[x.species1_term,first(x.species1_form)])
    c=spans(s,[x.species2_term,first(x.species2_form)])
    if a and c:
        lo,hi=sorted([a,c],key=lambda z:z[0])
        between=s[lo[1]:hi[0]]
        ntok=len(between.split())
        # clause boundary = ; or . or , + conjunction... use strong boundaries
        sameclause = not re.search(r"[;.]|\)\s*[A-Z]", between)
        rows.append((ntok,sameclause,True))
    else:
        rows.append((np.nan,False,False))
b["tok_between"]=[r[0] for r in rows]; b["same_clause"]=[r[1] for r in rows]; b["both_found"]=[r[2] for r in rows]
print("\n=== both taxa string-locatable in sentence ===")
print(pd.crosstab(b.triples_ok_full,b.both_found))
print("\n=== token distance between the two taxa (locatable rows) ===")
print(b[b.both_found].groupby("triples_ok_full").tok_between.describe()[["count","mean","50%","max"]])
print("\n=== same clause (no ; or . between) ===")
print(pd.crosstab(b.triples_ok_full,b.same_clause))
print("\n=== title vs abstract ===")
print(pd.crosstab(b.triples_ok_full,b.field))
print("\n=== canonical term literally present in sentence? ===")
b["t1_in"]=[str(t).lower() in str(s).lower() for t,s in zip(b.species1_term,b.sentence)]
b["t2_in"]=[str(t).lower() in str(s).lower() for t,s in zip(b.species2_term,b.sentence)]
b["ti_in"]=[str(t).lower() in str(s).lower() for t,s in zip(b.interaction_term,b.sentence)]
b["nterm_missing"]=(~b.t1_in).astype(int)+(~b.t2_in).astype(int)+(~b.ti_in).astype(int)
print("term literally absent counts (s1,s2,rel):",int((~b.t1_in).sum()),int((~b.t2_in).sum()),int((~b.ti_in).sum()))
print(pd.crosstab(b.cellV3,b.nterm_missing))
b.to_csv(OUT+"biotx100_audit.csv",index=False)
