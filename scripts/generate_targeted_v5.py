#!/usr/bin/env python3
"""Targeted training data built from V1's MEASURED error classes on the 150 scored rows.

V1's 15 false positives sort into four mechanisms, none of which the training
distribution can contain, because the pool only ever yields *extracted* pairs and
every pool passage carries exactly one distinct pair:

  N1 co_participant  two taxa each related to a THIRD taxon, not to each other
                     ("Arvicanthis neumanni ... and Mastomys natalensis as hosts of
                     L. donovani")                                   -- FP4,5,6,7,11,14
  N2 synonym         the pair is one organism named twice
                     ("sulla (Hedysarum coronarium)")                -- FP7,12,13
  N3 ancestor        one member is the other's own clade, from a parenthetical
                     annotation "(Acari: Prostigmata: Neothrombiidae)"  -- FP1
  N4 authority       one member is a taxonomic author, not an organism
                     ("Pterostichus melanarius (Illiger)")           -- FP2

The previous generator (generate_targeted_data.py) tried to build N1 by lexical
matching and instead emitted mostly N2 and N3 candidates *mislabelled as
distractors*, plus the true pair restated under a synonym ("Sacculina carcini"/
"Carcinus maenas"). 51.8% of its rows are provably a clade-parent or an apposition
partner, and the first nine rows contain four restatements of the true pair. It is
not reusable.

Here N2/N3/N4 become deliberate, correctly-signed negative families, and their span
detectors are subtracted from the passage before an N1 candidate is accepted, which
is exactly the failure the old guards missed. N1 is teacher-adjudicated (species-level
prompt, matching the evaluation label); N2/N3/N4 are negative by construction and a
sample of each is teacher-checked to measure purity.

The false-negative class (nominalised interaction: "human pathogens", "infested
with", "transmitted by") is NOT generated here -- it needs no new passages, only the
correct label on passages already in the training set, which is what
relabel_species_level.py produces.
"""
import argparse, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--n1-cap", type=int, default=9000)
ap.add_argument("--seed", type=int, default=20260923)
A = ap.parse_args()
rng = np.random.RandomState(A.seed)

def norm(s): return " ".join(str(s).split()).strip()
def first(x):
    if isinstance(x, (list, np.ndarray)): return str(x[0]).strip() if len(x) else ""
    return str(x).strip() if pd.notna(x) else ""

# ── benchmark guards: every evaluation passage and pair is forbidden ─────────
btxt, bpair = set(), set()
GUARDS = [("data/evaluation/unified_test_set.csv", "sentence", {}, ("species1", "species2")),
          ("data/evaluation/biotic_interaction_test_set.csv", "sentence", {}, None),
          ("data/evaluation/test500_paired.csv", "sentence", {}, ("species1", "species2")),
          ("data/evaluation/biotx_rejected_50_testset.csv", "sentence", {}, None),
          ("globi-relax_passages-triplets_2024-02-28_curation_EP.tsv", "sentence",
           dict(sep="\t", encoding="latin-1"), ("species1_term", "species2_term")),
          ("data/evaluation/biotx_retrieval_eval_100.csv", "sentence",
           dict(sep=";", encoding="utf-8-sig"), ("species1_term", "species2_term"))]
for f, tc, kw, pc in GUARDS:
    p = REPO/f
    if not p.exists(): continue
    x = pd.read_csv(p, **kw)
    if tc in x.columns: btxt |= set(x[tc].astype(str).map(norm))
    if pc and pc[0] in x.columns:
        bpair |= {tuple(sorted([norm(a).lower(), norm(b).lower()]))
                  for a, b in zip(x[pc[0]], x[pc[1]]) if pd.notna(a) and pd.notna(b)}
print(f"guards: {len(btxt):,} benchmark passages, {len(bpair):,} benchmark pairs", flush=True)

pool = pd.read_parquet(REPO/"data/training/distill/med25_pool.parquet")
for c in ("species1_form", "species2_form", "interaction_form"): pool[c] = pool[c].map(first)
pool["passage"] = pool.passage.map(norm)
pool = pool[(pool.species1_form != "") & (pool.species2_form != "") & (pool.interaction_form != "")]
pool["pk"] = [tuple(sorted([a.lower(), b.lower()])) for a, b in zip(pool.species1_form, pool.species2_form)]
pool = pool[~pool.passage.isin(btxt) & ~pool.pk.isin(bpair)]
pool = pool.drop_duplicates(subset=["passage"]).reset_index(drop=True)   # one row per passage
print(f"pool after guards: {len(pool):,} passages", flush=True)

# ── pattern detectors, shared by the negative families and by the N1 exclusion ──
AP = [re.compile(r"\b([a-z][a-z\- ]{3,25}?)\s*\(\s*([A-Z][a-z]+ [a-z]{3,})\b"),
      re.compile(r"\bthe\s+([a-z][a-z\- ]{3,25}?),\s+([A-Z][a-z]+ [a-z]{3,})\b"),
      re.compile(r"\b([A-Z][a-z]+ [a-z]{3,}),?\s+\(?the\s+([a-z][a-z\- ]{3,25}?)\)")]
ANC = [re.compile(r"\(((?:[A-Z][a-z]+\s*:\s*)+[A-Z][a-z]+)\)"),
       re.compile(r"\b(?:family|order|class|phylum|subfamily|superfamily|suborder|infraorder|tribe)\s+([A-Z][a-z]{3,})")]
AUTH = re.compile(r"\b([A-Z][a-z]+ [a-z]{3,})\s*\(?\b([A-Z][a-z]{3,})(?:\s*&\s*[A-Z][a-z]+)?,?\s*(?:1[6-9]\d\d|20[0-2]\d)\)?")
AUTH2 = re.compile(r"\b([A-Z][a-z]+ [a-z]{3,})\s*\(\s*([A-Z][a-z]{3,})\s*\)")
CLADE_SUF = re.compile(r"(aceae|ales|idae|inae|oidea|ida|iformes|mycota|mycetes|virales|viridae|phyta|opsida|acea)$", re.I)
STOP = {"species", "genus", "strain", "isolate", "sample", "study", "group", "family", "order",
        "class", "type", "form", "variety", "subspecies", "complex", "clade", "sequence", "gene"}


LEAD = {"the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "by", "with", "from",
        "at", "as", "that", "this", "these", "those", "its", "their", "our", "we", "study",
        "present", "here", "both", "two", "three", "system", "systems", "isolates", "strains",
        "clinical", "important", "novel", "new", "common", "other", "such", "including"}
def clean_common(a):
    """Trim leading function words and reject spans that are not a plausible organism name."""
    toks = [t for t in a.split() if t]
    while toks and toks[0].lower() in LEAD: toks.pop(0)
    if not toks or len(toks) > 4: return None
    a = " ".join(toks)
    if len(a) < 4: return None
    if toks[-1].lower() in STOP | LEAD: return None
    if not re.fullmatch(r"[a-z][a-z\- ]+", a): return None
    return a

rows = []
def ok_pair(a, b):
    al, bl = a.lower().strip(), b.lower().strip()
    if not al or not bl or al == bl: return False
    if al in bl or bl in al: return False
    if tuple(sorted([al, bl])) in bpair: return False
    return True

def emit(text, s1, s2, rel, kind, label=None):
    if not ok_pair(s1, s2): return False
    rows.append(dict(text=text, source_species=s1, target_species=s2,
                     interaction_type=rel, kind=kind, label=(-1 if label is None else label)))
    return True

# ── N2 / N3 / N4: rule negatives ────────────────────────────────────────────
n2 = n3 = n4 = 0
for psg, rel in zip(pool.passage, pool.interaction_form):
    for pat in AP:
        for m in pat.finditer(psg):
            a, b = norm(m.group(1)), norm(m.group(2))
            a = clean_common(a)
            if a is None: continue
            n2 += emit(psg, a, b, rel, "synonym", 0)
    clades = set()
    for pat in ANC:
        for m in pat.finditer(psg):
            for c in re.split(r"\s*:\s*", m.group(1)):
                c = norm(c)
                if len(c) >= 5: clades.add(c)
    if clades:
        pass  # filled below with the passage's own taxa
    seen = set()
    for pat in (AUTH, AUTH2):
        for m in pat.finditer(psg):
            binom, auth = norm(m.group(1)), norm(m.group(2))
            if auth.lower() in STOP or (binom, auth) in seen: continue
            seen.add((binom, auth))
            n4 += emit(psg, binom, auth, rel, "authority", 0)
# A clade annotation is only this taxon's OWN clade when the parenthetical
# immediately follows that taxon. When a passage annotates both taxa -- "Apanteles
# hemara (Hymenoptera: Braconidae), a Larval Endoparasitoid of Spoladea recurvalis
# (Lepidoptera: Crambidae)" -- the other taxon's clade IS an interaction partner, so
# pairing it is a false negative. Adjacency is what separates the two cases.
for psg, rel, s1, s2 in zip(pool.passage, pool.interaction_form, pool.species1_form, pool.species2_form):
    for t in (s1, s2):
        occ = [m.end() for m in re.finditer(r"\b" + re.escape(t) + r"\b", psg, re.I)]
        if not occ: continue
        for pat in ANC:
            for m in pat.finditer(psg):
                # parenthetical must start within 3 chars of an occurrence of this taxon
                if not any(0 <= m.start() - e <= 3 for e in occ): continue
                for c in re.split(r"\s*:\s*", m.group(1)):
                    c = norm(c)
                    if len(c) >= 5:
                        n3 += emit(psg, t, c, rel, "ancestor", 0)
print(f"N2 synonym {n2:,} | N3 ancestor {n3:,} | N4 authority {n4:,}", flush=True)

# ── N1 co-participant: a third taxon, with N2/N3/N4 spans subtracted ─────────
# Token-lookup against the pool's own surface-form vocabulary: O(passage length),
# not O(vocab) as in the previous generator.
vc = pd.concat([pool.species1_form, pool.species2_form]).value_counts()
vocab = {v.lower(): v for v in vc[vc >= 5].index
         if len(v) >= 5 and not CLADE_SUF.search(v) and v.lower() not in STOP}
print(f"taxon surface vocabulary: {len(vocab):,}", flush=True)
TOK = re.compile(r"[A-Za-z][A-Za-z\-]+")

def excluded_spans(psg):
    ex = []
    for pat in AP:
        for m in pat.finditer(psg): ex.append(m.span())
    for pat in ANC:
        for m in pat.finditer(psg): ex.append(m.span())
    for pat in (AUTH, AUTH2):
        for m in pat.finditer(psg): ex.append(m.span(2))
    return ex

def ovl(s, e, others): return any(not (e2 <= s or s2 >= e) for s2, e2 in others)

idx = rng.permutation(len(pool))
n1 = 0
for i in idx:
    if n1 >= A.n1_cap: break
    psg = pool.passage.iat[i]; rel = pool.interaction_form.iat[i]
    a, b = pool.species1_form.iat[i], pool.species2_form.iat[i]
    al, bl = a.lower(), b.lower()
    toks = [(m.group(0), m.start(), m.end()) for m in TOK.finditer(psg)]
    pair_spans = [(m.start(), m.end()) for nm in (al, bl)
                  for m in re.finditer(r"\b" + re.escape(nm) + r"\b", psg, re.I)]
    if not pair_spans: continue
    ex = excluded_spans(psg)
    got = 0
    for j in range(len(toks)):
        for k in (1, 2):                                  # 1- and 2-token surface forms
            if j + k > len(toks): continue
            cand = " ".join(t[0] for t in toks[j:j+k]).lower()
            orig = vocab.get(cand)
            if orig is None: continue
            if cand in (al, bl) or cand in al or cand in bl or al in cand or bl in cand: continue
            s, e = toks[j][1], toks[j+k-1][2]
            if ovl(s, e, pair_spans) or ovl(s, e, ex): continue
            # adjacency to a pair member => apposition, i.e. a synonym not a distractor
            if any(abs(s - q1) < 4 or abs(e - q0) < 4 for q0, q1 in pair_spans): continue
            if emit(psg, a, orig, rel, "co_participant"):
                n1 += 1; got += 1
            break
        if got >= 2: break
print(f"N1 co_participant: {n1:,}", flush=True)

D = pd.DataFrame(rows).drop_duplicates(subset=["text", "source_species", "target_species", "interaction_type"])
D = D[~D.text.isin(btxt)]
print(f"\ntotal candidates {len(D):,}")
print(D.groupby(["kind", "label"]).size())
out = REPO/"data/training/distill/v5_targeted_candidates.csv"
D.to_csv(out, index=False)
print(f"wrote {out}")
