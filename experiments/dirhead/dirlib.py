"""Shared pieces for the direction head: gold loading, baselines, span location.

Label convention (matches both annotation sheets): SUBJECT_IS / DIRECTION answer
"which of the two stored taxa is the SUBJECT of the relation as the sentence uses it".
Internally everything is expressed as `subj` in {1, 2} over the STORED order, and the
model-facing label is `subj_at` in {A, B} over the CANONICAL (alphabetically sorted)
order, so the label is invariant to how the pipeline happened to store the pair.
"""
import re, sys, json
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/home/egaillac/MetaP/classifier")
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "direction"))
import xenc_format


# ---------------------------------------------------------------- gold
def load_gold():
    """Every human direction judgement that exists, in one frame.

    220-sheet: DIRECTION in {F,R,N,U}.  v2 / SHORT sheets: SUBJECT_IS in {1,2,?}.
    Returns one row per item with subj in {1,2} or NaN (undecidable)."""
    out = []
    f = REPO / "data/evaluation/direction_curation_220.xlsx"
    d = pd.read_excel(f, sheet_name="annotate")
    for _, r in d[d.DIRECTION.notna()].iterrows():
        v = str(r.DIRECTION).strip().upper()
        subj = {"F": 1, "R": 2}.get(v, np.nan)
        out.append(dict(src_file="220", row_id=r.row_id, sentence=r.sentence,
                        species1=str(r.species1).split("|")[0], species2=str(r.species2).split("|")[0],
                        relation=r.relation, relation_canonical=r.relation_canonical,
                        raw=v, subj=subj))
    for tag, f in (("v2", "direction_curation_v2.xlsx"), ("short", "direction_curation_SHORT.xlsx")):
        p = REPO / "data/evaluation" / f
        if not p.exists():
            continue
        d = pd.read_excel(p, sheet_name="annotate")
        if "SUBJECT_IS" not in d.columns:
            continue
        for _, r in d[d.SUBJECT_IS.notna()].iterrows():
            v = str(r.SUBJECT_IS).strip()
            subj = {"1": 1, "2": 2}.get(v, np.nan)
            out.append(dict(src_file=tag, row_id=r.row_id, sentence=r.sentence,
                            species1=str(r.species1).split("|")[0], species2=str(r.species2).split("|")[0],
                            relation=r.relation, relation_canonical=r.get("relation", r.relation),
                            raw=v, subj=subj))
    g = pd.DataFrame(out)
    # de-duplicate: the v2/SHORT blocks are drawn from the same 193-row file as the 220 block
    g = g.drop_duplicates(subset=["row_id", "src_file"]).reset_index(drop=True)
    if len(g):
        g = g.drop_duplicates(subset=["sentence", "species1", "species2", "relation"], keep="first")
    return g.reset_index(drop=True)


# ---------------------------------------------------------------- spans
def spans(text, s1, s2, rel):
    """(start,end) of taxon1, taxon2 and the relation mention; None where absent."""
    t = str(text)
    a = xenc_format._locate(s1, t)
    b = xenc_format._locate(s2, t)
    r = None
    for part in str(rel).split("|"):
        part = part.strip()
        if not part:
            continue
        r = xenc_format._locate(part, t)
        if r:
            break
    return a, b, r


# ---------------------------------------------------------------- baselines
def base_stored(row):
    """The pipeline's current behaviour: trust the stored order."""
    return 1


def base_textorder(row):
    """First-mentioned taxon is the subject."""
    a, b, _ = spans(row.sentence, row.species1, row.species2, row.relation)
    if a is None or b is None:
        return None
    return 1 if a[0] <= b[0] else 2


PASSIVE = re.compile(r"(\bby\b\s*$)|(\bby\b)", re.I)


def voice(rel, rel_canon=None):
    """+1 active surface frame, -1 passive/receptive ('... by'), 0 unknown."""
    for s in (rel_canon, rel):
        if s is None or (isinstance(s, float) and np.isnan(s)):
            continue
        k = re.sub(r"\s+", " ", str(s).lower().strip())
        for part in k.split("|"):
            if re.search(r"\bby$|\bby\b", part):
                return -1
    return +1


def base_syntactic(row):
    """Rule W: the subject is the taxon immediately left of the relation mention,
    flipped when the relation surface frame is passive ('X ... infested by ... Y')."""
    a, b, r = spans(row.sentence, row.species1, row.species2, row.relation)
    if a is None or b is None or r is None:
        return None
    # which taxon is nearer on the left of the relation
    da = r[0] - a[1] if a[1] <= r[0] else np.inf
    db = r[0] - b[1] if b[1] <= r[0] else np.inf
    if not np.isfinite(da) and not np.isfinite(db):
        return None
    left = 1 if da < db else 2
    return left if voice(row.relation, row.get("relation_canonical")) > 0 else (3 - left)


def wilson(k, n, z=1.959963985):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def report(name, preds, truth):
    """preds/truth: iterables of {1,2,None}; None in preds = abstain."""
    n = fired = k = 0
    for p, t in zip(preds, truth):
        if t not in (1, 2):
            continue
        n += 1
        if p in (1, 2):
            fired += 1
            k += int(p == t)
    lo, hi = wilson(k, fired)
    return dict(name=name, n_decidable=n, fired=fired, coverage=fired / n if n else 0,
                correct=k, acc=k / fired if fired else float("nan"), lo=lo, hi=hi)
