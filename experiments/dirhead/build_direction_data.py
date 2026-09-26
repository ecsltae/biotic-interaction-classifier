#!/usr/bin/env python3
"""Build direction supervision for the med25 candidate pool, from scratch and reproducibly.

Three independent weak sources, each kept in its own column so the head can be trained on
any subset and each source's yield/accuracy can be reported separately:

  rule_role   role(taxon) x polarity(relation).  role comes from the OTT clade prior
              (a taxon-name lookup), polarity from the ROBI lexicon + morphology.
  globi       GloBI's directed source/target for the pair, composed with the same polarity.
              Independent of the clade prior; a taxon-PAIR lookup.
  syn         text position of the two mentions relative to the relation mention, flipped
              by the surface voice of the relation.  This one reads the passage.

Output: one parquet with every pool row and all three labels (1 = species1_form is the
subject, 2 = species2_form is), plus the canonical-order label the model is trained on.
"""
import sys, re, json
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/home/egaillac/MetaP/classifier")
sys.path.insert(0, str(REPO / "experiments" / "direction"))
sys.path.insert(0, str(REPO / "experiments" / "dirhead"))
sys.path.insert(0, str(REPO / "scripts"))
import polarity, ottlib, dirlib
import clade_prior

OUT = REPO / "experiments/dirhead/direction_pool.parquet"

# The clade prior, with the one defect the audit found fixed: virus taxa carry an
# `ncbitaxon_viruses:` id and have no OTT lineage, so kindscore() returned None for all
# 13,065 of them.  Viruses are unambiguously the agent side of every relation they appear
# in here, so they get the top score explicitly rather than by accident.
VIRUS_SCORE = 100


def role_scores(names, is_virus):
    cache = {}
    out = []
    for nm, v in zip(names, is_virus):
        if v:
            out.append(VIRUS_SCORE)
            continue
        key = str(nm).lower().strip()
        if key not in cache:
            c, s = clade_prior.kindscore(str(nm))
            cache[key] = s
        out.append(cache[key])
    return np.array([np.nan if x is None else float(x) for x in out])


def main():
    p = pd.read_parquet(REPO / "data/training/distill/med25_pool.parquet")
    kk = p.triplet_key.str.split(";")
    p["ro"] = kk.str[2]
    p["k1"], p["k2"] = kk.str[0], kk.str[1]
    p["s1"] = p.species1_form.apply(lambda x: str(x[0]) if len(x) else "")
    p["s2"] = p.species2_form.apply(lambda x: str(x[0]) if len(x) else "")
    p["rel"] = p.interaction_form.apply(lambda x: str(x[0]) if len(x) else "")
    print(f"pool {len(p)}", flush=True)

    # ---- polarity of the relation
    pol_cache = {}
    pol, polsrc = [], []
    for r, ro in zip(p.rel, p.ro):
        k = (r, ro)
        if k not in pol_cache:
            pol_cache[k] = polarity.polarity(r, ro)
        a, b = pol_cache[k]
        pol.append(np.nan if a is None else a)
        polsrc.append(b)
    p["pol"] = pol
    p["pol_src"] = polsrc
    print("polarity:", p.pol.value_counts(dropna=False).to_dict(), flush=True)

    # ---- role scores
    p["r1"] = role_scores(p.s1, p.k1.str.startswith("ncbitaxon_viruses"))
    p["r2"] = role_scores(p.s2, p.k2.str.startswith("ncbitaxon_viruses"))
    p["margin"] = (p.r1 - p.r2).abs()
    print(f"role resolved both taxa: {(p.r1.notna() & p.r2.notna()).mean():.4f}", flush=True)

    # ---- rule_role
    ok = p.pol.notna() & (p.pol != 0) & p.r1.notna() & p.r2.notna() & (p.r1 != p.r2)
    hi1 = p.r1 > p.r2
    agent_is_1 = hi1
    subj_is_1 = np.where(p.pol > 0, agent_is_1, ~agent_is_1)
    p["rule_role"] = np.where(ok, np.where(subj_is_1, 1, 2), 0).astype(int)
    p["sym"] = (p.pol == 0)

    # ---- syntactic rule (reads the passage)
    syn = []
    for txt, s1, s2, rel in zip(p.passage, p.s1, p.s2, p.rel):
        v = dirlib.base_syntactic(pd.Series(dict(sentence=txt, species1=s1, species2=s2,
                                                 relation=rel, relation_canonical=None)))
        syn.append(v if v in (1, 2) else 0)
    p["syn"] = np.array(syn, dtype=int)

    # ---- GloBI, composed with polarity
    gl = pd.read_csv(REPO / "data/globi/globi_trust_index.csv")
    gl["a"] = gl.source_species.astype(str).str.lower().str.strip()
    gl["b"] = gl.target_species.astype(str).str.lower().str.strip()
    # GloBI orientation: source is the parasite/pathogen/pollinator = the AGENT role
    g = {}
    for a, b in zip(gl.a, gl.b):
        key = tuple(sorted([a, b]))
        v = 1 if a == key[0] else 2          # which member of the sorted pair is the agent
        if key in g and g[key] != v:
            g[key] = 0                        # contradictory -> unusable
        else:
            g.setdefault(key, v)
    n_amb = sum(1 for v in g.values() if v == 0)
    print(f"globi pairs {len(g)} ({n_amb} contradictory)", flush=True)

    def gjoin(s1, s2, pol):
        if not np.isfinite(pol) or pol == 0:
            return 0
        a, b = str(s1).lower().strip(), str(s2).lower().strip()
        key = tuple(sorted([a, b]))
        v = g.get(key, 0)
        if v == 0:
            return 0
        agent_is_1 = (v == 1) == (key[0] == a)
        subj1 = agent_is_1 if pol > 0 else (not agent_is_1)
        return 1 if subj1 else 2

    p["globi"] = [gjoin(a, b, c) for a, b, c in zip(p.s1, p.s2, p.pol)]

    # ---- the canonical-order label the model sees: is span A (alphabetically first) the subject?
    can_first_is_1 = np.array([str(a).lower() <= str(b).lower() for a, b in zip(p.s1, p.s2)])
    for col in ("rule_role", "globi", "syn"):
        v = p[col].to_numpy()
        p[col + "_canon"] = np.where(v == 0, 0, np.where((v == 1) == can_first_is_1, 1, 2))
    p["can_first_is_1"] = can_first_is_1

    keep = ["triplet_key", "doc_id", "passage", "s1", "s2", "rel", "ro", "pol", "pol_src",
            "r1", "r2", "margin", "sym", "rule_role", "globi", "syn",
            "rule_role_canon", "globi_canon", "syn_canon", "can_first_is_1"]
    p[keep].to_parquet(OUT, index=False)
    print("\n--- yields (rows with a usable F/R label) ---")
    for col in ("rule_role", "globi", "syn"):
        n = int((p[col] != 0).sum())
        print(f"{col:10s} {n:7d}  {n/len(p):.4f}")
    print(f"{'sym':10s} {int(p.sym.sum()):7d}  {p.sym.mean():.4f}")
    both = (p.rule_role != 0) & (p.globi != 0)
    print(f"\nrule vs globi: overlap {int(both.sum())}, agree "
          f"{(p.loc[both, 'rule_role'] == p.loc[both, 'globi']).mean():.4f}")
    bs = (p.rule_role != 0) & (p.syn != 0)
    print(f"rule vs syn  : overlap {int(bs.sum())}, agree "
          f"{(p.loc[bs, 'rule_role'] == p.loc[bs, 'syn']).mean():.4f}")
    gs = (p.globi != 0) & (p.syn != 0)
    print(f"globi vs syn : overlap {int(gs.sum())}, agree "
          f"{(p.loc[gs, 'globi'] == p.loc[gs, 'syn']).mean():.4f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
