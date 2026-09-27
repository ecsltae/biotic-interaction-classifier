#!/usr/bin/env python3
"""Resample the presentation order of SemEval training pairs to hit a target pi.

pi is the marginal probability that the first-listed argument of a directed training pair is
its subject. Only the order in which the two arguments are presented changes: the passage, the
pair, the relation and the gold subject are untouched, so the intervention is purely
presentational and every other property of the corpus is held fixed. The test set is never
modified.
"""
import argparse
import numpy as np, pandas as pd


def retarget(df: pd.DataFrame, pi: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    idx = out.index[out.directed == 1].to_numpy()
    n_first = int(round(pi * len(idx)))
    perm = rng.permutation(idx)
    want_first = np.zeros(len(idx), bool)
    want_first[:n_first] = True
    target = dict(zip(perm, want_first))
    for i in idx:
        is_first = out.at[i, "subject_is"] == 1
        if is_first != target[i]:
            out.at[i, "s1"], out.at[i, "s2"] = out.at[i, "s2"], out.at[i, "s1"]
            out.at[i, "subject_is"] = 1 if out.at[i, "subject_is"] == 2 else 2
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default="semeval_train_ours.csv")
    ap.add_argument("--pi", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.train)
    r = retarget(df, a.pi, a.seed)
    d = r[r.directed == 1]
    got = (d.subject_is == 1).mean()
    # the pair set must be unchanged: same unordered pairs, same passages, same relations
    key = lambda f: sorted(tuple(sorted((str(x), str(y)))) for x, y in zip(f.s1, f.s2))
    assert key(df) == key(r), "pair multiset changed"
    assert (df.passage.values == r.passage.values).all(), "passages changed"
    assert (df.directed.values == r.directed.values).all(), "directedness changed"
    r.to_csv(a.out, index=False)
    print(f"{a.out}: target pi={a.pi:.2f}  achieved={got:.4f}  n_directed={len(d)}")
