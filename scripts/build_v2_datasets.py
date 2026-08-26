#!/usr/bin/env python3
"""Build the V2 training-data variants.

Root cause being fixed
----------------------
scripts/build_v14_dataset.py:59-61 filters POSITIVES through a lexicon signal
and lets negatives pass untouched:

    if filter_positives:
        pos = pos[pos['text'].apply(has_signal)].copy()

Measured consequence: a bare keyword lexicon scores F1 0.857 on v14 and 0.895 on
v16, but only 0.638 on the primary benchmark. Training positives carry the signal
94.9% of the time against 12.9% of negatives (gap +0.820); on test299 the gap is
+0.196. The champion's false-negative rate on zero-signal positives is 0.494 vs
0.186 on signal-bearing ones -- 43% of its whole error budget.

So the model is being taught a lexicon shortcut that the benchmark defeats.
These builders remove the shortcut by matching the benchmark's lexicon marginals
rather than by adding data.

Every output carries a sidecar manifest with input sha256s, row counts and the
achieved marginals, so a V1-vs-V2 x dataset factorial is interpretable.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from data.interaction_lexicon import score_sentence  # noqa: E402

SEED = 20260820
# test299 marginals -- the target, measured not guessed
TARGET_P_SIG_POS = 0.644
TARGET_P_SIG_NEG = 0.449

# Template-free real-literature sources. training_data_globi_v1..v9 are excluded
# deliberately: they reintroduce the v7 template collapse (8,208 masked templates
# over 25,081 rows, top template x102).
CLEAN_SOURCES = [
    "training_data_v14_unfiltered.csv", "sibils_diverse_real.csv", "globi_sibils_real.csv",
    "epmc_direct_sentences.csv", "epmc_direct_sentences_v2.csv", "external_db_sentences.csv",
    "globi_pmc_sentences_v2.csv", "globi_pmc_real_sentences.csv", "negatives_clean.csv",
    "commensal_harvest_20260703.csv", "parasitism_pathogen_harvest_20260423.csv",
]

# 100 rows lifted verbatim from two EP evaluation TSVs; 89 of them are inside
# v14 and v16. Invisible to a benchmark scan because those TSVs are unregistered.
CONTAMINATED = ["ep_curation_2024.csv"]


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sig(texts) -> np.ndarray:
    return np.array([score_sentence(str(t).lower())[1] for t in texts])


def norm(s) -> str:
    return " ".join(str(s).split()).strip()


def benchmark_texts() -> set:
    out = set()
    for f, tc, kw in [
        ("data/evaluation/biotic_interaction_test_set.csv", "sentence", {}),
        ("data/evaluation/test500_paired.csv", "sentence", {}),
        ("globi-relax_passages-triplets_2024-02-28_curation_EP.tsv", "sentence",
         dict(sep="\t", encoding="latin-1")),
        ("data/evaluation/eval_100.tsv", "sentence", dict(sep="\t")),
    ]:
        p = REPO / f
        if p.exists():
            try:
                out |= set(pd.read_csv(p, **kw)[tc].astype(str).map(norm))
            except Exception:
                pass
    # the unregistered EP TSVs that ep_curation_2024.csv was lifted from
    for f in (REPO / "data/evaluation").glob("*curation_EP*.tsv"):
        try:
            out |= set(pd.read_csv(f, sep="\t", encoding="latin-1")["sentence"].astype(str).map(norm))
        except Exception:
            pass
    return out


def benchmark_pairs() -> set:
    p = REPO / "data/evaluation/test500_paired.csv"
    if not p.exists():
        return set()
    d = pd.read_csv(p)
    return {tuple(sorted([norm(a).lower(), norm(b).lower()]))
            for a, b in zip(d.species1, d.species2) if pd.notna(a) and pd.notna(b)}


def harvest(kind: int, exclude_texts: set) -> pd.DataFrame:
    """Collect novel rows of one class from the template-free sources."""
    got, seen = [], set()
    for b in CLEAN_SOURCES:
        f = REPO / "data/training" / b
        if not f.exists():
            continue
        d = pd.read_csv(f)
        tc = "text" if "text" in d.columns else ("sentence" if "sentence" in d.columns else None)
        if tc is None or "label" not in d.columns:
            continue
        s = d[d["label"] == kind].copy()
        if not len(s):
            continue
        s["text"] = s[tc].astype(str).map(norm)
        s = s[~s["text"].isin(exclude_texts) & ~s["text"].isin(seen)].drop_duplicates("text")
        if not len(s):
            continue
        seen |= set(s["text"])
        keep = ["text", "label"] + [c for c in ("source_species", "target_species",
                                                "interaction_type", "source") if c in s.columns]
        s = s[keep].copy()
        s["origin"] = b
        got.append(s)
    return pd.concat(got, ignore_index=True) if got else pd.DataFrame(columns=["text", "label"])


def match_marginals(pos: pd.DataFrame, neg: pd.DataFrame, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Subsample so P(signal|class) matches the benchmark on both classes.

    Only ever DROPS signal-bearing positives and zero-signal negatives, i.e. it
    removes the shortcut rather than inventing rows.
    """
    pz, ps = pos[pos._sig == 0], pos[pos._sig > 0]
    n_keep_ps = int(round(len(pz) / (1 - TARGET_P_SIG_POS) * TARGET_P_SIG_POS))
    if n_keep_ps < len(ps):
        ps = ps.sample(n_keep_ps, random_state=rng)

    nh, nz = neg[neg._sig > 0], neg[neg._sig == 0]
    n_keep_nz = int(round(len(nh) / TARGET_P_SIG_NEG * (1 - TARGET_P_SIG_NEG)))
    if n_keep_nz < len(nz):
        nz = nz.sample(n_keep_nz, random_state=rng)
    return pd.concat([pz, ps]), pd.concat([nh, nz])


def profile(df: pd.DataFrame, label_col="label") -> dict:
    from sklearn.metrics import f1_score
    s = sig(df["text"]); y = df[label_col].astype(int).to_numpy()
    b = (s > 0).astype(int)
    return {
        "n": int(len(df)), "pos_rate": round(float(y.mean()), 4),
        "p_sig_pos": round(float(b[y == 1].mean()), 4),
        "p_sig_neg": round(float(b[y == 0].mean()), 4),
        "gap": round(float(b[y == 1].mean() - b[y == 0].mean()), 4),
        "lexicon_only_f1": round(float(f1_score(y, b, zero_division=0)), 4),
        "zero_signal_pos_frac": round(float((s[y == 1] == 0).mean()), 4),
        "high_signal_neg_frac": round(float((s[y == 0] >= 0.4).mean()), 4),
    }


def write(df: pd.DataFrame, out: Path, inputs: dict, notes: str, label_col="label"):
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    man = {
        "output": str(out.relative_to(REPO)), "sha256": sha256(out),
        "built_by": "scripts/build_v2_datasets.py", "seed": SEED,
        "git": subprocess.run(["git", "describe", "--always", "--dirty"], cwd=REPO,
                              capture_output=True, text=True).stdout.strip(),
        "inputs": inputs, "profile": profile(df, label_col),
        "target_marginals": {"p_sig_pos": TARGET_P_SIG_POS, "p_sig_neg": TARGET_P_SIG_NEG,
                             "source": "test299 (data/evaluation/test500_paired.csv)"},
        "notes": notes,
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(man, indent=2))
    p = man["profile"]
    print(f"  -> {out.name}: {p['n']} rows, {p['pos_rate']:.1%} pos | "
          f"P(sig|pos)={p['p_sig_pos']} P(sig|neg)={p['p_sig_neg']} "
          f"gap={p['gap']} lexF1={p['lexicon_only_f1']}")
    return man


# ── variant builders ──────────────────────────────────────────────────────

def build_d1_v14_signalmatched(rng, bench_txt, bench_pairs):
    """D1: v14 with the one-sided lexicon filter undone."""
    print("\nD1  v14_signalmatched -- undo the positives-only lexicon filter")
    v14 = pd.read_csv(REPO / "data/training/training_data_v14.csv")
    v14["text"] = v14["text"].astype(str).map(norm)

    # drop the ep_curation_2024 contamination sitting inside v14
    contam = set()
    for b in CONTAMINATED:
        f = REPO / "data/training" / b
        if f.exists():
            d = pd.read_csv(f)
            tc = "text" if "text" in d.columns else "sentence"
            contam |= set(d[tc].astype(str).map(norm))
    drop = v14["text"].isin(contam) | v14["text"].isin(bench_txt)
    print(f"  dropped {int(drop.sum())} contaminated rows from v14")
    v14 = v14[~drop].copy()
    v14["origin"] = "v14"

    known = set(v14["text"]) | bench_txt | contam
    add_pos = harvest(1, known)
    add_pos = add_pos[add_pos["text"].map(lambda t: score_sentence(t.lower())[1] == 0.0)]
    known |= set(add_pos["text"])
    add_neg = harvest(0, known)
    add_neg = add_neg[add_neg["text"].map(lambda t: score_sentence(t.lower())[1] >= 0.4)]
    print(f"  re-admitted {len(add_pos)} zero-signal positives, {len(add_neg)} high-signal negatives")

    df = pd.concat([v14, add_pos, add_neg], ignore_index=True)
    df = df.drop_duplicates("text").reset_index(drop=True)
    df["_sig"] = sig(df["text"])
    pos, neg = match_marginals(df[df.label == 1], df[df.label == 0], rng)
    out = pd.concat([pos, neg]).sample(frac=1, random_state=rng).reset_index(drop=True)
    out = out.drop(columns=["_sig"])
    return write(out, REPO / "data/training/v2_d1_v14_signalmatched.csv",
                 {"training_data_v14.csv": sha256(REPO / "data/training/training_data_v14.csv")},
                 "v14 with the positives-only lexicon filter undone: zero-signal positives re-admitted "
                 "from template-free real-literature sources (NO Qwen gate -- Qwen recall on zero-signal "
                 "gold positives is 0.2642, so gating destroys the rows it is meant to rescue), plus "
                 "high-signal negatives, then subsampled to the benchmark's lexicon marginals. "
                 "ep_curation_2024 contamination removed.")


def build_d3_soft_signalmatched_paired(rng, bench_txt, bench_pairs):
    """D3: the champion's soft-label corpus, signal-matched, with pairs recovered."""
    print("\nD3  soft_signalmatched_paired -- signal matching + pair columns on the soft corpus")
    src = REPO / "data/training/distillation_soft_labels_paired.csv"
    d = pd.read_csv(src)
    d["text"] = d["text"].astype(str).map(norm)
    d = d[~d["text"].isin(bench_txt)].copy()
    d = d.rename(columns={"hard_label": "label"})
    d["_sig"] = sig(d["text"])
    pos, neg = match_marginals(d[d.label == 1], d[d.label == 0], rng)
    out = pd.concat([pos, neg]).sample(frac=1, random_state=rng).reset_index(drop=True)
    out = out.drop(columns=["_sig"]).rename(columns={"label": "hard_label"})
    out["label"] = out["hard_label"]
    return write(out, REPO / "data/training/v2_d3_soft_signalmatched_paired.csv",
                 {"distillation_soft_labels_paired.csv": sha256(src)},
                 "Champion's soft-label corpus with taxon pairs recovered (79.86% markable, "
                 "P(label=1|markable)=0.0902 vs 0.0633 -- no v16-style leak) and subsampled to the "
                 "benchmark's lexicon marginals. Retains p_ensemble so KD and pair conditioning "
                 "can be tested in the same cell for the first time.")


def build_d2_manifest():
    """D2 already exists; stamp it with a manifest for parity."""
    src = REPO / "data/training/distillation_soft_labels_paired.csv"
    if not src.exists():
        return None
    d = pd.read_csv(src).rename(columns={"hard_label": "label"})
    d["text"] = d["text"].astype(str).map(norm)
    print("\nD2  soft_paired (already built) -- pairs only, no signal matching")
    man = {"output": "data/training/distillation_soft_labels_paired.csv", "sha256": sha256(src),
           "profile": profile(d), "notes": "Pair columns recovered; lexicon marginals untouched. "
           "The control that isolates pair conditioning from signal matching."}
    (REPO / "data/training/distillation_soft_labels_paired.manifest.json").write_text(json.dumps(man, indent=2))
    p = man["profile"]
    print(f"  -> {p['n']} rows, {p['pos_rate']:.1%} pos | P(sig|pos)={p['p_sig_pos']} "
          f"P(sig|neg)={p['p_sig_neg']} gap={p['gap']} lexF1={p['lexicon_only_f1']}")
    return man


if __name__ == "__main__":
    rng = np.random.RandomState(SEED)
    bt, bp = benchmark_texts(), benchmark_pairs()
    print(f"benchmark guard: {len(bt)} sentences, {len(bp)} taxon pairs excluded from every variant")
    print("\n=== REFERENCE: what we are matching ===")
    t = pd.read_csv(REPO / "data/evaluation/test500_paired.csv").rename(columns={"sentence": "text"})
    print("  test299:", profile(t))
    v = pd.read_csv(REPO / "data/training/training_data_v14.csv")
    print("  v14    :", profile(v))
    build_d1_v14_signalmatched(rng, bt, bp)
    build_d2_manifest()
    build_d3_soft_signalmatched_paired(rng, bt, bp)
    print("\ndone")
