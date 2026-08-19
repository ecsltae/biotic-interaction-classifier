"""
evaluate_reranking_extrinsic.py — Extrinsic validation of the ampliseq re-ranking (H4b).

This is the missing evaluator for the thesis's central open experiment (H4b): unlike
the *intrinsic* validation (validate_reranking_globi.py), which samples communities from
GloBI and scores them against GloBI (semi-circular, recovers KNOWN structure), this
script scores a reranked taxon list against an EXTERNALLY-established truth composition
(a mock community's spec sheet, or in-silico spike-ins of known-wrong taxa). Truth and
the interaction evidence come from disjoint origins, so it is not circular.

It consumes the EXISTING `reranked_per_sample.json` produced by ampliseq_rerank.py
(no change to the reranker), so it works on any of the six reranked datasets and on a
mock the instant one is available.

Two truth modes:
  --truth-mode closed : a defined mock. Every candidate NOT in the truth list is a
                        false positive (a spurious detection to be demoted).
  --truth-mode open   : real field data. Only the taxa in --injected are known-wrong
                        (FP); non-truth candidates are of unknown status and ignored.

Metrics (baseline order = original_rank, reranked order = new_rank):
  - Known-member recovery: mean Δrank of TRUTH (>0 = up), MRR, NDCG@k, recall@k, precision@k
  - False-positive reduction: mean Δrank of FP (<0 = down), FP-rate@k (count of FP in top-k)
  - Rank separation = mean(rank_FP) - mean(rank_TRUTH): should INCREASE after reranking
    (FP pushed below true members) — the single-number effect size.

Significance (reuses validate_reranking_globi.py's bootstrap + permutation machinery):
  - Bootstrap 95% CI on per-sample improvements.
  - Permutation test with the CORRECT extrinsic null: permute the coherence vector across
    the candidate taxa within each sample (breaking the taxon<->coherence assignment),
    recompute the reranked order and the separation metric. This tests "coherence
    assigned to the RIGHT taxa separates truth from FP better than coherence assigned to
    random taxa" — strictly stronger than the intrinsic random-ordering null.

Usage:
    # real mock (closed truth) or real data with spike-ins (open truth)
    python evaluate_reranking_extrinsic.py \\
        --reranked results/ampliseq_rerank/<dataset>/reranked_per_sample.json \\
        --truth    truth.tsv          # sample_id <tab> species   (known members)
        --injected intruders.tsv      # sample_id <tab> species   (known-wrong; optional)
        --truth-mode closed \\
        --beta 1.0 --n-perm 1000 --n-boot 1000 --seed 42 \\
        --output   results/validation/extrinsic_<dataset>.json

    # end-to-end self-test on synthetic truth (no real data needed):
    python evaluate_reranking_extrinsic.py --demo
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "classifier" / "experiments" / "knowledge_graph"))

# Reuse the intrinsic validator's metric + stats functions verbatim so H4a and H4b
# report on identical machinery (the thesis commits to this in App. A).
from validate_reranking_globi import reciprocal_rank, ndcg, bootstrap_ci


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _norm(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def load_reranked(path: Path) -> Dict[str, List[dict]]:
    """Load ampliseq_rerank.py's reranked_per_sample.json -> {sample_id: [rows]}.
    Each row carries species, reads, original_rank, new_rank, delta_rank, final_score."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):  # tolerate a flat single-sample list
        data = {"sample": data}
    # normalise species names on load
    for rows in data.values():
        for r in rows:
            r["species"] = _norm(r["species"])
    return data


def load_species_tsv(path: Optional[Path]) -> Dict[str, Set[str]]:
    """Load a `sample_id <tab> species` TSV -> {sample_id: {species, ...}} (normalised)."""
    out: Dict[str, Set[str]] = defaultdict(set)
    if path is None:
        return dict(out)
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            out[parts[0].strip()].add(_norm(parts[1]))
    return dict(out)


# ---------------------------------------------------------------------------
# Per-sample labelling and metrics
# ---------------------------------------------------------------------------

def label_sample(rows: List[dict], truth: Set[str], injected: Set[str],
                 truth_mode: str) -> Tuple[Set[str], Set[str]]:
    """Return (truth_present, fp_present) among this sample's candidate taxa."""
    candidates = {r["species"] for r in rows}
    truth_present = candidates & truth
    if truth_mode == "closed":
        # every non-truth candidate is a spurious detection; injected (if any) is a subset
        fp_present = (candidates - truth) | (candidates & injected)
    else:  # "open": only explicitly injected intruders are known-wrong
        fp_present = candidates & injected
    return truth_present, fp_present


def _order(rows: List[dict], key: str) -> List[str]:
    return [r["species"] for r in sorted(rows, key=lambda r: r[key])]


def _mean_rank(rows: List[dict], members: Set[str], key: str) -> Optional[float]:
    ranks = [r[key] for r in rows if r["species"] in members]
    return float(np.mean(ranks)) if ranks else None


def sample_metrics(rows: List[dict], truth: Set[str], fp: Set[str], k: int) -> Optional[dict]:
    if not truth and not fp:
        return None
    before = _order(rows, "original_rank")
    after = _order(rows, "new_rank")

    def topk(order, members):
        return sum(1 for s in order[:k] if s in members)

    m = {
        # known-member recovery
        "mrr_truth_before": reciprocal_rank(before, truth),
        "mrr_truth_after": reciprocal_rank(after, truth),
        "ndcg_truth_before": ndcg(before, truth, k),
        "ndcg_truth_after": ndcg(after, truth, k),
        "recall_truth_before": topk(before, truth) / len(truth) if truth else None,
        "recall_truth_after": topk(after, truth) / len(truth) if truth else None,
        "delta_rank_truth": float(np.mean([r["delta_rank"] for r in rows if r["species"] in truth])) if truth else None,
        # false-positive reduction
        "fp_at_k_before": topk(before, fp) if fp else None,
        "fp_at_k_after": topk(after, fp) if fp else None,
        "delta_rank_fp": float(np.mean([r["delta_rank"] for r in rows if r["species"] in fp])) if fp else None,
        # separation (want: FP ranked LOWER than truth => larger positive separation)
        "sep_before": _sep(rows, truth, fp, "original_rank"),
        "sep_after": _sep(rows, truth, fp, "new_rank"),
        "n_truth": len(truth), "n_fp": len(fp), "n_candidates": len(rows),
    }
    return m


def _sep(rows, truth, fp, key) -> Optional[float]:
    mt = _mean_rank(rows, truth, key)
    mf = _mean_rank(rows, fp, key)
    if mt is None or mf is None:
        return None
    return mf - mt  # FP mean rank minus truth mean rank; >0 means FP ranked below truth


# ---------------------------------------------------------------------------
# Extrinsic permutation null: permute coherence across candidate taxa
# ---------------------------------------------------------------------------

def _coherence(rows: List[dict], beta: float) -> np.ndarray:
    """Recover per-taxon coherence c from final_score = reads*(1+beta*c)  (beta from the run)."""
    reads = np.array([max(r["reads"], 1) for r in rows], dtype=float)
    final = np.array([r["final_score"] for r in rows], dtype=float)
    return np.maximum((final / reads - 1.0) / beta, 0.0)


def permutation_sep_improvement(samples: List[dict], beta: float,
                                n_perm: int, rng: np.random.Generator) -> Optional[dict]:
    """Null: within each sample, shuffle the coherence vector across taxa, recompute the
    reranked order, and recompute mean Δseparation. Compare to the observed Δseparation."""
    obs = [s["sep_after"] - s["sep_before"] for s in samples
           if s["sep_after"] is not None and s["sep_before"] is not None]
    if not obs:
        return None
    observed = float(np.mean(obs))

    null = np.empty(n_perm)
    for b in range(n_perm):
        diffs = []
        for s in samples:
            rows = s["_rows"]
            truth, fp = s["_truth"], s["_fp"]
            if s["sep_before"] is None:
                continue
            reads = np.array([max(r["reads"], 1) for r in rows], dtype=float)
            coh = s["_coh"].copy()
            rng.shuffle(coh)
            final = reads * (1.0 + beta * coh)
            order = [rows[i]["species"] for i in np.argsort(-final, kind="stable")]
            rank = {sp: i + 1 for i, sp in enumerate(order)}
            mt = np.mean([rank[t] for t in truth if t in rank]) if truth else None
            mf = np.mean([rank[f] for f in fp if f in rank]) if fp else None
            if mt is None or mf is None:
                continue
            diffs.append((mf - mt) - s["sep_before"])
        null[b] = np.mean(diffs) if diffs else 0.0
    p = (np.sum(null >= observed) + 1) / (n_perm + 1)
    return {"observed_delta_sep": round(observed, 4),
            "null_delta_sep_mean": round(float(null.mean()), 4),
            "perm_p_separation": float(p), "n_perm": n_perm}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def evaluate(reranked: Dict[str, List[dict]], truth_by_sample: Dict[str, Set[str]],
             injected_by_sample: Dict[str, Set[str]], truth_mode: str,
             k: int, beta: float, n_perm: int, n_boot: int, seed: int) -> dict:
    per_sample = []
    for sid, rows in reranked.items():
        truth = truth_by_sample.get(sid, set())
        inj = injected_by_sample.get(sid, set())
        truth_p, fp_p = label_sample(rows, truth, inj, truth_mode)
        m = sample_metrics(rows, truth_p, fp_p, k)
        if m is None:
            continue
        m["_rows"] = rows
        m["_truth"] = truth_p
        m["_fp"] = fp_p
        m["_coh"] = _coherence(rows, beta)
        m["sample_id"] = sid
        per_sample.append(m)

    if not per_sample:
        raise SystemExit("No sample had any truth/FP overlap with the reranked lists — "
                         "check name normalisation and sample-id matching.")

    def agg(field):
        vals = [s[field] for s in per_sample if s.get(field) is not None]
        return float(np.mean(vals)) if vals else None

    def improvement(after_f, before_f):
        return [s[after_f] - s[before_f] for s in per_sample
                if s.get(after_f) is not None and s.get(before_f) is not None]

    out = {
        "n_samples": len(per_sample),
        "truth_mode": truth_mode, "k": k, "beta": beta,
        "total_truth": sum(s["n_truth"] for s in per_sample),
        "total_fp": sum(s["n_fp"] for s in per_sample),
        # known-member recovery
        "MRR_truth_before": _r(agg("mrr_truth_before")),
        "MRR_truth_after": _r(agg("mrr_truth_after")),
        "NDCG_truth_before": _r(agg("ndcg_truth_before")),
        "NDCG_truth_after": _r(agg("ndcg_truth_after")),
        "recall_truth_before": _r(agg("recall_truth_before")),
        "recall_truth_after": _r(agg("recall_truth_after")),
        "mean_delta_rank_truth": _r(agg("delta_rank_truth")),
        # false-positive reduction
        "FP_at_k_before": _r(agg("fp_at_k_before")),
        "FP_at_k_after": _r(agg("fp_at_k_after")),
        "mean_delta_rank_fp": _r(agg("delta_rank_fp")),
        # separation
        "separation_before": _r(agg("sep_before")),
        "separation_after": _r(agg("sep_after")),
    }

    if n_boot > 0:
        out["MRR_truth_improvement_CI95"] = bootstrap_ci(
            improvement("mrr_truth_after", "mrr_truth_before"), n_boot, seed)
        sep_impr = [s["sep_after"] - s["sep_before"] for s in per_sample
                    if s["sep_after"] is not None and s["sep_before"] is not None]
        out["separation_improvement_CI95"] = bootstrap_ci(sep_impr, n_boot, seed + 1)
    if n_perm > 0:
        perm = permutation_sep_improvement(per_sample, beta, n_perm,
                                           np.random.default_rng(seed))
        if perm:
            out.update(perm)

    # strip private fields, keep a compact per-sample record
    out["per_sample"] = [{kk: s[kk] for kk in
                          ("sample_id", "n_truth", "n_fp", "n_candidates",
                           "mrr_truth_before", "mrr_truth_after",
                           "fp_at_k_before", "fp_at_k_after",
                           "sep_before", "sep_after")} for s in per_sample]
    return out


def _r(x, nd=4):
    return round(x, nd) if isinstance(x, float) else x


# ---------------------------------------------------------------------------
# Demo / self-test — synthetic mock, so the evaluator runs end-to-end today
# ---------------------------------------------------------------------------

def make_demo(seed: int = 42, n_samples: int = 12, n_truth: int = 12, n_fp: int = 6):
    """Build a synthetic reranked_per_sample dict + truth/injected maps that mimic a mock
    community: true members carry positive coherence (rise), injected intruders carry ~0
    coherence (fall). Proves the metrics + null compute correctly before real data exists."""
    rng = np.random.default_rng(seed)
    reranked: Dict[str, List[dict]] = {}
    truth_map: Dict[str, Set[str]] = {}
    inj_map: Dict[str, Set[str]] = {}
    for si in range(n_samples):
        sid = f"MOCK{si:02d}"
        truth = [f"true_species_{si}_{j}" for j in range(n_truth)]
        fps = [f"intruder_{si}_{j}" for j in range(n_fp)]
        species = truth + fps
        reads = rng.lognormal(5.0, 1.5, size=len(species)) + 1
        # true members get real coherence (documented partners present); intruders ~0
        coh = np.concatenate([rng.gamma(2.0, 1.0, size=n_truth),
                              rng.gamma(0.05, 0.2, size=n_fp)])
        final = reads * (1.0 + coh)
        rows = [{"species": species[i], "reads": int(reads[i]),
                 "final_score": float(final[i])} for i in range(len(species))]
        # baseline order = by reads desc; reranked order = by final_score desc
        for rk, i in enumerate(np.argsort(-reads, kind="stable"), 1):
            rows[i]["original_rank"] = rk
        for rk, i in enumerate(np.argsort(-final, kind="stable"), 1):
            rows[i]["new_rank"] = rk
        for r in rows:
            r["delta_rank"] = r["original_rank"] - r["new_rank"]
        reranked[sid] = rows
        truth_map[sid] = {_norm(t) for t in truth}
        inj_map[sid] = {_norm(f) for f in fps}
    return reranked, truth_map, inj_map


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Extrinsic (H4b) evaluation of ampliseq reranking")
    p.add_argument("--reranked", help="reranked_per_sample.json from ampliseq_rerank.py")
    p.add_argument("--truth", help="TSV: sample_id <tab> species (known members)")
    p.add_argument("--injected", help="TSV: sample_id <tab> species (known-wrong spike-ins)")
    p.add_argument("--truth-mode", choices=["closed", "open"], default="closed",
                   help="closed: non-truth candidates are FP (defined mock). "
                        "open: only --injected are FP (real field data + spike-ins).")
    p.add_argument("--k", type=int, default=10, help="top-k for @k metrics")
    p.add_argument("--beta", type=float, default=1.0,
                   help="beta used in the reranking run (to recover coherence for the null)")
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", help="output JSON path")
    p.add_argument("--demo", action="store_true",
                   help="run the synthetic-mock self-test (no real data needed)")
    args = p.parse_args()

    if args.demo:
        print("=== DEMO: synthetic mock community (self-test) ===", flush=True)
        reranked, truth_by, inj_by = make_demo(seed=args.seed)
    else:
        if not args.reranked or not args.truth:
            p.error("--reranked and --truth are required (or use --demo)")
        reranked = load_reranked(Path(args.reranked))
        truth_by = load_species_tsv(Path(args.truth))
        inj_by = load_species_tsv(Path(args.injected) if args.injected else None)

    res = evaluate(reranked, truth_by, inj_by, args.truth_mode,
                   args.k, args.beta, args.n_perm, args.n_boot, args.seed)

    print(f"\n{'='*62}")
    print(f"  samples={res['n_samples']}  truth={res['total_truth']}  FP={res['total_fp']}  "
          f"mode={res['truth_mode']}  k={res['k']}")
    print(f"  -- known-member recovery --")
    print(f"  MRR(truth):   {res['MRR_truth_before']:.4f} -> {res['MRR_truth_after']:.4f}"
          + (f"   CI95 {res['MRR_truth_improvement_CI95']}" if res.get('MRR_truth_improvement_CI95') else ""))
    print(f"  recall@{res['k']}(truth): {res['recall_truth_before']} -> {res['recall_truth_after']}")
    print(f"  mean Δrank(truth): {res['mean_delta_rank_truth']:+}  (>0 = moved up)")
    print(f"  -- false-positive reduction --")
    print(f"  FP@{res['k']}: {res['FP_at_k_before']} -> {res['FP_at_k_after']}  (lower = better)")
    print(f"  mean Δrank(FP):   {res['mean_delta_rank_fp']:+}  (<0 = moved down)")
    print(f"  -- separation (FP rank - truth rank; want it to grow) --")
    print(f"  separation: {res['separation_before']:.3f} -> {res['separation_after']:.3f}"
          + (f"   CI95 {res['separation_improvement_CI95']}" if res.get('separation_improvement_CI95') else ""))
    if 'perm_p_separation' in res:
        print(f"  permutation null (coherence shuffled across taxa, n={res['n_perm']}): "
              f"observed Δsep {res['observed_delta_sep']} vs null {res['null_delta_sep_mean']}  "
              f"p={res['perm_p_separation']:.4g}")
    print(f"{'='*62}\n")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Saved to {out}")


if __name__ == "__main__":
    main()
