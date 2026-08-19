"""
validate_reranking_globi.py — Intrinsic validation of the ampliseq re-ranking (H4a).

Methodology:
  1. Sample known ecological communities from GloBI: groups of species that share ≥N
     documented GloBI interaction partners (indicating a real co-occurring community).
  2. Simulate ASV tables for each community with random/uniform abundances.
  3. Apply the ampliseq re-ranking coherence scoring (GloBI-only, no BiotXplorer API calls).
  4. Measure whether known GloBI partners move UP in the ranking vs. a random baseline.

Metrics:
  - MRR  (Mean Reciprocal Rank): average of 1/rank for the first known partner
  - NDCG (Normalized Discounted Cumulative Gain): rank quality metric
  - Rank improvement: mean Δrank for species with known GloBI partners vs. those without
  - Per-interaction-type breakdown (PARASITISM, POLLINATION, HERBIVORY, etc.)

Usage:
    python validate_reranking_globi.py \\
        --globi-cache classifier/data/globi/globi_lookup_cache.pkl \\
        --n-communities 500 \\
        --min-partners 3 \\
        --community-size 20 \\
        --output results/validation/reranking_globi_validation.json \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "classifier"))
sys.path.insert(0, str(ROOT / "classifier" / "experiments" / "knowledge_graph"))

from ampliseq_rerank import coherence_score_globi, rerank_sample

import pandas as pd


# ---------------------------------------------------------------------------
# GloBI index loader
# ---------------------------------------------------------------------------

def load_globi_index(cache_path: Path) -> Tuple[Dict[str, Set[str]], Dict[str, str]]:
    """Load GloBI cache → (name→{partners}, (sp1,sp2)→itype) ."""
    print(f"Loading GloBI cache from {cache_path} ...", flush=True)
    with open(cache_path, "rb") as f:
        lookup: Dict[Tuple[str, str], str] = pickle.load(f)

    index: Dict[str, Set[str]] = defaultdict(set)
    pair_type: Dict[Tuple[str, str], str] = {}
    for (a, b), itype in lookup.items():
        index[a].add(b)
        index[b].add(a)
        pair_type[(a, b)] = itype
        pair_type[(b, a)] = itype

    print(f"  {len(index):,} species, {len(lookup):,} pairs", flush=True)
    return dict(index), pair_type


# ---------------------------------------------------------------------------
# Community sampling
# ---------------------------------------------------------------------------

def sample_communities(
    globi_index: Dict[str, Set[str]],
    pair_type: Dict[Tuple[str, str], str],
    n_communities: int,
    min_partners: int,
    community_size: int,
    rng: random.Random,
    interaction_filter: Optional[Set[str]] = None,
) -> List[Dict]:
    """
    Sample communities of `community_size` species where the focal species
    has ≥ min_partners known GloBI partners within the community.

    Returns a list of community dicts:
      {
        "focal": str,
        "species": [str, ...],       # all community members (focal included)
        "known_partners": [str, ...], # subset with documented GloBI links to focal
        "interaction_types": {sp: itype, ...},
      }
    """
    # Filter species with enough partners
    candidates = [
        sp for sp, partners in globi_index.items()
        if len(partners) >= min_partners and " " in sp  # rough binomial filter
    ]
    print(f"  {len(candidates):,} species with ≥{min_partners} GloBI partners", flush=True)

    all_species = list(globi_index.keys())
    communities = []
    attempts = 0
    max_attempts = n_communities * 20

    while len(communities) < n_communities and attempts < max_attempts:
        attempts += 1
        focal = rng.choice(candidates)
        partners = list(globi_index[focal])

        if interaction_filter:
            partners = [p for p in partners
                        if pair_type.get((focal, p), "") in interaction_filter
                        or pair_type.get((p, focal), "") in interaction_filter]
        if len(partners) < min_partners:
            continue

        # Choose min_partners known partners + fill rest with random species
        chosen_partners = rng.sample(partners, min(min_partners, len(partners)))
        n_noise = community_size - 1 - len(chosen_partners)
        noise = rng.sample(
            [s for s in all_species if s != focal and s not in chosen_partners],
            min(n_noise, len(all_species) - 1 - len(chosen_partners))
        )
        species_list = [focal] + chosen_partners + noise
        rng.shuffle(species_list)

        itypes = {p: pair_type.get((focal, p), pair_type.get((p, focal), "?"))
                  for p in chosen_partners}
        communities.append({
            "focal": focal,
            "species": species_list,
            "known_partners": chosen_partners,
            "interaction_types": itypes,
        })

    print(f"  Sampled {len(communities):,} communities ({attempts} attempts)", flush=True)
    return communities


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def reciprocal_rank(ranked_species: List[str], relevant: Set[str]) -> float:
    """MRR component: 1/rank of first relevant species in ranking."""
    for rank, sp in enumerate(ranked_species, start=1):
        if sp in relevant:
            return 1.0 / rank
    return 0.0


def ndcg(ranked_species: List[str], relevant: Set[str], k: int = 10) -> float:
    """NDCG@k: relevance = 1 if sp in relevant, else 0."""
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, sp in enumerate(ranked_species[:k], start=1)
        if sp in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def mean_rank(ranked_species: List[str], relevant: Set[str]) -> Optional[float]:
    ranks = [rank for rank, sp in enumerate(ranked_species, start=1) if sp in relevant]
    return sum(ranks) / len(ranks) if ranks else None


# ---------------------------------------------------------------------------
# Significance: permutation test (vs random ordering) + bootstrap CIs
# ---------------------------------------------------------------------------

def permutation_test(
    communities: List[Dict],
    observed_mrr: float,
    observed_ndcg: float,
    n_perm: int,
    rng: random.Random,
) -> Dict:
    """One-sided permutation test. Null: reranking carries no information, i.e.
    the community is randomly ordered. Builds a null distribution of the mean
    MRR/NDCG achieved by random orderings and asks whether the observed reranked
    means exceed it. p = (#{null >= observed} + 1) / (n_perm + 1)."""
    null_mrr = np.empty(n_perm)
    null_ndcg = np.empty(n_perm)
    for b in range(n_perm):
        mrr_b, ndcg_b = [], []
        for comm in communities:
            order = list(comm["species"])
            rng.shuffle(order)
            relevant = set(comm["known_partners"])
            mrr_b.append(reciprocal_rank(order, relevant))
            ndcg_b.append(ndcg(order, relevant))
        null_mrr[b] = np.mean(mrr_b)
        null_ndcg[b] = np.mean(ndcg_b)
    p_mrr = (np.sum(null_mrr >= observed_mrr) + 1) / (n_perm + 1)
    p_ndcg = (np.sum(null_ndcg >= observed_ndcg) + 1) / (n_perm + 1)
    return {
        "n_perm": n_perm,
        "null_MRR_mean": round(float(null_mrr.mean()), 4),
        "null_NDCG_mean": round(float(null_ndcg.mean()), 4),
        "perm_p_MRR": float(p_mrr),
        "perm_p_NDCG": float(p_ndcg),
    }


def bootstrap_ci(improvements: List[float], n_boot: int, seed: int,
                 alpha: float = 0.05) -> Optional[List[float]]:
    """Percentile bootstrap CI for the mean per-community improvement."""
    arr = np.asarray(improvements, dtype=float)
    if arr.size == 0:
        return None
    gen = np.random.default_rng(seed)
    boot = np.array([gen.choice(arr, size=arr.size, replace=True).mean()
                     for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [round(float(lo), 4), round(float(hi), 4)]


# ---------------------------------------------------------------------------
# Main validation loop
# ---------------------------------------------------------------------------

def run_validation(
    communities: List[Dict],
    globi_index: Dict[str, Set[str]],
    beta: float = 1.0,
    coherence: str = "sqrt",
    population: Optional[int] = None,
) -> Dict:
    mrr_before, mrr_after = [], []
    ndcg_before, ndcg_after = [], []
    delta_ranks = []
    per_type: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"before": [], "after": []})

    for comm in communities:
        focal = comm["focal"]
        species = comm["species"]
        known_partners = set(comm["known_partners"])
        itypes = comm["interaction_types"]

        # Simulate ASV table: uniform random reads (log-normal, common in ampliseq)
        n = len(species)
        reads = np.random.lognormal(mean=5.0, sigma=1.5, size=n).astype(int) + 1
        species_reads = pd.Series(reads, index=species)

        # Baseline: rank by reads only
        baseline_ranked = species_reads.sort_values(ascending=False).index.tolist()

        # Re-ranked: apply GloBI coherence
        reranked = rerank_sample(
            sample_id="sim",
            species_reads=species_reads,
            bx_index={},           # no BiotXplorer in intrinsic validation
            globi_index=globi_index,
            beta=beta,
            coherence=coherence,
            population=population,
        )
        reranked_list = [r["species"] for r in reranked]

        # Metrics
        mrr_b = reciprocal_rank(baseline_ranked, known_partners)
        mrr_a = reciprocal_rank(reranked_list, known_partners)
        ndcg_b = ndcg(baseline_ranked, known_partners)
        ndcg_a = ndcg(reranked_list, known_partners)

        mrr_before.append(mrr_b)
        mrr_after.append(mrr_a)
        ndcg_before.append(ndcg_b)
        ndcg_after.append(ndcg_a)

        # Δrank for known partners
        for sp in known_partners:
            b_rank = baseline_ranked.index(sp) + 1 if sp in baseline_ranked else None
            a_rank = reranked_list.index(sp) + 1 if sp in reranked_list else None
            if b_rank and a_rank:
                delta_ranks.append(b_rank - a_rank)  # positive = moved UP
                itype = itypes.get(sp, "unknown")
                per_type[itype]["before"].append(mrr_b)
                per_type[itype]["after"].append(mrr_a)

    n = len(communities)
    results = {
        "n_communities": n,
        "coherence": coherence,
        "population": population,
        "MRR_before": round(np.mean(mrr_before), 4),
        "MRR_after":  round(np.mean(mrr_after), 4),
        "MRR_improvement": round(np.mean(mrr_after) - np.mean(mrr_before), 4),
        "NDCG_before": round(np.mean(ndcg_before), 4),
        "NDCG_after":  round(np.mean(ndcg_after), 4),
        "NDCG_improvement": round(np.mean(ndcg_after) - np.mean(ndcg_before), 4),
        "mean_delta_rank": round(np.mean(delta_ranks), 3) if delta_ranks else None,
        "pct_partners_moved_up": round(
            100 * sum(1 for d in delta_ranks if d > 0) / len(delta_ranks), 1
        ) if delta_ranks else None,
        "per_interaction_type": {
            itype: {
                "MRR_before": round(np.mean(v["before"]), 4),
                "MRR_after":  round(np.mean(v["after"]), 4),
                "n": len(v["before"]),
            }
            for itype, v in sorted(per_type.items())
        },
        # per-community arrays for bootstrap (stripped before saving)
        "_arrays": {
            "mrr_before": mrr_before, "mrr_after": mrr_after,
            "ndcg_before": ndcg_before, "ndcg_after": ndcg_after,
        },
    }
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GloBI intrinsic validation of ampliseq re-ranking")
    parser.add_argument("--globi-cache", default=str(ROOT / "classifier/data/globi/globi_lookup_cache.pkl"))
    parser.add_argument("--n-communities", type=int, default=500)
    parser.add_argument("--min-partners", type=int, default=3,
                        help="Min GloBI partners within community for focal species")
    parser.add_argument("--community-size", type=int, default=20,
                        help="Number of species per simulated community")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Coherence boost multiplier (same as ampliseq_rerank.py)")
    parser.add_argument("--coherence", choices=["sqrt", "hypergeom"], default="sqrt",
                        help="GloBI coherence variant: sqrt (degree-normalised) or "
                             "hypergeom (calibrated -log10 upper-tail surprise under a "
                             "random-community null); default sqrt (backward-compatible).")
    parser.add_argument("--population", type=int, default=None,
                        help="Universe size N for the hypergeometric null (default: "
                             "number of GloBI species). Set to the per-sample candidate "
                             "count for a sensitivity check.")
    parser.add_argument("--output", default=str(ROOT / "classifier/results/validation/reranking_globi_validation.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-perm", type=int, default=1000,
                        help="Permutation-test iterations (0 to skip)")
    parser.add_argument("--n-boot", type=int, default=1000,
                        help="Bootstrap iterations for CIs (0 to skip)")
    parser.add_argument("--interaction-types", nargs="*",
                        help="Filter to these GloBI interaction types (e.g. PARASITISM POLLINATION)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    globi_index, pair_type = load_globi_index(Path(args.globi_cache))

    itype_filter = set(args.interaction_types) if args.interaction_types else None

    print(f"\nSampling {args.n_communities} communities (size={args.community_size}, "
          f"min_partners={args.min_partners}) ...", flush=True)
    communities = sample_communities(
        globi_index, pair_type,
        n_communities=args.n_communities,
        min_partners=args.min_partners,
        community_size=args.community_size,
        rng=rng,
        interaction_filter=itype_filter,
    )

    population = args.population if args.population else len(globi_index)
    print(f"\nRunning re-ranking validation (beta={args.beta}, "
          f"coherence={args.coherence}, population={population}) ...", flush=True)
    results = run_validation(communities, globi_index, beta=args.beta,
                             coherence=args.coherence, population=population)
    arrays = results.pop("_arrays")

    # Significance: bootstrap CIs on the improvement + permutation test vs random order
    if args.n_boot > 0:
        mrr_impr = [a - b for a, b in zip(arrays["mrr_after"], arrays["mrr_before"])]
        ndcg_impr = [a - b for a, b in zip(arrays["ndcg_after"], arrays["ndcg_before"])]
        results["MRR_improvement_CI95"] = bootstrap_ci(mrr_impr, args.n_boot, args.seed)
        results["NDCG_improvement_CI95"] = bootstrap_ci(ndcg_impr, args.n_boot, args.seed + 1)
    if args.n_perm > 0:
        print(f"  permutation test ({args.n_perm} iters) ...", flush=True)
        results.update(permutation_test(
            communities, results["MRR_after"], results["NDCG_after"],
            args.n_perm, random.Random(args.seed)))

    # Print summary
    print(f"\n{'='*60}")
    print(f"  N communities:     {results['n_communities']}")
    print(f"  MRR before:        {results['MRR_before']:.4f}")
    print(f"  MRR after:         {results['MRR_after']:.4f}  (Δ={results['MRR_improvement']:+.4f}"
          + (f", 95% CI {results['MRR_improvement_CI95']}" if 'MRR_improvement_CI95' in results else "") + ")")
    print(f"  NDCG before:       {results['NDCG_before']:.4f}")
    print(f"  NDCG after:        {results['NDCG_after']:.4f}  (Δ={results['NDCG_improvement']:+.4f}"
          + (f", 95% CI {results['NDCG_improvement_CI95']}" if 'NDCG_improvement_CI95' in results else "") + ")")
    if 'perm_p_MRR' in results:
        print(f"  Permutation test vs random order (n={results['n_perm']}):")
        print(f"    MRR:  observed {results['MRR_after']:.4f} vs null {results['null_MRR_mean']:.4f}  p={results['perm_p_MRR']:.4g}")
        print(f"    NDCG: observed {results['NDCG_after']:.4f} vs null {results['null_NDCG_mean']:.4f}  p={results['perm_p_NDCG']:.4g}")
    print(f"  Mean Δrank:        {results['mean_delta_rank']}")
    print(f"  Partners moved up: {results['pct_partners_moved_up']}%")
    print(f"\n  Per-interaction-type MRR:")
    for itype, v in results["per_interaction_type"].items():
        print(f"    {itype:30s}  before={v['MRR_before']:.3f}  after={v['MRR_after']:.3f}  n={v['n']}")
    print(f"{'='*60}\n")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
