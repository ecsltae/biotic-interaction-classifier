#!/usr/bin/env python3
"""
Ampliseq re-ranking via biotic interaction coherence.

Pipeline:
  1. Normalize species names → OTT IDs via SiBILS autocomplete API
  2. Query BiotXplorer for pairwise text-mined interactions between detected species
  3. Compute per-sample community coherence score using BiotXplorer + GloBI fallback
  4. Re-rank species by: original_abundance × (1 + β × coherence_score)

APIs:
  - SiBILS name→OTT: https://biodiversitypmc.dev.sibils.org/api/otf/concept/autocomplete?source=ott&text=...
  - BiotXplorer:     https://biotxplorer.sibils.org/api/search?taxon1=ott:X&taxon2=ott:Y&collection=medline
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests
import pandas as pd
from scipy.stats import hypergeom as _hypergeom

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "classifier" / "experiments" / "knowledge_graph"))
from kg_utils import load_kg, normalize_name
import improved_ott_resolver

SIBILS_AUTOCOMPLETE = "https://biodiversitypmc.dev.sibils.org/api/otf/concept/autocomplete"
BIOTXPLORER_SEARCH  = "https://biotxplorer.sibils.org/api/search"
COLLECTIONS         = ["medline", "pmc"]  # try both


# ── OTT name resolution ────────────────────────────────────────────────────────

def resolve_ott(species_name: str, cache: Dict[str, Optional[str]]) -> Optional[str]:
    """Resolve species name → 'ott:ID' using SiBILS autocomplete. Cached."""
    key = species_name.lower().strip()
    if key in cache:
        return cache[key]
    try:
        r = requests.get(
            SIBILS_AUTOCOMPLETE,
            params={"source": "ott", "text": species_name},
            timeout=10,
        )
        if r.ok:
            data = r.json()
            results = data.get("results", [])
            for res in results:
                pref = res.get("pref_term", "").lower()
                if pref == key or pref.startswith(key.split()[0]):
                    ott_id = f"ott:{res['concept_id']}"
                    cache[key] = ott_id
                    return ott_id
    except Exception:
        pass
    cache[key] = None
    return None


def resolve_all_ott(species_list: List[str]) -> Dict[str, Optional[str]]:
    """Resolve all species names to OTT IDs with progress reporting."""
    print(f"Resolving {len(species_list)} species names to OTT IDs ...", flush=True)
    cache: Dict[str, Optional[str]] = {}
    for i, sp in enumerate(species_list):
        ott = resolve_ott(sp, cache)
        status = ott if ott else "NOT FOUND"
        print(f"  [{i+1}/{len(species_list)}] {sp} → {status}", flush=True)
        time.sleep(0.1)  # gentle rate limit
    resolved = sum(1 for v in cache.values() if v)
    print(f"  Resolved: {resolved}/{len(species_list)}", flush=True)
    return cache


# ── BiotXplorer interaction queries ───────────────────────────────────────────

def query_biotxplorer(
    ott1: str,
    ott2: str,
    collection: str = "medline",
    timeout: int = 10,
) -> List[dict]:
    """Query BiotXplorer for interactions between two species (by OTT ID)."""
    try:
        r = requests.get(
            BIOTXPLORER_SEARCH,
            params={"taxon1": ott1, "taxon2": ott2, "collection": collection},
            timeout=timeout,
        )
        if r.ok:
            return r.json() if isinstance(r.json(), list) else []
    except Exception:
        pass
    return []


def build_biotxplorer_index(
    ott_map: Dict[str, Optional[str]],
    cache_path: Optional[Path] = None,
) -> Dict[Tuple[str, str], List[dict]]:
    """
    Query BiotXplorer for all pairs of resolved species.
    Returns {(sp1_norm, sp2_norm): [interaction_records]}.
    Saves/loads from cache to avoid re-querying.
    """
    if cache_path and cache_path.exists():
        print(f"Loading BiotXplorer cache from {cache_path} ...", flush=True)
        with open(cache_path, "rb") as f:
            result = pickle.load(f)
        print(f"  {len(result)} pairs loaded from cache", flush=True)
        return result

    resolved = {sp: ott for sp, ott in ott_map.items() if ott}
    species_list = list(resolved.keys())
    pairs = list(combinations(species_list, 2))
    print(f"Querying BiotXplorer for {len(pairs)} species pairs ...", flush=True)

    # Load partial cache if it exists (allows resuming interrupted runs)
    partial_path = cache_path.with_suffix(".partial.pkl") if cache_path else None
    result: Dict[Tuple[str, str], List[dict]] = {}
    start_i = 0
    if partial_path and partial_path.exists():
        with open(partial_path, "rb") as f:
            partial = pickle.load(f)
        result = partial["result"]
        start_i = partial["next_i"]
        print(f"  Resuming from pair {start_i}/{len(pairs)} (partial cache)", flush=True)

    for i, (sp1, sp2) in enumerate(pairs):
        if i < start_i:
            continue
        ott1, ott2 = resolved[sp1], resolved[sp2]
        interactions = []
        for coll in COLLECTIONS:
            hits = query_biotxplorer(ott1, ott2, collection=coll)
            interactions.extend(hits)
            time.sleep(0.15)
        if interactions:
            key = (normalize_name(sp1), normalize_name(sp2))
            result[key] = interactions
            result[(key[1], key[0])] = interactions  # bidirectional
            print(f"  [{i+1}/{len(pairs)}] {sp1} ↔ {sp2}: {len(interactions)} interactions", flush=True)
        else:
            print(f"  [{i+1}/{len(pairs)}] {sp1} ↔ {sp2}: none", flush=True)

        # Save partial cache every 200 pairs
        if partial_path and (i + 1) % 200 == 0:
            with open(partial_path, "wb") as f:
                pickle.dump({"result": result, "next_i": i + 1}, f, protocol=pickle.HIGHEST_PROTOCOL)

    if cache_path:
        with open(cache_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  BiotXplorer cache saved: {cache_path}", flush=True)
        if partial_path and partial_path.exists():
            partial_path.unlink()  # remove partial now that full cache is written

    return result


# ── GloBI fallback index ───────────────────────────────────────────────────────

def load_globi_cache(cache_path: Path) -> Dict[Tuple[str, str], str]:
    print(f"Loading GloBI cache from {cache_path} ...", flush=True)
    with open(cache_path, "rb") as f:
        lookup = pickle.load(f)
    print(f"  {len(lookup):,} pairs", flush=True)
    return lookup


def build_globi_index(
    lookup: Dict[Tuple[str, str], str],
    kg_path: Optional[Path],
) -> Dict[str, Set[str]]:
    """GloBI + KG: {species_norm → set of known partners}."""
    index: Dict[str, Set[str]] = defaultdict(set)
    for (sp1, sp2) in lookup:
        index[sp1].add(sp2)
        index[sp2].add(sp1)
    if kg_path and kg_path.exists():
        G = load_kg(kg_path)
        for u, v, _ in G.edges(data=True):
            un, vn = normalize_name(u), normalize_name(v)
            index[un].add(vn)
            index[vn].add(un)
    return dict(index)


# ── Coherence scoring ──────────────────────────────────────────────────────────

def coherence_score_biotxplorer(
    species_norm: str,
    detected_set: Set[str],
    bx_index: Dict[Tuple[str, str], List[dict]],
) -> Tuple[float, int, List[str]]:
    """
    BiotXplorer-based coherence: sum of interaction scores for partners in sample.
    Returns (score, n_partners_found, partner_names).
    """
    total_score = 0.0
    partners_found = []
    for partner in detected_set:
        if partner == species_norm:
            continue
        hits = bx_index.get((species_norm, partner), [])
        if hits:
            best_score = max(h.get("score", 0) for h in hits)
            total_score += best_score
            interaction_types = list({h["interaction"]["preferred_term"] for h in hits})
            partners_found.append(f"{partner} ({', '.join(interaction_types)})")
    return total_score, len(partners_found), partners_found


def coherence_score_globi(
    species_norm: str,
    detected_set: Set[str],
    globi_index: Dict[str, Set[str]],
    method: str = "sqrt",
    population: Optional[int] = None,
) -> Tuple[float, int, List[str]]:
    """GloBI community-coherence score.

    method="sqrt" (default, backward-compatible): sublinear degree-normalised
        local density,  k / sqrt(max(1, K)).
    method="hypergeom": calibrated surprise. Under a null in which the detected
        community S is a random size-|S| subset of the taxon universe, the number
        of v's K known partners falling in S is Hypergeometric(N, K, |S|); the
        score is the upper-tail surprise -log10 P(X >= k). Parameter-free, and
        gives the score a probabilistic meaning (math roadmap sec. 3a).
    """
    partners = globi_index.get(species_norm, set())
    in_community = [p for p in partners if p in detected_set and p != species_norm]
    k = len(in_community)
    K = len(partners)
    if method == "hypergeom":
        N = population if population else len(globi_index)  # universe size
        n = len(detected_set)
        if k == 0 or K == 0 or N <= 0:
            score = 0.0
        else:
            sf = _hypergeom.sf(k - 1, N, min(K, N), min(n, N))  # P(X >= k)
            score = -math.log10(max(sf, 1e-300))
    else:  # "sqrt"
        score = k / math.sqrt(max(1, K))
    return score, k, in_community


def rerank_sample(
    sample_id: str,
    species_reads: pd.Series,
    bx_index: Dict[Tuple[str, str], List[dict]],
    globi_index: Dict[str, Set[str]],
    beta: float,
    coherence: str = "sqrt",
    population: Optional[int] = None,
) -> List[dict]:
    present = species_reads[species_reads > 0]
    if present.empty:
        return []

    detected_set = set(present.index.tolist())
    results = []

    for orig_rank, (sp, reads) in enumerate(
        present.sort_values(ascending=False).items(), start=1
    ):
        bx_score, bx_support, bx_partners = coherence_score_biotxplorer(
            sp, detected_set, bx_index
        )
        gl_score, gl_support, gl_partners = coherence_score_globi(
            sp, detected_set, globi_index, method=coherence, population=population
        )
        # BiotXplorer is primary; GloBI fills in when BiotXplorer has no data
        combined_score = bx_score if bx_score > 0 else gl_score
        final_score = reads * (1 + beta * combined_score)
        results.append({
            "species": sp,
            "reads": int(reads),
            "original_rank": orig_rank,
            "biotxplorer_score": round(bx_score, 4),
            "biotxplorer_support": bx_support,
            "biotxplorer_partners": bx_partners,
            "globi_support": gl_support,
            "globi_partners": gl_partners[:5],
            "final_score": round(final_score, 4),
        })

    results.sort(key=lambda x: -x["final_score"])
    for new_rank, r in enumerate(results, start=1):
        r["new_rank"] = new_rank
        r["delta_rank"] = r["original_rank"] - new_rank

    return results


# ── Data loading ───────────────────────────────────────────────────────────────

def load_taxonomy(taxonomy_path: Path, fmt: str = "auto") -> Dict[str, str]:
    """Load ASV→species mapping from BOLD CSV, SILVA TSV, or NCBI BLAST TSV.

    BOLD format (CSV): columns Status, ASV_ID, Species — keeps Status=="Success" rows.
    SILVA format (TSV): columns ASV_ID, Kingdom, ..., Genus, Species — uses best available rank.
    BLAST format (TSV): columns ASV_ID, species_blast, identity_pct, evalue — extracts first two words.
    fmt: "auto" detects from extension (.tsv → silva/blast by header, .csv → bold), or pass "silva"/"bold"/"blast".
    """
    if fmt == "auto":
        if not str(taxonomy_path).endswith(".tsv"):
            fmt = "bold"
        else:
            # Detect blast vs silva by header
            with open(taxonomy_path) as fh:
                header = fh.readline()
            fmt = "blast" if "species_blast" in header else "silva"

    mapping: Dict[str, str] = {}
    if fmt == "bold":
        df = pd.read_csv(taxonomy_path)
        df = df[df["Status"] == "Success"].dropna(subset=["Species"])
        for _, row in df.iterrows():
            asv = str(row["ASV_ID"]).strip()
            sp = str(row["Species"]).strip()
            if asv and sp and sp not in ("", "nan"):
                mapping[asv] = sp
    elif fmt == "blast":
        import re
        df = pd.read_csv(taxonomy_path, sep="\t")
        for _, row in df.iterrows():
            asv = str(row["ASV_ID"]).strip()
            raw = str(row.get("species_blast", "")).strip()
            if raw in ("", "nan", "No hit"):
                continue
            # Extract first binomial name (Genus species) from NCBI title
            m = re.search(r"([A-Z][a-z]+ [a-z]+)", raw)
            sp = m.group(1) if m else raw.split("|")[-1].strip()[:50]
            if asv and sp:
                mapping[asv] = sp
    else:  # silva TSV
        df = pd.read_csv(taxonomy_path, sep="\t")
        for _, row in df.iterrows():
            asv = str(row["ASV_ID"]).strip()
            # Prefer species > genus > family — never go above family for re-ranking
            sp = None
            for col in ("Species", "Genus", "Family"):
                val = str(row.get(col, "")).strip()
                if val and val not in ("", "nan", "None"):
                    sp = val
                    break
            if asv and sp:
                mapping[asv] = sp.replace("_", " ")  # UNITE uses Genus_species format

    print(f"  Taxonomy ({fmt}): {len(mapping)} ASVs → {len(set(mapping.values()))} unique taxa", flush=True)
    return mapping


def load_asv_table(asv_table_path: Path, taxonomy: Dict[str, str]) -> pd.DataFrame:
    print(f"Loading ASV table ...", flush=True)
    df = pd.read_csv(asv_table_path, sep="\t", index_col=0)
    assigned = [a for a in df.index if a in taxonomy]
    df = df.loc[assigned]
    df.index = [normalize_name(taxonomy[a]) for a in df.index]
    df.index.name = "species"
    df = df.groupby("species").sum()
    print(f"  {len(df)} species × {df.shape[1]} samples", flush=True)
    return df


# ── Report ─────────────────────────────────────────────────────────────────────

def generate_report(all_results: Dict[str, List[dict]], out_dir: Path) -> None:
    species_stats: Dict[str, dict] = defaultdict(lambda: {
        "deltas": [], "bx_scores": [], "samples_boosted": 0
    })
    for rows in all_results.values():
        for r in rows:
            sp = r["species"]
            species_stats[sp]["deltas"].append(r["delta_rank"])
            species_stats[sp]["bx_scores"].append(r["biotxplorer_score"])
            if r["delta_rank"] > 0:
                species_stats[sp]["samples_boosted"] += 1

    summary_rows = []
    for sp, s in species_stats.items():
        summary_rows.append({
            "species": sp,
            "avg_delta_rank": round(sum(s["deltas"]) / len(s["deltas"]), 3),
            "n_samples": len(s["deltas"]),
            "n_samples_boosted": s["samples_boosted"],
            "avg_biotxplorer_score": round(sum(s["bx_scores"]) / len(s["bx_scores"]), 4),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("avg_delta_rank", ascending=False)
    summary_df.to_csv(out_dir / "rank_changes_summary.csv", index=False)

    n_samples_active = sum(1 for rows in all_results.values() if rows)
    n_changed = sum(1 for rows in all_results.values() if any(r["delta_rank"] != 0 for r in rows))

    lines = [
        "# Ampliseq Re-ranking — Ecological Coherence Report\n",
        f"**Samples with detections:** {n_samples_active}  ",
        f"**Samples with rank changes:** {n_changed}  ",
        f"**Species:** {len(species_stats)}\n",
        "## Interaction data sources",
        "- **BiotXplorer** (primary): text-mined interactions from MEDLINE/PMC via SiBILS",
        "- **GloBI** (fallback): structured interaction database (4.2M species pairs)\n",
        "## Species rank changes\n",
        "| Species | Avg Δrank | Samples boosted | Avg BiotXplorer score |",
        "|---|---|---|---|",
    ]
    for _, row in summary_df.iterrows():
        sign = "+" if row["avg_delta_rank"] > 0 else ""
        lines.append(
            f"| {row['species']} | {sign}{row['avg_delta_rank']} | "
            f"{row['n_samples_boosted']}/{row['n_samples']} | "
            f"{row['avg_biotxplorer_score']} |"
        )

    lines += [
        "\n## Interpretation",
        "- **Δrank > 0**: species moved UP (interaction partners co-detected in same sample)",
        "- **Δrank < 0**: species moved DOWN (relatively less community support)",
        "- Score combines BiotXplorer text-mined evidence + GloBI database pairs\n",
    ]

    (out_dir / "report.md").write_text("\n".join(lines))
    print("\n=== Re-ranking Summary ===", flush=True)
    print(summary_df.to_string(index=False), flush=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def _annotate_trust(
    bx_index: Dict[Tuple[str, str], List[dict]],
    trust_service_url: str = "http://localhost:8003",
) -> Dict[Tuple[str, str], List[dict]]:
    """Call trust service /score_biotxplorer_hits for each species pair.

    Adds trust_score, retracted, retraction_type fields to each hit.
    Falls back gracefully if the trust service is unavailable.
    """
    try:
        import requests as _req
        _req.get(f"{trust_service_url}/health", timeout=3).raise_for_status()
    except Exception:
        print("  Trust service unavailable — skipping trust annotation", flush=True)
        return bx_index

    annotated = {}
    n_annotated = 0
    for (sp1, sp2), hits in bx_index.items():
        if not hits:
            annotated[(sp1, sp2)] = hits
            continue
        try:
            payload = {"hits": hits, "species1": sp1, "species2": sp2}
            r = requests.post(
                f"{trust_service_url}/score_biotxplorer_hits",
                json=payload,
                timeout=15,
            )
            if r.ok:
                annotated_hits = r.json().get("hits", hits)
                annotated[(sp1, sp2)] = annotated_hits
                n_annotated += 1
            else:
                annotated[(sp1, sp2)] = hits
        except Exception:
            annotated[(sp1, sp2)] = hits

    print(f"  Trust-annotated {n_annotated}/{len(bx_index)} species pairs", flush=True)
    return annotated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asv-table",   required=True)
    parser.add_argument("--taxonomy",    required=True)
    parser.add_argument("--globi-cache", required=True)
    parser.add_argument("--kg-path",     default=None)
    parser.add_argument("--output-dir",  required=True)
    parser.add_argument("--beta",        type=float, default=1.0)
    parser.add_argument("--coherence",   choices=["sqrt", "hypergeom"], default="sqrt",
                        help="GloBI coherence score: 'sqrt' (degree-normalised, default) "
                             "or 'hypergeom' (calibrated surprise).")
    parser.add_argument("--top-n",           type=int, default=20)
    parser.add_argument("--taxonomy-format", choices=["auto", "bold", "silva", "blast"], default="auto",
                        help="Taxonomy file format: auto-detected from extension by default")
    parser.add_argument("--skip-biotxplorer", action="store_true",
                        help="Skip BiotXplorer queries and use GloBI only (much faster)")
    parser.add_argument("--trust-annotate", action="store_true",
                        help="Annotate BiotXplorer hits with trust scores via port 8003 service")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bx_cache = out / "biotxplorer_cache.pkl"

    # 1. Load taxonomy + ASV table
    print("=== Step 1: Load ampliseq data ===", flush=True)
    taxonomy = load_taxonomy(Path(args.taxonomy), fmt=args.taxonomy_format)
    species_df = load_asv_table(Path(args.asv_table), taxonomy)
    unique_species = list(set(taxonomy.values()))

    # 2. Resolve OTT IDs (cache-aware top-up: keep existing good resolutions,
    #    fill the misses with the offline OTT dump + gnverifier resolver).
    print("\n=== Step 2: Resolve OTT IDs ===", flush=True)
    ott_cache_path = out / "ott_cache.json"
    ott_map = json.load(open(ott_cache_path)) if ott_cache_path.exists() else {}
    norm_species = sorted({normalize_name(s) for s in unique_species})
    unresolved = [n for n in norm_species if not ott_map.get(n)]
    if unresolved:
        print(f"  {len(unresolved)}/{len(norm_species)} unresolved; "
              f"topping up via offline OTT dump + gnverifier ...", flush=True)
        filled = improved_ott_resolver.resolve(unresolved)  # keys: normalized names
        for n in unresolved:
            ott_map[n] = filled.get(n)
        json.dump(ott_map, open(ott_cache_path, "w"), indent=2)
    resolved = sum(1 for v in ott_map.values() if v)
    print(f"  Resolved: {resolved}/{len(norm_species)} "
          f"({100*resolved/max(1,len(norm_species)):.0f}%)", flush=True)

    # 3. Query BiotXplorer for all pairs
    if args.skip_biotxplorer:
        print("\n=== Step 3: BiotXplorer SKIPPED (--skip-biotxplorer) ===", flush=True)
        bx_index = {}
    else:
        print("\n=== Step 3: Query BiotXplorer ===", flush=True)
        bx_index = build_biotxplorer_index(ott_map, bx_cache)
        if args.trust_annotate and bx_index:
            print("\n=== Step 3b: Trust annotation via port 8003 ===", flush=True)
            bx_index = _annotate_trust(bx_index)

    # 4. Build GloBI fallback index
    print("\n=== Step 4: Load GloBI fallback ===", flush=True)
    globi = load_globi_cache(Path(args.globi_cache))
    kg_path = Path(args.kg_path) if args.kg_path else None
    globi_index = build_globi_index(globi, kg_path)

    # 5. Re-rank per sample
    print(f"\n=== Step 5: Re-rank {species_df.shape[1]} samples "
          f"(β={args.beta}, coherence={args.coherence}) ===", flush=True)
    population = len(globi_index)
    all_results: Dict[str, List[dict]] = {}
    for sample_id in species_df.columns:
        rows = rerank_sample(sample_id, species_df[sample_id], bx_index, globi_index,
                             args.beta, coherence=args.coherence, population=population)
        if rows:
            all_results[sample_id] = rows[:args.top_n]

    print(f"  {len(all_results)} samples with detected species", flush=True)

    json.dump(all_results, open(out / "reranked_per_sample.json", "w"), indent=2)

    # 6. Report
    print("\n=== Step 6: Report ===", flush=True)
    generate_report(all_results, out)
    print(f"\nDone. Outputs in {out}", flush=True)


if __name__ == "__main__":
    main()
