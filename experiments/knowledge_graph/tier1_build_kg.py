#!/usr/bin/env python3
"""
Tier 1: Build a Knowledge Graph from existing v14 training data.

Input:  classifier/data/training/training_data_v14.csv
Output: results/tier1/kg_v14.pkl   — NetworkX MultiDiGraph (pickle)
        results/tier1/kg_stats.json — statistics
"""

import argparse
import json
import sys
import time
from pathlib import Path

import networkx as nx
import pandas as pd

# Bootstrap path so kg_utils and classifier imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kg_utils import (
    ROOT, CLASSIFIER,
    map_interaction_type_to_category,
    normalize_name, parse_species_list,
    save_kg, compute_kg_stats,
)


def load_taxonomy_cache(cache_path: Path) -> dict:
    """Load globi_taxonomy_cache.json → {lower_name: {kingdom, phylum, class, order}}."""
    if not cache_path.exists():
        print(f"  Taxonomy cache not found: {cache_path}", flush=True)
        return {}
    print(f"  Loading taxonomy cache ({cache_path.stat().st_size // 1_000_000} MB)...", flush=True)
    t0 = time.time()
    data = json.load(open(cache_path))
    taxonomy = data.get("species_taxonomy", {})
    print(f"  Loaded {len(taxonomy):,} species in {time.time()-t0:.1f}s", flush=True)
    return taxonomy


def build_kg(df: pd.DataFrame) -> nx.MultiDiGraph:
    """Build a directed multigraph from v14 positive rows with species annotations."""
    G = nx.MultiDiGraph()

    pos = df[
        (df["label"] == 1)
        & df["source_species"].notna()
        & df["source_species"].astype(str).str.strip().ne("")
        & df["target_species"].notna()
        & df["target_species"].astype(str).str.strip().ne("")
    ]
    print(f"  Positive rows with species: {len(pos):,}", flush=True)

    n_edges = 0
    for _, row in pos.iterrows():
        src_list = parse_species_list(row["source_species"])
        tgt_list = parse_species_list(row["target_species"])
        category = map_interaction_type_to_category(str(row.get("interaction_type", "")))
        source_tag = str(row.get("source", "v14"))
        text_snippet = str(row.get("text", ""))[:300]

        for src in src_list:
            src_n = normalize_name(src)
            if not src_n:
                continue
            G.add_node(src_n, name=src_n)
            for tgt in tgt_list:
                tgt_n = normalize_name(tgt)
                if not tgt_n or tgt_n == src_n:
                    continue
                G.add_node(tgt_n, name=tgt_n)
                G.add_edge(
                    src_n, tgt_n,
                    interaction_type=str(row.get("interaction_type", "")),
                    category=category,
                    source=source_tag,
                    text=text_snippet,
                )
                n_edges += 1

    print(f"  Graph: {G.number_of_nodes():,} nodes, {n_edges:,} edges", flush=True)
    return G


def enrich_with_taxonomy(G: nx.MultiDiGraph, taxonomy: dict) -> None:
    """Add kingdom/phylum/class to nodes from the GloBI taxonomy cache."""
    enriched = 0
    for node in G.nodes():
        info = taxonomy.get(node, {})
        if info:
            G.nodes[node].update({
                "kingdom": info.get("kingdom", ""),
                "phylum":  info.get("phylum", ""),
                "class":   info.get("class", ""),
                "order":   info.get("order", ""),
            })
            enriched += 1
    print(f"  Taxonomy enrichment: {enriched:,}/{G.number_of_nodes():,} nodes", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Tier 1: Build KG from v14 training data")
    parser.add_argument("--v14-path",      required=True)
    parser.add_argument("--taxonomy-cache", required=True)
    parser.add_argument("--output-dir",    required=True)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=== Tier 1: Building KG from v14 training data ===", flush=True)
    t0 = time.time()

    # Load training data
    print(f"Loading {args.v14_path} ...", flush=True)
    df = pd.read_csv(args.v14_path)
    print(f"  {len(df):,} total rows", flush=True)

    # Build graph
    G = build_kg(df)

    # Taxonomy enrichment
    taxonomy = load_taxonomy_cache(Path(args.taxonomy_cache))
    if taxonomy:
        enrich_with_taxonomy(G, taxonomy)

    # Save KG
    kg_path = out / "kg_v14.pkl"
    save_kg(G, kg_path)
    print(f"  KG saved: {kg_path}", flush=True)

    # Compute and save stats
    print("Computing KG statistics...", flush=True)
    stats = compute_kg_stats(G)
    stats_path = out / "kg_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved: {stats_path}", flush=True)

    # Print summary
    print(f"\n  Nodes: {stats['n_nodes']:,}", flush=True)
    print(f"  Edges: {stats['n_edges']:,}", flush=True)
    print(f"  Unique pairs: {stats['n_unique_pairs']:,}", flush=True)
    print(f"  Categories: {stats['category_counts']}", flush=True)
    print(f"  Top hosts (out-degree): {stats['top_hosts_by_outdegree'][:5]}", flush=True)
    print(f"  Top pathogens (in-degree): {stats['top_pathogens_by_indegree'][:5]}", flush=True)
    print(f"\nTier 1 complete in {time.time()-t0:.1f}s", flush=True)

    # Write sentinel
    (out / "tier1_done").touch()


if __name__ == "__main__":
    main()
