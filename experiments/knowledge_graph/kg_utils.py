#!/usr/bin/env python3
"""Shared utilities for the KG sandbox experiment."""

from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx

# ── Project root & paths ───────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[3]  # MetaP/
CLASSIFIER = ROOT / "classifier"

# Add classifier to path so we can import from src/
for p in [str(ROOT), str(CLASSIFIER), str(CLASSIFIER / "experiments" / "multitask")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Category → SpERT relation mapping ─────────────────────────────────────────

CATEGORY_TO_SPERT_RELATION = {
    "INFECTION":      "INFECTS",        # source=PATHOGEN infects target=HOST
    "ENDOPARASITISM": "COLONIZED_BY",   # target HOST colonised_by source (endoparasite)
    "PARASITISM":     "INFECTED_BY",    # target HOST infected_by source
    "PARASITOIDISM":  "INFECTED_BY",
    "VECTOR":         "TRANSMITTED_BY", # source VECTOR → target HOST
}

SPERT_TRAINING_CATEGORIES = set(CATEGORY_TO_SPERT_RELATION.keys())

# ENTITY_TYPES order from relation_extractor.py: 'O','HOST','PATHOGEN','VECTOR','RESERVOIR','DISEASE'
SPERT_ENTITY_INDEX = {
    "O": 0, "HOST": 1, "PATHOGEN": 2, "VECTOR": 3, "RESERVOIR": 4, "DISEASE": 5,
}

SPERT_RELATION_INDEX = {
    "NO_RELATION": 0,
    "INFECTED_BY": 1, "INFECTS": 2,
    "TRANSMITS": 3, "TRANSMITTED_BY": 4,
    "VECTOR_OF": 5, "HAS_VECTOR": 6,
    "RESERVOIR_FOR": 7, "HAS_RESERVOIR": 8,
    "SUSCEPTIBLE_TO": 9, "RESISTANT_TO": 10,
    "COLONIZED_BY": 11, "COLONIZES": 12,
    "CAUSES_DISEASE": 13, "DISEASE_CAUSED_BY": 14,
    "CO_INFECTS_WITH": 15,
}

# ── Interaction type → category mapping (v14 column values) ───────────────────

def map_interaction_type_to_category(interaction_type: str) -> str:
    """Map a v14 interaction_type string to a canonical category (always returns plain str)."""
    try:
        from src.data.interaction_taxonomy import classify_interaction_type, _ensure_loaded
        _ensure_loaded()
        result = classify_interaction_type([], globi_type=str(interaction_type))
        # InteractionCategory is a str Enum; .value gives the plain string
        return result.value if hasattr(result, "value") else str(result)
    except Exception:
        pass
    # Fallback hardcoded mapping
    _FALLBACK = {
        "pathogenOf": "INFECTION", "endoparasiteOf": "ENDOPARASITISM",
        "parasiteOf": "PARASITISM", "ectoparasiteOf": "PARASITISM",
        "kleptoparasiteOf": "PARASITISM", "parasitoidOf": "PARASITOIDISM",
        "hasHost": "PARASITISM", "preysOn": "PREDATION", "eats": "PREDATION",
        "vectorOf": "VECTOR", "transmits": "VECTOR",
        "pollinates": "POLLINATION", "visitsFlowersOf": "POLLINATION",
        "grazesOn": "HERBIVORY", "feedsOn": "HERBIVORY",
        "dispersesSeedsOf": "DISPERSAL", "symbioticWith": "SYMBIOSIS",
        "mutualistOf": "SYMBIOSIS", "interactsWith": "GENERIC",
        "none": "GENERIC", "none_two_species": "GENERIC", "none_three_species": "GENERIC",
    }
    return _FALLBACK.get(str(interaction_type), "GENERIC")


# ── Species name normalisation ─────────────────────────────────────────────────

def normalize_name(name: str) -> str:
    return name.strip().lower()


def parse_species_list(val) -> List[str]:
    """Parse comma or JSON-encoded species list from DataFrame cell."""
    if not val or (isinstance(val, float)):
        return []
    s = str(val).strip()
    if not s or s in ("nan", "[]", ""):
        return []
    # Try JSON list
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [x.strip() for x in parsed if str(x).strip()]
    except (json.JSONDecodeError, ValueError):
        pass
    return [x.strip() for x in s.split(",") if x.strip()]


def find_abbreviated_spans(text: str, name: str) -> List[Tuple[int, int]]:
    """Find 'P. falciparum' style abbreviated forms of 'Plasmodium falciparum'."""
    parts = name.strip().split()
    if len(parts) != 2:
        return []
    genus, epithet = parts
    pattern = re.compile(
        r"\b" + re.escape(genus[0]) + r"\.\s*" + re.escape(epithet) + r"\b",
        re.IGNORECASE,
    )
    return [(m.start(), m.end()) for m in pattern.finditer(text)]


# ── KG persistence ─────────────────────────────────────────────────────────────

def save_kg(G: nx.MultiDiGraph, path: Path) -> None:
    """Save KG with pickle (fast, preserves all attributes)."""
    path = Path(path)
    with open(path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_kg(path: Path) -> nx.MultiDiGraph:
    """Load KG from pickle."""
    with open(str(path), "rb") as f:
        return pickle.load(f)


# ── KG statistics ──────────────────────────────────────────────────────────────

def compute_kg_stats(G: nx.MultiDiGraph) -> dict:
    """Compute PhD-relevant KG statistics."""
    stats: dict = {}

    nodes = list(G.nodes())
    edges = list(G.edges(data=True))
    stats["n_nodes"] = len(nodes)
    stats["n_edges"] = len(edges)

    # Unique (source, target) pairs regardless of parallel edges
    unique_pairs = set((u, v) for u, v, _ in edges)
    stats["n_unique_pairs"] = len(unique_pairs)

    # Category breakdown
    cats: Dict[str, int] = {}
    for _, _, d in edges:
        cat = str(d.get("category", "GENERIC"))
        if hasattr(cat, "value"):
            cat = cat.value  # type: ignore
        cats[cat] = cats.get(cat, 0) + 1
    stats["category_counts"] = dict(sorted(cats.items(), key=lambda x: -x[1]))

    # Source breakdown
    srcs: Dict[str, int] = {}
    for _, _, d in edges:
        src = d.get("source", "unknown")
        srcs[src] = srcs.get(src, 0) + 1
    stats["source_breakdown"] = dict(sorted(srcs.items(), key=lambda x: -x[1]))

    # Kingdom breakdown (node attributes)
    kingdoms: Dict[str, int] = {}
    for n in nodes:
        k = G.nodes[n].get("kingdom", "")
        if k:
            kingdoms[k] = kingdoms.get(k, 0) + 1
    stats["kingdom_breakdown"] = dict(sorted(kingdoms.items(), key=lambda x: -x[1]))

    # Degree stats (in simple graph for degree)
    SG = nx.DiGraph()
    for u, v, _ in edges:
        SG.add_edge(u, v)

    out_deg = sorted(SG.out_degree(), key=lambda x: -x[1])
    in_deg = sorted(SG.in_degree(), key=lambda x: -x[1])
    stats["top_hosts_by_outdegree"] = [(n, d) for n, d in out_deg[:20] if d > 0]
    stats["top_pathogens_by_indegree"] = [(n, d) for n, d in in_deg[:20] if d > 0]

    # Multi-host pathogens (pathogens with many hosts = generalists)
    path_hosts: Dict[str, set] = {}
    for u, v, d in edges:
        cat = d.get("category", "")
        if cat in ("INFECTION", "PARASITISM", "ENDOPARASITISM", "PARASITOIDISM"):
            path_hosts.setdefault(v, set()).add(u)
    stats["multi_host_pathogens"] = sorted(
        [(sp, len(hosts)) for sp, hosts in path_hosts.items()],
        key=lambda x: -x[1]
    )[:20]

    # Connected components
    undirected = SG.to_undirected()
    ccs = list(nx.connected_components(undirected))
    stats["n_weakly_connected_components"] = len(ccs)
    stats["largest_component_size"] = max(len(c) for c in ccs) if ccs else 0
    stats["n_isolated_nodes"] = sum(1 for n in SG.nodes() if SG.degree(n) == 0)

    # Hub species (betweenness centrality — skip for very large graphs)
    if len(SG) <= 5000:
        bc = nx.betweenness_centrality(SG, normalized=True)
        stats["hub_species"] = sorted(bc.items(), key=lambda x: -x[1])[:20]
    else:
        stats["hub_species"] = []

    return stats


# ── Query helpers ──────────────────────────────────────────────────────────────

def query_pathogen_hosts(G: nx.MultiDiGraph, pathogen: str) -> List[dict]:
    """Return all hosts of a given pathogen (species that have an edge to pathogen)."""
    pathogen = normalize_name(pathogen)
    results = []
    for u, v, d in G.edges(data=True):
        if normalize_name(v) == pathogen:
            results.append({"host": u, "category": d.get("category"), "source": d.get("source")})
    return results


def query_shared_pathogens(G: nx.MultiDiGraph, host1: str, host2: str) -> List[str]:
    """Pathogens shared between two hosts — potential zoonotic bridges."""
    h1, h2 = normalize_name(host1), normalize_name(host2)
    h1_pathogens = {normalize_name(v) for u, v, _ in G.edges(data=True) if normalize_name(u) == h1}
    h2_pathogens = {normalize_name(v) for u, v, _ in G.edges(data=True) if normalize_name(u) == h2}
    return sorted(h1_pathogens & h2_pathogens)
