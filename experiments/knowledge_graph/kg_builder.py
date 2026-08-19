#!/usr/bin/env python3
"""
kg_builder.py — Typed directed Knowledge Graph from classifier + NER output.

Takes sentence-level predictions from the enriched pipeline (port 8002) and
builds a KG where edges are typed, directed (A infects B ≠ B infects A), and
symmetric interactions (symbioticWith) are explicitly flagged as undirected.

Direction inference (4-step cascade):
  1. NER entity type-pair table  (PATHOGEN+HOST → infects, directed)
  2. robiext_v2025 RO concept lookup (pollinates → directed, plant=object)
  3. Explicit symmetric set  (symbioticWith → undirected)
  4. Fallback (undirected, flag="ambiguous")

Validation gates (both reused from existing modules):
  - robi_prefilter.is_pair_plausible()   kingdom-pair plausibility
  - RobiValidator.validate_interaction() ROBI domain rules

Usage (CLI):
  python kg_builder.py --input eval_100.tsv    --mode sentences --output kg.csv
  python kg_builder.py --input flagged.csv     --mode pairs     --output kg.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT       = Path(__file__).resolve().parents[3]
CLASSIFIER = ROOT / "classifier"

for _p in [str(ROOT), str(CLASSIFIER), str(CLASSIFIER / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Reused modules ─────────────────────────────────────────────────────────────

from src.data.robi_validator import RobiValidator  # noqa: E402

_ROBI_PREFILTER = ROOT / "biotx_community_check" / "biotx_block" / "robi_prefilter.py"
if _ROBI_PREFILTER.exists():
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("robi_prefilter", _ROBI_PREFILTER)
    _mod  = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _is_pair_plausible = _mod.is_pair_plausible
else:
    def _is_pair_plausible(ka, kb):  # graceful fallback
        return True, []

# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class KGEntity:
    text:        str
    entity_type: str            # HOST / PATHOGEN / VECTOR / RESERVOIR / DISEASE / ORGANISM
    normalized:  str = ""       # canonical name (OTT cache or lower-stripped)
    kingdom:     Optional[str] = None

    def __post_init__(self):
        if not self.normalized:
            self.normalized = self.text.strip().lower()


@dataclass
class KGEdge:
    subject:    KGEntity
    predicate:  str
    object:     KGEntity
    directed:   bool            # False = symmetric / undirected
    ro_id:      Optional[str] = None
    confidence: float = 0.0
    n_sources:  int   = 1
    sources:    List[Dict] = field(default_factory=list)
    flag:       Optional[str] = None   # ambiguous / kingdom_mismatch / unvalidated

    @property
    def triple_key(self) -> Tuple[str, str, str]:
        """Canonical key for deduplication (direction-aware for directed edges)."""
        s, o = self.subject.normalized, self.object.normalized
        if not self.directed:
            s, o = min(s, o), max(s, o)   # canonical order for symmetric
        return (s, self.predicate, o)


# ── RelationMapper ─────────────────────────────────────────────────────────────

# Step 1: NER type-pair → (predicate, directed, ro_id)
# Covers the host-pathogen-vector axis where NER types encode direction directly.
NER_PAIR_TABLE: Dict[Tuple[str, str], Tuple[str, bool, Optional[str]]] = {
    ("PATHOGEN", "HOST"):     ("infects",       True,  "RO_0002556"),
    ("HOST",     "PATHOGEN"): ("infected_by",   True,  "RO_0002557"),
    ("VECTOR",   "HOST"):     ("transmits_to",  True,  "RO_0002459"),
    ("VECTOR",   "PATHOGEN"): ("carries",       True,  None),
    ("PATHOGEN", "DISEASE"):  ("causes",        True,  "RO_0003304"),
    ("DISEASE",  "HOST"):     ("affects",       True,  None),
    ("RESERVOIR","PATHOGEN"): ("reservoir_for", True,  None),
    ("PATHOGEN", "RESERVOIR"):("uses_reservoir",True,  None),
    ("HOST",     "HOST"):     ("coInfects_with",False, None),
}

# Step 3: explicit symmetric predicates (not derivable from NER types alone)
SYMMETRIC_PREDICATES = {
    "symbioticWith", "symbiotically interacts with",
    "mutualistOf", "mutualistically interacts with",
    "interactsWith", "biotically interacts with",
    "coHabitsWith", "ecologically co-occurs with",
    "amensalism", "amensalistically interacts with",
    "antibiosis",
    "commensually interacts with",   # note: commensalistOf IS directed
    "coInfects_with",
}

# RO IDs that are explicitly symmetric (from robiext_v2025)
SYMMETRIC_RO_IDS = {
    "RO_0002437",  # biotically interacts with
    "RO_0002438",  # trophically interacts with
    "RO_0002440",  # symbiotically interacts with
    "RO_0002441",  # commensually interacts with
    "RO_0002442",  # mutualistically interacts with
    "RO_0002434",  # interacts with
    "RO_0002466",  # is commensalism
    "RO_0008506",  # ecologically co-occurs with
    "RO_e000028",  # amensalistically interacts with
}

# Inverse marker keywords — concepts containing these are passive/inverse forms
_INVERSE_MARKERS = (
    "has ", " by", "pollinated", "infected", "visited by",
    "preyed", "eaten by", "killed by", "dispersed by",
    "parasitized by", "is host", "is prey",
)


def _is_inverse_term(term: str) -> bool:
    t = term.lower()
    return any(t.startswith("has ") or marker in t for marker in _INVERSE_MARKERS)


class RelationMapper:
    """
    Maps (NER entities, interaction_type_string) → (predicate, directed, ro_id).

    Builds lookup tables from robiext_v2025.json at init time.
    """

    def __init__(self, robiext_path: Optional[Path] = None):
        p = robiext_path or (CLASSIFIER / "data/taxonomies/robiext_v2025.json")
        self._term_table: Dict[str, Tuple[str, bool, str]] = {}  # term → (pred, directed, ro_id)
        self._ro_id_table: Dict[str, Tuple[str, bool]] = {}      # ro_id → (pred, directed)
        if p.exists():
            self._build_from_robiext(p)

    def _build_from_robiext(self, path: Path) -> None:
        with open(path) as f:
            concepts = json.load(f)["concepts"]

        for concept in concepts:
            ro_id   = concept["id"]
            term    = concept["preferred_term"]["term"]
            synonyms = [s["term"] for s in concept.get("synonyms", [])
                        if s.get("relevance", True)]

            symmetric = ro_id in SYMMETRIC_RO_IDS
            inverse   = (not symmetric) and _is_inverse_term(term)
            directed  = not symmetric

            # Predicate: clean active form (lowercase, strip trailing " of" for compactness)
            pred = term.lower().rstrip()

            self._ro_id_table[ro_id] = (pred, directed)

            for t in [term] + synonyms:
                key = t.strip().lower()
                self._term_table[key] = (pred, directed, ro_id)

    def lookup_term(self, interaction_type: str) -> Optional[Tuple[str, bool, str]]:
        """Return (predicate, directed, ro_id) for an interaction_type string, or None."""
        key = interaction_type.strip().lower()
        if key in self._term_table:
            return self._term_table[key]
        # Partial match on significant words
        for stored_key, val in self._term_table.items():
            if key in stored_key or stored_key in key:
                return val
        return None

    def infer_relation(
        self,
        entities: List[KGEntity],
        interaction_type: str,
        confidence: float,
        source: Optional[Dict] = None,
    ) -> Optional[KGEdge]:
        """
        Infer a typed directed/undirected KGEdge from entities + interaction_type.

        Returns None if fewer than 2 entities are provided.
        """
        if len(entities) < 2:
            return None

        src = source or {}
        sources = [{"sentence": src.get("sentence",""), "doi": src.get("doi"),
                    "score": confidence}]

        # ── Step 1: NER type-pair lookup ──────────────────────────────────────
        type_pair = (entities[0].entity_type, entities[1].entity_type)
        if type_pair in NER_PAIR_TABLE:
            pred, directed, ro_id = NER_PAIR_TABLE[type_pair]
            return KGEdge(
                subject=entities[0], predicate=pred, object=entities[1],
                directed=directed, ro_id=ro_id,
                confidence=confidence, sources=sources,
            )

        # Try reverse pair (model may emit in either order)
        rev_pair = (entities[1].entity_type, entities[0].entity_type)
        if rev_pair in NER_PAIR_TABLE:
            pred, directed, ro_id = NER_PAIR_TABLE[rev_pair]
            return KGEdge(
                subject=entities[1], predicate=pred, object=entities[0],
                directed=directed, ro_id=ro_id,
                confidence=confidence, sources=sources,
            )

        # ── Step 2: robiext term lookup ───────────────────────────────────────
        if interaction_type:
            match = self.lookup_term(interaction_type)
            if match:
                pred, directed, ro_id = match
                # Orient subject/object using NER types if informative, else keep order
                subj, obj = _orient_by_ner(entities[0], entities[1], pred)
                return KGEdge(
                    subject=subj, predicate=pred, object=obj,
                    directed=directed, ro_id=ro_id,
                    confidence=confidence, sources=sources,
                )

        # ── Step 3: explicit symmetric set ───────────────────────────────────
        if not interaction_type or interaction_type.lower() in {
            s.lower() for s in SYMMETRIC_PREDICATES
        }:
            return KGEdge(
                subject=entities[0], predicate=interaction_type or "interactsWith",
                object=entities[1], directed=False,
                confidence=confidence, sources=sources, flag="ambiguous",
            )

        # ── Step 4: fallback — undirected, unvalidated ────────────────────────
        return KGEdge(
            subject=entities[0], predicate=interaction_type,
            object=entities[1], directed=False,
            confidence=confidence, sources=sources, flag="unvalidated",
        )


def _orient_by_ner(a: KGEntity, b: KGEntity, predicate: str) -> Tuple[KGEntity, KGEntity]:
    """
    Use NER entity types to correctly orient subject → object for a predicate.
    E.g. pollinates: ORGANISM→PLANT, so if b.entity_type contains 'HOST' and
    predicate is about pathogen, swap.
    Returns (subject, object).
    """
    pred = predicate.lower()
    # pollinates: animal/insect pollinates plant
    if "pollinat" in pred:
        if b.kingdom in ("Plantae", "Viridiplantae") or b.entity_type == "HOST":
            return a, b   # a=pollinator, b=plant ✓
        if a.kingdom in ("Plantae", "Viridiplantae") or a.entity_type == "HOST":
            return b, a   # swap
    # Default: keep order from sentence (model output order is meaningful)
    return a, b


# ── EntityNormalizer ───────────────────────────────────────────────────────────

class EntityNormalizer:
    """
    Resolve entity text to canonical taxon names via OTT cache.
    Falls back to lowercase stripped text.
    OTT cache format: {name_lower: ott_id_str} — built by ampliseq reranker.
    """

    def __init__(self, ott_cache_path: Optional[Path] = None):
        self._cache: Dict[str, str] = {}
        if ott_cache_path and ott_cache_path.exists():
            try:
                with open(ott_cache_path) as f:
                    raw = json.load(f)
                # raw values may be "ott:12345" or just the name
                self._cache = {k.lower(): str(v) for k, v in raw.items()}
            except Exception:
                pass

    def normalize(self, text: str) -> str:
        key = text.strip().lower()
        return self._cache.get(key, key)


# ── KnowledgeGraph accumulator ─────────────────────────────────────────────────

class KnowledgeGraph:
    """
    Accumulates KGEdge objects across sentences, deduplicates, and exports.

    Merge semantics: same (subject.normalized, predicate, object.normalized) triple
    → accumulate sources, confidence = mean(scores), n_sources = count.
    """

    def __init__(
        self,
        robiext_path: Optional[Path] = None,
        ott_cache_path: Optional[Path] = None,
    ):
        self.mapper     = RelationMapper(robiext_path)
        self.normalizer = EntityNormalizer(ott_cache_path)
        self.validator  = RobiValidator()
        self._edges: List[KGEdge] = []
        self._merged: Optional[List[KGEdge]] = None

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def add_from_api_response(self, response: Dict[str, Any]) -> Optional[KGEdge]:
        """
        Parse a port 8002 /predict_enriched response dict and add the resulting edge.

        Expected keys: entities (list of {text, type, kingdom?}),
                       interaction_type, confidence/score, sentence, doi (optional).
        """
        raw_entities = response.get("entities", [])
        entities = [
            KGEntity(
                text=e.get("text", ""),
                entity_type=e.get("type", "ORGANISM"),
                normalized=self.normalizer.normalize(e.get("text", "")),
                kingdom=e.get("kingdom"),
            )
            for e in raw_entities
            if e.get("text")
        ]

        interaction_type = response.get("interaction_type", "")
        confidence       = float(response.get("confidence", response.get("score", 0.0)))
        source = {"sentence": response.get("sentence",""), "doi": response.get("doi")}

        edge = self.mapper.infer_relation(entities, interaction_type, confidence, source)
        if edge is None:
            return None

        # Validation gates
        plausible, _ = _is_pair_plausible(edge.subject.kingdom, edge.object.kingdom)
        if not plausible:
            edge.flag = "kingdom_mismatch"

        valid, violations = self.validator.validate_interaction(
            edge.subject.text, edge.object.text,
            edge.predicate,
            edge.subject.kingdom, edge.object.kingdom,
        )
        if not valid:
            if edge.flag is None:
                edge.flag = "kingdom_mismatch"

        self._edges.append(edge)
        self._merged = None   # invalidate merge cache
        return edge

    def add_edge(self, edge: KGEdge) -> None:
        self._edges.append(edge)
        self._merged = None

    # ── Merge ─────────────────────────────────────────────────────────────────

    def merge_edges(self) -> List[KGEdge]:
        """
        Deduplicate edges by triple_key.
        Merged edge: confidence = mean(scores), n_sources = count, sources accumulated.
        """
        groups: Dict[Tuple, List[KGEdge]] = defaultdict(list)
        for e in self._edges:
            groups[e.triple_key].append(e)

        merged = []
        for key, group in groups.items():
            base = group[0]
            all_sources = [s for e in group for s in e.sources]
            scores = [s.get("score", 0.0) for s in all_sources if s.get("score") is not None]
            merged.append(KGEdge(
                subject=base.subject,
                predicate=base.predicate,
                object=base.object,
                directed=base.directed,
                ro_id=base.ro_id,
                confidence=sum(scores) / len(scores) if scores else 0.0,
                n_sources=len(group),
                sources=all_sources,
                flag=base.flag,
            ))
        self._merged = merged
        return merged

    @property
    def edges(self) -> List[KGEdge]:
        if self._merged is None:
            self.merge_edges()
        return self._merged

    # ── Export ────────────────────────────────────────────────────────────────

    def to_networkx(self) -> nx.DiGraph:
        """
        DiGraph for directed edges.
        Symmetric edges are added as both A→B and B→A with edge attr directed=False.
        """
        G = nx.DiGraph()
        for e in self.edges:
            attrs = {
                "predicate": e.predicate,
                "directed":  e.directed,
                "ro_id":     e.ro_id,
                "confidence":e.confidence,
                "n_sources": e.n_sources,
                "flag":      e.flag,
            }
            G.add_node(e.subject.normalized,
                       entity_type=e.subject.entity_type,
                       kingdom=e.subject.kingdom,
                       text=e.subject.text)
            G.add_node(e.object.normalized,
                       entity_type=e.object.entity_type,
                       kingdom=e.object.kingdom,
                       text=e.object.text)
            G.add_edge(e.subject.normalized, e.object.normalized, **attrs)
            if not e.directed:   # symmetric → both directions
                G.add_edge(e.object.normalized, e.subject.normalized, **attrs)
        return G

    def export_csv(self, path: Path) -> pd.DataFrame:
        """Export edges as CSV: subject, predicate, object, directed, ro_id, confidence, n_sources, flag."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for e in self.edges:
            rows.append({
                "subject":        e.subject.text,
                "subject_norm":   e.subject.normalized,
                "subject_type":   e.subject.entity_type,
                "subject_kingdom":e.subject.kingdom,
                "predicate":      e.predicate,
                "object":         e.object.text,
                "object_norm":    e.object.normalized,
                "object_type":    e.object.entity_type,
                "object_kingdom": e.object.kingdom,
                "directed":       e.directed,
                "ro_id":          e.ro_id,
                "confidence":     round(e.confidence, 4),
                "n_sources":      e.n_sources,
                "flag":           e.flag or "",
            })
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        return df

    def to_jsonld(self, path: Path) -> Dict:
        """
        Export as JSON-LD with @context mapping predicates to RO IRIs.
        """
        RO_BASE = "http://purl.obolibrary.org/obo/"
        context = {
            "@vocab":        "http://purl.obolibrary.org/obo/",
            "subject":       "@id",
            "object":        "@id",
            "predicate":     "@type",
            "directed":      "schema:direction",
            "confidence":    "schema:certainty",
        }
        # Add ro_id → IRI mappings found in the graph
        for e in self.edges:
            if e.ro_id:
                label = e.predicate.replace(" ", "_")
                context[label] = RO_BASE + e.ro_id.replace(":", "_")

        graph = []
        for e in self.edges:
            graph.append({
                "@type":      e.predicate,
                "subject":    e.subject.normalized,
                "object":     e.object.normalized,
                "directed":   e.directed,
                "ro_id":      e.ro_id,
                "confidence": e.confidence,
                "n_sources":  e.n_sources,
                "flag":       e.flag,
            })

        doc = {"@context": context, "@graph": graph}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=2))
        return doc

    def stats(self) -> Dict:
        directed   = sum(1 for e in self.edges if e.directed)
        undirected = sum(1 for e in self.edges if not e.directed)
        flagged    = sum(1 for e in self.edges if e.flag)
        return {
            "total_edges": len(self.edges),
            "directed":    directed,
            "undirected":  undirected,
            "flagged":     flagged,
            "unique_subjects": len({e.subject.normalized for e in self.edges}),
            "unique_objects":  len({e.object.normalized for e in self.edges}),
        }


# ── CLI ────────────────────────────────────────────────────────────────────────

def _run_sentences_mode(input_path: Path, kg: KnowledgeGraph) -> None:
    """
    Load eval TSV / CSV with columns: text, label, source_species, target_species,
    interaction_type (all optional except text). Simulate API response per row.
    """
    df = pd.read_csv(input_path, sep=None, engine="python")
    col = "text" if "text" in df.columns else df.columns[0]
    for _, row in df.iterrows():
        text = str(row.get("text", row.iloc[0]))
        sp1  = str(row.get("source_species", "") or "")
        sp2  = str(row.get("target_species",  "") or "")
        itype = str(row.get("interaction_type", "") or "")
        score = float(row.get("score", row.get("confidence", 0.8)))

        entities = []
        if sp1 and sp1 != "nan":
            entities.append({"text": sp1, "type": "HOST"})
        if sp2 and sp2 != "nan":
            entities.append({"text": sp2, "type": "PATHOGEN"})

        kg.add_from_api_response({
            "entities": entities,
            "interaction_type": itype,
            "confidence": score,
            "sentence": text,
        })


def _run_pairs_mode(input_path: Path, kg: KnowledgeGraph) -> None:
    """
    Load flagged_species CSV (from sintax.py flag output).
    Columns: species, globi_partners (semicolon-separated), n_globi_partners, ...
    Emits one edge per (species, partner) pair with interaction_type=pollinates.
    """
    df = pd.read_csv(input_path)
    for _, row in df.iterrows():
        sp = str(row.get("species", ""))
        partners_raw = str(row.get("globi_partners", ""))
        if not sp or not partners_raw or partners_raw == "nan":
            continue
        for partner in partners_raw.split(";"):
            partner = partner.strip()
            if not partner:
                continue
            bx_score = float(row.get("biotxplorer_score", 0.0) or 0.0)
            kg.add_from_api_response({
                "entities": [
                    {"text": partner, "type": "ORGANISM"},
                    {"text": sp,      "type": "ORGANISM"},
                ],
                "interaction_type": "pollinates",
                "confidence": bx_score if bx_score > 0 else 0.5,
                "sentence": f"{partner} visits {sp}",
            })


def main() -> None:
    p = argparse.ArgumentParser(description="Build KG from classifier/NER output")
    p.add_argument("--input",   required=True, help="Input CSV/TSV")
    p.add_argument("--output",  required=True, help="Output CSV path")
    p.add_argument("--mode",    choices=["sentences", "pairs"], default="sentences")
    p.add_argument("--jsonld",  default=None, help="Optional JSON-LD output path")
    p.add_argument("--ott-cache", default=None, help="Path to ott_cache.json")
    p.add_argument("--min-confidence", type=float, default=0.0)
    args = p.parse_args()

    ott_path = Path(args.ott_cache) if args.ott_cache else None
    kg = KnowledgeGraph(ott_cache_path=ott_path)

    input_path = Path(args.input)
    if args.mode == "sentences":
        _run_sentences_mode(input_path, kg)
    else:
        _run_pairs_mode(input_path, kg)

    # Filter by confidence
    if args.min_confidence > 0:
        kg._edges = [e for e in kg._edges if e.confidence >= args.min_confidence]

    kg.merge_edges()
    df = kg.export_csv(Path(args.output))

    s = kg.stats()
    print(f"KG built: {s['total_edges']} edges "
          f"({s['directed']} directed, {s['undirected']} undirected, "
          f"{s['flagged']} flagged)")
    print(f"  Unique subjects: {s['unique_subjects']}  objects: {s['unique_objects']}")
    print(f"  Output: {args.output}")

    if args.jsonld:
        kg.to_jsonld(Path(args.jsonld))
        print(f"  JSON-LD: {args.jsonld}")


if __name__ == "__main__":
    main()
