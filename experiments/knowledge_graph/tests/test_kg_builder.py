"""Tests for kg_builder.py"""
from __future__ import annotations

import sys
import json
import tempfile
from pathlib import Path

import pytest
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "classifier" / "experiments" / "knowledge_graph"))

from kg_builder import (
    KGEntity, KGEdge, RelationMapper, KnowledgeGraph,
    NER_PAIR_TABLE, SYMMETRIC_PREDICATES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mapper():
    return RelationMapper()


@pytest.fixture
def kg():
    return KnowledgeGraph()


def _entity(text, etype, kingdom=None):
    return KGEntity(text=text, entity_type=etype, kingdom=kingdom)


def _make_response(sp1, type1, sp2, type2, interaction_type="", confidence=0.9,
                   kingdom1=None, kingdom2=None):
    return {
        "entities": [
            {"text": sp1, "type": type1, "kingdom": kingdom1},
            {"text": sp2, "type": type2, "kingdom": kingdom2},
        ],
        "interaction_type": interaction_type,
        "confidence": confidence,
        "sentence": f"{sp1} {interaction_type} {sp2}",
    }


# ---------------------------------------------------------------------------
# 1. PATHOGEN+HOST NER pair → infects, directed=True
# ---------------------------------------------------------------------------

def test_pathogen_host_direction(mapper):
    entities = [
        _entity("Plasmodium falciparum", "PATHOGEN"),
        _entity("Homo sapiens",          "HOST"),
    ]
    edge = mapper.infer_relation(entities, "pathogenOf", 0.95)
    assert edge is not None
    assert edge.directed is True
    assert edge.predicate == "infects"
    assert edge.subject.text == "Plasmodium falciparum"
    assert edge.object.text  == "Homo sapiens"
    assert edge.ro_id == "RO_0002556"


# ---------------------------------------------------------------------------
# 2. pollinates → directed=True, plant should be object
# ---------------------------------------------------------------------------

def test_pollinates_direction(mapper):
    entities = [
        _entity("Apis mellifera",    "ORGANISM", kingdom="Animalia"),
        _entity("Lotus corniculatus","ORGANISM", kingdom="Plantae"),
    ]
    edge = mapper.infer_relation(entities, "pollinates", 0.8)
    assert edge is not None
    assert edge.directed is True
    assert "pollinat" in edge.predicate.lower()


# ---------------------------------------------------------------------------
# 3. symbioticWith → directed=False
# ---------------------------------------------------------------------------

def test_symbiosis_undirected(mapper):
    entities = [
        _entity("Rhizobium leguminosarum", "ORGANISM"),
        _entity("Lotus corniculatus",       "ORGANISM"),
    ]
    edge = mapper.infer_relation(entities, "symbioticWith", 0.7)
    assert edge is not None
    assert edge.directed is False


# ---------------------------------------------------------------------------
# 4. Impossible kingdom pair (plant+plant + predator) → flag=kingdom_mismatch
# ---------------------------------------------------------------------------

def test_kingdom_gate_rejects(kg):
    response = _make_response(
        "Arabidopsis thaliana", "ORGANISM",
        "Mus musculus",         "ORGANISM",
        interaction_type="preysOn",
        kingdom1="Plantae", kingdom2="Animalia",
    )
    # RobiValidator: predator requires Animalia as source — Plantae as source = violation
    edge = kg.add_from_api_response(response)
    assert edge is not None
    assert edge.flag == "kingdom_mismatch"


# ---------------------------------------------------------------------------
# 5. Duplicate edges → n_sources=2, confidence=mean
# ---------------------------------------------------------------------------

def test_merge_duplicate_edges(kg):
    r1 = _make_response("Borrelia burgdorferi","PATHOGEN","Ixodes scapularis","VECTOR",
                        "vectorOf", confidence=0.8)
    r2 = _make_response("Borrelia burgdorferi","PATHOGEN","Ixodes scapularis","VECTOR",
                        "vectorOf", confidence=0.6)
    kg.add_from_api_response(r1)
    kg.add_from_api_response(r2)
    merged = kg.merge_edges()
    # Both edges should collapse to one triple
    matches = [e for e in merged
               if "borrelia" in e.subject.normalized and "ixodes" in e.object.normalized
               or "ixodes" in e.subject.normalized and "borrelia" in e.object.normalized]
    assert len(matches) == 1
    assert matches[0].n_sources == 2
    assert abs(matches[0].confidence - 0.7) < 1e-6


# ---------------------------------------------------------------------------
# 6. export_csv smoke test
# ---------------------------------------------------------------------------

def test_export_csv(kg, tmp_path):
    kg.add_from_api_response(_make_response(
        "Anopheles gambiae","VECTOR","Plasmodium falciparum","PATHOGEN",
        "transmits", confidence=0.9,
    ))
    out = tmp_path / "kg.csv"
    df = kg.export_csv(out)
    assert out.exists()
    assert len(df) >= 1
    assert "subject" in df.columns
    assert "predicate" in df.columns
    assert "directed" in df.columns
    assert "ro_id" in df.columns


# ---------------------------------------------------------------------------
# 7. to_networkx → DiGraph with correct nodes/edges
# ---------------------------------------------------------------------------

def test_to_networkx(kg):
    kg.add_from_api_response(_make_response(
        "Plasmodium falciparum","PATHOGEN","Homo sapiens","HOST",
        "infects", confidence=0.95,
    ))
    G = kg.to_networkx()
    assert isinstance(G, __import__("networkx").DiGraph)
    assert G.number_of_nodes() >= 2
    assert G.number_of_edges() >= 1


# ---------------------------------------------------------------------------
# 8. JSON-LD output has @context and RO IRI mapping
# ---------------------------------------------------------------------------

def test_jsonld_ro_context(kg, tmp_path):
    kg.add_from_api_response(_make_response(
        "Plasmodium falciparum","PATHOGEN","Homo sapiens","HOST",
        "infects", confidence=0.95,
    ))
    out = tmp_path / "kg.jsonld"
    doc = kg.to_jsonld(out)
    assert "@context" in doc
    assert "@graph" in doc
    assert len(doc["@graph"]) >= 1
    assert out.exists()
    # Context should have @vocab
    assert "@vocab" in doc["@context"]
