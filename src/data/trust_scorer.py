"""
trust_scorer.py — Composite trust/credibility score for interaction claims.

Scores a claim on four axes and combines them into a single composite score:
  - temporal_score:      publication recency (exponential decay)
  - evidence_score:      evidence type (field > in_vivo > lab > genomic > review)
  - contradiction_risk:  conflict in training data or GloBI (0=none, 1=high)
  - kg_confirmation:     whether the pair is known to GloBI

Integrates with:
  - evidence_classifier.classify_evidence()
  - contradiction_detector.get_detector().check()
  - quality_filter.calculate_quality_score() (lexical quality)

Usage (standalone):
    from classifier.src.data.trust_scorer import TrustScorer
    scorer = TrustScorer()
    result = scorer.score(
        text="Ixodes ricinus transmits Borrelia burgdorferi to humans.",
        species1="Ixodes ricinus",
        species2="Borrelia burgdorferi",
        interaction_type="pathogenOf",
        pub_year=2018,
    )
    print(result.composite_trust, result.explanation)
"""

from __future__ import annotations

import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "classifier"))

from src.data.evidence_classifier import classify_evidence, EvidenceResult
from src.data.contradiction_detector import get_detector, ConflictResult
from src.data.retraction_checker import check_retraction, retraction_score, SCORE_UNKNOWN


# ---------------------------------------------------------------------------
# Weights for composite score
# ---------------------------------------------------------------------------

WEIGHTS: Dict[str, float] = {
    "temporal":        0.20,
    "evidence":        0.20,
    "contradiction":   0.20,  # inverted: (1 - contradiction_risk)
    "kg_confirmation": 0.20,
    "retraction":      0.20,  # NEW — 5th axis
}

# Half-life for temporal decay (years). Claims lose 50% trust after this many years.
HALF_LIFE_YEARS = 15.0

CURRENT_YEAR = 2026


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrustResult:
    composite_trust: float        # 0.0–1.0
    temporal_score: float         # 0.0–1.0
    evidence_type: str
    evidence_score: float         # 0.0–1.0
    contradiction_risk: float     # 0.0–1.0
    kg_confirmed: bool
    kg_confirmation_score: float  # 0.0–1.0
    retraction_score: float       # 0.0 (retracted) / 0.6 (corrected) / 1.0 (clean) / 0.85 (unknown)
    retracted: bool
    retraction_type: Optional[str]
    pub_year: Optional[int]
    doi: Optional[str]
    species1: str
    species2: str
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "composite_trust": round(self.composite_trust, 3),
            "temporal_score": round(self.temporal_score, 3),
            "evidence_type": self.evidence_type,
            "evidence_score": round(self.evidence_score, 3),
            "contradiction_risk": round(self.contradiction_risk, 3),
            "kg_confirmed": self.kg_confirmed,
            "kg_confirmation_score": round(self.kg_confirmation_score, 3),
            "retraction_score": round(self.retraction_score, 3),
            "retracted": self.retracted,
            "retraction_type": self.retraction_type,
            "pub_year": self.pub_year,
            "doi": self.doi,
            "explanation": self.explanation,
        }


# ---------------------------------------------------------------------------
# Temporal decay
# ---------------------------------------------------------------------------

def temporal_score(pub_year: Optional[int], current_year: int = CURRENT_YEAR) -> float:
    """Exponential decay: score = exp(-λ * age), λ = ln(2) / half_life."""
    if pub_year is None:
        return 0.4  # unknown date — penalise but don't discard
    age = max(0, current_year - pub_year)
    lam = math.log(2) / HALF_LIFE_YEARS
    return round(math.exp(-lam * age), 4)


# ---------------------------------------------------------------------------
# GloBI confirmation
# ---------------------------------------------------------------------------

def _load_globi_cache() -> Optional[dict]:
    path = ROOT / "classifier/data/globi/globi_lookup_cache.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


_globi_cache: Optional[dict] = None
_globi_loaded = False

# Trust index: maps (source_species_lower, target_species_lower, interaction_type) → pub_year
_trust_index: Optional[dict] = None
_trust_index_loaded = False

# DOI index: maps doi.lower() → {pub_year, article_type, citation_count}
_trust_doi_index: Optional[dict] = None


def _normalize(s: str) -> str:
    return s.strip().lower().replace("_", " ")


def _load_trust_index() -> tuple:
    """Returns (species_index, doi_index). Both None if file missing."""
    path = ROOT / "classifier/data/globi/globi_trust_index.csv"
    if not path.exists():
        return None, None
    import csv as _csv
    sp_idx: dict = {}
    doi_idx: dict = {}
    with path.open("r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            year_str = row.get("pub_year", "")
            year = int(year_str) if year_str and year_str.isdigit() else None
            key = (
                _normalize(row.get("source_species", "")),
                _normalize(row.get("target_species", "")),
                row.get("interaction_type", ""),
            )
            if year and key not in sp_idx:
                sp_idx[key] = year
            doi = (row.get("doi") or "").strip().lower()
            if doi and doi not in doi_idx:
                doi_idx[doi] = {
                    "pub_year": year,
                    "article_type": row.get("article_type", ""),
                    "citation_count": int(row.get("citation_count", 0) or 0),
                }
    return sp_idx, doi_idx


def _ensure_trust_index() -> None:
    global _trust_index, _trust_doi_index, _trust_index_loaded
    if not _trust_index_loaded:
        _trust_index, _trust_doi_index = _load_trust_index()
        _trust_index_loaded = True


def lookup_pub_year(sp1: str, sp2: str, interaction_type: str) -> Optional[int]:
    """Look up publication year from trust index for a given species pair."""
    _ensure_trust_index()
    if _trust_index is None:
        return None
    n1, n2, itype = _normalize(sp1), _normalize(sp2), interaction_type
    return _trust_index.get((n1, n2, itype)) or _trust_index.get((n2, n1, itype))


def lookup_by_doi(doi: str) -> Optional[Dict[str, Any]]:
    """Look up pub_year, article_type, citation_count from trust index by DOI."""
    _ensure_trust_index()
    if _trust_doi_index is None or not doi:
        return None
    return _trust_doi_index.get(doi.strip().lower())


def kg_confirmed(sp1: str, sp2: str) -> bool:
    global _globi_cache, _globi_loaded
    if not _globi_loaded:
        _globi_cache = _load_globi_cache()
        _globi_loaded = True
    if _globi_cache is None:
        return False
    n1, n2 = _normalize(sp1), _normalize(sp2)
    return (n1, n2) in _globi_cache or (n2, n1) in _globi_cache


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class TrustScorer:
    """Composite trust scorer for biotic interaction claims."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        training_csv: Optional[str] = None,
        globi_cache: Optional[str] = None,
    ) -> None:
        self.weights = weights or WEIGHTS
        self._detector = get_detector(training_csv, globi_cache)

    def score(
        self,
        text: str,
        species1: str = "",
        species2: str = "",
        interaction_type: str = "",
        pub_year: Optional[int] = None,
        doi: Optional[str] = None,
    ) -> TrustResult:
        # Enrich from DOI index first (may provide pub_year if not given)
        doi_meta = lookup_by_doi(doi) if doi else None
        if pub_year is None and doi_meta:
            pub_year = doi_meta.get("pub_year")

        # 1. Temporal — fall back to species-pair trust index lookup
        if pub_year is None and species1 and species2:
            pub_year = lookup_pub_year(species1, species2, interaction_type)
        t_score = temporal_score(pub_year)

        # 2. Evidence type
        ev: EvidenceResult = classify_evidence(text)

        # 3. Contradiction
        conflict: ConflictResult = self._detector.check(
            species1, species2, interaction_type, text
        )
        contradiction = conflict.contradiction_risk

        # 4. GloBI confirmation
        confirmed = kg_confirmed(species1, species2) if species1 and species2 else False
        kg_score = 1.0 if confirmed else 0.35

        # 5. Retraction check
        if doi:
            ret_result = check_retraction(doi)
            r_score = ret_result.score
            r_retracted = ret_result.retracted
            r_type = ret_result.retraction_type
        else:
            r_score = SCORE_UNKNOWN
            r_retracted = False
            r_type = None

        # Weighted composite (contradiction inverted)
        w = self.weights
        composite = (
            w["temporal"]        * t_score +
            w["evidence"]        * ev.evidence_score +
            w["contradiction"]   * (1.0 - contradiction) +
            w["kg_confirmation"] * kg_score +
            w.get("retraction", 0.20) * r_score
        )
        composite = round(min(1.0, max(0.0, composite)), 3)

        # Human-readable explanation
        parts = []
        if r_retracted:
            parts.append("RETRACTED PAPER")
        if pub_year:
            age = CURRENT_YEAR - pub_year
            parts.append(f"{age}-year-old {ev.evidence_type.replace('_', ' ')} study")
        else:
            parts.append(f"Unknown publication date ({ev.evidence_type.replace('_', ' ')})")
        if confirmed:
            parts.append("confirmed in GloBI")
        else:
            parts.append("not in GloBI")
        if conflict.sentence_negated:
            parts.append(f"sentence negated ('{conflict.negation_pattern}')")
        if contradiction > 0.3:
            parts.append(f"contradiction risk {contradiction:.0%} ({conflict.n_negative_claims} conflicting claims)")
        explanation = "; ".join(parts) + "."

        return TrustResult(
            composite_trust=composite,
            temporal_score=t_score,
            evidence_type=ev.evidence_type,
            evidence_score=ev.evidence_score,
            contradiction_risk=contradiction,
            kg_confirmed=confirmed,
            kg_confirmation_score=kg_score,
            retraction_score=r_score,
            retracted=r_retracted,
            retraction_type=r_type,
            pub_year=pub_year,
            doi=doi,
            species1=species1,
            species2=species2,
            explanation=explanation,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scorer = TrustScorer()

    examples = [
        {
            "text": "Ixodes ricinus ticks were collected from the field and found to carry Borrelia burgdorferi.",
            "species1": "Ixodes ricinus", "species2": "Borrelia burgdorferi",
            "interaction_type": "pathogenOf", "pub_year": 2021,
        },
        {
            "text": "Plasmodium falciparum was cultured in vitro with human erythrocytes.",
            "species1": "Plasmodium falciparum", "species2": "Homo sapiens",
            "interaction_type": "pathogenOf", "pub_year": 1998,
        },
        {
            "text": "No evidence of Toxoplasma gondii infection was detected in this bat population.",
            "species1": "Toxoplasma gondii", "species2": "Chiroptera",
            "interaction_type": "pathogenOf", "pub_year": 2015,
        },
        {
            "text": "It has long been known that fleas serve as vectors of Yersinia pestis.",
            "species1": "Siphonaptera", "species2": "Yersinia pestis",
            "interaction_type": "vectorOf", "pub_year": 1973,
        },
    ]

    print(f"{'Trust':>6}  {'Temp':>5}  {'Evid':>5}  {'Contr':>5}  {'KG':>5}  Explanation")
    print("-" * 120)
    for ex in examples:
        r = scorer.score(**ex)
        print(f"{r.composite_trust:6.3f}  {r.temporal_score:5.3f}  "
              f"{r.evidence_score:5.3f}  {r.contradiction_risk:5.3f}  "
              f"{'Y' if r.kg_confirmed else 'N':>5}  {r.explanation[:80]}")
