"""
contradiction_detector.py — Detect contradictory interaction claims.

Two detection levels:
  1. Sentence-level negation: detect explicit negation within a single sentence
     (extends the pattern already in outcome_codes.py)
  2. Cross-source conflict: given an interaction claim (sp1, sp2, type),
     check whether the GloBI cache or training index contains contradicting claims

Usage:
    detector = ContradictionDetector()
    detector.build_index_from_csv("data/training/training_data_v14.csv")
    result = detector.check("Plasmodium falciparum", "Anopheles gambiae", "pathogenOf", text)
"""

from __future__ import annotations

import csv
import pickle
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# Sentence-level negation patterns
# ---------------------------------------------------------------------------

_NEGATION_PATTERNS = [
    # Classic verbal negation
    r"\b(does not|do not|did not|does n't|do n't|did n't)\s+\w+",
    r"\b(is not|are not|was not|were not|isn't|aren't|wasn't|weren't)\s+\w+",
    r"\b(cannot|can't|could not|couldn't)\s+\w+",
    r"\b(will not|won't|would not|wouldn't)\s+\w+",
    # Interaction-specific negations
    r"\bno (evidence of|sign of|indication of|record of)\b",
    r"\bfailed to (infect|parasiti|transmit|coloniz)\w*",
    r"\bnot (found|detected|observed|isolated)\b",
    r"\bunable to (infect|colonize|parasiti)\w*",
    r"\babsence of (infection|parasitism|interaction)\b",
    r"\b(refut|contradict|disprove|refute)\w*",
    r"\blacks? (the ability|capacity) to\b",
    r"\bis (not|no longer) considered (a )?(host|vector|reservoir|pathogen)\b",
]

_NEGATION_COMPILED = [re.compile(p, re.IGNORECASE) for p in _NEGATION_PATTERNS]


def detect_sentence_negation(text: str) -> Tuple[bool, Optional[str]]:
    """Return (is_negated, matched_pattern) for a sentence."""
    for pat in _NEGATION_COMPILED:
        m = pat.search(text)
        if m:
            return True, m.group(0)
    return False, None


# ---------------------------------------------------------------------------
# Cross-source conflict index
# ---------------------------------------------------------------------------

def _normalize(name: str) -> str:
    return name.strip().lower().replace("_", " ")


@dataclass
class ConflictResult:
    conflict_detected: bool
    contradiction_risk: float  # 0.0 = no conflict, 1.0 = direct contradiction
    n_positive_claims: int = 0
    n_negative_claims: int = 0
    sources_positive: List[str] = field(default_factory=list)
    sources_negative: List[str] = field(default_factory=list)
    sentence_negated: bool = False
    negation_pattern: Optional[str] = None


class ContradictionDetector:
    """
    Builds and queries an index of (sp1, sp2, interaction_type) → {positive, negative} claims.

    A 'positive claim' = a training sentence with label=1 for that pair.
    A 'negative claim' = a sentence with label=0 explicitly denying the interaction
      (detected via negation patterns), or a sentence marked NEGATED in the dataset.
    """

    def __init__(self) -> None:
        # {(norm_sp1, norm_sp2, itype): {"pos": [sources], "neg": [sources]}}
        self._index: Dict[Tuple[str, str, str], Dict[str, List[str]]] = {}
        self._built = False

    def build_index_from_csv(self, csv_path: str | Path) -> int:
        """Parse training CSV and build the conflict index. Returns number of rows indexed."""
        csv_path = Path(csv_path)
        count = 0
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = int(row.get("label", 0))
                sp1 = _normalize(row.get("source_species", ""))
                sp2 = _normalize(row.get("target_species", ""))
                itype = row.get("interaction_type", "").strip()
                source = row.get("source", "training")
                text = row.get("text", "")

                if not sp1 or not sp2:
                    continue

                key = (sp1, sp2, itype)
                if key not in self._index:
                    self._index[key] = {"pos": [], "neg": []}

                is_neg, _ = detect_sentence_negation(text)
                if label == 1 and not is_neg:
                    self._index[key]["pos"].append(source)
                elif label == 0 or is_neg:
                    self._index[key]["neg"].append(source)
                count += 1

        self._built = True
        return count

    def build_index_from_globi_cache(self, cache_path: str | Path) -> int:
        """Enrich index with GloBI known pairs (all treated as positive claims)."""
        cache_path = Path(cache_path)
        if not cache_path.exists():
            return 0
        with open(cache_path, "rb") as f:
            lookup: dict = pickle.load(f)

        count = 0
        for (sp1, sp2), itype in lookup.items():
            key = (_normalize(sp1), _normalize(sp2), itype)
            if key not in self._index:
                self._index[key] = {"pos": [], "neg": []}
            self._index[key]["pos"].append("globi")
            count += 1
        return count

    def check(
        self,
        sp1: str,
        sp2: str,
        interaction_type: str,
        text: str = "",
    ) -> ConflictResult:
        """Check for contradictions for a given species pair + interaction type."""
        # Sentence-level negation
        sent_neg, neg_pat = detect_sentence_negation(text)

        # Index lookup — try both orderings
        key1 = (_normalize(sp1), _normalize(sp2), interaction_type)
        key2 = (_normalize(sp2), _normalize(sp1), interaction_type)
        entry = self._index.get(key1) or self._index.get(key2)

        if entry is None:
            return ConflictResult(
                conflict_detected=False,
                contradiction_risk=0.1 if sent_neg else 0.0,
                sentence_negated=sent_neg,
                negation_pattern=neg_pat,
            )

        n_pos = len(entry["pos"])
        n_neg = len(entry["neg"])
        total = n_pos + n_neg

        if total == 0:
            risk = 0.0
        elif n_pos == 0:
            risk = 1.0  # only contradicting claims
        elif n_neg == 0:
            risk = 0.0  # purely confirmed
        else:
            # Contradiction ratio: fraction of claims that deny the interaction
            # Capped at 0.9 since some contradictions are outdated data artefacts
            risk = min(0.9, n_neg / total)

        # Sentence-level negation bumps risk
        if sent_neg:
            risk = min(1.0, risk + 0.3)

        return ConflictResult(
            conflict_detected=(risk > 0.3),
            contradiction_risk=round(risk, 3),
            n_positive_claims=n_pos,
            n_negative_claims=n_neg,
            sources_positive=entry["pos"][:5],
            sources_negative=entry["neg"][:5],
            sentence_negated=sent_neg,
            negation_pattern=neg_pat,
        )

    @property
    def index_size(self) -> int:
        return len(self._index)


# ---------------------------------------------------------------------------
# Singleton loader (lazy, for API use)
# ---------------------------------------------------------------------------

_detector: Optional[ContradictionDetector] = None


def get_detector(
    training_csv: Optional[str] = None,
    globi_cache: Optional[str] = None,
) -> ContradictionDetector:
    """Return a shared ContradictionDetector, built lazily on first call."""
    global _detector
    if _detector is not None:
        return _detector

    _detector = ContradictionDetector()

    default_csv = ROOT / "classifier/data/training/training_data_v14.csv"
    default_globi = ROOT / "classifier/data/globi/globi_lookup_cache.pkl"

    csv_path = Path(training_csv) if training_csv else default_csv
    if csv_path.exists():
        n = _detector.build_index_from_csv(csv_path)
        print(f"[ContradictionDetector] Indexed {n:,} training rows", flush=True)

    globi_path = Path(globi_cache) if globi_cache else default_globi
    if globi_path.exists():
        n = _detector.build_index_from_globi_cache(globi_path)
        print(f"[ContradictionDetector] Added {n:,} GloBI pairs", flush=True)

    return _detector


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    detector = get_detector()
    print(f"Index size: {detector.index_size:,} species-pair-type triples\n")

    test_cases = [
        ("Plasmodium falciparum", "Anopheles gambiae", "pathogenOf",
         "Plasmodium falciparum is transmitted by Anopheles gambiae mosquitoes."),
        ("Borrelia burgdorferi", "Ixodes ricinus", "pathogenOf",
         "Borrelia burgdorferi does not infect Ixodes ricinus in this study."),
        ("Toxoplasma gondii", "Felis catus", "pathogenOf",
         "No evidence of Toxoplasma gondii infection was detected in these cats."),
    ]

    for sp1, sp2, itype, text in test_cases:
        r = detector.check(sp1, sp2, itype, text)
        print(f"{sp1} × {sp2} ({itype})")
        print(f"  risk={r.contradiction_risk:.2f}  conflict={r.conflict_detected}"
              f"  pos={r.n_positive_claims}  neg={r.n_negative_claims}"
              f"  sent_neg={r.sentence_negated}")
        if r.negation_pattern:
            print(f"  pattern: '{r.negation_pattern}'")
        print()
