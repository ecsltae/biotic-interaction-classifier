"""
evidence_classifier.py — Classify the type of scientific evidence in a sentence.

Evidence types (in decreasing ecological reliability order):
  field_observation  — direct observation in nature (most trustworthy for ecology)
  in_vivo_experiment — controlled experiment in living organisms
  lab_experiment     — controlled lab conditions (in vitro, culture)
  genomic_inference  — inferred from sequence/phylogenetic analysis
  meta_analysis      — review or synthesis of multiple studies
  unknown            — no clear signal

Maps each type to a numeric score usable as a trust component.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Evidence type → trust weight (ecological/interaction validity perspective)
# ---------------------------------------------------------------------------

EVIDENCE_WEIGHTS: Dict[str, float] = {
    "field_observation": 1.0,
    "in_vivo_experiment": 0.75,
    "lab_experiment": 0.55,
    "genomic_inference": 0.50,
    "meta_analysis": 0.40,   # reviews summarise older claims; decay propagates
    "unknown": 0.30,
}

# Keyword patterns per evidence type (order matters — first match wins)
_PATTERNS: List[Tuple[str, List[str]]] = [
    ("field_observation", [
        r"\bwas found (to|in|on)\b",
        r"\bwere found (to|in|on)\b",
        r"\bcollected from\b",
        r"\bobserved (in|on|to)\b",
        r"\bfield (survey|study|collection|observation)\b",
        r"\bwild(life)?\b.*\b(collected|captured|sampled)\b",
        r"\b(trapped|captured|sampled) (in|from) (the )?(field|nature|wild)\b",
        r"\bnaturally (infect|parasiti|predat)\w+\b",
    ]),
    ("in_vivo_experiment", [
        r"\bin vivo\b",
        r"\b(mice|rats|rabbits|hamsters|guinea pigs|monkeys)\b.{0,80}\b(infected|inoculated|challenged|administered)\b",
        r"\b(infected|inoculated|challenged|administered)\b.{0,80}\b(mice|rats|rabbits)\b",
        r"\banimal model\b",
        r"\bexperimental(ly)? infect\w*\b",
        r"\bexperimental(ly)? inoculat\w*\b",
    ]),
    ("lab_experiment", [
        r"\bin vitro\b",
        r"\bcell (culture|line)\b",
        r"\bunder laboratory conditions\b",
        r"\bcultur(ed|ing|e) (with|in)\b",
        r"\b(petri dish|flask|well plate|culture medium)\b",
        r"\binoculated (with|into) (a )?(culture|medium|agar)\b",
    ]),
    ("genomic_inference", [
        r"\bphylogenetic analysis\b",
        r"\bgenome (comparison|analysis|sequencing)\b",
        r"\b(16S|18S|ITS|COI|rbcL)\b.{0,40}\bidentif\w+\b",
        r"\bwhole[- ]genome\b",
        r"\bmolecular clock\b",
        r"\bsequence similarity\b",
        r"\bmetagenom\w+\b",
    ]),
    ("meta_analysis", [
        r"\breview of\b",
        r"\bmeta[- ]analysis\b",
        r"\bsystematic review\b",
        r"\bsurvey(ed|ing)? (studies|literature|reports)\b",
        r"\b(previous|earlier|prior) studi(es|ed)\b.{0,60}\breported\b",
        r"\bhas been reported (to|in)\b",
        r"\bhas long been known\b",
        r"\bwidely (known|accepted|reported)\b",
    ]),
]

_COMPILED: List[Tuple[str, List[re.Pattern]]] = [
    (etype, [re.compile(pat, re.IGNORECASE) for pat in pats])
    for etype, pats in _PATTERNS
]


@dataclass
class EvidenceResult:
    evidence_type: str
    evidence_score: float
    matched_pattern: str | None


def classify_evidence(text: str) -> EvidenceResult:
    """Return the evidence type and trust score for a sentence."""
    for etype, compiled_pats in _COMPILED:
        for pat in compiled_pats:
            m = pat.search(text)
            if m:
                return EvidenceResult(
                    evidence_type=etype,
                    evidence_score=EVIDENCE_WEIGHTS[etype],
                    matched_pattern=m.group(0),
                )
    return EvidenceResult(
        evidence_type="unknown",
        evidence_score=EVIDENCE_WEIGHTS["unknown"],
        matched_pattern=None,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    examples = [
        "Ixodes ricinus ticks were collected from the field and found to carry Borrelia burgdorferi.",
        "We infected C57BL/6 mice intraperitoneally with 1×10^6 Trypanosoma cruzi trypomastigotes.",
        "Leishmania donovani was cultured in vitro in M199 medium at 26°C.",
        "Phylogenetic analysis of 16S rRNA sequences identified Wolbachia as the endosymbiont.",
        "It has long been known that Plasmodium falciparum is transmitted by Anopheles gambiae.",
        "The bat was sleeping in the cave with other species.",
    ]

    for sent in examples:
        r = classify_evidence(sent)
        print(f"[{r.evidence_type:20s}] ({r.evidence_score:.2f}) {sent[:80]}")
        if r.matched_pattern:
            print(f"    matched: '{r.matched_pattern}'")
