"""Tests for the trust scoring system (Track B)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.evidence_classifier import classify_evidence
from src.data.contradiction_detector import detect_sentence_negation, ContradictionDetector
from src.data.trust_scorer import TrustScorer, temporal_score


# ---------------------------------------------------------------------------
# Evidence classifier
# ---------------------------------------------------------------------------

class TestEvidenceClassifier:
    def test_field_observation(self):
        r = classify_evidence("Ticks were collected from the field and found to carry bacteria.")
        assert r.evidence_type == "field_observation"
        assert r.evidence_score == 1.0

    def test_lab_experiment(self):
        r = classify_evidence("Leishmania was cultured in vitro in M199 medium.")
        assert r.evidence_type == "lab_experiment"
        assert r.evidence_score < 0.8

    def test_in_vivo(self):
        r = classify_evidence("We infected C57BL/6 mice with Trypanosoma cruzi.")
        assert r.evidence_type == "in_vivo_experiment"

    def test_genomic(self):
        r = classify_evidence("Phylogenetic analysis of 16S rRNA identified Wolbachia.")
        assert r.evidence_type == "genomic_inference"

    def test_meta_analysis(self):
        r = classify_evidence("It has long been known that fleas transmit Yersinia pestis.")
        assert r.evidence_type == "meta_analysis"

    def test_unknown(self):
        r = classify_evidence("The bat was near other species.")
        assert r.evidence_type == "unknown"
        assert r.matched_pattern is None


# ---------------------------------------------------------------------------
# Negation detection
# ---------------------------------------------------------------------------

class TestNegationDetection:
    def test_does_not_infect(self):
        is_neg, pat = detect_sentence_negation("This organism does not infect humans.")
        assert is_neg
        assert "does not" in pat

    def test_no_evidence(self):
        is_neg, pat = detect_sentence_negation("No evidence of parasitism was found.")
        assert is_neg

    def test_positive_sentence(self):
        is_neg, _ = detect_sentence_negation("Ixodes transmits Borrelia burgdorferi.")
        assert not is_neg

    def test_failed_to_infect(self):
        is_neg, _ = detect_sentence_negation("The strain failed to infect the host cells.")
        assert is_neg


# ---------------------------------------------------------------------------
# Contradiction detector (without GloBI — unit test only)
# ---------------------------------------------------------------------------

class TestContradictionDetector:
    def test_no_data_gives_low_risk(self):
        detector = ContradictionDetector()
        r = detector.check("Species A", "Species B", "parasiteOf")
        assert r.contradiction_risk == 0.0
        assert not r.conflict_detected

    def test_sentence_negation_bumps_risk(self):
        detector = ContradictionDetector()
        r = detector.check("Species A", "Species B", "parasiteOf",
                           text="Species A does not infect Species B.")
        assert r.sentence_negated
        assert r.contradiction_risk > 0.0

    def test_conflict_in_index(self):
        detector = ContradictionDetector()
        # Manually inject conflicting claims
        key = ("species a", "species b", "parasiteOf")
        detector._index[key] = {"pos": ["src1", "src2"], "neg": ["src3"]}
        r = detector.check("Species A", "Species B", "parasiteOf")
        assert r.n_positive_claims == 2
        assert r.n_negative_claims == 1
        assert 0.0 < r.contradiction_risk < 0.5


# ---------------------------------------------------------------------------
# Temporal score
# ---------------------------------------------------------------------------

class TestTemporalScore:
    def test_recent_paper_high(self):
        assert temporal_score(2024) > 0.88

    def test_old_paper_lower(self):
        assert temporal_score(1990) < 0.5

    def test_very_old_paper(self):
        assert temporal_score(1960) < 0.15

    def test_unknown_date_midrange(self):
        score = temporal_score(None)
        assert 0.3 <= score <= 0.5


# ---------------------------------------------------------------------------
# TrustScorer integration
# ---------------------------------------------------------------------------

class TestTrustScorer:
    def setup_method(self):
        # Use empty training index to avoid loading 4M GloBI pairs in unit tests
        self.scorer = TrustScorer()

    def test_recent_field_study_scores_high(self):
        r = self.scorer.score(
            text="Ticks were collected from the field and were found to carry Borrelia.",
            species1="Ixodes ricinus",
            species2="Borrelia burgdorferi",
            interaction_type="pathogenOf",
            pub_year=2022,
        )
        assert r.composite_trust > 0.6
        assert r.evidence_type == "field_observation"
        assert r.temporal_score > 0.8

    def test_negated_old_review_scores_low(self):
        r = self.scorer.score(
            text="It has long been known that no infection occurs between these species.",
            species1="Sp1", species2="Sp2",
            interaction_type="parasiteOf",
            pub_year=1970,
        )
        assert r.composite_trust < 0.55
        assert r.contradiction_risk > 0.0 or r.temporal_score < 0.3

    def test_composite_between_0_and_1(self):
        r = self.scorer.score(text="A eats B.", species1="A", species2="B")
        assert 0.0 <= r.composite_trust <= 1.0

    def test_explanation_not_empty(self):
        r = self.scorer.score(text="Ixodes ricinus transmits bacteria.", pub_year=2020)
        assert len(r.explanation) > 10

    def test_to_dict_keys(self):
        r = self.scorer.score(text="Test sentence.")
        d = r.to_dict()
        assert "composite_trust" in d
        assert "evidence_type" in d
        assert "contradiction_risk" in d
        assert "retraction_score" in d
        assert "retracted" in d
        assert "explanation" in d

    def test_doi_parameter_accepted(self):
        r = self.scorer.score(
            text="Ixodes ricinus transmits bacteria.",
            species1="Ixodes ricinus",
            doi="10.3354/dao002147",
        )
        assert r.doi == "10.3354/dao002147"
        assert 0.0 <= r.composite_trust <= 1.0

    def test_retraction_score_in_composite(self):
        # Without DOI → retraction_score should be SCORE_UNKNOWN (0.85)
        r = self.scorer.score(text="A infects B.")
        assert r.retraction_score == 0.85
        assert not r.retracted
