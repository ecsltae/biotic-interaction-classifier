"""The polarity index the handoff scorer feeds the direction head must be the one training used.

A mismatch is train/serve skew that no evaluation would flag: the head simply receives an embedding
it never saw for that relation class. Training (train_direction.py, eval_direction.py,
predict_joint.py) maps agent -> 1 and everything else -> 0 for the 2-valued head;
train_direction_v3.py adds symmetric -> 2 for the 3-valued one.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "handoff" / "biotic_verifier"))
import polarity as P  # noqa: E402


@pytest.mark.parametrize("p, n_pol, expected", [
    (1, 2, 1), (-1, 2, 0), (0, 2, 0), (None, 2, 0),            # 2-valued head
    (1, 3, 1), (-1, 3, 0), (0, 3, 2), (None, 3, 0),            # 3-valued head
])
def test_polarity_index_matches_training(p, n_pol, expected):
    assert P.polarity_to_index(p, n_pol) == expected


@pytest.mark.parametrize("rel", ["interacts with", "mutualism", "competes with", "co-infection",
                                 "associated with", "co-occurs with"])
def test_mutual_relations_are_bidirectional(rel):
    p, _ = P.polarity(rel)
    assert P.is_symmetric(p), f"{rel!r} should be symmetric, got {p}"


@pytest.mark.parametrize("rel", ["parasite of", "host of", "feeds on", "pollinates", "infects"])
def test_directed_relations_are_not_bidirectional(rel):
    p, _ = P.polarity(rel)
    assert not P.is_symmetric(p), f"{rel!r} should be directed, got {p}"
