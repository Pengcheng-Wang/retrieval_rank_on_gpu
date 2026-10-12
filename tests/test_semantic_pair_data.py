import numpy as np
import pytest

from rrbench.semantic_rankers import prepare_pairs


def test_training_pair_builder_excludes_all_known_positives_from_weak_negatives():
    panel = {"query_ids": ["q"], "qrels": {"q": {"1": 1, "2": 1}}}
    pairs = prepare_pairs(panel, np.array([[1, 2, 3, 4, -1]], dtype=np.int64), 17, negatives=2)
    assert set(pairs[pairs[:, 2] == 1, 1]) == {1, 2}
    assert set(pairs[pairs[:, 2] == 0, 1]) == {3, 4}


def test_empty_or_invalid_hard_negative_pool_is_not_a_silent_training_default():
    panel = {"query_ids": ["q"], "qrels": {"q": {"1": 1}}}
    with pytest.raises(ValueError, match="hard-negative"):
        prepare_pairs(panel, np.array([[1, -1]], dtype=np.int64), 17)
