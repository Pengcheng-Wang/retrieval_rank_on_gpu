import numpy as np
import pytest

from rrbench.semantic_history import frozen_exposure_history


def test_histories_use_only_strictly_prior_training_clock_envelopes():
    items = np.array([10, 11, 12, 13])
    upper = np.array([100, 200, 300, 400])
    history = frozen_exposure_history(items, upper, target_lower=300, maximum=2)
    assert history.tolist() == [11, 12]
    assert 13 not in history


def test_same_day_and_same_block_targets_cannot_read_later_feedback():
    items = np.array([10, 11, 12])
    upper = np.array([200, 200, 300])
    assert frozen_exposure_history(items, upper, target_lower=199, maximum=10).size == 0
    assert frozen_exposure_history(items, upper, target_lower=200, maximum=10).tolist() == [10, 11]


def test_history_limits_and_clock_order_are_explicit():
    with pytest.raises(ValueError, match="ordered"):
        frozen_exposure_history(np.array([1, 2]), np.array([200, 100]), 300, 10)
