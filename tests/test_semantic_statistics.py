import numpy as np
import pytest

from rrbench.semantic_statistics import paired_cluster_effect


def test_personalized_uncertainty_resamples_users_not_independent_impressions():
    left = np.array([1.0, 1.0, 0.0])
    right = np.array([0.0, 0.0, 0.0])
    users = np.array([11, 11, 12])
    effect = paired_cluster_effect(left, right, users, seed=17, draws=200)
    assert effect["users_or_query_groups"] == 2
    assert effect["mean_user_effect"] == 0.5
    assert effect["mean_event_or_query_effect"] == pytest.approx(2 / 3)
    assert effect["ci95_mean_user_effect"] == [0.0, 1.0]
    assert effect["ci95_mean_event_or_query_effect"] == [0.0, 1.0]


def test_misaligned_or_nonfinite_paired_data_is_not_an_uncertainty_estimate():
    with pytest.raises(ValueError):
        paired_cluster_effect(np.ones(2), np.ones(1), np.array([1, 2]))
