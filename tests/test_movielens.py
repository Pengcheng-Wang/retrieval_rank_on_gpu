import numpy as np
import pytest
import torch

from rrbench.movielens import chronological_split
from rrbench.quality import epoch_examples


def ratings_fixture():
    return np.array([
        [1, 10, 5, 1], [1, 11, 2, 2], [1, 12, 5, 3],
        [1, 13, 4, 4], [1, 14, 5, 4],
        [2, 12, 5, 1], [2, 15, 5, 2], [2, 10, 5, 3], [2, 14, 5, 4],
    ])


def test_temporal_split_excludes_same_timestamp_fitting_events():
    data = chronological_split(ratings_fixture())
    assert len(data.eval_users) == 2
    assert len(data.train_users) == 4
    assert data.train_labels.tolist() == [1, 0, 1, 1]
    assert np.array_equal(data.train_seen, data.validation_seen)
    assert data.test_seen.sum() > data.validation_seen.sum()


def test_no_future_label_blacklist_for_negatives():
    data = chronological_split(ratings_fixture())
    users, items, labels = epoch_examples(data, 10, 17, torch.device("cpu"))
    for user, item in zip(users[labels == 0].tolist(), items[labels == 0].tolist()):
        # Existing low-rated observations are explicitly supervised negatives;
        # generated negatives exclude every observed training item.
        if data.train_seen[user, item]:
            assert any((data.train_users == user) & (data.train_items == item) & (data.train_labels == 0))
    assert len(users) == len(data.train_users) + 10 * int(data.train_labels.sum())


def test_sparse_timestamp_history_fails_explicitly():
    with pytest.raises(ValueError, match="distinct positive timestamps"):
        chronological_split(np.array([[1, 1, 5, 1], [1, 2, 5, 1], [1, 3, 5, 2]]))


def test_zero_negative_sampling_keeps_original_observations():
    data = chronological_split(ratings_fixture())
    users, items, labels = epoch_examples(data, 0, 17, torch.device("cpu"))
    assert len(users) == len(data.train_users)
    assert labels.sum() == data.train_labels.sum()
