import numpy as np
import pytest

from rrbench.semantic_recommendation_baselines import mapped_logged_pairs


def test_logged_recommendation_rows_keep_labels_and_only_training_catalog_ids():
    items = {100: 0, 200: 1}
    users = {11: 0}
    rows = mapped_logged_pairs([(11, 100, 1), (11, 200, 0), (11, 999, 1)], users, items)
    np.testing.assert_array_equal(rows, [[0, 0, 1], [0, 1, 0]])


def test_unknown_training_user_is_not_silently_mapped_to_another_person():
    with pytest.raises(ValueError, match="user"):
        mapped_logged_pairs([(12, 100, 1)], {11: 0}, {100: 0})
