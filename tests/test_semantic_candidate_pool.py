import numpy as np
import pytest

from rrbench.semantic_candidates import common_candidate_pool


def test_label_independent_common_pool_preserves_rank_interleaved_union_without_injected_targets():
    first = np.array([[4, 2, 8, -1], [1, 3, 4, 5]], dtype=np.int64)
    second = np.array([[2, 7, 9, -1], [3, 6, 7, -1]], dtype=np.int64)
    pool = common_candidate_pool([first, second], 6)
    assert pool.tolist() == [[4, 2, 7, 8, 9, -1], [1, 3, 6, 4, 7, 5]]


def test_missing_query_rows_cannot_make_a_common_reranker_pool():
    with pytest.raises(ValueError, match="aligned"):
        common_candidate_pool([np.ones((2, 4), dtype=np.int64), np.ones((3, 4), dtype=np.int64)], 5)


def test_finite_common_cap_does_not_exclude_every_candidate_from_later_retrievers():
    first = np.arange(1000, dtype=np.int64)[None]
    second = np.arange(1000, 2000, dtype=np.int64)[None]
    pool = common_candidate_pool([first, second], 1000)[0]
    assert sum(pool < 1000) == 500
    assert sum(pool >= 1000) == 500
