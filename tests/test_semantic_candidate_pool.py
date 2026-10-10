import numpy as np
import pytest

from rrbench.semantic_candidates import common_candidate_pool


def test_label_independent_common_pool_preserves_first_retriever_union_order_without_injected_targets():
    first = np.array([[4, 2, 8, -1], [1, 3, 4, 5]], dtype=np.int64)
    second = np.array([[2, 7, 9, -1], [3, 6, 7, -1]], dtype=np.int64)
    pool = common_candidate_pool([first, second], 6)
    assert pool.tolist() == [[4, 2, 8, 7, 9, -1], [1, 3, 4, 5, 6, 7]]


def test_missing_query_rows_cannot_make_a_common_reranker_pool():
    with pytest.raises(ValueError, match="aligned"):
        common_candidate_pool([np.ones((2, 4), dtype=np.int64), np.ones((3, 4), dtype=np.int64)], 5)
