import numpy as np
import torch

from rrbench.semantic_cascade import score_cached_candidates


class Dot(torch.nn.Module):
    def forward(self, query, item, sparse):
        return (query * item).sum(dim=1)


def test_cached_scoring_preserves_query_rows_and_invalid_candidate_padding():
    items = np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype=np.float16)
    queries = np.array([[1, 0], [0, 1]], dtype=np.float16)
    candidates = np.array([[0, 2, -1], [2, 1, 0]], dtype=np.int64)
    centroids = np.eye(2, dtype=np.float32)
    scores = score_cached_candidates(
        Dot(), items, queries, candidates, np.array([0, 1, 1]), np.array([5, 8, 9]),
        centroids, torch.device("cpu"), pair_batch=2,
    )
    assert np.isneginf(scores[0, 2])
    np.testing.assert_allclose(scores[:, :2], [[1, float(items[2, 0])], [float(items[2, 1]), 1]])
