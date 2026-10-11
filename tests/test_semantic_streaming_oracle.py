import numpy as np
import torch

from rrbench.exact import exact_topk
from rrbench.semantic_ann import item_major_exact_topk


def test_item_major_fp32_oracle_matches_query_major_without_retransferring_the_catalog():
    rng = np.random.default_rng(17)
    items = rng.normal(size=(19, 8)).astype(np.float16)
    query = torch.tensor(rng.normal(size=(7, 8)), dtype=torch.float32)
    expected_scores, expected_ids = exact_topk(items, query, 5, chunk_size=6)
    actual_scores, actual_ids = item_major_exact_topk(items, query, 5, chunk_size=6, query_batch=3)
    assert actual_scores.dtype == torch.float32
    torch.testing.assert_close(actual_scores, expected_scores, rtol=1e-6, atol=1e-6)
    assert torch.equal(actual_ids, expected_ids)
