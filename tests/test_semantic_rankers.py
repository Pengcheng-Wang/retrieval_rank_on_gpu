import pytest
import torch

from rrbench.semantic_rankers import CachedSemanticMLP, semantic_features


def test_cached_ranker_uses_transferable_query_item_features_not_query_ids():
    query = torch.tensor([[1.0, 0], [0, 1]])
    item = torch.tensor([[0.6, 0.8], [0.8, 0.6]])
    features = semantic_features(query, item)
    assert features.shape == (2, 8)
    torch.testing.assert_close(features[:, 4:6], query * item)
    model = CachedSemanticMLP(2)
    assert model(query, item).shape == (2,)
    assert torch.isfinite(model(query, item)).all()


def test_misaligned_candidate_features_fail_explicitly():
    with pytest.raises(ValueError, match="aligned"):
        semantic_features(torch.ones(2, 3), torch.ones(3, 3))
