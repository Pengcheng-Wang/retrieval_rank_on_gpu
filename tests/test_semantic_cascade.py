import numpy as np
import pytest
import torch

from rrbench.semantic_cascade import (
    pipeline_memory_bytes, rank_candidates, transfer_candidate_payload, transfer_candidates,
)


def test_same_candidate_ids_and_scores_produce_same_final_topk_for_both_stage_boundaries():
    candidates = torch.tensor([[5, 3, 8, -1], [8, 5, 1, 4]], dtype=torch.int64)
    scores = torch.tensor([[0.3, 0.8, 0.4, -torch.inf], [0.1, 0.2, 0.9, 0.3]])
    resident = transfer_candidates(candidates, "integrated", torch.device("cpu"))
    cascade = transfer_candidates(candidates, "local-cascade", torch.device("cpu"))
    np.testing.assert_array_equal(rank_candidates(resident, scores, 2).numpy(), [[3, 8], [1, 4]])
    torch.testing.assert_close(rank_candidates(resident, scores, 2), rank_candidates(cascade, scores, 2))


def test_unknown_boundary_or_misaligned_scorer_is_not_a_success_fallback():
    with pytest.raises(ValueError, match="boundary"):
        transfer_candidates(torch.tensor([[1]]), "assumed-rpc", torch.device("cpu"))
    with pytest.raises(ValueError, match="aligned"):
        rank_candidates(torch.tensor([[1, 2]]), torch.tensor([[0.1]]), 1)


def test_native_4096_index_plus_two_large_models_does_not_pretend_to_fit_one_gpu():
    native = pipeline_memory_bytes(8841823, 4096, 2, 7567295488, 8188548096, cached_features=False)
    mrl = pipeline_memory_bytes(8841823, 1024, 2, 7567295488, 8188548096, cached_features=False)
    assert native > 80 * (1 << 30)
    assert mrl < 80 * (1 << 30)


def test_local_cascade_reuses_the_received_host_ids_for_feature_fetch_without_a_second_roundtrip():
    source = torch.tensor([[5, 3, 8]], dtype=torch.int64)
    device_ids, host_ids = transfer_candidate_payload(source, "local-cascade", torch.device("cpu"))
    assert host_ids is not None
    assert host_ids.device.type == "cpu"
    torch.testing.assert_close(device_ids, host_ids)
    _, integrated_host_ids = transfer_candidate_payload(source, "integrated", torch.device("cpu"))
    assert integrated_host_ids is None
