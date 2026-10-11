import numpy as np
import pytest
import torch
import json

from rrbench.ann import Partition
from rrbench.io import sha256, write_json
from rrbench.semantic_ann import (
    make_shared_ivf, resident_exact_topk, validate_backends, validate_complete_ann,
    validate_journal, validate_oracle, validate_partition, is_faiss_gpu_allocation_failure,
)


def test_fp16_faiss_representation_is_an_actual_half_scalar_code_not_ignored_clone_option():
    faiss = pytest.importorskip("faiss")
    items = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=np.float32)
    partition = Partition(
        np.eye(2, dtype=np.float32), np.array([0, 0, 1, 1]),
        np.arange(4), np.array([0, 2, 4]),
    )
    index = make_shared_ivf(items, partition, half=True)
    assert isinstance(index, faiss.IndexIVFScalarQuantizer)
    assert index.code_size == items.shape[1] * 2
    index.nprobe = 2
    _, identifiers = index.search(np.array([[1, 0]], dtype=np.float32), 2)
    assert identifiers.tolist() == [[0, 1]]


def test_resident_exact_oracle_uses_all_vectors_in_fp32():
    items = np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype=np.float16)
    queries = torch.tensor([[1.0, 0]])
    scores, identifiers = resident_exact_topk(items, queries, 3, chunk_size=2)
    assert scores.dtype == torch.float32
    assert identifiers.tolist() == [[0, 2, 1]]
    assert scores[0, 1].item() == float(items[2, 0])


def test_cached_assignment_order_mismatch_cannot_claim_shared_ivf_lists():
    partition = Partition(
        np.eye(2, dtype=np.float32), np.array([0, 0, 1, 1], dtype=np.int64),
        np.array([2, 3, 0, 1], dtype=np.int64), np.array([0, 2, 4], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="ordered"):
        validate_partition(partition, 4, 2, 2)


def test_groundtruth_data_bytes_are_verified_even_when_receipt_is_unchanged(tmp_path):
    np.save(tmp_path / "ids.npy", np.array([[0, 1]], dtype=np.int64))
    np.save(tmp_path / "scores.npy", np.array([[1, 0]], dtype=np.float32))
    write_json(tmp_path / "completed.json", {
        "complete": True, "queries": 1, "catalog_items": 2, "dimension": 2,
        "ids.npy_sha256": sha256(tmp_path / "ids.npy"), "scores.npy_sha256": sha256(tmp_path / "scores.npy"),
    })
    validate_oracle(tmp_path, 1, 2, 2, 2)
    np.save(tmp_path / "ids.npy", np.array([[1, 0]], dtype=np.int64))
    with pytest.raises(ValueError, match="checksum"):
        validate_oracle(tmp_path, 1, 2, 2, 2)


def test_missing_candidate_file_is_not_a_completed_ann_condition(tmp_path):
    row = {"condition": "silver-fp32-p2-k1", "status": "ok", "prediction_ids_sha256": "missing"}
    with pytest.raises(FileNotFoundError):
        validate_journal([row], tmp_path, {"silver-fp32-p2-k1"}, 1, 2)


def test_undeclared_faiss_int8_arm_is_rejected_before_expensive_work():
    with pytest.raises(ValueError, match="backend"):
        validate_backends(["faiss-gpu-int8"])


def test_changed_metric_or_timing_cannot_recertify_an_old_ann_completion(tmp_path):
    journal = tmp_path / "ann.jsonl"
    journal.write_text(json.dumps({"condition": "one", "geometric_recall_at_k": 1.0}) + "\n", encoding="utf-8")
    write_json(tmp_path / "completed.json", {
        "complete": True, "ann_sha256": sha256(journal), "catalog_items": 10,
        "dimension": 2, "panel_queries": 1, "conditions": ["one"],
    })
    validate_complete_ann(tmp_path, 10, 2, 1, {"one"})
    journal.write_text(json.dumps({"condition": "one", "geometric_recall_at_k": 0.0}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="journal"):
        validate_complete_ann(tmp_path, 10, 2, 1, {"one"})


def test_real_faiss_gpu_allocation_failure_is_explicit_without_hiding_other_backend_errors():
    assert is_faiss_gpu_allocation_failure(RuntimeError(
        "StandardGpuResources: alloc fail type IVFLists cudaMalloc error out of memory [2]"
    ))
    assert not is_faiss_gpu_allocation_failure(RuntimeError("Unsupported scalar quantizer type"))
