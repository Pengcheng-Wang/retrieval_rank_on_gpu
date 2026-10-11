import copy

import pytest

from rrbench.semantic_reporting import public_quality, public_timing
from rrbench.semantic_reporting import checked_receipt
from rrbench.semantic_reporting import verify_online_parity
from rrbench.semantic_reporting import verify_online_protocol
from rrbench.semantic_reporting import public_online_cell
from rrbench.io import sha256, write_json


def test_public_metrics_keep_aggregates_and_hit_counts_without_private_query_identifiers():
    quality = {
        "queries": 2, "recall@10": 0.5, "ndcg@10": 0.4,
        "per_query": [{"query_id": "3:private-user:time", "recall@10": 1.0, "ndcg@10": 0.8},
                      {"query_id": "3:another-user:time", "recall@10": 0.0, "ndcg@10": 0.0}],
    }
    result = public_quality(quality)
    assert result["recall@10"] == 0.5
    assert result["hit_queries@10"] == 1
    assert "per_query" not in result
    assert "private-user" not in str(result)


def test_primary_three_batch_timing_publication_does_not_imply_p99_or_slo():
    result = public_timing({
        "samples_ms": [1.0, 2.0, 3.0], "sample_count": 3, "latency_mean_ms": 2.0,
        "latency_p50_ms": 2.0, "observed_min_ms": 1.0, "observed_max_ms": 3.0,
    })
    assert result["samples_ms"] == [1.0, 2.0, 3.0]
    assert "latency_p99_ms" not in result
    assert "ci95_ms" not in result


def test_mismatched_recorded_metric_is_not_sanitized_into_success():
    with pytest.raises(ValueError, match="mean"):
        public_quality({
            "queries": 1, "recall@10": 0.9, "per_query": [{"query_id": "q", "recall@10": 0.0}],
        })


def test_quality_payload_corruption_is_checked_before_public_export(tmp_path):
    scores = tmp_path / "scores.npy"
    scores.write_bytes(b"verified-score-fixture")
    write_json(tmp_path / "completed.json", {"complete": True, "scores_sha256": sha256(scores)})
    checked_receipt(tmp_path / "completed.json")
    scores.write_bytes(b"changed")
    with pytest.raises(ValueError, match="payload"):
        checked_receipt(tmp_path / "completed.json")


def test_full_online_result_cannot_claim_success_with_a_false_same_scorer_parity_condition():
    rows = [{
        "query_batch": batch, "candidate_cap": cap, "same_candidates_scores_final_ids": True,
    } for batch in (1, 16) for cap in (10, 50, 100, 500)]
    verify_online_parity({"parity": rows})
    rows[-1]["same_candidates_scores_final_ids"] = False
    with pytest.raises(ValueError, match="parity"):
        verify_online_parity({"parity": rows})


def test_stale_remote_online_budget_cannot_be_exported_as_the_amended_primary_measurement():
    with pytest.raises(ValueError, match="protocol"):
        verify_online_protocol({"conditions": []}, {})


def online_cell():
    conditions = []
    for batch in (1, 16):
        rows = [list(range(start, start + batch)) for start in range(0, batch * 4, batch)]
        for cap in (10, 50, 100, 500):
            for boundary in ("integrated", "local-cascade"):
                conditions.append({
                    "query_batch": batch, "candidate_cap": cap, "boundary": boundary,
                    "warmup_query_rows": rows[0], "measured_query_row_batches": rows[1:],
                    "full_online_latency": {
                        "samples_ms": [1.0, 2.0, 3.0], "sample_count": 3, "latency_mean_ms": 2.0,
                    },
                })
    return {
        "conditions": conditions,
        "parity": [{
            "query_batch": batch, "candidate_cap": cap, "same_candidates_scores_final_ids": True,
        } for batch in (1, 16) for cap in (10, 50, 100, 500)],
        "device_memory_samples": [{
            "memory_used_bytes": 80, "device_memory_bytes": 100, "gpu_utilization_percent": 75,
        }],
        "maximum_observed_device_memory_used_bytes": 80,
        "device_memory_sampling": "Outside timers; not continuous.",
        "peak_cuda_allocated_bytes": 20, "memory_caveat": "Excludes external allocations.",
        "timing_scope": "Full pipeline", "network": "Local serialization",
        "text_integrated_boundary": "Host tokenization",
    }, {
        "primary_warmup_batches": 1, "primary_measured_batches": 3,
        "timing_query_policy": "Fourdisjointhash-selectedbatches, shared acrossmodels/caps/boundaries.",
        "history_bound_preprocessing": "Unique raw queries per scorerbatch; identical bounded tokens reused for repeated candidatepairs.",
        "source_root": "/private/operational/path",
    }, {"corpus": "kuairand", "ranker": "qwen8-rerank"}


def test_online_publication_preserves_memory_and_parity_but_only_hashes_query_schedules():
    value, config, context = online_cell()
    result = public_online_cell(value, config, context)
    assert result["parity"] == value["parity"]
    assert result["device_memory_samples"] == value["device_memory_samples"]
    assert set(result["query_schedule_sha256_by_batch"]) == {"1", "16"}
    assert all(len(digest) == 64 for digest in result["query_schedule_sha256_by_batch"].values())
    assert "source_root" not in result["configuration"]
    assert "measured_query_row_batches" not in str(result)


def test_online_publication_rejects_naive_history_preprocessing_as_a_final_optimized_cell():
    value, config, context = online_cell()
    config.pop("history_bound_preprocessing")
    with pytest.raises(ValueError, match="optimized history"):
        public_online_cell(value, config, context)


def test_online_publication_rejects_mismatched_candidate_cap_query_schedules():
    value, config, context = online_cell()
    value["conditions"][-1] = copy.deepcopy(value["conditions"][-1])
    value["conditions"][-1]["measured_query_row_batches"][-1][-1] = 1000
    with pytest.raises(ValueError, match="different measured query"):
        public_online_cell(value, config, context)


def test_online_publication_rejects_a_misstated_device_memory_maximum():
    value, config, context = online_cell()
    value["maximum_observed_device_memory_used_bytes"] = 79
    with pytest.raises(ValueError, match="memory"):
        public_online_cell(value, config, context)
