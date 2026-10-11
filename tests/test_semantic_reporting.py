import pytest

from rrbench.semantic_reporting import public_quality, public_timing
from rrbench.semantic_reporting import checked_receipt
from rrbench.semantic_reporting import verify_online_parity
from rrbench.semantic_reporting import verify_online_protocol
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
