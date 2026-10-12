import numpy as np
import pytest

from rrbench import semantic_extension_reporting as reporting


def test_noninferiority_requires_the_lower_bound_not_a_small_point_loss():
    differences = np.array([-0.10, 0.098, 0.0, 0.0])
    result = reporting.paired_quality_interval(
        differences, np.zeros_like(differences), margin=0.002, confidence=0.975, draws=1000, seed=17,
    )
    assert result["mean_difference"] > -0.002
    assert result["lower_bound"] < -0.002
    assert result["status"] == "inconclusive"


def test_exactly_matching_observed_metrics_can_pass_a_scoped_margin():
    result = reporting.paired_quality_interval(
        np.ones(20), np.ones(20), margin=0.002, confidence=0.975, draws=100, seed=17,
    )
    assert result["status"] == "within_declared_margin"
    assert "observed" in result["interpretation"]


def test_quality_with_established_excess_loss_fails_instead_of_inconclusive():
    result = reporting.paired_quality_interval(
        np.zeros(20), np.full(20, 0.1), margin=0.002, confidence=0.975, draws=100, seed=17,
    )
    assert result["status"] == "outside_declared_margin"


def test_bootstrap_cannot_pair_different_populations_or_nonfinite_metrics():
    with pytest.raises(ValueError, match="aligned"):
        reporting.paired_quality_interval(np.array([1.0]), np.array([1.0, 0.0]), draws=10)
    with pytest.raises(ValueError, match="finite"):
        reporting.paired_quality_interval(np.array([np.nan]), np.array([0.0]), draws=10)


def test_device_headroom_counts_external_allocations_and_not_only_torch():
    result = reporting.observed_headroom([{
        "memory_used_bytes": 75 * (1 << 30), "device_memory_bytes": 80 * (1 << 30),
        "torch_allocated_bytes": 20 * (1 << 30), "torch_peak_allocated_bytes": 22 * (1 << 30),
    }])
    assert result["minimum_conservative_headroom_bytes"] == 3 * (1 << 30)
    assert result["passes_8GiB_observed_gate"] is False


def test_descriptive_session_ratio_reports_every_restart():
    result = reporting.session_latency_ratios(
        {"0": 8.0, "1": 9.0, "2": 10.0}, {"0": 10.0, "1": 10.0, "2": 10.0},
    )
    assert result["ratios"] == {"0": 0.8, "1": 0.9, "2": 1.0}
    assert result["mean_ratio"] == pytest.approx(0.9)
    assert "not" in result["scope"]
    with pytest.raises(ValueError, match="sessions"):
        reporting.session_latency_ratios({"0": 8}, {"0": 10, "1": 10})


def test_context_averaging_does_not_triple_the_independent_query_count():
    candidate = {str(session): np.array([0.1 + session * 0.1, 0.0]) for session in range(3)}
    reference = {str(session): np.zeros(2) for session in range(3)}
    query_ids = {str(session): ["dev-small:a", "dev-small:b"] for session in range(3)}
    result = reporting.context_averaged_quality_interval(
        candidate, reference, query_ids, query_ids, expected_queries=2, draws=100, seed=17,
    )
    assert result["queries"] == 2
    assert result["execution_contexts"] == 3
    assert result["mean_difference"] == pytest.approx(0.1)
    assert len(result["context_point_differences"]) == 3


def test_context_averaging_rejects_changed_query_order_or_a_pooled_trec_panel():
    candidate = {str(session): np.ones(2) for session in range(3)}
    reference = {str(session): np.ones(2) for session in range(3)}
    query_ids = {str(session): ["dev-small:a", "dev-small:b"] for session in range(3)}
    changed = {**query_ids, "1": ["dev-small:b", "dev-small:a"]}
    with pytest.raises(ValueError, match="identit"):
        reporting.context_averaged_quality_interval(
            candidate, reference, query_ids, changed, expected_queries=2, draws=10,
        )
    mixed = {str(session): ["dev-small:a", "trec19:b"] for session in range(3)}
    with pytest.raises(ValueError, match="dev"):
        reporting.context_averaged_quality_interval(
            candidate, reference, mixed, mixed, expected_queries=2, draws=10,
        )
