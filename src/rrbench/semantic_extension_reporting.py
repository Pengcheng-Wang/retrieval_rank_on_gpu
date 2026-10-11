from __future__ import annotations

from typing import Any

import numpy as np


def paired_quality_interval(
    candidate: np.ndarray, reference: np.ndarray, margin: float = 0.002,
    confidence: float = 0.975, draws: int = 20000, seed: int = 20261011,
) -> dict[str, Any]:
    if candidate.ndim != 1 or reference.shape != candidate.shape or not len(candidate):
        raise ValueError("Quality margin requires aligned nonempty query observations.")
    if (
        not np.isfinite(candidate).all() or not np.isfinite(reference).all()
        or not np.isfinite(margin) or margin <= 0 or not 0 < confidence < 1 or draws <= 0
    ):
        raise ValueError("Quality margin inputs and bootstrap settings must be finite and valid.")
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 256):
        count = min(256, draws - start)
        selections = rng.integers(0, len(difference), size=(count, len(difference)))
        samples[start:start + count] = difference[selections].mean(axis=1)
    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(samples, [alpha, 1 - alpha])
    status = (
        "within_declared_margin" if lower > -margin
        else ("outside_declared_margin" if upper < -margin else "inconclusive")
    )
    return {
        "queries": len(difference), "mean_difference": float(difference.mean()),
        "lower_bound": float(lower), "upper_bound": float(upper), "confidence": confidence,
        "margin": margin, "status": status, "bootstrap_draws": draws, "seed": seed,
        "method": "Paired query percentile bootstrap on unchanged compatible query population.",
        "interpretation": "Scoped observed post-test exploratory margin assessment, not universal quality equivalence or confirmatory generalization.",
    }


def context_averaged_quality_interval(
    candidate: dict[str, np.ndarray], reference: dict[str, np.ndarray],
    candidate_ids: dict[str, list[str]], reference_ids: dict[str, list[str]],
    expected_queries: int = 1000, margin: float = 0.002, confidence: float = 0.975,
    draws: int = 20000, seed: int = 20261011,
) -> dict[str, Any]:
    sessions = {"0", "1", "2"}
    if any(set(value) != sessions for value in (candidate, reference, candidate_ids, reference_ids)):
        raise ValueError("Three fixed execution contexts and their query identities are required.")
    ids = candidate_ids["0"]
    if (
        expected_queries <= 0 or len(ids) != expected_queries or len(set(ids)) != len(ids)
        or any(candidate_ids[session] != ids or reference_ids[session] != ids for session in sessions)
    ):
        raise ValueError("Execution contexts change the independent query identities or order.")
    if any(not isinstance(query, str) or not query.startswith("dev-small:") for query in ids):
        raise ValueError("Primary margin population is selected dev-small, not pooled TREC/IR panels.")
    candidate_values, reference_values, individual = [], [], {}
    for session in sorted(sessions):
        left, right = candidate[session], reference[session]
        if (
            left.shape != (expected_queries,) or right.shape != left.shape
            or not np.isfinite(left).all() or not np.isfinite(right).all()
        ):
            raise ValueError("Context metrics are not aligned finite observations of the same query population.")
        candidate_values.append(left)
        reference_values.append(right)
        individual[session] = float((left - right).mean())
    result = paired_quality_interval(
        np.mean(candidate_values, axis=0), np.mean(reference_values, axis=0),
        margin=margin, confidence=confidence, draws=draws, seed=seed,
    )
    return {
        **result, "execution_contexts": 3, "context_point_differences": individual,
        "context_aggregation": "Average per-query metrics across three fixed execution contexts, then resample the original query identities once.",
        "independence_warning": "Repeated session predictions are not additional independent query observations.",
    }


def observed_headroom(samples: list[dict[str, int]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Observed headroom assessment needs actual device-wide samples.")
    observed, conservative = [], []
    for sample in samples:
        used, total = sample["memory_used_bytes"], sample["device_memory_bytes"]
        allocated = sample.get("torch_allocated_bytes", 0)
        peak = sample.get("torch_peak_allocated_bytes", allocated)
        if (
            any(type(value) is not int for value in (used, total, allocated, peak))
            or not 0 <= used <= total or not 0 <= allocated <= peak or total <= 0
        ):
            raise ValueError("Device-wide memory/allocator observations are invalid.")
        observed.append(total - used)
        conservative.append(total - used - (peak - allocated))
    minimum = min(conservative)
    return {
        "samples": len(samples), "maximum_observed_device_used_bytes": max(row["memory_used_bytes"] for row in samples),
        "minimum_sampled_free_bytes": min(observed),
        "minimum_conservative_headroom_bytes": minimum,
        "passes_8GiB_observed_gate": minimum >= 8 * (1 << 30),
        "scope": "Sampled NVML device memory plus recorded allocator-peak sensitivity; not a continuous peak or unseen-workload guarantee.",
    }


def session_latency_ratios(candidate: dict[str, float], reference: dict[str, float]) -> dict[str, Any]:
    if set(candidate) != {"0", "1", "2"} or set(reference) != set(candidate):
        raise ValueError("All three restarted paired sessions must be present.")
    if any(not np.isfinite(value) or value <= 0 for value in (*candidate.values(), *reference.values())):
        raise ValueError("Session latency means must be finite and positive.")
    ratios = {session: candidate[session] / reference[session] for session in sorted(candidate)}
    mean = float(np.mean(list(ratios.values())))
    return {
        "ratios": ratios, "mean_ratio": mean,
        "minimum_10_percent_mean_gain": mean <= 0.9,
        "within_5_percent_slower_mean": mean <= 1.05,
        "scope": "Descriptive ratios of freshly paired restarted-session means, not serving tails, latency confidence intervals or production speedup.",
    }
