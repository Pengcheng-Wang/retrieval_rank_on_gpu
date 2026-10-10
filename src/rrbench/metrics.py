from __future__ import annotations

import numpy as np


def recall_at_k(predicted: np.ndarray, truth: np.ndarray, k: int) -> np.ndarray:
    if predicted.ndim != 2 or truth.ndim != 2 or len(predicted) != len(truth):
        raise ValueError("Predictions and truth must be aligned 2-D arrays.")
    if k <= 0:
        raise ValueError("k must be positive.")
    recalls = []
    for row, target in zip(predicted[:, :k], truth[:, :k]):
        relevant = set(int(i) for i in target if i >= 0)
        retrieved = set(int(i) for i in row if i >= 0)
        recalls.append(len(relevant & retrieved) / len(relevant) if relevant else 0.0)
    return np.asarray(recalls, dtype=np.float64)


def single_positive_metrics(
    rankings: np.ndarray, targets: np.ndarray, ks: list[int]
) -> dict[str, float]:
    if rankings.ndim != 2 or targets.shape != (len(rankings),):
        raise ValueError("Targets must have one item per ranking row.")
    result = {}
    for k in ks:
        if k <= 0:
            raise ValueError("Cutoffs must be positive.")
        hits = rankings[:, :k] == targets[:, None]
        found = hits.any(axis=1)
        positions = hits.argmax(axis=1) + 1
        gain = np.where(found, 1.0 / np.log2(positions + 1.0), 0.0)
        result[f"recall@{k}"] = float(found.mean())
        result[f"ndcg@{k}"] = float(gain.mean())
    return result


def bootstrap_mean_ci(values: np.ndarray, seed: int = 0, draws: int = 1000) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("A nonempty finite 1-D sample is required.")
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    rng = np.random.default_rng(seed)
    means = np.array(
        [values[rng.integers(0, len(values), len(values))].mean() for _ in range(draws)]
    )
    return np.quantile(means, [0.025, 0.975]).tolist()


def latency_summary(samples_ms: list[float], batch_size: int) -> dict[str, object]:
    values = np.asarray(samples_ms, dtype=np.float64)
    if batch_size <= 0 or len(values) == 0 or (values <= 0).any():
        raise ValueError("Positive latencies and batch size are required.")
    return {
        "latency_mean_ms": float(values.mean()),
        "latency_p50_ms": float(np.quantile(values, 0.5)),
        "latency_p95_ms": float(np.quantile(values, 0.95)),
        "latency_p99_ms": float(np.quantile(values, 0.99)),
        "latency_mean_ci95_ms": bootstrap_mean_ci(values),
        "offline_queries_per_second": float(batch_size * 1000 / values.mean()),
        "samples_ms": samples_ms,
        "sample_count": len(samples_ms),
        "throughput_kind": "sequential offline batches; not serving QPS under an SLO",
        "p99_warning": "descriptive only; small samples cannot establish a serving tail SLO",
    }
