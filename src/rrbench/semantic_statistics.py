from __future__ import annotations

from typing import Any

import numpy as np


def paired_cluster_effect(
    left: np.ndarray, right: np.ndarray, groups: np.ndarray, seed: int = 17, draws: int = 2000,
) -> dict[str, Any]:
    if (
        left.ndim != 1 or left.shape != right.shape or left.shape != groups.shape
        or not len(left) or not np.isfinite(left).all() or not np.isfinite(right).all() or draws <= 0
    ):
        raise ValueError("Paired uncertainty requires aligned finite measurements/groups and positive draws.")
    unique, inverse = np.unique(groups, return_inverse=True)
    difference = left.astype(np.float64) - right.astype(np.float64)
    counts = np.bincount(inverse)
    effects = np.bincount(inverse, weights=difference) / counts
    rng = np.random.default_rng(seed)
    samples = np.asarray([
        effects[rng.integers(0, len(unique), len(unique))].mean() for _ in range(draws)
    ])
    return {
        "mean_user_effect": float(effects.mean()),
        "mean_event_or_query_effect": float(difference.mean()),
        "ci95_mean_user_effect": np.quantile(samples, [0.025, 0.975]).tolist(),
        "users_or_query_groups": len(unique), "paired_rows": len(left),
        "bootstrap_draws": draws, "seed": seed,
        "estimand": "Equal group-weighted paired effect; groups are users for recommendation or query groups for IR.",
        "warning": "This conditional sampled-panel effect cannot be multiplied by full-catalog event coverage to claim all-target recall.",
    }
