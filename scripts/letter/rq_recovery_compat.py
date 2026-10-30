"""Fail-closed endpoint extension for the frozen k-means-constrained 0.7.3."""

from __future__ import annotations

import math

import numpy as np


POLICY = "rq-kmeans-cdf-endpoint-v1"
CLAIM = "compatibility_extension_at_native_index_error"


def endpoint_candidates(distances, potential, draws, candidates):
    """Leave defined native outcomes untouched; extend only a rounded CDF overrun."""
    weights = np.asarray(distances)
    ids = np.asarray(candidates)
    count = weights.size
    if ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids > count):
        raise ValueError("Unexpected native candidate index")
    bad = ids == count
    if not np.any(bad):
        return candidates, None
    values = np.asarray(draws)
    p = np.asarray(potential)
    if (count < 2 or weights.ndim not in (1, 2)
            or (weights.ndim == 2 and weights.shape[0] != 1)
            or weights.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or p.shape != () or p.dtype != weights.dtype
            or values.ndim != 1 or values.shape != ids.shape or values.dtype != np.float64
            or not np.isfinite(weights).all() or np.any(weights < 0)
            or not np.isfinite(values).all() or not np.isfinite(p)):
        raise ValueError("Unsupported or nonfinite native CDF arithmetic")
    native_sum = weights.sum()
    if native_sum.tobytes() != p.tobytes():
        raise ValueError("Potential differs from the unchanged native reduction")
    tiny = np.finfo(weights.dtype).tiny
    if float(p) < tiny or np.any((weights > 0) & (weights < tiny)):
        raise FloatingPointError("Subnormal CDF arithmetic is not admitted by this policy")
    cumulative = np.cumsum(weights, dtype=np.float64)
    if (not np.isfinite(cumulative).all() or np.any(np.diff(cumulative) < 0)
            or not np.array_equal(np.searchsorted(cumulative, values), ids)):
        raise ValueError("Candidate indices do not match the finite native CDF")
    total = float(cumulative[-1])
    native_total = float(p)
    if (total <= 0 or not total < native_total or np.any(values[bad] <= total)
            or np.any(values[bad] > native_total)):
        raise FloatingPointError("Not a rounded positive endpoint overrun")
    m = count - 1
    up = float(np.finfo(weights.dtype).eps) / 2
    uc = float(np.finfo(np.float64).eps) / 2
    if m * up >= 1 or m * uc >= 1:
        raise FloatingPointError("Summation error bound is not defined")
    gp, gc = m * up / (1 - m * up), m * uc / (1 - m * uc)
    bound = total * (gp + gc) / (1 - gc)
    if native_total - total > math.nextafter(bound, math.inf):
        raise FloatingPointError("Endpoint discrepancy exceeds nonnegative-sum roundoff")
    endpoint = int(np.searchsorted(cumulative, total, side="left"))
    repaired = ids.copy()
    repaired[bad] = endpoint
    return repaired, {
        "policy": POLICY, "claim": CLAIM, "samples": count,
        "distance_dtype": str(weights.dtype), "native_potential": native_total,
        "cdf_endpoint": total, "roundoff_ceiling": bound,
        "affected_trials": np.flatnonzero(bad).tolist(), "endpoint_index": endpoint,
        "native_candidates": ids.tolist(), "corrected_candidates": repaired.tolist(),
        "draws": values.tolist(),
    }
