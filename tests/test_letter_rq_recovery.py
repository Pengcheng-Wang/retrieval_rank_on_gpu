"""Narrow numerical and provenance gates for the two stopped RQ continuations."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "letter"))


def correction():
    assert importlib.util.find_spec("rq_recovery_compat") is not None, (
        "A private, guarded compatibility correction is required"
    )
    from rq_recovery_compat import endpoint_candidates
    return endpoint_candidates


def endpoint_case(trailing_zero=False):
    weights = np.zeros((1, 256), dtype=np.float32)
    weights[0, 0] = 1
    weights[0, 254 if trailing_zero else 255] = 3 * 2.0 ** -25
    potential = weights.sum()
    draws = np.array([0, 0.5, np.nextafter(float(potential), 0)])
    ids = np.searchsorted(np.cumsum(weights, dtype=np.float64), draws)
    assert ids[-1] == 256
    return weights, potential, draws, ids


@pytest.mark.parametrize("trailing_zero,endpoint", [(False, 255), (True, 254)])
def test_roundoff_overrun_uses_existing_cdf_endpoint(trailing_zero, endpoint):
    weights, potential, draws, ids = endpoint_case(trailing_zero)
    original = ids.copy()
    corrected, evidence = correction()(weights, potential, draws, ids)
    assert corrected.tolist() == [0, 0, endpoint]
    assert np.array_equal(ids, original)
    assert evidence["affected_trials"] == [2]
    assert evidence["endpoint_index"] == endpoint
    assert evidence["claim"] == "compatibility_extension_at_native_index_error"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_successful_candidates_and_rng_are_exactly_unchanged(dtype):
    rng = np.random.RandomState(71)
    weights = rng.uniform(0.01, 4, size=(1, 256)).astype(dtype)
    potential = weights.sum()
    draws = rng.random_sample(19) * potential
    ids = np.searchsorted(np.cumsum(weights, dtype=np.float64), draws)
    before = rng.get_state()
    result, evidence = correction()(weights, potential, draws, ids)
    assert result is ids
    assert evidence is None
    after = rng.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_zero_potential_and_exact_endpoint_tie_keep_native_indices():
    weights = np.zeros((1, 256), dtype=np.float32)
    ids = np.zeros(4, dtype=np.int64)
    result, evidence = correction()(weights, weights.sum(), np.zeros(4), ids)
    assert result is ids and evidence is None
    weights, potential, _, _ = endpoint_case(True)
    draws = np.array([np.cumsum(weights, dtype=np.float64)[-1]])
    ids = np.array([254])
    result, evidence = correction()(weights, potential, draws, ids)
    assert result is ids and evidence is None


@pytest.mark.parametrize("fault", [
    "negative", "nan", "infinity", "wrong_potential", "over_budget",
    "wrong_indices", "subnormal", "index_above_n", "wrong_dtype",
])
def test_unexplained_out_of_range_cases_fail_closed(fault):
    weights, potential, draws, ids = endpoint_case()
    if fault == "negative":
        weights[0, 8] = -1
    elif fault == "nan":
        weights[0, 8] = np.nan
    elif fault == "infinity":
        draws[-1] = np.inf
    elif fault == "wrong_potential":
        potential = np.float32(float(potential) * 2)
    elif fault == "over_budget":
        draws[-1] = float(potential) * 2
    elif fault == "wrong_indices":
        ids[1] = 256
    elif fault == "subnormal":
        weights[0, 8] = np.nextafter(np.float32(0), np.float32(1))
    elif fault == "index_above_n":
        ids[-1] = 257
    elif fault == "wrong_dtype":
        weights = weights.astype(np.float16)
    with pytest.raises((ValueError, FloatingPointError)):
        correction()(weights, potential, draws, ids)


def test_overlay_rejects_unpinned_package_without_writing(tmp_path):
    assert importlib.util.find_spec("rq_recovery_runtime") is not None, (
        "Private package construction must be checksum guarded"
    )
    from rq_recovery_runtime import make_overlay
    source = tmp_path / "source"
    module = source / "sklearn_import" / "cluster" / "k_means_.py"
    module.parent.mkdir(parents=True)
    module.write_text("def _k_init(): return None\n")
    destination = tmp_path / "overlay"
    with pytest.raises(ValueError, match="pinned"):
        make_overlay(source, destination)
    assert not destination.exists()


@pytest.mark.parametrize("variant,gpu", [("semantic", "1"), ("cf", "6")])
def test_controller_scope_only_admits_owned_leases(tmp_path, variant, gpu):
    assert importlib.util.find_spec("rq_recovery_control") is not None, (
        "Continuation must have an isolated fail-closed lease controller"
    )
    from rq_recovery_control import validate_scope
    run = tmp_path / "runs" / f"openllama-v2-rq-Instruments-{variant}-s42"
    assert validate_scope(tmp_path, variant, run, gpu, "node-1") == "production"
    probe = tmp_path / "rq_recovery_v1" / "probes" / variant / "run"
    assert validate_scope(tmp_path, variant, probe, gpu, "node-1") == "probe"
    for output, device, host in (
        (run, "0", "node-1"), (run, gpu, "node-0"),
        (tmp_path / "runs" / "openllama-v2-rq-Beauty-semantic-s42", gpu, "node-1"),
        (tmp_path / "outside", gpu, "node-1"),
    ):
        with pytest.raises(ValueError):
            validate_scope(tmp_path, variant, output, device, host)
