"""Explicit native-only differential and real captured-task recovery gates."""

import ast
import json
import os
from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    not os.environ.get("RQ_RECOVERY_NATIVE_ROOT"), reason="Explicit isolated native evidence required",
)


def functions():
    import k_means_constrained.sklearn_import.cluster.k_means_ as module
    from rq_recovery_compat import endpoint_candidates
    from rq_recovery_runtime import ANCHOR, INSERT
    root = Path(os.environ["RQ_RECOVERY_NATIVE_ROOT"])
    source = (root / "originals" / "k_means_constrained" /
              "sklearn_import" / "cluster" / "k_means_.py").read_bytes()
    result = []
    for content in (source, source.replace(ANCHOR, ANCHOR + INSERT)):
        node = next(n for n in ast.parse(content).body
                    if isinstance(n, ast.FunctionDef) and n.name == "_k_init")
        namespace = dict(module.__dict__)
        namespace["_rq_endpoint_candidates"] = lambda *args: endpoint_candidates(*args)[0]
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<frozen-k-init>", "exec"), namespace)
        result.append(namespace["_k_init"])
    return result


def equal_rng(left, right):
    assert left[0] == right[0] and np.array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def test_actual_private_overlay_is_imported():
    import k_means_constrained.k_means_constrained_ as estimator
    import k_means_constrained.sklearn_import.cluster.k_means_ as module
    assert hasattr(module, "_rq_compat_k_init"), "Native jobs must import the private overlay"
    assert estimator._init_centroids.__globals__["_k_init"] is module._k_init
    assert str(Path(os.environ["RQ_RECOVERY_NATIVE_ROOT"])) in module.__file__


@pytest.mark.parametrize("variant", ["semantic", "cf"])
def test_captured_native_fault_fixed_with_exact_native_draw_budget(variant):
    original, corrected = functions()
    root = Path(os.environ["RQ_RECOVERY_NATIVE_ROOT"]) / "probes" / f"{variant}-observe/events"
    event = next(json.loads(p.read_text()) for p in root.glob("task-*.json")
                 if json.loads(p.read_text())["status"] == "native-fault")
    archive = np.load(root / event["archive"], allow_pickle=False)
    initial = (event["rng_algorithm"], archive["rng_before_keys"], event["rng_before_position"],
               *event["rng_before_gauss"])
    rng_a, rng_b, expected = np.random.RandomState(), np.random.RandomState(), np.random.RandomState()
    for rng in (rng_a, rng_b, expected):
        rng.set_state(initial)
    with pytest.raises(IndexError, match="256"):
        original(archive["X"], 10, archive["x_squared_norms"], rng_a)
    centers = corrected(archive["X"], 10, archive["x_squared_norms"], rng_b)
    assert centers.shape == (10, 32) and np.isfinite(centers).all()
    expected.randint(256)
    for _ in range(9):
        expected.random_sample(4)
    equal_rng(rng_b.get_state(), expected.get_state())
    assert rng_a.get_state()[2] == event["rng_after_position"]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("case", ["normal", "small", "large", "duplicates", "zero"])
@pytest.mark.parametrize("trials", [None, 1, 8])
def test_all_defined_native_initializations_are_bitwise_identical(dtype, sparse, case, trials):
    import scipy.sparse as sp
    from k_means_constrained.sklearn_import.utils.extmath import row_norms
    original, corrected = functions()
    values = np.random.RandomState(881).normal(size=(256, 32)).astype(dtype)
    if case == "small":
        values *= 0.001
    elif case == "large":
        values *= 1000
    elif case == "duplicates":
        values[32:] = values[np.arange(224) % 32]
    elif case == "zero":
        values[:] = 0
    X = sp.csr_matrix(values) if sparse else np.asfortranarray(values)
    norms = row_norms(X, squared=True)
    left, right = np.random.RandomState(619), np.random.RandomState(619)
    a, b = original(X, 10, norms, left, trials), corrected(X, 10, norms, right, trials)
    assert a.dtype == b.dtype and np.array_equal(a, b)
    equal_rng(left.get_state(), right.get_state())


@pytest.mark.parametrize("variant", ["semantic", "cf"])
def test_real_ten_initialization_workers_and_unaffected_tasks_match(variant, tmp_path, monkeypatch):
    import torch
    from k_means_constrained import KMeansConstrained
    root = Path(os.environ["RQ_RECOVERY_NATIVE_ROOT"])
    base = json.loads((root / "originals" / variant / "base.json").read_text())
    state = torch.load(Path(base["artifact"]["path"]) / "payload/checkpoints/epoch-16400/state.pt",
                       map_location="cpu", weights_only=False)
    values = state["model"]["rq.vq_layers.0.embedding.weight"].numpy().copy()
    events = tmp_path / f"{variant}-tasks"
    events.mkdir()
    monkeypatch.setenv("RQ_RECOVERY_EVENTS", str(events))
    # Reusable loky processes preserve their original environment, so this test
    # uses a fresh executor for each independent native captured-state case.
    from joblib.externals.loky import get_reusable_executor
    get_reusable_executor().shutdown(wait=True, kill_workers=False)
    expected = np.random.RandomState()
    expected.set_state(state["numpy_rng"])
    expected.randint(np.iinfo(np.int32).max, size=10)
    np.random.set_state(state["numpy_rng"])
    model = KMeansConstrained(n_clusters=10, size_min=10, size_max=60,
                             max_iter=10, n_init=10, n_jobs=10, verbose=False)
    model.fit(values)
    equal_rng(np.random.get_state(), expected.get_state())
    tasks = [json.loads(line) for path in events.glob("tasks-*.jsonl")
             for line in path.read_text().splitlines()]
    assert len(tasks) == 10 and all(t["status"] == "success" for t in tasks)
    assert len({t["pid"] for t in tasks}) > 1
    assert all(t["actual_init_centroids_binding"] for t in tasks)
    assert sum(len(t["corrections"]) for t in tasks) == 1
    observed = [json.loads(p.read_text()) for p in
                (root / "probes" / f"{variant}-observe/events").glob("task-*.json")]
    for old in observed:
        task = next(t for t in tasks if t["rng_before_keys_sha256"] == old["rng_before_sha256"])
        assert task["input_sha256"] == old["input_sha256"]
        if old["status"] == "native-success":
            assert task["centers_sha256"] == old["centers_sha256"]
            assert task["rng_after_keys_sha256"] == old["rng_after_sha256"]
            assert task["rng_after_position"] == old["rng_after_position"]
    counts = np.bincount(model.labels_)
    assert len(counts) == 10 and counts.min() >= 10 and counts.max() <= 60
