"""Private package construction and actual loky-task compatibility attestation."""

from __future__ import annotations

import contextvars
import hashlib
import json
import marshal
import os
from pathlib import Path
import shutil
import time

import numpy as np

from rq_recovery_compat import POLICY, endpoint_candidates


MODULE = "sklearn_import/cluster/k_means_.py"
NATIVE_SHA256 = "7d1dd15a050d22fcde3b4e427959be7031cab0345b715ada96563f7dc016f2db"
ANCHOR = (
    b"        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),\n"
    b"                                        rand_vals)\n"
)
INSERT = (
    b"        candidate_ids = _rq_endpoint_candidates(\n"
    b"            closest_dist_sq, current_pot, rand_vals, candidate_ids)\n"
)
SUFFIX = b"""

from rq_recovery_runtime import correct_candidates as _rq_endpoint_candidates
from rq_recovery_runtime import audit_k_init as _rq_audit_k_init
_rq_compat_k_init = _k_init
def _k_init(X, n_clusters, x_squared_norms, random_state, n_local_trials=None):
    return _rq_audit_k_init(_rq_compat_k_init, X, n_clusters, x_squared_norms,
                            random_state, n_local_trials)
"""
_TASK = contextvars.ContextVar("rq_recovery_initialization", default=None)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_manifest(root: Path) -> dict:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Compatibility evidence must not contain symlinks")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = {
                "sha256": sha(path), "bytes": path.stat().st_size,
            }
    return result


def make_overlay(source: Path, destination: Path) -> dict:
    original = (source / MODULE).read_bytes()
    if hashlib.sha256(original).hexdigest() != NATIVE_SHA256 or original.count(ANCHOR) != 1:
        raise ValueError("Native k-means++ source is not the pinned 0.7.3 implementation")
    if destination.exists():
        raise FileExistsError("Compatibility overlays are immutable")
    original_files = file_manifest(source)
    package = destination / "k_means_constrained"
    shutil.copytree(source, package)
    (package / MODULE).write_bytes(original.replace(ANCHOR, ANCHOR + INSERT) + SUFFIX)
    overlay_files = file_manifest(package)
    changed = [name for name in original_files if original_files[name] != overlay_files[name]]
    if changed != [MODULE] or set(original_files) != set(overlay_files):
        raise ValueError("Overlay changed more than the guarded native initialization module")
    receipt = {
        "policy": POLICY, "native_module_sha256": NATIVE_SHA256, "changed": changed,
        "original_files": original_files, "overlay_files": overlay_files,
    }
    (destination / "overlay.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def array_sha(value) -> str:
    value = np.asarray(value)
    return hashlib.sha256(value.dtype.str.encode() + str(value.shape).encode()
                          + value.tobytes(order="C")).hexdigest()


def rng_sha(state) -> str:
    return hashlib.sha256(
        state[0].encode() + state[1].tobytes()
        + repr(state[2:]).encode()
    ).hexdigest()


def _append(root: Path, event: dict) -> None:
    content = (json.dumps(event, sort_keys=True, allow_nan=False) + "\n").encode()
    with (root / f"tasks-{os.getpid()}.jsonl").open("ab") as stream:
        stream.write(content)
        stream.flush()


def correct_candidates(distances, potential, draws, candidates):
    result, correction = endpoint_candidates(distances, potential, draws, candidates)
    if correction is None:
        return result
    context = _TASK.get()
    if context is None:
        raise RuntimeError("A corrected native task must have its input/RNG attested")
    frame = __import__("sys")._getframe(1).f_locals
    serial = len(context["corrections"])
    name = f'correction-{context["task_id"]}-{serial}.npz'
    rng = frame["random_state"].get_state()
    np.savez(
        context["root"] / name, X=frame["X"], x_squared_norms=frame["x_squared_norms"],
        distances=distances, potential=potential, draws=draws, candidates=candidates,
        rng_before_keys=context["rng_before"][1], rng_after_draw_keys=rng[1],
    )
    correction.update(
        center=int(frame["c"]), evidence=name, evidence_sha256=sha(context["root"] / name),
        rng_after_draw_sha256=rng_sha(rng), rng_after_draw_position=int(rng[2]),
    )
    context["corrections"].append(correction)
    return result


def audit_k_init(native, X, n_clusters, x_squared_norms, random_state, n_local_trials=None):
    import k_means_constrained.k_means_constrained_ as estimator
    import k_means_constrained.sklearn_import.cluster.k_means_ as module

    root = Path(os.environ["RQ_RECOVERY_EVENTS"])
    if (not root.is_dir()
            or estimator._init_centroids.__globals__["_k_init"] is not module._k_init
            or native is not module._rq_compat_k_init):
        raise RuntimeError("Actual native initialization is not bound to the private overlay")
    before = random_state.get_state()
    task_id = f"{os.getpid()}-{time.time_ns()}"
    context = {
        "root": root, "task_id": task_id, "rng_before": before, "corrections": [],
    }
    event = {
        "task_id": task_id, "pid": os.getpid(), "ppid": os.getppid(), "policy": POLICY,
        "input_sha256": array_sha(X.toarray() if hasattr(X, "toarray") else X),
        "shape": list(X.shape), "dtype": str(X.dtype),
        "n_clusters": n_clusters, "n_local_trials": n_local_trials,
        "rng_before_sha256": rng_sha(before), "rng_before_position": int(before[2]),
        "rng_before_keys_sha256": array_sha(before[1]),
        "initial_seed": int(before[1][0]) if before[2] == 624 else None,
        "module_origin": str(Path(module.__file__).resolve()),
        "module_sha256": sha(Path(module.__file__)),
        "function_code_sha256": hashlib.sha256(marshal.dumps(native.__code__)).hexdigest(),
        "actual_init_centroids_binding": True,
    }
    context_path = root / "task-context.json"
    if context_path.is_file():
        call = json.loads(context_path.read_text())
        if (call["expected_input_sha256"] != event["input_sha256"]
                or event["initial_seed"] not in call["initialization_seeds"]):
            raise ValueError("Actual worker input/seed differs from its parent clustering call")
        event["context"] = call
        event["context_sha256"] = sha(context_path)
    elif os.environ.get("RQ_RECOVERY_REQUIRE_CONTEXT") == "1":
        raise RuntimeError("Production initialization lacks its epoch/phase context")
    token = _TASK.set(context)
    try:
        result = native(X, n_clusters, x_squared_norms, random_state, n_local_trials)
        event.update(status="success", centers_sha256=array_sha(result))
        return result
    except BaseException as error:
        event.update(status="failure", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        event.update(
            rng_after_sha256=rng_sha(random_state.get_state()),
            rng_after_keys_sha256=array_sha(random_state.get_state()[1]),
            rng_after_position=int(random_state.get_state()[2]),
            corrections=context["corrections"],
        )
        _TASK.reset(token)
        _append(root, event)
