"""Pinned k-means endpoint overlay for two recovered Figure 8 tokenizer arms."""

from __future__ import annotations

import ast
import contextvars
import hashlib
import json
import marshal
import os
from pathlib import Path
import re
import shutil
import time
import zipfile

import numpy as np

from rq_recovery_compat import POLICY, endpoint_candidates
from rq_recovery_lineage import audit_event_archive as audit_compatibility_archive
from rq_recovery_runtime import array_sha, file_manifest, rng_sha, sha


MODULE = "sklearn_import/cluster/k_means_.py"
NATIVE_SHA256 = "7d1dd15a050d22fcde3b4e427959be7031cab0345b715ada96563f7dc016f2db"
ANCHOR = (
    b"        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),\n"
    b"                                        rand_vals)\n"
)
INSERT = (
    b"        candidate_ids = _figure8_endpoint_candidates(\n"
    b"            closest_dist_sq, current_pot, rand_vals, candidate_ids)\n"
)
SUFFIX = b"""

from figure8_tokenizer_recovery_runtime import correct_candidates as _figure8_endpoint_candidates
from figure8_tokenizer_recovery_runtime import audit_k_init as _figure8_audit_k_init
_figure8_compat_k_init = _k_init
def _k_init(X, n_clusters, x_squared_norms, random_state, n_local_trials=None):
    return _figure8_audit_k_init(
        _figure8_compat_k_init, X, n_clusters, x_squared_norms,
        random_state, n_local_trials,
    )
"""
ENDPOINT_ERROR = re.compile(
    r"^index (?P<index>\d+) is out of bounds for axis 0 with size (?P<size>\d+)$"
)
_TASK = contextvars.ContextVar("figure8_tokenizer_recovery_initialization", default=None)
_PRISTINE = {}


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
        "policy": POLICY,
        "native_module_sha256": NATIVE_SHA256,
        "changed": changed,
        "original_files": original_files,
        "overlay_files": overlay_files,
    }
    (destination / "overlay.json").write_text(
        __import__("json").dumps(receipt, indent=2) + "\n",
        encoding="utf-8",
    )
    return receipt


def same_rng(left, right) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def validate_differential(
    *,
    native_error,
    native_centers,
    native_rng,
    corrected_centers,
    corrected_rng,
    corrections,
    sample_count: int,
) -> str:
    if native_error is None:
        if corrections:
            raise ValueError("A defined native initialization unexpectedly needed a correction")
        if not np.array_equal(native_centers, corrected_centers):
            raise ValueError("Defined native and compatibility centers differ")
        if not same_rng(native_rng, corrected_rng):
            raise ValueError("Defined native and compatibility RNG states differ")
        return "native-defined-bitwise-equal"
    if not isinstance(native_error, IndexError):
        raise native_error
    match = ENDPOINT_ERROR.fullmatch(str(native_error))
    if (
        match is None
        or int(match.group("index")) != sample_count
        or int(match.group("size")) != sample_count
    ):
        raise native_error
    if not corrections:
        raise ValueError("The exact native endpoint fault was not repaired by a guarded correction")
    for correction in corrections:
        native = correction.get("native_candidates")
        corrected = correction.get("corrected_candidates")
        affected = correction.get("affected_trials")
        if (
            not isinstance(native, list)
            or not isinstance(corrected, list)
            or not isinstance(affected, list)
            or not affected
            or any(native[index] != sample_count for index in affected)
            or any(not 0 <= corrected[index] < sample_count for index in affected)
        ):
            raise ValueError("Endpoint correction evidence does not match the native fault")
    return "native-endpoint-fault-corrected"


def execute_differential(
    corrected,
    pristine,
    X,
    n_clusters: int,
    x_squared_norms,
    random_state,
    n_local_trials,
    corrections: list[dict],
):
    native_rng = np.random.RandomState()
    native_rng.set_state(random_state.get_state())
    native_centers = None
    native_error = None
    try:
        native_centers = pristine(
            X, n_clusters, x_squared_norms, native_rng, n_local_trials,
        )
    except BaseException as error:
        native_error = error
    if native_error is not None:
        match = ENDPOINT_ERROR.fullmatch(str(native_error)) if isinstance(native_error, IndexError) else None
        if (
            match is None
            or int(match.group("index")) != len(X)
            or int(match.group("size")) != len(X)
        ):
            raise native_error
    corrected_centers = corrected(
        X, n_clusters, x_squared_norms, random_state, n_local_trials,
    )
    classification = validate_differential(
        native_error=native_error,
        native_centers=native_centers,
        native_rng=native_rng.get_state(),
        corrected_centers=corrected_centers,
        corrected_rng=random_state.get_state(),
        corrections=corrections,
        sample_count=len(X),
    )
    return corrected_centers, {
        "classification": classification,
        "native_error": (
            f"{type(native_error).__name__}: {native_error}"
            if native_error is not None else None
        ),
        "centers_equal": native_error is None,
        "rng_equal": native_error is None,
        "native_rng_after_sha256": rng_sha(native_rng.get_state()),
        "native_rng_after_position": int(native_rng.get_state()[2]),
    }


def audit_event_archive(path: Path, invocation: str) -> dict:
    audited = audit_compatibility_archive(path)
    native_defined = 0
    native_faults = 0
    with zipfile.ZipFile(path) as archive:
        task_files = sorted(
            name for name in archive.namelist()
            if Path(name).name.startswith("tasks-") and Path(name).suffix == ".jsonl"
        )
        for name in task_files:
            for line in archive.read(name).splitlines():
                event = json.loads(line)
                context = event.get("context", {})
                if context.get("invocation") != invocation:
                    raise ValueError("Compatibility task belongs to another invocation")
                differential = event.get("differential")
                if not isinstance(differential, dict):
                    raise ValueError("Compatibility task lacks an exact native differential")
                classification = differential.get("classification")
                if classification == "native-defined-bitwise-equal":
                    if (
                        differential.get("centers_equal") is not True
                        or differential.get("rng_equal") is not True
                        or differential.get("native_error") is not None
                        or event.get("corrections")
                    ):
                        raise ValueError("Defined native task differs under compatibility execution")
                    native_defined += 1
                elif classification == "native-endpoint-fault-corrected":
                    if (
                        not isinstance(differential.get("native_error"), str)
                        or not event.get("corrections")
                    ):
                        raise ValueError("Endpoint fault lacks its compatibility differential")
                    native_faults += 1
                else:
                    raise ValueError("Unknown compatibility differential classification")
    summary = {
        **audited["summary"],
        "native_defined": native_defined,
        "native_faults": native_faults,
    }
    if summary["tasks"] != native_defined + native_faults:
        raise ValueError("Compatibility differential inventory is incomplete")
    return {
        **audited,
        "summary": summary,
    }


def _append(root: Path, event: dict) -> None:
    content = (json.dumps(event, sort_keys=True, allow_nan=False) + "\n").encode()
    with (root / f"tasks-{os.getpid()}.jsonl").open("ab") as stream:
        stream.write(content)
        stream.flush()


def _pristine_k_init(module):
    source = Path(os.environ["FIGURE8_TOKENIZER_NATIVE_KMEANS_MODULE"]).resolve()
    if source.is_symlink() or sha(source) != NATIVE_SHA256:
        raise ValueError("Pristine k-means++ source differs from the pinned native module")
    key = str(source)
    cached = _PRISTINE.get(key)
    if cached is not None:
        return cached
    content = source.read_bytes()
    node = next(
        candidate for candidate in ast.parse(content).body
        if isinstance(candidate, ast.FunctionDef) and candidate.name == "_k_init"
    )
    namespace = dict(module.__dict__)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    _PRISTINE[key] = namespace["_k_init"]
    return _PRISTINE[key]


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
        context["root"] / name,
        X=frame["X"],
        x_squared_norms=frame["x_squared_norms"],
        distances=distances,
        potential=potential,
        draws=draws,
        candidates=candidates,
        rng_before_keys=context["rng_before"][1],
        rng_after_draw_keys=rng[1],
    )
    correction.update(
        center=int(frame["c"]),
        evidence=name,
        evidence_sha256=sha(context["root"] / name),
        rng_after_draw_sha256=rng_sha(rng),
        rng_after_draw_position=int(rng[2]),
    )
    context["corrections"].append(correction)
    return result


def audit_k_init(native, X, n_clusters, x_squared_norms, random_state, n_local_trials=None):
    import k_means_constrained.k_means_constrained_ as estimator
    import k_means_constrained.sklearn_import.cluster.k_means_ as module

    root = Path(os.environ["FIGURE8_TOKENIZER_RECOVERY_EVENTS"])
    overlay = Path(os.environ["FIGURE8_TOKENIZER_RECOVERY_OVERLAY"]).resolve()
    module_path = Path(module.__file__).resolve()
    if (
        not root.is_dir()
        or module_path != overlay / "k_means_constrained" / MODULE
        or estimator._init_centroids.__globals__["_k_init"] is not module._k_init
        or native is not module._figure8_compat_k_init
    ):
        raise RuntimeError("Actual native initialization is not bound to the Figure 8 overlay")
    before = random_state.get_state()
    task_id = f"{os.getpid()}-{time.time_ns()}"
    context = {
        "root": root,
        "task_id": task_id,
        "rng_before": before,
        "corrections": [],
    }
    event = {
        "task_id": task_id,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "policy": POLICY,
        "input_sha256": array_sha(X.toarray() if hasattr(X, "toarray") else X),
        "shape": list(X.shape),
        "dtype": str(X.dtype),
        "n_clusters": n_clusters,
        "n_local_trials": n_local_trials,
        "rng_before_sha256": rng_sha(before),
        "rng_before_position": int(before[2]),
        "rng_before_keys_sha256": array_sha(before[1]),
        "initial_seed": int(before[1][0]) if before[2] == 624 else None,
        "module_origin": str(module_path),
        "module_sha256": sha(module_path),
        "function_code_sha256": hashlib.sha256(marshal.dumps(native.__code__)).hexdigest(),
        "pristine_module_sha256": NATIVE_SHA256,
        "actual_init_centroids_binding": True,
    }
    context_path = root / "task-context.json"
    if not context_path.is_file():
        raise RuntimeError("Figure 8 recovery initialization lacks its parent context")
    context_bytes = context_path.read_bytes()
    call = json.loads(context_bytes)
    if (
        call.get("expected_input_sha256") != event["input_sha256"]
        or event["initial_seed"] not in call.get("initialization_seeds", [])
    ):
        raise ValueError("Actual worker input/seed differs from its parent clustering call")
    event["context"] = call
    event["context_sha256"] = hashlib.sha256(context_bytes).hexdigest()
    token = _TASK.set(context)
    try:
        result, differential = execute_differential(
            native,
            _pristine_k_init(module),
            X,
            n_clusters,
            x_squared_norms,
            random_state,
            n_local_trials,
            context["corrections"],
        )
        event.update(status="success", centers_sha256=array_sha(result), differential=differential)
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
