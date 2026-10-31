"""Compatibility evidence is mandatory without rewriting native RQ contracts."""

import hashlib
import io
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import types
import zipfile

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "letter"))


def helper():
    assert importlib.util.find_spec("rq_recovery_lineage") is not None, (
        "Repaired native states require additive verified lineage"
    )
    from rq_recovery_lineage import verify_checkpoint_attestation
    return verify_checkpoint_attestation


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def write_event_archive(path, fault=None):
    from rq_recovery_compat import endpoint_candidates, POLICY
    from rq_recovery_runtime import array_sha

    weights = np.zeros((1, 256), dtype=np.float32)
    weights[0, 0] = 1
    weights[0, 255] = 3 * 2.0 ** -25
    potential = weights.sum()
    draws = np.array([0, 0.5, np.nextafter(float(potential), 0)])
    candidates = np.searchsorted(np.cumsum(weights, dtype=np.float64), draws)
    _, correction = endpoint_candidates(weights, potential, draws, candidates)
    before_keys = np.arange(624, dtype=np.uint32)
    after_keys = before_keys[::-1].copy()
    source_input = np.ones((2, 2))
    evidence_input = np.full((2, 2), 2.0) if fault == "wrong_input_evidence" else source_input
    evidence = io.BytesIO()
    np.savez(
        evidence, X=evidence_input, x_squared_norms=np.ones(2),
        distances=weights, potential=potential, draws=draws, candidates=candidates,
        rng_before_keys=before_keys, rng_after_draw_keys=after_keys,
    )
    evidence_bytes = evidence.getvalue()
    evidence_name = "correction-task-1-0.npz"
    context = {
        "call_id": "call-1", "epoch": 16401, "phase": "training", "level": 0,
        "expected_input_sha256": array_sha(source_input), "initialization_seeds": [17],
        "rng_before_sha256": "b" * 64,
    }
    context_bytes = (json.dumps(context, sort_keys=True) + "\n").encode()
    correction.update({
        "center": 6, "evidence": evidence_name,
        "evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
        "rng_after_draw_sha256": "c" * 64, "rng_after_draw_position": 7,
    })
    event = {
        "task_id": "task-1", "pid": 7, "ppid": 1, "policy": POLICY,
        "input_sha256": context["expected_input_sha256"], "shape": [2, 2],
        "dtype": "float32", "n_clusters": 10, "n_local_trials": 3,
        "rng_before_sha256": "d" * 64, "rng_before_position": 624,
        "rng_before_keys_sha256": array_sha(before_keys), "initial_seed": 17,
        "module_origin": "/private/overlay/k_means_.py", "module_sha256": "e" * 64,
        "function_code_sha256": "f" * 64, "actual_init_centroids_binding": True,
        "context": context, "context_sha256": hashlib.sha256(context_bytes).hexdigest(),
        "status": "success", "centers_sha256": "1" * 64,
        "rng_after_sha256": "2" * 64, "rng_after_keys_sha256": array_sha(after_keys),
        "rng_after_position": 8, "corrections": [correction],
    }
    if fault == "failed_task":
        event["status"] = "failure"
    elif fault == "wrong_context":
        event["context"]["expected_input_sha256"] = "9" * 64
    elif fault == "wrong_policy":
        event["corrections"][0]["policy"] = "unapproved"
    elif fault == "wrong_evidence":
        evidence_bytes += b"tampered"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("contexts/call-1.json", context_bytes)
        archive.writestr(evidence_name, evidence_bytes)
        archive.writestr("tasks-7.jsonl", json.dumps(event, sort_keys=True) + "\n")


def test_event_archive_is_independently_recomputed(tmp_path):
    from rq_recovery_lineage import audit_event_archive

    archive = tmp_path / "events.zip"
    write_event_archive(archive)
    result = audit_event_archive(archive)

    assert result["summary"] == {
        "tasks": 1, "failed_tasks": 0, "corrections": 1,
        "phases": {"training": 1}, "worker_pids": [7],
    }
    assert result["corrections_by_epoch_level_phase"] == {
        "16401": {"training": {"0": 1}},
    }


@pytest.mark.parametrize("fault", [
    "failed_task", "wrong_context", "wrong_policy", "wrong_evidence",
    "wrong_input_evidence",
])
def test_event_archive_rejects_failed_or_unbound_corrections(tmp_path, fault):
    from rq_recovery_lineage import audit_event_archive

    archive = tmp_path / "events.zip"
    write_event_archive(archive, fault)
    with pytest.raises(ValueError):
        audit_event_archive(archive)


@pytest.fixture
def checkpoint(tmp_path):
    from rq_recovery_control import HARNESS, ORIGINAL_SEAL, VARIANTS
    from rq_recovery_compat import POLICY
    from rq_recovery_runtime import file_manifest, sha
    directory = tmp_path / "epoch-16402"
    directory.mkdir()
    (directory / "state.pt").write_bytes(b"unit boundary fixture, not a native torch state")
    write_json(directory / "completed.json", {
        "status": "complete", "kind": "tokenizer_epoch_checkpoint", "epoch": 16401,
        "state_sha256": sha(directory / "state.pt"),
    })
    target = directory / "compatibility"
    target.mkdir()
    for name in ("source.zip", "gates.zip"):
        (target / name).write_bytes(b"unit artifact " + name.encode())
    write_event_archive(target / "events.zip")
    admission = {
        "status": "admitted", "policy": POLICY, "variants": VARIANTS,
        "original_seal_sha256": ORIGINAL_SEAL,
        "artifacts": {name: sha(target / name) for name in ("source.zip", "gates.zip")},
        "native_k_init_sha256": "1" * 64, "executed_k_init_sha256": "2" * 64,
        "banked_evidence": {"unit_fixture": True},
    }
    write_json(target / "admission.json", admission)
    trusted = sha(target / "admission.json")
    attestation = {
        "kind": "rq_compatibility_attestation", "purpose": "production", "artifact": "checkpoint",
        "policy": POLICY, "run_id": "openllama-v2-rq-Instruments-semantic-s42",
        "run_config_sha256": VARIANTS["semantic"]["config"], "base_epoch": 16400,
        "base_state_sha256": VARIANTS["semantic"]["base_state"], "original_seal_sha256": ORIGINAL_SEAL,
        "native_harness_sha256": HARNESS, "admission_sha256": trusted, "epoch": 16402,
        "state_sha256": sha(directory / "state.pt"), "selected_checkpoint": "checkpoints/epoch-08000",
        "runtime_summary": {
            "tasks": 1, "failed_tasks": 0, "corrections": 1,
            "phases": {"training": 1}, "worker_pids": [7],
        },
        "files": file_manifest(target),
    }
    write_json(target / "attestation.json", attestation)
    return directory, trusted, attestation


def test_bound_checkpoint_provenance_is_accepted(checkpoint):
    path, trusted, expected = checkpoint
    assert helper()(path, expected["run_id"], expected["run_config_sha256"], trusted) == expected


@pytest.mark.parametrize("fault", [
    "missing_attestation", "changed_state", "changed_source", "untracked_file",
    "wrong_admission", "old_purpose", "wrong_base", "wrong_epoch", "foreign_run",
    "wrong_runtime_summary",
])
def test_missing_or_unbound_provenance_fails_closed(checkpoint, fault):
    path, trusted, expected = checkpoint
    metadata = path / "compatibility" / "attestation.json"
    if fault == "missing_attestation":
        metadata.unlink()
    elif fault == "changed_state":
        (path / "state.pt").write_bytes(b"changed")
    elif fault == "changed_source":
        (path / "compatibility" / "source.zip").write_bytes(b"changed")
    elif fault == "untracked_file":
        (path / "compatibility" / "untracked.py").write_bytes(b"unexpected")
    elif fault == "wrong_admission":
        trusted = hashlib.sha256(b"another admission").hexdigest()
    else:
        changed = dict(expected)
        if fault == "old_purpose":
            changed["purpose"] = "probe"
        elif fault == "wrong_base":
            changed["base_state_sha256"] = "0" * 64
        elif fault == "wrong_epoch":
            changed["epoch"] = 16403
        elif fault == "foreign_run":
            changed["run_id"] = "openllama-v2-rq-Beauty-semantic-s42"
        elif fault == "wrong_runtime_summary":
            changed["runtime_summary"] = {
                "tasks": 1, "failed_tasks": 0, "corrections": 0,
                "phases": {}, "worker_pids": [7],
            }
        write_json(metadata, changed)
    verifier = helper()
    with pytest.raises((ValueError, FileNotFoundError)):
        verifier(path, expected["run_id"], expected["run_config_sha256"], trusted)


@pytest.fixture
def final_bundle(checkpoint, monkeypatch, tmp_path):
    from rq_recovery_runtime import file_manifest, sha
    from rq_recovery_compat import POLICY
    old, trusted, attestation = checkpoint
    bundle = tmp_path / "synthetic-boundary-bundle"
    latest = bundle / "payload" / "checkpoints" / "epoch-20000"
    latest.parent.mkdir(parents=True)
    old.rename(latest)
    completed = json.loads((latest / "completed.json").read_text())
    completed["epoch"] = 19999
    write_json(latest / "completed.json", completed)
    attestation["epoch"] = 20000
    write_json(latest / "compatibility" / "attestation.json", attestation)
    export = bundle / "payload" / "export"
    export.mkdir()
    (export / "export.json").write_bytes(b"synthetic export boundary only")
    shutil.copytree(latest / "compatibility", export / "compatibility")
    metadata = export / "compatibility" / "attestation.json"
    metadata.unlink()
    exported = {
        **attestation, "artifact": "export", "selected_epoch": 8000, "index_sha256": "3" * 64,
        "export_sha256": sha(export / "export.json"),
        "final_checkpoint_attestation_sha256": sha(latest / "compatibility" / "attestation.json"),
        "files": file_manifest(export / "compatibility"),
    }
    write_json(metadata, exported)
    write_json(bundle / "payload" / "completed.json", {"rq_compatibility": {
        "policy": POLICY, "admission_sha256": trusted, "base_state_sha256": attestation["base_state_sha256"],
        "export_attestation_sha256": sha(metadata),
    }})
    lineage = {
        "rq_run_id": attestation["run_id"], "run_config_sha256": attestation["run_config_sha256"],
        "selected_epoch": 8000, "selected_checkpoint": "checkpoints/epoch-08000",
        "index_sha256": "3" * 64, "codes_sha256": "4" * 64,
        "export_sha256": sha(export / "export.json"),
    }
    calls = []
    def native(*args):
        calls.append(args)
        return lineage
    monkeypatch.setitem(sys.modules, "fresh_ids", types.SimpleNamespace(verify_rq=native))
    return bundle, trusted, calls


def test_native_gate_composed_and_old_selected_checkpoint_not_rewritten(final_bundle):
    from rq_recovery_lineage import verify_rq_with_compatibility
    bundle, trusted, calls = final_bundle
    result = verify_rq_with_compatibility(bundle, {}, {}, "Instruments", "semantic",
                                        admission_sha256=trusted)
    assert len(calls) == 1
    assert result["rq_compatibility"]["selected_model_predates_correction"] is True
    assert result["rq_compatibility"]["export_corrections"] == 0
    assert result["rq_compatibility"]["export_was_corrected"] is False
    assert result["rq_compatibility"]["native_artifact_identity_claimed"] is False
    assert result["rq_compatibility"]["downstream_ids_repair_dependent"] is False
    assert result["rq_compatibility"]["corrected_index_sha256"] == "3" * 64
    assert result["rq_compatibility"]["corrected_codes_sha256"] == "4" * 64
    assert result["rq_compatibility"]["corrections_by_epoch_level_phase"] == {
        "16401": {"training": {"0": 1}},
    }
    assert not (bundle / "payload" / "checkpoints" / "epoch-08000").exists()


@pytest.mark.parametrize("fault", ["omitted_admission", "missing_final_metadata", "stale_final_binding"])
def test_export_cannot_lose_or_rebind_compatibility(final_bundle, fault):
    from rq_recovery_lineage import verify_rq_with_compatibility
    bundle, trusted, _ = final_bundle
    if fault == "omitted_admission":
        trusted = None
    elif fault == "missing_final_metadata":
        write_json(bundle / "payload" / "completed.json", {})
    else:
        path = bundle / "payload" / "export" / "compatibility" / "attestation.json"
        metadata = json.loads(path.read_text())
        metadata["final_checkpoint_attestation_sha256"] = "0" * 64
        write_json(path, metadata)
    with pytest.raises(ValueError):
        verify_rq_with_compatibility(bundle, {}, {}, "Instruments", "semantic", admission_sha256=trusted)


def test_unrelated_native_lineage_is_unchanged(monkeypatch):
    from rq_recovery_lineage import verify_rq_with_compatibility
    lineage = {"rq_run_id": "openllama-v2-rq-Beauty-full-s42"}
    monkeypatch.setitem(sys.modules, "fresh_ids", types.SimpleNamespace(verify_rq=lambda *args: lineage))
    assert verify_rq_with_compatibility(Path("unused"), {}, {}, "Beauty", "full") is lineage
