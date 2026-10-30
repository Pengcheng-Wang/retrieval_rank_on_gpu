"""Compatibility evidence is mandatory without rewriting native RQ contracts."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import types

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
    for name in ("source.zip", "gates.zip", "events.zip"):
        (target / name).write_bytes(b"unit artifact " + name.encode())
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
        "runtime_summary": {"corrections": 1, "phases": {"training": 1}},
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
        "index_sha256": "3" * 64, "export_sha256": sha(export / "export.json"),
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
