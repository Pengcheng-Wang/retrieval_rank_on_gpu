"""Operational admission tests; fixtures are not trained RQ/T5 artifacts."""

import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "letter"
sys.path.insert(0, str(SCRIPTS))
import fresh_ids
import fresh_pipeline
from bank import copy_artifact, source_manifest
from protocol import file_sha256, write_json


def module():
    assert (SCRIPTS / "fresh_rq_admission.py").is_file(), "Mandatory repaired-RQ queue guard is missing"
    return importlib.import_module("fresh_rq_admission")


@pytest.fixture
def repaired(tmp_path, monkeypatch):
    from rq_recovery_control import VARIANTS, HARNESS, ORIGINAL_SEAL
    from rq_recovery_compat import POLICY
    from rq_recovery_runtime import file_manifest
    importlib.import_module("rq_recovery_lineage")

    root = tmp_path / "owned"
    helper = root / "helpers"
    helper.mkdir(parents=True)
    names = ("rq_recovery_compat.py", "rq_recovery_runtime.py",
             "rq_recovery_control.py", "rq_recovery_lineage.py")
    for name in names:
        shutil.copyfile(SCRIPTS / name, helper / name)
        monkeypatch.delitem(sys.modules, Path(name).stem, raising=False)
    capsule = root / "capsule"
    capsule.mkdir()
    for name in ("source.zip", "gates.zip", "preparation.json"):
        (capsule / name).write_bytes(("unit artifact, not package/data: " + name).encode())
    admission = {
        "status": "admitted", "policy": POLICY, "variants": VARIANTS,
        "original_seal_sha256": ORIGINAL_SEAL, "native_k_init_sha256": "1" * 64,
        "executed_k_init_sha256": "2" * 64,
        "artifacts": {name: file_sha256(capsule / name) for name in ("source.zip", "gates.zip")},
        "prepared_sha256": file_sha256(capsule / "preparation.json"),
        "banked_evidence": {"unit_fixture_only": True},
    }
    write_json(capsule / "admission.json", admission)
    trusted = file_sha256(capsule / "admission.json")
    destination = tmp_path / "durable"
    durable = destination / "rq-recovery-v1" / "capsule"
    copy_artifact(capsule, durable)
    run_id = "openllama-v2-rq-Instruments-semantic-s42"
    bundle = root / "bundle"
    latest = bundle / "payload" / "checkpoints" / "epoch-20000"
    latest.mkdir(parents=True)
    (latest / "state.pt").write_bytes(b"unit native gate stub; NOT a real checkpoint")
    write_json(latest / "completed.json", {
        "status": "complete", "kind": "tokenizer_epoch_checkpoint", "epoch": 19999,
        "state_sha256": file_sha256(latest / "state.pt"),
    })
    compatibility = latest / "compatibility"
    shutil.copytree(capsule, compatibility)
    (compatibility / "preparation.json").unlink()
    (compatibility / "events.zip").write_bytes(b"unit worker events")
    base = {
        "kind": "rq_compatibility_attestation", "purpose": "production",
        "artifact": "checkpoint", "policy": POLICY, "run_id": run_id,
        "run_config_sha256": VARIANTS["semantic"]["config"],
        "base_epoch": 16400, "base_state_sha256": VARIANTS["semantic"]["base_state"],
        "original_seal_sha256": ORIGINAL_SEAL, "native_harness_sha256": HARNESS,
        "admission_sha256": trusted, "epoch": 20000,
        "state_sha256": file_sha256(latest / "state.pt"),
        "selected_checkpoint": "checkpoints/epoch-08000",
        "runtime_summary": {"corrections": 1, "phases": {"training": 1}},
        "files": file_manifest(compatibility),
    }
    write_json(compatibility / "attestation.json", base)
    selected = bundle / "payload" / "checkpoints" / "epoch-08000"
    selected.mkdir()
    (selected / "state.pt").write_bytes(b"untouched pre-correction selected-state fixture")
    export = bundle / "payload" / "export"
    export.mkdir()
    (export / "export.json").write_bytes(b"unit export receipt")
    shutil.copytree(compatibility, export / "compatibility")
    marker = export / "compatibility" / "attestation.json"
    marker.unlink()
    exported = {
        **base, "artifact": "export", "selected_epoch": 8000, "index_sha256": "3" * 64,
        "export_sha256": file_sha256(export / "export.json"),
        "final_checkpoint_attestation_sha256": file_sha256(compatibility / "attestation.json"),
        "files": file_manifest(export / "compatibility"),
    }
    write_json(marker, exported)
    write_json(bundle / "payload" / "completed.json", {"rq_compatibility": {
        "policy": POLICY, "admission_sha256": trusted,
        "base_state_sha256": base["base_state_sha256"],
        "export_attestation_sha256": file_sha256(marker),
    }})
    lineage = {
        "rq_run_id": run_id, "run_config_sha256": base["run_config_sha256"],
        "selected_epoch": 8000, "selected_checkpoint": "checkpoints/epoch-08000",
        "selected_state_sha256": file_sha256(selected / "state.pt"),
        "index_sha256": "3" * 64, "export_sha256": file_sha256(export / "export.json"),
    }
    calls = []

    def native(*args):
        calls.append("native")
        return lineage

    monkeypatch.setattr(fresh_ids, "verify_rq", native)
    settings = {
        "scope": ["openllama-v2-rq-Instruments-semantic-s42", "openllama-v2-rq-Instruments-cf-s42"],
        "helper_directory_on_node0": str(helper), "helper": "rq_recovery_lineage.verify_rq_with_compatibility",
        "helper_sha256": file_sha256(helper / "rq_recovery_lineage.py"),
        "helper_files": {name: file_sha256(helper / name) for name in names},
        "trusted_admission_sha256": trusted, "capsule": str(capsule), "durable_capsule": str(durable),
    }
    integration = root / "integration.json"
    write_json(integration, settings)
    return SimpleNamespace(
        root=root, destination=destination, helper=helper, capsule=capsule, durable=durable,
        bundle=bundle, integration=integration, settings=settings, calls=calls,
        native=native, lineage=lineage, selected=selected, latest=latest, export=export,
    )


def make_guard(case, monkeypatch):
    m = module()
    monkeypatch.setattr(m, "bank_lock", lambda *args: nullcontext())
    return m.RepairAdmission(case.integration, case.root, case.destination)


def test_native_first_verified_capsule_and_old_selected_state_are_preserved(repaired, monkeypatch):
    guard = make_guard(repaired, monkeypatch)
    before = source_manifest(repaired.bundle)
    paths, environment = list(sys.path), dict(os.environ)
    result = guard(repaired.bundle, {}, {}, "Instruments", "semantic")
    assert repaired.calls == ["native"]
    assert fresh_ids.verify_rq is repaired.native
    assert result["rq_compatibility"]["selected_model_predates_correction"] is True
    assert result["rq_compatibility"]["queue_admission"]["capsule_bank_receipt_sha256"]
    assert source_manifest(repaired.bundle) == before
    assert not (repaired.selected / "compatibility").exists()
    assert sys.path == paths and dict(os.environ) == environment


@pytest.mark.parametrize("fault", [
    "wrong_admission", "missing_admission", "missing_capsule", "missing_capsule_bank",
    "wrong_capsule_bank", "changed_overlay_archive", "changed_banked_file",
    "missing_checkpoint_attestation", "missing_export_attestation", "changed_export_binding",
    "changed_helper",
])
def test_repaired_completion_never_falls_back_to_native_only(repaired, monkeypatch, fault):
    if fault == "wrong_admission":
        repaired.settings["trusted_admission_sha256"] = "e" * 64
        write_json(repaired.integration, repaired.settings)
    elif fault == "missing_admission":
        (repaired.export / "compatibility" / "admission.json").unlink()
    elif fault == "missing_capsule":
        (repaired.capsule / "admission.json").unlink()
    elif fault == "missing_capsule_bank":
        (repaired.durable / "bank-receipt.json").unlink()
    elif fault == "wrong_capsule_bank":
        write_json(repaired.durable / "bank-receipt.json", {"files": {}})
    elif fault == "changed_overlay_archive":
        (repaired.export / "compatibility" / "source.zip").write_bytes(b"wrong overlay")
    elif fault == "changed_banked_file":
        (repaired.durable / "source.zip").write_bytes(b"wrong durable overlay")
    elif fault == "missing_checkpoint_attestation":
        (repaired.latest / "compatibility" / "attestation.json").unlink()
    elif fault == "missing_export_attestation":
        (repaired.export / "compatibility" / "attestation.json").unlink()
    elif fault == "changed_export_binding":
        write_json(repaired.bundle / "payload" / "completed.json", {"rq_compatibility": {}})
    else:
        path = repaired.helper / "rq_recovery_runtime.py"
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises((ValueError, FileNotFoundError)):
        guard = make_guard(repaired, monkeypatch)
        guard(repaired.bundle, {}, {}, "Instruments", "semantic")
    if fault != "changed_helper":
        assert repaired.calls == ["native"]


def test_native_partial_rejection_precedes_all_compatibility_reads(repaired, monkeypatch):
    def reject(*args):
        repaired.calls.append("native partial rejected")
        raise ValueError("unfinished native 20000 epoch gate")
    monkeypatch.setattr(fresh_ids, "verify_rq", reject)
    guard = make_guard(repaired, monkeypatch)
    (repaired.capsule / "admission.json").unlink()
    with pytest.raises(ValueError, match="unfinished native"):
        guard(repaired.bundle, {}, {}, "Instruments", "semantic")
    assert repaired.calls == ["native partial rejected"]


def test_unrelated_native_result_is_unchanged_without_capsule(repaired, monkeypatch):
    repaired.lineage["rq_run_id"] = "openllama-v2-rq-Beauty-full-s42"
    guard = make_guard(repaired, monkeypatch)
    shutil.rmtree(repaired.capsule)
    result = guard(repaired.bundle, {}, {}, "Beauty", "full")
    assert result is repaired.lineage and repaired.calls == ["native"]
    assert "rq_compatibility" not in result


def test_alias_rebinding_is_rejected_instead_of_recursing(repaired, monkeypatch):
    guard = make_guard(repaired, monkeypatch)
    monkeypatch.setattr(fresh_ids, "verify_rq", guard)
    with pytest.raises(ValueError, match="native verifier"):
        guard(repaired.bundle, {}, {}, "Instruments", "semantic")


def test_integration_contract_cannot_change_after_guard_initialization(repaired, monkeypatch):
    guard = make_guard(repaired, monkeypatch)
    repaired.integration.write_bytes(repaired.integration.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="checksum"):
        guard(repaired.bundle, {}, {}, "Instruments", "semantic")


def test_existing_repaired_stage_is_never_rewritten_to_add_lineage(repaired, monkeypatch):
    m = module()
    runtime = object.__new__(m.CompatibilityRuntime)
    runtime.work, runtime.spec = repaired.root / "work", {}
    runtime.manifest = repaired.root / "manifest.json"
    write_json(runtime.manifest, {})
    runtime.rq_verifier = make_guard(repaired, monkeypatch)
    cell = {"rq_run_id": repaired.lineage["rq_run_id"], "dataset": "Instruments",
            "variant": "semantic", "stage_id": "semantic-ids"}
    stage = runtime.work / "stages" / cell["stage_id"]
    write_json(stage / "manifest.json", {"immutable": "preexisting contract"})
    before = source_manifest(stage)
    monkeypatch.setattr(fresh_ids, "verify_stage",
                        lambda _: {"experiment_lineage": {"rq": repaired.lineage}})
    with pytest.raises(ValueError, match="existing repaired-ID stage"):
        runtime.admit(cell, repaired.bundle)
    assert source_manifest(stage) == before


def test_controller_is_cpu_only_but_children_keep_the_assigned_gpu_lease(tmp_path, monkeypatch):
    m = module()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    with pytest.raises(RuntimeError, match="CPU-only"):
        m.CompatibilityRuntime(*([tmp_path] * 5))
    runtime = object.__new__(m.CompatibilityRuntime)
    runtime.env = tmp_path / "env.sh"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    command, env = runtime.command(["python", "native-run.py"], seed=17)
    assert env["CUDA_VISIBLE_DEVICES"] == "2,3" and env["LETTER_SEED"] == "17"
    assert "rq_recovery" not in repr(command)


def test_compatibility_metadata_is_composed_before_manifest_construction():
    assert hasattr(fresh_ids, "attach_rq_compatibility"), "Stage lineage needs the admitted correction"
    correction = {"policy": "fixture-policy", "source_archive_sha256": "f" * 64,
                  "queue_admission": {"capsule_bank_receipt_sha256": "e" * 64}}
    native = {"rq_run_id": "fixture-run", "rq_compatibility": correction}
    lineage = {"rq": copy.deepcopy(native), "kind": "fixture"}
    fresh_ids.attach_rq_compatibility(lineage, native)
    assert lineage["rq_compatibility"] == correction == lineage["rq"]["rq_compatibility"]
    encoded = json.loads(json.dumps({"experiment_lineage": lineage}))
    source_run = {"experiment_lineage": encoded["experiment_lineage"]}
    assert {"source_run": source_run}["source_run"]["experiment_lineage"]["rq_compatibility"] == correction
    assert native["rq_compatibility"] == correction


def test_stage_hook_propagates_the_entire_verified_correction_before_build(tmp_path, monkeypatch):
    root, destination = tmp_path / "owned", tmp_path / "durable"
    root.mkdir()
    bundle = root / "bundle"
    args = SimpleNamespace(
        binding=root / "private.json", bundle=bundle, output=root / "new-stage",
        durable_bundle=destination / "openllama-v2/tokenizers/bundle", dataset="Instruments",
        variant="semantic", manifest=root / "manifest.json", execution=root / "execution.json",
        method_spec=root / "method.json", tokenizer_recipe=root / "recipe.json",
        upstream=root / "upstream", project=root / "control", semantic=root / "semantic",
        teacher=root / "teacher",
    )
    write_json(args.manifest, {})
    write_json(args.execution, {"unit_fixture_only": True})
    spec = {"claim_type": "fixture", "semantic_encoder": {}, "generator": {},
            "method_spec_canonical_sha256": "1" * 64, "tokenizer_recipe_canonical_sha256": "2" * 64}
    correction = {"policy": "fixture", "selected_model_predates_correction": True,
                  "continuation_runtime": {"phases": {"training": 1}},
                  "queue_admission": {"capsule_bank_receipt_sha256": "3" * 64}}
    rq = {"index_sha256": "4" * 64, "codes_sha256": "5" * 64, "rq_compatibility": correction}
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(LOCK_EX=1, flock=lambda *a: None))
    monkeypatch.setattr(fresh_ids, "validate_scope", lambda _: (root, destination))
    monkeypatch.setattr(fresh_ids, "load_execution", lambda *a: spec)
    monkeypatch.setattr(fresh_ids, "verify_original", lambda *a: None)
    monkeypatch.setattr(fresh_ids, "verify_inputs", lambda *a: {"fixture": True})
    monkeypatch.setattr(fresh_ids, "verify_durable", lambda *a: {"fixture": True})
    original = fresh_ids.verify_rq
    calls = []

    def verified(*args):
        calls.append("mandatory compatibility verifier")
        return rq

    def build(*values):
        calls.append("build immutable new stage")
        lineage = values[-1]
        assert lineage["rq_compatibility"] == lineage["rq"]["rq_compatibility"] == correction
        return {"experiment_lineage": json.loads(json.dumps(lineage))}

    monkeypatch.setattr(fresh_ids, "build_stage", build)
    result = fresh_ids.stage_real(args, rq_verifier=verified)
    assert calls == ["mandatory compatibility verifier", "build immutable new stage"]
    assert result["experiment_lineage"]["rq_compatibility"] == correction
    assert fresh_ids.verify_rq is original


def adoption_fixture(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    old = {
        "execution_sha256": "1" * 64, "job_binding_sha256": "2" * 64,
        "adapter_sha256": "3" * 64, "driver_sha256": "4" * 64, "control": "old-control",
        "root": "owned", "work": str(work), "durable_root": "durable", "cuda_visible_devices": [2, 3],
        "queue": [{"run_id": "fixture", "seed": 42, "temperature": 1.0}],
    }
    write_json(work / "queue-binding.json", old)
    existing = work / "stages" / "old" / "manifest.json"
    write_json(existing, {"immutable": "existing stage and native contract"})
    snapshot = tmp_path / "protected.json"
    write_json(snapshot, {
        "queue-binding.json": {
            "sha256": file_sha256(work / "queue-binding.json"),
            "bytes": (work / "queue-binding.json").stat().st_size,
        },
        "stages/old/manifest.json": {
            "sha256": file_sha256(existing),
            "bytes": existing.stat().st_size,
        },
    })
    new = copy.deepcopy(old)
    new.update(adapter_sha256="5" * 64, driver_sha256="6" * 64, control="new-control",
               operational_admission={"integration_sha256": "7" * 64})
    adoption = {
        "kind": "fresh_queue_operational_adoption", "work": str(work),
        "previous_binding_file": "queue-binding.json",
        "previous_binding_sha256": file_sha256(work / "queue-binding.json"),
        "new_binding_file": "queue-binding-rq-compat-v1.json", "new_binding": new,
        "protected_files": str(snapshot), "protected_files_sha256": file_sha256(snapshot),
    }
    path = tmp_path / "adoption.json"
    write_json(path, adoption)
    return work, old, new, path


def test_versioned_queue_adoption_preserves_old_binding_work_and_stages(tmp_path):
    assert hasattr(fresh_pipeline, "bind_queue"), "Same-work operational adoption must be explicit"
    work, old, new, adoption = adoption_fixture(tmp_path)
    before = source_manifest(work)
    marker = fresh_pipeline.bind_queue(work, new, adoption)
    assert marker == work / "queue-binding-rq-compat-v1.json"
    assert json.loads((work / "queue-binding.json").read_text()) == old
    assert json.loads(marker.read_text()) == new
    assert fresh_pipeline.bind_queue(work, new, adoption) == marker
    after = source_manifest(work)["files"]
    for name, metadata in before["files"].items():
        assert after[name] == metadata


@pytest.mark.parametrize("fault", [
    "recipe", "queue", "work", "old_digest", "stage_changed", "overwrite_old",
    "unlisted_file",
])
def test_adoption_cannot_change_science_or_rewrite_frozen_progress(tmp_path, fault):
    assert hasattr(fresh_pipeline, "bind_queue"), "Operational adoption contract is missing"
    work, old, new, path = adoption_fixture(tmp_path)
    adoption = json.loads(path.read_text())
    if fault == "recipe":
        new["execution_sha256"] = "8" * 64
    elif fault == "queue":
        new["queue"][0]["seed"] = 17
    elif fault == "work":
        new["work"] = "another-work"
    elif fault == "old_digest":
        adoption["previous_binding_sha256"] = "0" * 64
    elif fault == "stage_changed":
        write_json(work / "stages/old/manifest.json", {"changed": True})
    elif fault == "unlisted_file":
        write_json(work / "unlisted.json", {"not": "in the protected manifest"})
    else:
        adoption["new_binding_file"] = "queue-binding.json"
    adoption["new_binding"] = new
    write_json(path, adoption)
    with pytest.raises(ValueError):
        fresh_pipeline.bind_queue(work, new, path)
    assert json.loads((work / "queue-binding.json").read_text()) == old
