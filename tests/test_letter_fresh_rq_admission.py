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
        "invocation": "fixture-invocation",
        "resumed_from_epoch": 16400,
        "resumed_from_state_sha256": VARIANTS["semantic"]["base_state"],
        "runtime_summary": {
            "tasks": 1, "failed_tasks": 0, "corrections": 1,
            "phases": {"training": 1}, "worker_pids": [7],
        },
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
    lineage["codes_sha256"] = "4" * 64
    chain = root / "resume-chains" / run_id
    receipt = chain / "receipts" / "fixture-invocation.json"
    write_json(receipt, {
        "status": "exact", "epoch": 16400,
        "state_sha256": VARIANTS["semantic"]["base_state"],
        "model_optimizer_python_numpy_torch_cuda_rng": "all_exact",
    })
    chain_manifest = chain / "manifest.json"
    write_json(chain_manifest, {
        "kind": "rq_recovery_resume_chain", "run_id": run_id,
        "base_epoch": 16400, "base_state_sha256": VARIANTS["semantic"]["base_state"],
        "final_epoch": 20000,
        "checkpoints": {
            "20000": {
                "attestation_sha256": file_sha256(compatibility / "attestation.json"),
                "state_sha256": file_sha256(latest / "state.pt"),
                "invocation": "fixture-invocation", "resumed_from_epoch": 16400,
                "resumed_from_state_sha256": VARIANTS["semantic"]["base_state"],
            },
        },
        "invocations": [{
            "invocation": "fixture-invocation", "resumed_from_epoch": 16400,
            "resumed_from_state_sha256": VARIANTS["semantic"]["base_state"],
            "restore_receipt": "receipts/fixture-invocation.json",
            "restore_receipt_sha256": file_sha256(receipt),
            "produced_epochs": [20000],
        }],
    })
    durable_chain = destination / "rq-recovery-v1" / "resume-chains" / run_id
    copy_artifact(chain, durable_chain)
    cf_run_id = "openllama-v2-rq-Instruments-cf-s42"
    cf_chain = root / "resume-chains" / cf_run_id
    cf_receipt = cf_chain / "receipts" / "fixture-invocation.json"
    write_json(cf_receipt, {
        "status": "exact", "epoch": 16400,
        "state_sha256": VARIANTS["cf"]["base_state"],
        "model_optimizer_python_numpy_torch_cuda_rng": "all_exact",
    })
    cf_manifest = cf_chain / "manifest.json"
    write_json(cf_manifest, {
        "kind": "rq_recovery_resume_chain", "run_id": cf_run_id,
        "base_epoch": 16400, "base_state_sha256": VARIANTS["cf"]["base_state"],
        "final_epoch": 20000, "checkpoints": {}, "invocations": [],
    })
    cf_durable_chain = destination / "rq-recovery-v1" / "resume-chains" / cf_run_id
    copy_artifact(cf_chain, cf_durable_chain)
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
        "resume_chains": {
            run_id: {
                "source": str(chain), "manifest_sha256": file_sha256(chain_manifest),
                "durable": str(durable_chain),
            },
            cf_run_id: {
                "source": str(cf_chain), "manifest_sha256": file_sha256(cf_manifest),
                "durable": str(cf_durable_chain),
            },
        },
    }
    integration = root / "integration.json"
    write_json(integration, settings)
    return SimpleNamespace(
        root=root, destination=destination, helper=helper, capsule=capsule, durable=durable,
        bundle=bundle, integration=integration, settings=settings, calls=calls,
        native=native, lineage=lineage, selected=selected, latest=latest, export=export,
        chain=chain, chain_manifest=chain_manifest, durable_chain=durable_chain,
    )


def make_guard(case, monkeypatch):
    m = module()
    monkeypatch.setattr(m, "bank_lock", lambda *args: nullcontext())
    guard = m.RepairAdmission(
        case.integration, case.root, case.destination, file_sha256(case.integration),
    )
    monkeypatch.setattr(
        guard.helper, "audit_event_archive",
        lambda _: {
            "summary": {
                "tasks": 1, "failed_tasks": 0, "corrections": 1,
                "phases": {"training": 1}, "worker_pids": [7],
            },
            "corrections_by_epoch_level_phase": {"16401": {"training": {"0": 1}}},
        },
    )
    return guard


def test_external_integration_digest_is_mandatory(repaired, monkeypatch):
    m = module()
    monkeypatch.setattr(m, "bank_lock", lambda *args: nullcontext())
    with pytest.raises(ValueError, match="externally authorized"):
        m.RepairAdmission(repaired.integration, repaired.root, repaired.destination, "0" * 64)


def test_compatibility_path_components_may_not_redirect(repaired, monkeypatch):
    m = module()
    original = Path.is_symlink
    monkeypatch.setattr(
        Path, "is_symlink",
        lambda path: path == repaired.helper or original(path),
    )
    with pytest.raises(ValueError, match="redirect"):
        m.RepairAdmission(
            repaired.integration, repaired.root, repaired.destination,
            file_sha256(repaired.integration),
        )


def test_native_first_verified_capsule_and_old_selected_state_are_preserved(repaired, monkeypatch):
    guard = make_guard(repaired, monkeypatch)
    original = guard.helper.verify_rq_with_compatibility
    observed = []

    def compatibility(*args, **kwargs):
        native_lineage = kwargs["native_lineage"]
        observed.append(native_lineage)
        assert native_lineage == repaired.lineage
        assert native_lineage is not repaired.lineage
        return original(*args, **kwargs)

    monkeypatch.setattr(guard.helper, "verify_rq_with_compatibility", compatibility)
    before = source_manifest(repaired.bundle)
    paths, environment = list(sys.path), dict(os.environ)
    result = guard(repaired.bundle, {}, {}, "Instruments", "semantic")
    assert repaired.calls == ["native"]
    assert len(observed) == 1
    assert fresh_ids.verify_rq is repaired.native
    assert result["rq_compatibility"]["selected_model_predates_correction"] is True
    assert result["rq_compatibility"]["queue_admission"]["capsule_bank_receipt_sha256"]
    resume = result["rq_compatibility"]["queue_admission"]["resume_chain"]
    assert resume["status"] == "exact_full_state_chain"
    assert resume["final_epoch"] == 20000
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
        (repaired.chain / "manifest.json").unlink()
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
    monkeypatch.setattr(
        guard.helper, "verify_rq_with_compatibility",
        lambda *args, **kwargs: pytest.fail("compatibility evidence was read before native rejection"),
    )
    (repaired.capsule / "admission.json").unlink()
    with pytest.raises(ValueError, match="unfinished native"):
        guard(repaired.bundle, {}, {}, "Instruments", "semantic")
    assert repaired.calls == ["native partial rejected"]


@pytest.mark.parametrize("fault", [
    "missing_receipt", "model_only_receipt", "wrong_resume_state",
    "missing_checkpoint", "changed_durable_chain",
])
def test_resume_chain_is_complete_exact_and_durable(repaired, monkeypatch, fault):
    manifest = json.loads(repaired.chain_manifest.read_text())
    if fault == "missing_receipt":
        (repaired.chain / manifest["invocations"][0]["restore_receipt"]).unlink()
    elif fault == "model_only_receipt":
        receipt = repaired.chain / manifest["invocations"][0]["restore_receipt"]
        value = json.loads(receipt.read_text())
        value["model_optimizer_python_numpy_torch_cuda_rng"] = "model_only"
        write_json(receipt, value)
    elif fault == "wrong_resume_state":
        manifest["invocations"][0]["resumed_from_state_sha256"] = "0" * 64
        write_json(repaired.chain_manifest, manifest)
    elif fault == "missing_checkpoint":
        manifest["checkpoints"] = {}
        write_json(repaired.chain_manifest, manifest)
    else:
        (repaired.durable_chain / "manifest.json").write_bytes(b"changed")
    with pytest.raises((ValueError, FileNotFoundError)):
        make_guard(repaired, monkeypatch)(
            repaired.bundle, {}, {}, "Instruments", "semantic",
        )


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


def test_existing_repaired_stage_accepts_only_its_immutable_legacy_evidence(repaired, monkeypatch):
    m = module()
    runtime = object.__new__(m.CompatibilityRuntime)
    runtime.work, runtime.spec = repaired.root / "work", {}
    runtime.manifest = repaired.root / "manifest.json"
    write_json(runtime.manifest, {})
    cell = {
        "rq_run_id": repaired.lineage["rq_run_id"], "dataset": "Instruments",
        "variant": "semantic", "stage_id": "semantic-ids",
    }
    stage = runtime.work / "stages" / cell["stage_id"]
    write_json(stage / "manifest.json", {"immutable": "legacy repaired stage"})
    new = {
        "policy": "rq-kmeans-cdf-endpoint-v1",
        "classification": "compatibility correction at undefined native indexing",
        "not_claimed": "bitwise identity with a native execution past its crash",
        "admission_sha256": "1" * 64, "base_epoch": 16400,
        "base_state_sha256": "2" * 64, "native_k_init_sha256": "3" * 64,
        "executed_k_init_sha256": "4" * 64, "final_attestation_sha256": "5" * 64,
        "export_attestation_sha256": "6" * 64, "source_archive_sha256": "7" * 64,
        "gates_archive_sha256": "8" * 64, "banked_evidence": {"exact": True},
        "selected_model_predates_correction": False,
        "compatibility_enabled_for_continuation_and_export": True,
        "export_corrections": 0, "continuation_runtime": {"corrections": 1},
        "cumulative_runtime_at_export": {"corrections": 1},
        "original_scientific_recipe_and_selection_unchanged": True,
        "export_was_corrected": False, "native_artifact_identity_claimed": False,
        "downstream_ids_repair_dependent": True,
        "corrected_index_sha256": "9" * 64, "corrected_codes_sha256": "a" * 64,
        "corrections_by_epoch_level_phase": {"16401": {"training": {"0": 1}}},
        "queue_admission": {"resume_chain": {"status": "exact_full_state_chain"}},
    }
    legacy = {
        key: copy.deepcopy(value) for key, value in new.items()
        if key not in {
            "export_was_corrected", "native_artifact_identity_claimed",
            "downstream_ids_repair_dependent", "corrected_index_sha256",
            "corrected_codes_sha256", "corrections_by_epoch_level_phase",
            "queue_admission",
        }
    }
    legacy["queue_admission"] = {
        "capsule_bank_receipt_sha256": "b" * 64,
        "integration_sha256": "c" * 64,
    }
    staged = {
        "experiment_lineage": {
            "rq_compatibility": copy.deepcopy(legacy),
            "rq": {"rq_compatibility": copy.deepcopy(legacy)},
        },
    }
    def verifier(*args):
        return {**repaired.lineage, "rq_compatibility": copy.deepcopy(new)}

    verifier.scope = {cell["rq_run_id"]}
    runtime.rq_verifier = verifier
    monkeypatch.setattr(m.native, "verify_stage", lambda _: staged)
    monkeypatch.setattr(m.Runtime, "admit", lambda self, *args: stage)
    before = source_manifest(stage)

    assert runtime.admit(cell, repaired.bundle) == stage
    assert source_manifest(stage) == before

    staged["experiment_lineage"]["rq_compatibility"]["base_state_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="existing repaired-ID stage"):
        runtime.admit(cell, repaired.bundle)


def test_controller_is_cpu_only_but_children_keep_the_assigned_gpu_lease(tmp_path, monkeypatch):
    m = module()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    with pytest.raises(RuntimeError, match="CPU-only"):
        m.CompatibilityRuntime(*([tmp_path] * 6))
    runtime = object.__new__(m.CompatibilityRuntime)
    runtime.env = tmp_path / "env.sh"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    command, env = runtime.command(["python", "native-run.py"], seed=17, gpu=True)
    assert env["CUDA_VISIBLE_DEVICES"] == "2,3" and env["LETTER_SEED"] == "17"
    assert "CUDA_VISIBLE_DEVICES=2,3" in command
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
