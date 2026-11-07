import copy
import importlib.util
import json
import shutil
import sys
import uuid
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "letter"


def protocol(workspace=None):
    path = SCRIPTS / "p5_xnode_state.py"
    assert path.exists(), "A checkpoint-bound, fenced cross-node protocol is required"
    spec = importlib.util.spec_from_file_location("p5_xnode_protocol_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if workspace is not None:
        from test_letter_p5_xnode_configuration import configure
        configure(module, workspace)
    return module


@pytest.fixture
def workspace(monkeypatch):
    monkeypatch.delitem(sys.modules, "p5_xnode_state", raising=False)
    path = ROOT / ".local" / "letter" / "p5-xnode-tests" / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path)


def lease():
    return {
        "kind": "p5-xnode-lease-v1", "cell": "example-cell",
        "owner_node": "worker.invalid", "gpu": 0, "gpu_uuid": "fixture-device-zero",
        "lease_id": "lease-v2", "ownership_epoch": 2,
        "root": {
            "artifact": "p5-state-000000100-000-111111111111",
            "seal_sha256": "1" * 64, "contract_sha256": "2" * 64,
            "step": 100, "epoch": 0, "phase": "train",
            "model_sha256": "3" * 64, "training_state_sha256": "4" * 64,
        },
        "contract_file_sha256": "5" * 64,
        "input_manifest_sha256": "6" * 64,
        "control_manifest_sha256": "7" * 64,
        "microbatch": 16, "accumulation": 8,
    }


def envelope(module, current_lease=None):
    current_lease = current_lease or lease()
    return {
        "kind": "p5-xnode-checkpoint-v1", "purpose": "production",
        "cell": current_lease["cell"], "owner_node": current_lease["owner_node"],
        "gpu": current_lease["gpu"], "lease_id": current_lease["lease_id"],
        "ownership_epoch": current_lease["ownership_epoch"],
        "lease_sha256": module.digest(current_lease), "generation": 1,
        "parent_seal_sha256": current_lease["root"]["seal_sha256"],
        "root_seal_sha256": current_lease["root"]["seal_sha256"],
        "checkpoint": {
            "artifact": "p5-state-000000110-000-222222222222",
            "seal_sha256": "8" * 64, "contract_sha256": current_lease["root"]["contract_sha256"],
            "step": 110, "epoch": 0, "phase": "train",
            "model_sha256": "9" * 64, "training_state_sha256": "a" * 64,
        },
        "selected": None, "execution_attestation_sha256": "b" * 64,
    }


def test_stale_writer_cannot_bank_a_migrated_cell(workspace):
    module = protocol(workspace)
    current = lease()
    registry = {"kind": "p5-xnode-ownership-v1", "status": "committed",
                "owners": {current["cell"]: current}}
    assert module.require_owner(registry, current["cell"], module.WORKER_HOST, "lease-v2") == current
    with pytest.raises(ValueError, match="owner|fenced"):
        module.require_owner(registry, current["cell"], module.WRITER_HOST, "lease-v1")
    registry["status"] = "prepared"
    with pytest.raises(ValueError, match="committed"):
        module.require_owner(registry, current["cell"], module.WORKER_HOST, "lease-v2")


@pytest.mark.parametrize("field,value", [
    ("owner_node", "foreign.invalid"), ("gpu", 1), ("lease_id", "stale"),
    ("ownership_epoch", 1), ("root_seal_sha256", "0" * 64),
    ("parent_seal_sha256", "0" * 64), ("generation", 3), ("purpose", "probe"),
])
def test_import_rejects_stale_owner_or_broken_ancestry(workspace, field, value):
    module = protocol(workspace)
    current = lease()
    value_ = envelope(module, current)
    value_[field] = value
    with pytest.raises(ValueError):
        module.validate_envelope(value_, current, module.root_head(current))


def test_monotonic_generation_and_identical_replay_only(workspace):
    module = protocol(workspace)
    current = lease()
    value = envelope(module, current)
    head = module.root_head(current)
    assert module.validate_envelope(value, current, head) == "next"
    advanced = module.envelope_head(value)
    assert module.validate_envelope(value, current, advanced) == "replay"
    divergent = copy.deepcopy(value)
    divergent["checkpoint"]["seal_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="diverg"):
        module.validate_envelope(divergent, current, advanced)
    backwards = copy.deepcopy(value)
    backwards["checkpoint"]["step"] = 99
    with pytest.raises(ValueError, match="progress|backward"):
        module.validate_envelope(backwards, current, head)


def test_same_step_validation_transition_is_not_a_rollback(workspace):
    module = protocol(workspace)
    current = lease()
    value = envelope(module, current)
    value["checkpoint"].update(step=100, phase="validate")
    assert module.validate_envelope(value, current, module.root_head(current)) == "next"
    parent = module.envelope_head(value)
    after_validation = copy.deepcopy(value)
    after_validation["generation"] = 2
    after_validation["parent_seal_sha256"] = value["checkpoint"]["seal_sha256"]
    after_validation["checkpoint"].update(
        artifact="p5-state-000000100-001-333333333333", seal_sha256="d" * 64, epoch=1, phase="train",
    )
    assert module.validate_envelope(after_validation, current, parent) == "next"


def test_same_step_divergent_training_weights_are_rejected(workspace):
    module = protocol(workspace)
    current = lease()
    value = envelope(module, current)
    value["checkpoint"]["step"] = 100
    with pytest.raises(ValueError, match="progress|diverg"):
        module.validate_envelope(value, current, module.root_head(current))


def test_source_contract_or_microbatch_cannot_change(workspace):
    module = protocol(workspace)
    current = lease()
    changed = envelope(module, current)
    changed["checkpoint"]["contract_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="contract"):
        module.validate_envelope(changed, current, module.root_head(current))
    with pytest.raises(ValueError):
        module.require_assignment(current, module.WORKER_HOST, 1, "fixture-device-zero")


def test_tree_digest_compares_tensor_bits_not_pickle_container_bytes():
    import numpy as np
    import torch
    module = protocol()
    a = {"sampler": torch.tensor([1, 0, 3]), "rng": np.array([4, 5], dtype=np.uint32),
         "optimizer": {0: {"exp_avg": torch.tensor([1., -0.])}}, "state": (2, .5, None)}
    assert module.tree_digest(a) == module.tree_digest(copy.deepcopy(a))
    b = copy.deepcopy(a)
    b["optimizer"][0]["exp_avg"][1] = 0.
    assert module.tree_digest(a) != module.tree_digest(b)
    b = copy.deepcopy(a)
    b["sampler"][0] = 2
    assert module.tree_digest(a) != module.tree_digest(b)


def test_lease_reassignment_is_only_the_explicit_private_cells(workspace):
    module = protocol(workspace)
    assert module.DESTINATIONS == {"example-cell": 0}
    value = lease()
    value["cell"] = "foreign-cell"
    with pytest.raises(ValueError):
        module.require_assignment(value, module.WORKER_HOST, 0, "fixture-device-zero")


def fake_checkpoint(module, run, step, suffix):
    path = run / "states" / f"p5-state-{step:09d}-000-{suffix}"
    path.mkdir(parents=True)
    (path / "model.safetensors").write_bytes(("model-" + str(step)).encode())
    (path / "training-state.pt").write_bytes(("optimizer-rng-" + str(step)).encode())
    module.write(path / "p5-seal.json", {
        "kind": "p5-full-state-v1", "status": "sealed",
        "contract_sha256": module.digest(module.read(run / "p5-run-config.json")),
        "global_step": step, "epoch": 0, "phase": "train",
        "files_sha256": {name: module.sha(path / name) for name in ("model.safetensors", "training-state.pt")},
    })
    return path


def package_fixture(module, workspace):
    run = workspace / "run"
    module.write(run / "p5-run-config.json", {"test_scientific_contract": True})
    root = fake_checkpoint(module, run, 100, "111111111111")
    current = lease()
    current["root"] = module.checkpoint_header(root)
    current["contract_file_sha256"] = module.sha(run / "p5-run-config.json")
    module.write(run / "crossnode-attestation.json", {"lease_sha256": module.digest(current)})
    next_ = fake_checkpoint(module, run, 110, "222222222222")
    return run, current, next_


def test_publisher_recovers_renamed_package_before_head_commit(workspace, monkeypatch):
    module = protocol(workspace)
    run, current, next_ = package_fixture(module, workspace)
    original_write = module.write

    def interrupted(path, value):
        if Path(path).name == "head.json":
            raise RuntimeError("crash after immutable package rename")
        original_write(path, value)

    monkeypatch.setattr(module, "write", interrupted)
    with pytest.raises(RuntimeError, match="crash"):
        module.publish_checkpoint(run, next_, current)
    monkeypatch.setattr(module, "write", original_write)
    package = module.publish_checkpoint(run, next_, current)
    saved = module.verify_package(package)
    assert saved["generation"] == 1
    assert len(list((run / ".xnode" / "outbox").glob("g*"))) == 1
    assert module.read(run / ".xnode" / "head.json")["seal_sha256"] == module.sha(next_ / "p5-seal.json")


def test_package_rejects_tampered_tensor_and_foreign_contract(workspace):
    module = protocol(workspace)
    run, current, next_ = package_fixture(module, workspace)
    package = module.publish_checkpoint(run, next_, current)
    with (package / "checkpoint" / "model.safetensors").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        module.verify_package(package)


def test_ack_is_bound_to_owner_and_every_native_state_file(workspace):
    module = protocol(workspace)
    run, current, next_ = package_fixture(module, workspace)
    package = module.publish_checkpoint(run, next_, current)
    package_manifest = module.file_manifest(package)
    receipt = {"files": package_manifest}
    ack = module.make_ack(package, current, receipt, "private-durable-destination", "c" * 64)
    header = module.checkpoint_header(next_)
    assert module.validate_ack(ack, current, header)
    bad = copy.deepcopy(ack)
    bad["lease_id"] = "retired-owner"
    with pytest.raises(ValueError):
        module.validate_ack(bad, current, header)
    bad = copy.deepcopy(ack)
    del bad["bank_receipt"]["files"]["checkpoint/training-state.pt"]
    bad["bank_receipt_sha256"] = module.digest(bad["bank_receipt"])
    with pytest.raises(ValueError):
        module.validate_ack(bad, current, header)


def test_selected_readmission_ack_is_explicitly_nonhistorical(workspace):
    module = protocol(workspace)
    run, current, next_ = package_fixture(module, workspace)
    package = module.publish_checkpoint(run, next_, current)
    receipt = {"files": module.file_manifest(package)}
    checkpoint = module.checkpoint_header(next_)
    ack = {
        "kind": "p5-xnode-selected-readmission-ack-v1",
        "status": "verified",
        "writer_host": module.WRITER_HOST,
        "cell": current["cell"],
        "lease_id": current["lease_id"],
        "lease_sha256": module.digest(current),
        "ownership_epoch": current["ownership_epoch"],
        "registry_event_sha256": "c" * 64,
        "checkpoint": checkpoint,
        "destination": "private-selected-readmission",
        "bank_receipt": receipt,
        "bank_receipt_sha256": module.digest(receipt),
        "readmission": {
            "kind": "p5-selected-lineage-readmission-v1",
            "historically_accepted": False,
            "wire_generation": 0,
            "root_seal_sha256": current["root"]["seal_sha256"],
            "source_event_sha256": "d" * 64,
            "selected_lineage_deployment_sha256": "e" * 64,
            "package_seal_sha256": module.sha(package / "package-seal.json"),
        },
    }

    assert module.validate_ack(ack, current, checkpoint)

    bad = copy.deepcopy(ack)
    bad["readmission"]["historically_accepted"] = True
    with pytest.raises(ValueError, match="readmission|historical"):
        module.validate_ack(bad, current, checkpoint)
    bad = copy.deepcopy(ack)
    bad["readmission"]["root_seal_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="readmission|root"):
        module.validate_ack(bad, current, checkpoint)


def test_probe_packages_cannot_be_accepted_as_production(workspace):
    module = protocol(workspace)
    run, current, next_ = package_fixture(module, workspace)
    package = module.publish_checkpoint(run, next_, current, purpose="probe")
    envelope_ = module.verify_package(package)
    with pytest.raises(ValueError):
        module.validate_envelope(envelope_, current, module.root_head(current))


def test_result_ack_rejects_complete_receipt_without_frozen_head(workspace):
    module = protocol(workspace)
    snapshot = workspace / "result"
    module.write(snapshot / "test-result.json", {"status": "complete"})
    (snapshot / "test-predictions.jsonl").write_text('{"complete":"toy fixture"}\n')
    module.write(snapshot / "p5-result-seal.json", {
        "kind": "p5-reference-results-v1", "status": "sealed",
        "files_sha256": {p.name: module.sha(p) for p in snapshot.iterdir() if p.is_file()},
    })
    current = lease()
    receipt = {"files": module.file_manifest(snapshot)}
    ack = {
        "kind": "p5-xnode-result-ack-v1", "status": "verified", "writer_host": module.WRITER_HOST,
        "lease_sha256": module.digest(current), "lease_id": current["lease_id"],
        "result_seal_sha256": module.sha(snapshot / "p5-result-seal.json"),
        "bank_receipt": receipt, "bank_receipt_sha256": module.digest(receipt),
    }
    with pytest.raises(ValueError, match="frozen|binding"):
        module.validate_result_ack(ack, current, snapshot)
    bad = copy.deepcopy(ack)
    bad["writer_host"] = module.WORKER_HOST
    with pytest.raises(ValueError):
        module.validate_result_ack(bad, current, snapshot)
