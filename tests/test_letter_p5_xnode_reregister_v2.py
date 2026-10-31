import copy
import importlib.util
import json
import shutil
import socket
import sys
from pathlib import Path

import pytest
import torch

from test_letter_p5_xnode_reregister import fixture as v1_fixture
from test_letter_p5_xnode_reregister import sha, write


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "letter"
sys.path.insert(0, str(SCRIPTS))


def module():
    path = SCRIPTS / "p5_xnode_reregister_v2.py"
    assert path.exists(), "A deployable v2 UUID re-registration tool is required"
    spec = importlib.util.spec_from_file_location("p5_xnode_reregister_v2_test", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def v2_fixture(tmp_path):
    input_manifest = tmp_path / "input-manifest.json"
    write(input_manifest, {"fixture": input_manifest.stem})
    v1, _, request, old_event, certificate = v1_fixture(
        tmp_path, input_manifest_sha256=sha(input_manifest),
    )
    old_bindings = json.loads(Path(request["old_bindings"]).read_text())
    for cell in request["cells"]:
        if old_bindings["cells"][cell]["role"] == "writer":
            root = old_event["owners"][cell]["root"]
            root.update(
                artifact=f"{cell.lower()}-root-90", seal_sha256=("1" * 64),
                step=90, model_sha256=("2" * 64), training_state_sha256=("3" * 64),
            )
    prior_sha = v1.event_sha(old_event)
    prior_receipt = {"files": {"ownership.json": {"sha256": prior_sha, "bytes": 1}}}
    certificate.update(
        event=old_event, event_sha256=prior_sha, bank_receipt=prior_receipt,
        bank_receipt_sha256=v1.digest(prior_receipt),
    )
    write(Path(request["registry"]) / "committed" / "0001.json", certificate)
    for cell in request["cells"]:
        if old_bindings["cells"][cell]["role"] == "worker":
            lease_id = old_event["owners"][cell]["lease_id"]
            record_path = (Path(request["registry"]).parent / "ledger" / cell / lease_id
                           / "accepted" / "g00000001.json")
            record = json.loads(record_path.read_text())
            record["ack"]["registry_event_sha256"] = prior_sha
            write(record_path, record)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    for name in ("p5_xnode_control.py", "p5_xnode_state.py"):
        shutil.copyfile(SCRIPTS / name, runtime / name)
    maintenance_manifest = tmp_path / "maintenance-manifest.json"
    maintenance_proof = tmp_path / "maintenance-proof.json"
    base = tmp_path / "base"
    base.mkdir()
    for path in (maintenance_manifest, maintenance_proof):
        write(path, {"fixture": path.stem})
    template = {
        "kind": "p5-xnode-runtime-v1",
        "root": str(tmp_path), "p5_root": str(tmp_path / "p5"),
        "control_root": str(tmp_path / "control"),
        "durable_root": str(tmp_path.parent / (tmp_path.name + "-durable")),
        "python": str(Path(sys.executable).resolve()), "base": str(base),
        "input_manifest": str(input_manifest), "input_manifest_sha256": sha(input_manifest),
        "maintenance_manifest": str(maintenance_manifest),
        "maintenance_proof": str(maintenance_proof),
        "ownership_certificate": str(tmp_path / "obsolete-certificate.json"),
        "crossnode_proof": str(tmp_path / "obsolete-proof.json"),
        "files_sha256": {
            name: sha(runtime / name)
            for name in ("p5_xnode_control.py", "p5_xnode_state.py")
        },
    }
    template_path = runtime / "runtime-template.json"
    write(template_path, template)
    for cell in request["cells"]:
        if old_event["owners"][cell]["owner_node"] != request["old_bindings"]:
            pass
        assignment = json.loads(Path(request["old_bindings"]).read_text())["cells"][cell]
        if assignment["role"] == "writer":
            old = request["evidence"][cell]
            source = Path(old["package"]) / "checkpoint"
            durable = (tmp_path.parent / (tmp_path.name + "-durable") / "runs" / cell
                       / old_event["owners"][cell]["root"]["artifact"])
            durable.mkdir(parents=True)
            for member in source.iterdir():
                shutil.copyfile(member, durable / member.name)
            receipt = {"files": {
                member.name: {"sha256": sha(durable / member.name), "bytes": member.stat().st_size}
                for member in durable.iterdir() if member.is_file()
            }}
            write(durable / "bank-receipt.json", receipt)
            ack = {
                "status": "verified", "kind": "p5-full-state-v1", "artifact": durable.name,
                "seal_sha256": sha(source / "p5-seal.json"),
                "contract_sha256": old_event["owners"][cell]["root"]["contract_sha256"],
                "destination": str(durable), "bank_receipt_sha256": v1.digest(receipt),
                "writer_host": json.loads(Path(request["old_bindings"]).read_text())["writer_host"],
            }
            ack_path = tmp_path / "evidence" / cell / "bank-ack.json"
            write(ack_path, ack)
            request["evidence"][cell] = {
                "kind": "p5-uuid-reregistration-writer-root-v2", "cell": cell,
                "checkpoint": str(durable), "ack": str(ack_path),
                "contract": old["contract"],
            }
        else:
            request["evidence"][cell]["kind"] = "p5-uuid-reregistration-worker-root-v2"
    request.update(
        kind="p5-uuid-reregistration-request-v2",
        runtime_template=str(template_path),
    )
    request_path = tmp_path / "request-v2.json"
    write(request_path, request)
    return module(), v1, request_path, request, old_event, certificate


def prepare(tmp_path):
    v2, _, request_path, request, old_event, certificate = v2_fixture(tmp_path)
    return v2, request_path, request, old_event, certificate, v2.prepare_transition(request_path)


def test_prepare_emits_guardian_compatible_proof_runtime_and_registration(tmp_path):
    v2, _, request, _, _, plan = prepare(tmp_path)
    proof = plan["aggregate_proof"]
    assert set(proof) == {
        "kind", "status", "source_node", "destination_node",
        "runtime_manifest_sha256", "cells",
    }
    assert proof["kind"] == "p5-xnode-crossnode-proof-v1"
    assert proof["runtime_manifest_sha256"] == plan["runtime_manifest_sha256"]
    assert plan["event"]["crossnode_proof_sha256"] == v2.document_sha(proof)
    assert set(proof["cells"]) == {"worker-tid-small", "worker-tid-large"}
    assert set(plan["event"]["replacement_proofs_sha256"]) == set(request["cells"])
    transitioned = plan["event"]["transition_cells"]
    assert all(
        plan["event"]["owners"][cell]["control_manifest_sha256"]
        == plan["runtime_manifest_sha256"]
        for cell in transitioned
    )
    registration = v2.worker_registration(plan, "worker-tid-small")
    assert set(registration) == {
        "kind", "writer_host", "lease_sha256", "event_sha256",
        "input_manifest_sha256",
    }
    assert plan["event"]["sequence"] == 1


def test_prepare_uses_writer_bank_receipts_and_worker_xnode_ledgers(tmp_path):
    v2, _, _, _, _, plan = prepare(tmp_path)
    assert plan["cells"]["writer-cid"]["evidence_kind"] == "writer-bank"
    assert plan["cells"]["worker-tid-small"]["evidence_kind"] == "worker-xnode"
    assert "accepted_record_sha256" not in plan["cells"]["writer-cid"]
    assert "accepted_record_sha256" in plan["cells"]["worker-tid-small"]


def test_prepare_accepts_receipt_valid_restored_sequence_two_root_then_three(tmp_path):
    v2, _, request_path, request, old_event, certificate = v2_fixture(tmp_path)
    registry = Path(request["registry"])
    event2 = copy.deepcopy(old_event)
    event2["sequence"] = 2
    event2["previous_event_sha256"] = None
    event2_sha = v2.event_sha(event2)
    receipt2 = {"files": {"ownership.json": {"sha256": event2_sha, "bytes": 1}}}
    certificate2 = {
        "event": event2,
        "event_sha256": event2_sha,
        "writer_host": certificate["writer_host"],
        "bank_receipt": receipt2,
        "bank_receipt_sha256": v2.digest(receipt2),
    }
    event3 = copy.deepcopy(old_event)
    event3["sequence"] = 3
    event3["previous_event_sha256"] = event2_sha
    event3_sha = v2.event_sha(event3)
    receipt3 = {"files": {"ownership.json": {"sha256": event3_sha, "bytes": 1}}}
    certificate3 = {
        "event": event3,
        "event_sha256": event3_sha,
        "writer_host": certificate["writer_host"],
        "bank_receipt": receipt3,
        "bank_receipt_sha256": v2.digest(receipt3),
    }
    committed = registry / "committed"
    for path in committed.glob("*.json"):
        path.unlink()
    write(committed / "0002.json", certificate2)
    write(committed / "0003.json", certificate3)
    for cell in request["cells"]:
        bindings = json.loads(Path(request["old_bindings"]).read_text())
        if bindings["cells"][cell]["role"] == "worker":
            owner = event3["owners"][cell]
            record_path = (
                registry.parent / "ledger" / cell / owner["lease_id"]
                / "accepted" / "g00000001.json"
            )
            record = json.loads(record_path.read_text())
            record["ack"]["registry_event_sha256"] = event3_sha
            write(record_path, record)

    plan = v2.prepare_transition(request_path)

    assert plan["event"]["previous_event_sha256"] == event3_sha
    assert plan["event"]["sequence"] == 4


@pytest.mark.parametrize("damage", ["first-previous", "sequence-gap", "wrong-receipt"])
def test_restored_positive_sequence_root_remains_strictly_bound(tmp_path, damage):
    v2, _, request_path, request, old_event, certificate = v2_fixture(tmp_path)
    registry = Path(request["registry"])
    event2 = copy.deepcopy(old_event)
    event2["sequence"] = 2
    event2["previous_event_sha256"] = "0" * 64 if damage == "first-previous" else None
    event2_sha = v2.event_sha(event2)
    receipt2 = {"files": {"ownership.json": {"sha256": event2_sha, "bytes": 1}}}
    certificate2 = {
        "event": event2, "event_sha256": event2_sha,
        "writer_host": certificate["writer_host"],
        "bank_receipt": receipt2,
        "bank_receipt_sha256": v2.digest(receipt2),
    }
    event3 = copy.deepcopy(old_event)
    event3["sequence"] = 4 if damage == "sequence-gap" else 3
    event3["previous_event_sha256"] = event2_sha
    event3_sha = v2.event_sha(event3)
    receipt3 = {"files": {"ownership.json": {"sha256": event3_sha, "bytes": 1}}}
    certificate3 = {
        "event": event3, "event_sha256": event3_sha,
        "writer_host": certificate["writer_host"],
        "bank_receipt": receipt3,
        "bank_receipt_sha256": (
            "0" * 64 if damage == "wrong-receipt" else v2.digest(receipt3)
        ),
    }
    committed = registry / "committed"
    for path in committed.glob("*.json"):
        path.unlink()
    write(committed / "0002.json", certificate2)
    write(committed / "0003.json", certificate3)

    with pytest.raises(ValueError, match="ancestry|certificate|sequence|receipt"):
        v2.prepare_transition(request_path)


def compact_writer_owners(v2, request, old_event, certificate):
    bindings = json.loads(Path(request["old_bindings"]).read_text())
    compact = {}
    for cell, owner in old_event["owners"].items():
        if bindings["cells"][cell]["role"] == "writer":
            compact[cell] = {
                key: owner[key] for key in (
                    "cell", "gpu", "lease_id", "owner_node", "ownership_epoch",
                )
            }
    old_event["owners"].update(compact)
    event_sha = v2.event_sha(old_event)
    receipt = {"files": {"ownership.json": {"sha256": event_sha, "bytes": 1}}}
    certificate.update(
        event=old_event, event_sha256=event_sha, bank_receipt=receipt,
        bank_receipt_sha256=v2.digest(receipt),
    )
    write(Path(request["registry"]) / "committed" / "0001.json", certificate)
    for cell, owner in old_event["owners"].items():
        if cell in request["cells"] and bindings["cells"][cell]["role"] == "worker":
            record_path = (
                Path(request["registry"]).parent / "ledger" / cell / owner["lease_id"]
                / "accepted" / "g00000001.json"
            )
            record = json.loads(record_path.read_text())
            record["ack"]["registry_event_sha256"] = event_sha
            write(record_path, record)
    return compact


def writer_durable_recovery(v2, request, old_event, cell, generations=2):
    source = Path(request["evidence"][cell]["checkpoint"])
    contract = Path(request["evidence"][cell]["contract"])
    destination = source.parent / "receipt-recovery"
    checkpoints = []
    for generation in range(1, generations + 1):
        checkpoint = destination / f"writer-state-{generation}"
        checkpoint.mkdir(parents=True)
        shutil.copyfile(source / "model.safetensors", checkpoint / "model.safetensors")
        contract_value = json.loads(contract.read_text())
        torch.save({
            "schema_version": 1,
            "contract": contract_value,
            "optimizer": {"state": {"fixture": generation}},
            "scheduler": {"fixture": generation},
            "sampler": {"fixture": generation},
            "rng": {"fixture": generation},
            "state": {
                "global_step": 90 + generation,
                "epoch": 9,
                "phase": "train",
            },
            "checkpoint_boundary": "after_complete_optimizer_update_no_partial_gradients",
        }, checkpoint / "training-state.pt")
        seal = json.loads((source / "p5-seal.json").read_text())
        seal.update(
            global_step=90 + generation,
            epoch=9,
            phase="train",
            files_sha256={
                "model.safetensors": sha(checkpoint / "model.safetensors"),
                "training-state.pt": sha(checkpoint / "training-state.pt"),
            },
        )
        write(checkpoint / "p5-seal.json", seal)
        receipt = {"files": {
            member.name: {"sha256": sha(member), "bytes": member.stat().st_size}
            for member in checkpoint.iterdir() if member.is_file()
        }}
        write(checkpoint / "bank-receipt.json", receipt)
        checkpoints.append(str(checkpoint))
    request["evidence"][cell] = {
        "kind": "p5-uuid-reregistration-writer-durable-root-v2",
        "cell": cell,
        "checkpoints": checkpoints,
        "contract": str(contract),
        "registry_event_sha256": v2.event_sha(old_event),
        "destination_root": str(destination),
        "selection": {"policy": "latest"},
    }
    chosen = Path(checkpoints[-1])
    proof = json.loads(Path(request["proofs"][cell]).read_text())
    proof.update(
        source_checkpoint_seal_sha256=sha(chosen / "p5-seal.json"),
        model_sha256=sha(chosen / "model.safetensors"),
        training_state_sha256=sha(chosen / "training-state.pt"),
    )
    write(Path(request["proofs"][cell]), proof)
    return checkpoints


def test_prepare_reconstructs_compact_retained_writer_from_receipt_and_bindings(tmp_path):
    v2, _, request_path, request, old_event, certificate = v2_fixture(tmp_path)
    compact = compact_writer_owners(v2, request, old_event, certificate)

    plan = v2.prepare_transition(request_path)

    old_bindings = json.loads(Path(request["old_bindings"]).read_text())
    runtime = plan["runtime_manifest"]
    for cell, historical in compact.items():
        lease = plan["event"]["owners"][cell]
        assignment = old_bindings["cells"][cell]
        assert old_event["owners"][cell] == historical
        assert lease["kind"] == "p5-xnode-lease-v1"
        assert lease["cell"] == cell
        assert lease["owner_node"] == historical["owner_node"]
        assert lease["gpu"] == historical["gpu"] == assignment["gpu"]
        assert lease["microbatch"] == assignment["microbatch"]
        assert lease["accumulation"] == assignment["accumulation"]
        assert lease["contract_file_sha256"] == sha(request["evidence"][cell]["contract"])
        assert lease["input_manifest_sha256"] == runtime["input_manifest_sha256"]
        assert lease["control_manifest_sha256"] == plan["runtime_manifest_sha256"]
        reconstruction = plan["cells"][cell]["writer_lease_reconstruction"]
        assert reconstruction["historical_owner_sha256"] == v2.digest(historical)
        assert reconstruction["writer_evidence_sha256"] == v2.digest({
            "checkpoint": sha(Path(request["evidence"][cell]["checkpoint"]) / "p5-seal.json"),
            "ack": sha(request["evidence"][cell]["ack"]),
            "contract": sha(request["evidence"][cell]["contract"]),
        })


def test_compact_writer_without_ack_uses_nonhistorical_receipt_recovery_latest(tmp_path):
    v2, _, request_path, request, old_event, certificate = v2_fixture(tmp_path)
    compact = compact_writer_owners(v2, request, old_event, certificate)
    checkpoints = {}
    for cell in compact:
        checkpoints[cell] = writer_durable_recovery(v2, request, old_event, cell)
    write(request_path, request)

    plan = v2.prepare_transition(request_path)

    for cell in compact:
        metadata = plan["cells"][cell]
        assert metadata["evidence_kind"] == "writer-durable-recovery"
        assert metadata["recovery_kind"] == "p5-writer-durable-recovery-v1"
        assert metadata["historically_accepted"] is False
        assert metadata["selection_policy"] == "latest"
        assert metadata["chosen_checkpoint"] == checkpoints[cell][-1]
        assert metadata["abandoned_verified_checkpoints"] == []
        assert plan["event"]["durable_recovery"][cell] == metadata


def test_writer_durable_recovery_requires_explicit_rollback_and_records_newer_root(tmp_path):
    v2, _, request_path, request, old_event, certificate = v2_fixture(tmp_path)
    compact = compact_writer_owners(v2, request, old_event, certificate)
    cell = sorted(compact)[0]
    checkpoints = writer_durable_recovery(v2, request, old_event, cell)
    for other in compact:
        if other != cell:
            writer_durable_recovery(v2, request, old_event, other)
    request["evidence"][cell]["selection"] = {"policy": "latest", "generation": 1}
    write(request_path, request)
    with pytest.raises(ValueError, match="rollback|latest"):
        v2.prepare_transition(request_path)
    request["evidence"][cell]["selection"] = {
        "policy": "intentional_rollback",
        "generation": 1,
        "reason": "operator-selected durable writer rollback",
    }
    chosen = Path(checkpoints[0])
    proof = json.loads(Path(request["proofs"][cell]).read_text())
    proof.update(
        source_checkpoint_seal_sha256=sha(chosen / "p5-seal.json"),
        model_sha256=sha(chosen / "model.safetensors"),
        training_state_sha256=sha(chosen / "training-state.pt"),
    )
    write(Path(request["proofs"][cell]), proof)
    write(request_path, request)

    plan = v2.prepare_transition(request_path)

    assert plan["cells"][cell]["selection_policy"] == "intentional_rollback"
    assert [
        item["path"] for item in plan["cells"][cell]["abandoned_verified_checkpoints"]
    ] == [checkpoints[1]]


@pytest.mark.parametrize("damage", [
    "missing-receipt", "partial-state", "embedded-contract", "event", "destination",
])
def test_writer_durable_recovery_fails_closed_on_unbound_or_partial_root(tmp_path, damage):
    v2, _, request_path, request, old_event, certificate = v2_fixture(tmp_path)
    compact = compact_writer_owners(v2, request, old_event, certificate)
    cell = sorted(compact)[0]
    checkpoints = writer_durable_recovery(v2, request, old_event, cell)
    for other in compact:
        if other != cell:
            writer_durable_recovery(v2, request, old_event, other)
    checkpoint = Path(checkpoints[-1])
    if damage == "missing-receipt":
        (checkpoint / "bank-receipt.json").unlink()
    elif damage == "partial-state":
        (checkpoint / "training-state.pt").unlink()
    elif damage == "embedded-contract":
        payload = torch.load(checkpoint / "training-state.pt", map_location="cpu", weights_only=False)
        payload["contract"] = {"foreign": True}
        torch.save(payload, checkpoint / "training-state.pt")
        seal = json.loads((checkpoint / "p5-seal.json").read_text())
        seal["files_sha256"]["training-state.pt"] = sha(checkpoint / "training-state.pt")
        write(checkpoint / "p5-seal.json", seal)
        receipt = {"files": {
            member.name: {"sha256": sha(member), "bytes": member.stat().st_size}
            for member in checkpoint.iterdir() if member.name != "bank-receipt.json"
        }}
        write(checkpoint / "bank-receipt.json", receipt)
    elif damage == "event":
        request["evidence"][cell]["registry_event_sha256"] = "0" * 64
    else:
        request["evidence"][cell]["destination_root"] = str(checkpoint.parent.parent)
    write(request_path, request)

    with pytest.raises(ValueError, match="writer|receipt|contract|event|destination|state"):
        v2.prepare_transition(request_path)


@pytest.mark.parametrize("damage", [
    "owner-gpu", "owner-epoch", "new-binding-epoch", "ack-writer",
])
def test_compact_writer_reconstruction_fails_closed_on_ambiguous_identity(tmp_path, damage):
    v2, _, request_path, request, old_event, certificate = v2_fixture(tmp_path)
    compact = compact_writer_owners(v2, request, old_event, certificate)
    cell = sorted(compact)[0]
    if damage == "owner-gpu":
        old_event["owners"][cell]["gpu"] += 1
    elif damage == "owner-epoch":
        old_event["owners"][cell]["ownership_epoch"] += 1
    elif damage == "new-binding-epoch":
        bindings = json.loads(Path(request["new_bindings"]).read_text())
        bindings["cells"][cell]["ownership_epoch"] += 1
        write(Path(request["new_bindings"]), bindings)
    else:
        ack = json.loads(Path(request["evidence"][cell]["ack"]).read_text())
        ack["writer_host"] = "foreign-writer"
        write(Path(request["evidence"][cell]["ack"]), ack)
    if damage != "ack-writer":
        event_sha = v2.event_sha(old_event)
        receipt = {"files": {"ownership.json": {"sha256": event_sha, "bytes": 1}}}
        certificate.update(
            event=old_event, event_sha256=event_sha, bank_receipt=receipt,
            bank_receipt_sha256=v2.digest(receipt),
        )
        write(Path(request["registry"]) / "committed" / "0001.json", certificate)

    with pytest.raises(ValueError, match="writer|binding|identity|acknowledgement"):
        v2.prepare_transition(request_path)


def test_commit_publishes_canonical_state_and_activation_last(tmp_path, monkeypatch):
    v2, request_path, request, _, _, plan = prepare(tmp_path)
    bindings = plan["bindings"]
    monkeypatch.setattr(socket, "gethostname", lambda: bindings["writer_host"])
    result = v2.commit_transition(request_path, plan)
    xroot = Path(bindings["control_root"])
    assert Path(result["activation"]).samefile(xroot / "activation" / "last-published.json")
    activation = json.loads((xroot / "activation" / "last-published.json").read_text())
    assert activation["sequence"] == plan["event"]["sequence"]
    for relative, checksum in activation["files_sha256"].items():
        assert sha(xroot / relative) == checksum
    for cell in request["cells"]:
        assert (xroot / "leases" / f"{cell}.json").is_file()
        assert (xroot / "roots" / cell / "checkpoint" / "training-state.pt").is_file()
        assert (xroot / "roots" / cell / "p5-run-config.json").is_file()
        assert (xroot / "roots" / cell / "latest.json").is_file()
    for cell in ("worker-tid-small", "worker-tid-large"):
        transport = xroot / "worker-bootstrap" / cell / ".xnode"
        assert {path.name for path in transport.iterdir()} == {"registration.json"}


def test_interrupted_commit_is_not_activated(tmp_path, monkeypatch):
    v2, request_path, _, _, _, plan = prepare(tmp_path)
    monkeypatch.setattr(socket, "gethostname", lambda: plan["bindings"]["writer_host"])
    with pytest.raises(RuntimeError, match="activation"):
        v2.commit_transition(request_path, plan, fail_before_activation=True)
    with pytest.raises(ValueError, match="activated|activation"):
        v2.verify_activation(Path(plan["bindings"]["control_root"]),
                             plan["event"]["broker_release_sha256"])
    result = v2.activate_transition(request_path, plan)
    assert Path(result["activation"]).is_file()
    assert v2.verify_activation(
        Path(plan["bindings"]["control_root"]),
        plan["event"]["broker_release_sha256"],
    )["event_sha256"] == plan["event_sha256"]


def test_activation_rejects_stale_canonical_file(tmp_path, monkeypatch):
    v2, request_path, _, _, _, plan = prepare(tmp_path)
    monkeypatch.setattr(socket, "gethostname", lambda: plan["bindings"]["writer_host"])
    v2.commit_transition(request_path, plan)
    xroot = Path(plan["bindings"]["control_root"])
    lease = xroot / "leases" / "worker-tid-small.json"
    lease.write_text(lease.read_text() + "\n")
    with pytest.raises(ValueError, match="activation|canonical|hash"):
        v2.verify_activation(xroot, plan["event"]["broker_release_sha256"])


def test_activation_recovery_cannot_bless_stale_staged_file(tmp_path, monkeypatch):
    v2, request_path, _, _, _, plan = prepare(tmp_path)
    monkeypatch.setattr(socket, "gethostname", lambda: plan["bindings"]["writer_host"])
    with pytest.raises(RuntimeError, match="activation"):
        v2.commit_transition(request_path, plan, fail_before_activation=True)
    xroot = Path(plan["bindings"]["control_root"])
    lease = xroot / "leases" / "worker-tid-small.json"
    lease.write_text(lease.read_text() + "\n")
    with pytest.raises(ValueError, match="canonical|staged|lease|hash"):
        v2.activate_transition(request_path, plan)


def test_broker_v3_authorizes_current_certificate_only_after_activation(tmp_path, monkeypatch):
    v2, request_path, _, _, _, plan = prepare(tmp_path)
    monkeypatch.setattr(socket, "gethostname", lambda: plan["bindings"]["writer_host"])
    v2.commit_transition(request_path, plan)
    broker_path = SCRIPTS / "p5_xnode_broker.py"
    spec = importlib.util.spec_from_file_location("p5_xnode_broker_v3_test", broker_path)
    broker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(broker)
    broker.state.configure_bindings(
        Path(plan["bindings"]["control_root"]) / "current" / "bindings.json"
    )
    certificate = broker.activated_current(
        Path(plan["bindings"]["control_root"]),
        plan["event"]["broker_release_sha256"],
    )
    assert certificate["event_sha256"] == plan["event_sha256"]


def test_staged_runtime_preserves_guardian_registration_semantics(tmp_path, monkeypatch):
    v2, request_path, _, _, _, plan = prepare(tmp_path)
    monkeypatch.setattr(socket, "gethostname", lambda: plan["bindings"]["writer_host"])
    v2.commit_transition(request_path, plan)
    xroot = Path(plan["bindings"]["control_root"])
    control_path = SCRIPTS / "p5_xnode_control.py"
    spec = importlib.util.spec_from_file_location("p5_xnode_control_v2_test", control_path)
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
    control.state._BINDINGS = None
    control.state.configure_bindings(xroot / "current" / "bindings.json")
    config = control.verify_runtime(xroot / "current" / "runtime-manifest.json")
    assert sha(xroot / "current" / "runtime-manifest.json") == plan["runtime_manifest_sha256"]
    assert config["files_sha256"] == plan["runtime_manifest"]["files_sha256"]
    cell = "worker-tid-small"
    run = Path(plan["bindings"]["p5_root"]) / "runs" / cell
    (run / ".xnode").mkdir(parents=True)
    shutil.copyfile(
        xroot / "worker-bootstrap" / cell / ".xnode" / "registration.json",
        run / ".xnode" / "registration.json",
    )
    lease = json.loads((xroot / "leases" / f"{cell}.json").read_text())
    registration = json.loads((run / ".xnode" / "registration.json").read_text())
    assert registration["lease_sha256"] == control.state.digest(lease)
    assert registration["event_sha256"] == json.loads(
        (xroot / "current" / "ownership-certificate.json").read_text()
    )["event_sha256"]
    assert registration["input_manifest_sha256"] == config["input_manifest_sha256"]
    assert registration["writer_host"] == control.state.WRITER_HOST
    assert control.registration(run, lease, config) == v2.worker_registration(plan, cell)


def test_broker_v3_rejects_incomplete_activation_inventory(tmp_path, monkeypatch):
    v2, request_path, _, _, _, plan = prepare(tmp_path)
    monkeypatch.setattr(socket, "gethostname", lambda: plan["bindings"]["writer_host"])
    v2.commit_transition(request_path, plan)
    xroot = Path(plan["bindings"]["control_root"])
    activation_path = xroot / "activation" / "last-published.json"
    activation = json.loads(activation_path.read_text())
    activation["files_sha256"].pop("leases/worker-tid-small.json")
    write(activation_path, activation)
    broker_path = SCRIPTS / "p5_xnode_broker.py"
    spec = importlib.util.spec_from_file_location("p5_xnode_broker_v3_inventory_test", broker_path)
    broker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(broker)
    broker.state._BINDINGS = None
    broker.state.configure_bindings(xroot / "current" / "bindings.json")
    with pytest.raises(ValueError, match="activation|inventory|lease"):
        broker.activated_current(xroot, plan["event"]["broker_release_sha256"])


def test_broker_v3_release_rejects_certificate_hash_cycle(tmp_path):
    v2, _, _, _, _, plan = prepare(tmp_path)
    release = {
        "kind": "p5-xnode-broker-v3",
        "files_sha256": {
            "p5_xnode_broker.py": "a" * 64,
            "p5_xnode_state.py": "b" * 64,
            "p5_xnode_adopt.py": "c" * 64,
        },
        "runtime_manifest_sha256": plan["runtime_manifest_sha256"],
        "ownership_certificate_sha256": "d" * 64,
    }
    broker_path = SCRIPTS / "p5_xnode_broker.py"
    spec = importlib.util.spec_from_file_location("p5_xnode_broker_v3_release_test", broker_path)
    broker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(broker)
    with pytest.raises(ValueError, match="cycle|certificate|anchor"):
        broker.validate_broker_v3_release(release)
