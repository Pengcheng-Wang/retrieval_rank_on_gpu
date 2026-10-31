import copy
import hashlib
import importlib.util
import json
import socket
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "letter"
sys.path.insert(0, str(SCRIPTS))


def module():
    path = SCRIPTS / "p5_xnode_reregister.py"
    assert path.exists(), "A versioned offline UUID re-registration tool is required"
    spec = importlib.util.spec_from_file_location("p5_xnode_reregister_test", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def binding(root, uuids):
    cells = {}
    assignments = {
        "writer-cid": ("writer", 4, 64, 2),
        "writer-tid": ("writer", 5, 64, 2),
        "worker-tid-small": ("worker", 5, 16, 8),
        "worker-tid-large": ("worker", 6, 64, 2),
        "completed-cell": ("worker", 7, 64, 2),
    }
    for cell, (role, gpu, micro, accumulation) in assignments.items():
        cells[cell] = {
            "role": role, "gpu": gpu, "gpu_uuid": uuids[cell],
            "microbatch": micro, "accumulation": accumulation,
            "root_step": 100, "ownership_epoch": 2,
        }
    return {
        "kind": "p5-xnode-bindings-v1",
        "root": str(root), "p5_root": str(root / "p5"),
        "control_root": str(root / "control"),
        "durable_root": str(root.parent / (root.name + "-durable")),
        "python": str(Path(sys.executable).resolve()),
        "writer_host": socket.gethostname(), "worker_host": "worker.invalid",
        "cells": cells, "adopted_cell": None, "learner_sha256": "a" * 64,
        "legacy_device_guard": "True", "legacy_transport_guard": "True",
    }


def checkpoint_header(cell):
    token = hashlib.sha256(cell.encode()).hexdigest()
    return {
        "artifact": f"p5-state-000000100-000-{token[:12]}",
        "seal_sha256": token, "contract_sha256": hashlib.sha256((cell + "-contract").encode()).hexdigest(),
        "step": 100, "epoch": 0, "phase": "train",
        "model_sha256": hashlib.sha256((cell + "-model").encode()).hexdigest(),
        "training_state_sha256": hashlib.sha256((cell + "-state").encode()).hexdigest(),
    }


def old_owner(cell, assignment, header, input_manifest_sha256="b" * 64):
    return {
        "kind": "p5-xnode-lease-v1", "cell": cell,
        "owner_node": socket.gethostname() if assignment["role"] == "writer" else "worker.invalid",
        "gpu": assignment["gpu"], "gpu_uuid": assignment["gpu_uuid"],
        "lease_id": cell.lower() + "-lease-old", "ownership_epoch": 2,
        "root": copy.deepcopy(header),
        "contract_file_sha256": hashlib.sha256((cell + "-contract-file").encode()).hexdigest(),
        "input_manifest_sha256": input_manifest_sha256,
        "control_manifest_sha256": "c" * 64,
        "native_gate": "/private/native-gate",
        "microbatch": assignment["microbatch"], "accumulation": assignment["accumulation"],
    }


def fixture(tmp_path, input_manifest_sha256="b" * 64):
    reregister = module()
    cells = [
        "writer-cid", "writer-tid", "worker-tid-small", "worker-tid-large",
    ]
    old_uuids = {cell: "old-" + cell for cell in cells}
    old_uuids["completed-cell"] = "old-completed"
    new_uuids = {cell: "new-" + cell for cell in cells}
    new_uuids["completed-cell"] = "old-completed"
    old_binding = binding(tmp_path, old_uuids)
    new_binding = binding(tmp_path, new_uuids)
    owners, evidence, proofs = {}, {}, {}
    for cell, assignment in old_binding["cells"].items():
        header = checkpoint_header(cell)
        owner = old_owner(cell, assignment, header, input_manifest_sha256)
        owners[cell] = owner
        if cell not in cells:
            continue
        contract = tmp_path / "evidence" / cell / "scientific-contract.json"
        write(contract, {"cell": cell, "contract": "unchanged"})
        owner["contract_file_sha256"] = sha(contract)
        owner["root"]["contract_sha256"] = digest(json.loads(contract.read_text()))
        package = tmp_path / "evidence" / cell / "package"
        (package / "checkpoint").mkdir(parents=True)
        (package / "checkpoint" / "model.safetensors").write_bytes((cell + "-model").encode())
        (package / "checkpoint" / "training-state.pt").write_bytes((cell + "-state").encode())
        seal = {
            "kind": "p5-full-state-v1", "status": "sealed",
            "contract_sha256": owner["root"]["contract_sha256"],
            "global_step": 100, "epoch": 0, "phase": "train",
            "files_sha256": {
                "model.safetensors": sha(package / "checkpoint" / "model.safetensors"),
                "training-state.pt": sha(package / "checkpoint" / "training-state.pt"),
            },
        }
        write(package / "checkpoint" / "p5-seal.json", seal)
        owner["root"].update({
            "seal_sha256": sha(package / "checkpoint" / "p5-seal.json"),
            "model_sha256": seal["files_sha256"]["model.safetensors"],
            "training_state_sha256": seal["files_sha256"]["training-state.pt"],
        })
        attestation = {
            "lease_sha256": digest(owner), "host": owner["owner_node"],
            "gpu": owner["gpu"], "purpose": "production",
            "root_checkpoint_seal_sha256": owner["root"]["seal_sha256"],
            "scientific_contract_rewritten": False,
        }
        write(package / "execution-attestation.json", attestation)
        envelope = {
            "kind": "p5-xnode-checkpoint-v1", "purpose": "production",
            "cell": cell, "owner_node": owner["owner_node"], "gpu": owner["gpu"],
            "lease_id": owner["lease_id"], "ownership_epoch": owner["ownership_epoch"],
            "lease_sha256": digest(owner), "generation": 1,
            "parent_seal_sha256": owner["root"]["seal_sha256"],
            "root_seal_sha256": owner["root"]["seal_sha256"],
            "checkpoint": copy.deepcopy(owner["root"]), "selected": None,
            "execution_attestation_sha256": sha(package / "execution-attestation.json"),
        }
        write(package / "scientific-contract.json", json.loads(contract.read_text()))
        write(package / "envelope.json", envelope)
        files = {}
        for path in package.rglob("*"):
            if path.is_file():
                files[path.relative_to(package).as_posix()] = {
                    "sha256": sha(path), "bytes": path.stat().st_size,
                }
        write(package / "package-seal.json", {
            "kind": "p5-xnode-package-v1", "status": "sealed", "files": files,
        })
        receipt = {"files": {
            path.relative_to(package).as_posix(): {"sha256": sha(path), "bytes": path.stat().st_size}
            for path in package.rglob("*") if path.is_file()
        }}
        ack = {
            "kind": "p5-xnode-bank-ack-v1", "status": "verified",
            "writer_host": old_binding["writer_host"], "cell": cell,
            "lease_id": owner["lease_id"], "lease_sha256": digest(owner),
            "ownership_epoch": owner["ownership_epoch"], "registry_event_sha256": "e" * 64,
            "generation": 1, "envelope": envelope, "checkpoint": envelope["checkpoint"],
            "destination": str(package), "bank_receipt": receipt,
            "bank_receipt_sha256": digest(receipt),
        }
        record = {"lease_sha256": digest(owner), "envelope": envelope, "ack": ack}
        ledger = tmp_path / "control" / "ledger" / cell / owner["lease_id"]
        write(ledger / "accepted" / "g00000001.json", record)
        write(ledger / "head.json", {**envelope["checkpoint"], "generation": 1,
                                     "envelope_sha256": digest(envelope)})
        evidence[cell] = {
            "kind": "p5-uuid-reregistration-root-v1", "cell": cell,
            "ledger": str(ledger), "package": str(package), "contract": str(contract),
        }
        proof = {
            "kind": "p5-replacement-gpu-proof-v1", "status": "pass", "cell": cell,
            "host": owner["owner_node"], "gpu": owner["gpu"], "gpu_uuid": new_uuids[cell],
            "source_checkpoint_seal_sha256": owner["root"]["seal_sha256"],
            "contract_sha256": owner["root"]["contract_sha256"],
            "model_sha256": owner["root"]["model_sha256"],
            "training_state_sha256": owner["root"]["training_state_sha256"],
            "cold_save_reload_bitwise_equal": True,
            "next_sampled_rows_and_losses_equal": True,
            "populated_optimizer_memory_pass": True,
            "complete_20beam_geometry_pass": True,
        }
        proof_path = tmp_path / "proofs" / f"{cell}.json"
        write(proof_path, proof)
        proofs[cell] = str(proof_path)
    old_event = {
        "kind": "p5-xnode-ownership-v1", "status": "committed",
        "owners": owners, "previous_event_sha256": None,
        "broker_release_sha256": "f" * 64,
    }
    event_sha = reregister.event_sha(old_event)
    for cell in cells:
        lease_id = owners[cell]["lease_id"]
        record_path = (tmp_path / "control" / "ledger" / cell / lease_id
                       / "accepted" / "g00000001.json")
        record = json.loads(record_path.read_text())
        record["ack"]["registry_event_sha256"] = event_sha
        write(record_path, record)
    old_receipt = {"files": {"ownership.json": {"sha256": event_sha, "bytes": 1}}}
    certificate = {
        "event": old_event, "event_sha256": event_sha,
        "writer_host": old_binding["writer_host"], "bank_receipt": old_receipt,
        "bank_receipt_sha256": digest(old_receipt),
    }
    committed = tmp_path / "control" / "ownership" / "committed" / "0001.json"
    write(committed, certificate)
    old_binding_path, new_binding_path = tmp_path / "old-bindings.json", tmp_path / "new-bindings.json"
    write(old_binding_path, old_binding)
    write(new_binding_path, new_binding)
    request = {
        "kind": "p5-uuid-reregistration-request-v1",
        "cells": cells, "old_bindings": str(old_binding_path),
        "new_bindings": str(new_binding_path),
        "registry": str(tmp_path / "control" / "ownership"),
        "evidence": evidence, "proofs": proofs,
        "broker_release_sha256": "9" * 64,
    }
    request_path = tmp_path / "request.json"
    write(request_path, request)
    return reregister, request_path, request, old_event, certificate


def prepare(tmp_path):
    reregister, request_path, request, old_event, certificate = fixture(tmp_path)
    plan = reregister.prepare_transition(request_path)
    return reregister, request_path, request, old_event, certificate, plan


def test_prepare_binds_all_cells_and_increments_only_transition_epochs(tmp_path):
    reregister, _, request, old_event, _, plan = prepare(tmp_path)
    assert plan["kind"] == "p5-uuid-reregistration-plan-v1"
    assert set(plan["event"]["owners"]) == set(old_event["owners"])
    assert plan["event"]["previous_event_sha256"] == reregister.event_sha(old_event)
    assert plan["event"]["broker_release_sha256"] == request["broker_release_sha256"]
    for cell in request["cells"]:
        old, new = old_event["owners"][cell], plan["event"]["owners"][cell]
        assert new["ownership_epoch"] == old["ownership_epoch"] + 1
        assert new["gpu"] == old["gpu"]
        assert new["owner_node"] == old["owner_node"]
        assert new["microbatch"] == old["microbatch"]
        assert new["accumulation"] == old["accumulation"]
        assert new["root"] == plan["cells"][cell]["old_accepted_head"]
        assert new["lease_id"] != old["lease_id"]
    assert plan["event"]["owners"]["completed-cell"] == old_event["owners"]["completed-cell"]


def test_prepare_rejects_pending_transition_intent(tmp_path):
    reregister, request_path, request, old_event, _ = fixture(tmp_path)
    write(Path(request["registry"]) / "transition-intent.json", {
        "event_sha256": "0" * 64, "previous_event_sha256": reregister.event_sha(old_event),
    })
    with pytest.raises(ValueError, match="pending"):
        reregister.prepare_transition(request_path)


@pytest.mark.parametrize("damage", [
    "parent", "uuid", "partial", "proof", "epoch", "receipt", "ack-event",
])
def test_verify_rejects_faulted_transition(tmp_path, damage):
    reregister, request_path, request, old_event, certificate, plan = prepare(tmp_path)
    if damage == "parent":
        plan["event"]["previous_event_sha256"] = "0" * 64
    elif damage == "uuid":
        cell = request["cells"][0]
        plan["event"]["owners"][cell]["gpu_uuid"] = "stale-uuid"
    elif damage == "partial":
        cell = request["cells"][0]
        package = Path(request["evidence"][cell]["package"])
        (package / "checkpoint" / "training-state.pt").unlink()
    elif damage == "proof":
        cell = request["cells"][0]
        proof = json.loads(Path(request["proofs"][cell]).read_text())
        proof["complete_20beam_geometry_pass"] = False
        write(Path(request["proofs"][cell]), proof)
    elif damage == "epoch":
        cell = request["cells"][0]
        plan["event"]["owners"][cell]["ownership_epoch"] += 1
    elif damage == "receipt":
        certificate["bank_receipt_sha256"] = "0" * 64
        write(Path(request["registry"]) / "committed" / "0001.json", certificate)
    else:
        cell = request["cells"][0]
        lease_id = old_event["owners"][cell]["lease_id"]
        record_path = (Path(request["registry"]).parent / "ledger" / cell / lease_id
                       / "accepted" / "g00000001.json")
        record = json.loads(record_path.read_text())
        record["ack"]["registry_event_sha256"] = "0" * 64
        write(record_path, record)
    with pytest.raises(ValueError):
        reregister.verify_transition(request_path, plan)


def test_prepare_rejects_cross_cell_partial_transition(tmp_path):
    reregister, request_path, request, _, _ = fixture(tmp_path)
    request["cells"] = request["cells"][:-1]
    write(request_path, request)
    with pytest.raises(ValueError, match="exact transition set"):
        reregister.prepare_transition(request_path)


def test_prepare_rejects_ack_from_different_registry_event(tmp_path):
    reregister, request_path, request, old_event, _ = fixture(tmp_path)
    cell = request["cells"][0]
    lease_id = old_event["owners"][cell]["lease_id"]
    record_path = (Path(request["registry"]).parent / "ledger" / cell / lease_id
                   / "accepted" / "g00000001.json")
    record = json.loads(record_path.read_text())
    record["ack"]["registry_event_sha256"] = "0" * 64
    write(record_path, record)
    with pytest.raises(ValueError, match="acknowledgement"):
        reregister.prepare_transition(request_path)


def test_v1_commit_is_historical_and_fails_closed(tmp_path):
    reregister, request_path, _, _, _, plan = prepare(tmp_path)
    output = tmp_path / "transition-output"
    with pytest.raises(RuntimeError, match="nondeployable|v2"):
        reregister.commit_transition(request_path, plan, output)
    assert not output.exists()


def test_old_registration_cannot_be_reused_for_new_lease(tmp_path):
    reregister, request_path, request, _, _, plan = prepare(tmp_path)
    cell = "worker-tid-small"
    old = {
        "kind": "p5-xnode-registration-v1", "writer_host": plan["event"]["owners"][cell]["owner_node"],
        "lease_sha256": plan["cells"][cell]["old_lease_sha256"],
        "event_sha256": plan["event"]["previous_event_sha256"],
        "input_manifest_sha256": plan["event"]["owners"][cell]["input_manifest_sha256"],
    }
    with pytest.raises(ValueError, match="old|stale|registration"):
        reregister.verify_worker_registration(old, plan, cell)


def test_worker_bootstrap_rejects_old_outbox_generation(tmp_path):
    reregister, _, _, _, _, plan = prepare(tmp_path)
    cell = "worker-tid-small"
    transport = tmp_path / "worker-run" / ".xnode"
    write(transport / "registration.json", reregister.worker_registration(plan, cell))
    write(transport / "outbox" / "g00000001-old" / "envelope.json", {
        "lease_sha256": plan["cells"][cell]["old_lease_sha256"],
    })
    with pytest.raises(ValueError, match="outbox|generation|transport"):
        reregister.verify_worker_bootstrap(transport, plan, cell)
    (transport / "outbox" / "g00000001-old" / "envelope.json").unlink()
    (transport / "outbox" / "g00000001-old").rmdir()
    (transport / "outbox").rmdir()
    assert reregister.verify_worker_bootstrap(transport, plan, cell)
