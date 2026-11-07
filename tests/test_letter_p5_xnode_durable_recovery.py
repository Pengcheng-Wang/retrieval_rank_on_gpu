import copy
import json
import shutil
from pathlib import Path

import pytest

from test_letter_p5_xnode_reregister import digest, sha, write
from test_letter_p5_xnode_reregister_v2 import prepare, v2_fixture


def reseal(package):
    package = Path(package)
    files = {
        path.relative_to(package).as_posix(): {
            "sha256": sha(path), "bytes": path.stat().st_size,
        }
        for path in package.rglob("*")
        if path.is_file() and path.name not in {"package-seal.json", "bank-receipt.json"}
    }
    write(package / "package-seal.json", {
        "kind": "p5-xnode-package-v1", "status": "sealed", "files": files,
    })
    receipt = {"files": {
        path.relative_to(package).as_posix(): {
            "sha256": sha(path), "bytes": path.stat().st_size,
        }
        for path in package.rglob("*")
        if path.is_file() and path.name != "bank-receipt.json"
    }}
    write(package / "bank-receipt.json", receipt)


def durable_chain(tmp_path, generations=3):
    v2, _, request_path, request, old_event, _ = v2_fixture(tmp_path)
    cell = "worker-tid-small"
    old_lease = old_event["owners"][cell]
    source_package = Path(request["evidence"][cell]["package"])
    durable = Path(request["new_bindings"]).parent.parent / (
        Path(request["new_bindings"]).parent.name + "-durable"
    ) / "worker-recovery" / cell
    packages = []
    parent = old_lease["root"]
    for generation in range(1, generations + 1):
        package = durable / f"g{generation:08d}-fixture"
        package.mkdir(parents=True)
        for path in source_package.rglob("*"):
            if path.is_file():
                target = package / path.relative_to(source_package)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
        envelope = json.loads((package / "envelope.json").read_text())
        checkpoint = copy.deepcopy(parent)
        checkpoint.update(
            artifact=f"durable-worker-state-{generation}",
            step=parent["step"] + 10,
        )
        seal = json.loads((package / "checkpoint" / "p5-seal.json").read_text())
        seal.update(
            global_step=checkpoint["step"],
        )
        (package / "checkpoint" / "model.safetensors").write_bytes(
            f"model-{generation}".encode()
        )
        (package / "checkpoint" / "training-state.pt").write_bytes(
            f"state-{generation}".encode()
        )
        seal["files_sha256"] = {
            "model.safetensors": sha(package / "checkpoint" / "model.safetensors"),
            "training-state.pt": sha(package / "checkpoint" / "training-state.pt"),
        }
        write(package / "checkpoint" / "p5-seal.json", seal)
        checkpoint.update(
            seal_sha256=sha(package / "checkpoint" / "p5-seal.json"),
            model_sha256=sha(package / "checkpoint" / "model.safetensors"),
            training_state_sha256=sha(package / "checkpoint" / "training-state.pt"),
        )
        envelope.update(
            generation=generation,
            parent_seal_sha256=parent["seal_sha256"],
            root_seal_sha256=old_lease["root"]["seal_sha256"],
            checkpoint=checkpoint,
        )
        write(package / "envelope.json", envelope)
        reseal(package)
        packages.append(str(package))
        parent = checkpoint
    evidence = {
        "kind": "p5-uuid-reregistration-worker-durable-root-v2",
        "cell": cell, "packages": packages,
        "contract": request["evidence"][cell]["contract"],
        "registry_event_sha256": v2.event_sha(old_event),
        "destination_root": str(durable),
        "selection": {"policy": "latest"},
        "successor": None,
    }
    request["evidence"][cell] = evidence
    proof = json.loads(Path(request["proofs"][cell]).read_text())
    proof.update(
        source_checkpoint_seal_sha256=parent["seal_sha256"],
        model_sha256=parent["model_sha256"],
        training_state_sha256=parent["training_state_sha256"],
    )
    write(Path(request["proofs"][cell]), proof)
    write(request_path, request)
    return v2, request_path, request, old_event, cell, packages


def test_durable_recovery_defaults_to_highest_contiguous_receipt_verified_package(tmp_path):
    v2, request_path, _, _, cell, packages = durable_chain(tmp_path)
    plan = v2.prepare_transition(request_path)
    metadata = plan["cells"][cell]
    assert metadata["evidence_kind"] == "worker-durable-recovery"
    assert metadata["historically_accepted"] is False
    assert metadata["selection_policy"] == "latest"
    assert metadata["chosen_generation"] == 3
    assert [item["path"] for item in metadata["verified_packages"]] == packages
    assert metadata["abandoned_verified_packages"] == []
    assert plan["event"]["durable_recovery"][cell] == metadata


def test_successor_without_ack_is_optional_evidence_not_acceptance(tmp_path):
    v2, request_path, request, _, cell, _ = durable_chain(tmp_path)
    request["evidence"][cell]["successor"] = {
        "generation": 4, "parent_seal_sha256": "0" * 64,
    }
    write(request_path, request)
    plan = v2.prepare_transition(request_path)
    assert plan["cells"][cell]["successor_observation"] == request["evidence"][cell]["successor"]
    assert plan["cells"][cell]["historically_accepted"] is False


@pytest.mark.parametrize("damage", [
    "missing-receipt", "partial-package", "gap", "divergence",
    "mixed-lease", "wrong-event", "wrong-destination", "unlisted-package",
])
def test_durable_recovery_rejects_crash_window_or_foreign_chain(tmp_path, damage):
    v2, request_path, request, old_event, cell, packages = durable_chain(tmp_path)
    package = Path(packages[1])
    if damage == "missing-receipt":
        (package / "bank-receipt.json").unlink()
    elif damage == "partial-package":
        (package / "checkpoint" / "training-state.pt").unlink()
    elif damage == "gap":
        request["evidence"][cell]["packages"].pop(1)
    elif damage == "divergence":
        envelope = json.loads((package / "envelope.json").read_text())
        envelope["parent_seal_sha256"] = "0" * 64
        write(package / "envelope.json", envelope)
        reseal(package)
    elif damage == "mixed-lease":
        envelope = json.loads((package / "envelope.json").read_text())
        envelope["lease_id"] = "foreign-lease"
        write(package / "envelope.json", envelope)
        reseal(package)
    elif damage == "wrong-event":
        request["evidence"][cell]["registry_event_sha256"] = "0" * 64
    elif damage == "wrong-destination":
        request["evidence"][cell]["destination_root"] = str(Path(packages[0]).parent.parent)
    else:
        extra = Path(packages[0]).parent / "g00000004-unlisted"
        extra.mkdir()
    write(request_path, request)
    with pytest.raises(ValueError):
        v2.prepare_transition(request_path)


def test_lower_durable_choice_requires_explicit_rollback_and_records_abandoned(tmp_path):
    v2, request_path, request, _, cell, packages = durable_chain(tmp_path)
    request["evidence"][cell]["selection"] = {"policy": "latest", "generation": 2}
    write(request_path, request)
    with pytest.raises(ValueError, match="rollback|latest"):
        v2.prepare_transition(request_path)
    request["evidence"][cell]["selection"] = {
        "policy": "intentional_rollback", "generation": 2,
        "reason": "operator-selected scientific rollback",
    }
    proof = json.loads(Path(request["proofs"][cell]).read_text())
    chosen = json.loads((Path(packages[1]) / "envelope.json").read_text())["checkpoint"]
    proof.update(
        source_checkpoint_seal_sha256=chosen["seal_sha256"],
        model_sha256=chosen["model_sha256"],
        training_state_sha256=chosen["training_state_sha256"],
    )
    write(Path(request["proofs"][cell]), proof)
    write(request_path, request)
    plan = v2.prepare_transition(request_path)
    assert plan["cells"][cell]["selection_policy"] == "intentional_rollback"
    assert [
        item["path"] for item in plan["cells"][cell]["abandoned_verified_packages"]
    ] == [packages[2]]


def test_durable_recovery_does_not_require_or_reconstruct_old_accepted_ledger(tmp_path):
    v2, request_path, request, _, cell, _ = durable_chain(tmp_path)
    ledger = Path(request["registry"]).parent / "ledger" / cell
    shutil.rmtree(ledger)
    v2.prepare_transition(request_path)
    assert not ledger.exists()


def test_committed_certificate_labels_receipt_verified_durable_recovery(tmp_path, monkeypatch):
    v2, request_path, _, _, cell, _ = durable_chain(tmp_path)
    plan = v2.prepare_transition(request_path)
    monkeypatch.setattr(v2.socket, "gethostname", lambda: plan["bindings"]["writer_host"])
    result = v2.commit_transition(request_path, plan)
    certificate = json.loads(Path(result["certificate"]).read_text())
    assert certificate["recovery"]["kind"] == "p5-ownership-durable-recovery-v1"
    assert certificate["recovery"]["cells"][cell]["historically_accepted"] is False
    assert certificate["event"]["durable_recovery"][cell] == plan["cells"][cell]
