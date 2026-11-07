import copy
import importlib.util
import json
import shutil
import socket
import sys
import uuid
from pathlib import Path

import pytest

from test_letter_p5_xnode_reregister import sha, write
from test_letter_p5_xnode_reregister_v2 import prepare as prepare_v2


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "letter"
sys.path.insert(0, str(SCRIPTS))


def module():
    path = SCRIPTS / "p5_protocol_repair.py"
    assert path.exists(), "A versioned P5 protocol-repair transition is required"
    spec = importlib.util.spec_from_file_location("p5_protocol_repair_test", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def bank(directory):
    directory = Path(directory)
    receipt = {"files": {
        path.relative_to(directory).as_posix(): {
            "sha256": sha(path),
            "bytes": path.stat().st_size,
        }
        for path in directory.rglob("*")
        if path.is_file() and path.name != "bank-receipt.json"
    }}
    write(directory / "bank-receipt.json", receipt)
    return receipt


def repair_fixture(tmp_path, monkeypatch, repair):
    v2, parent_request_path, parent_request, _, _, parent = prepare_v2(
        tmp_path / "parent"
    )
    monkeypatch.setattr(socket, "gethostname", lambda: parent["bindings"]["writer_host"])
    v2.commit_transition(parent_request_path, parent)
    xroot = Path(parent["bindings"]["control_root"])
    active = parent["event"]["transition_cells"]

    bindings = copy.deepcopy(parent["bindings"])
    for cell in active:
        bindings["cells"][cell]["ownership_epoch"] += 1
    bindings_path = tmp_path / "new-bindings.json"
    write(bindings_path, bindings)

    runtime = tmp_path / "runtime-v3"
    runtime.mkdir()
    for name in ("p5_xnode_control.py", "p5_xnode_state.py"):
        shutil.copyfile(SCRIPTS / name, runtime / name)
    with (runtime / "p5_xnode_state.py").open("a", encoding="utf-8") as stream:
        stream.write("\n# protocol-repair fixture\n")
    template = copy.deepcopy(json.loads(Path(parent_request["runtime_template"]).read_text()))
    template["files_sha256"] = {
        name: sha(runtime / name)
        for name in ("p5_xnode_control.py", "p5_xnode_state.py")
    }
    template_path = runtime / "runtime-template.json"
    write(template_path, template)
    runtime_manifest = v2._runtime_manifest(
        {"runtime_template": str(template_path)}, bindings,
    )
    runtime_manifest["ownership_registry"] = str(
        Path(parent_request["registry"]).resolve()
    )

    broker_dir = tmp_path / "broker-v4"
    broker_dir.mkdir()
    for name in ("p5_xnode_broker.py", "p5_xnode_state.py", "p5_xnode_adopt.py"):
        source = (
            runtime / name
            if name == "p5_xnode_state.py"
            else SCRIPTS / name
        )
        shutil.copyfile(source, broker_dir / name)
    broker_release = {
        "kind": "p5-xnode-broker-v4",
        "files_sha256": {
            name: sha(broker_dir / name)
            for name in (
                "p5_xnode_broker.py", "p5_xnode_state.py", "p5_xnode_adopt.py",
            )
        },
        "runtime_manifest_sha256": v2.document_sha(runtime_manifest),
    }
    broker_release_path = broker_dir / "broker-release.json"
    write(broker_release_path, broker_release)
    validation = {
        "kind": "p5-broker-v4-validation-v1",
        "status": "pass",
        "broker_release_sha256": sha(broker_release_path),
        "runtime_manifest_sha256": v2.document_sha(runtime_manifest),
        "checks": {
            "dynamic_binding_api": True,
            "legacy_sequence_root": True,
            "selected_readmission": True,
            "active_cell_fence": True,
        },
    }
    validation_path = broker_dir / "broker-validation.json"
    write(validation_path, validation)

    evidence = Path(bindings["durable_root"]) / "selected-lineage-evidence"
    receipts = evidence / "receipts"
    receipts.mkdir(parents=True)
    cells = {}
    for cell in active:
        owner = parent["event"]["owners"][cell]
        selected = copy.deepcopy(owner["root"])
        cells[cell] = {
            "host": owner["owner_node"],
            "role": bindings["cells"][cell]["role"],
            "receipt_sha256": "",
            "root": {
                "artifact": owner["root"]["artifact"],
                "seal_sha256": owner["root"]["seal_sha256"],
            },
            "selected": {
                "artifact": selected["artifact"],
                "seal_sha256": selected["seal_sha256"],
                "best_metric": 0.5,
            },
        }
        receipt = {
            "kind": "p5-selected-lineage-materialization-v1",
            "status": "materialized",
            "cell": cell,
            "event_sha256": parent["event_sha256"],
            "root": cells[cell]["root"],
            "selected": cells[cell]["selected"],
            "source": {
                "kind": "writer-checkpoint",
                "path": parent["roots"][cell]["checkpoint"],
                "artifact": selected["artifact"],
                "checkpoint_seal_sha256": selected["seal_sha256"],
                "bank_receipt_sha256": "0" * 64,
            },
        }
        receipt_path = receipts / f"{cell}.json"
        write(receipt_path, receipt)
        cells[cell]["receipt_sha256"] = sha(receipt_path)
    deployment = {
        "kind": "p5-selected-lineage-deployment-v1",
        "status": "verified",
        "event_sha256": parent["event_sha256"],
        "activation_sha256": sha(xroot / "activation" / "last-published.json"),
        "cells": cells,
    }
    write(evidence / "deployment.json", deployment)
    packet_plan = evidence / "plan-v2.json"
    write(packet_plan, parent)
    write(evidence / "request.json", {
        "kind": "p5-selected-lineage-materialization-request-v1",
        "transition_plan": str(packet_plan),
        "event_sha256": parent["event_sha256"],
        "broker_release_sha256": parent["event"]["broker_release_sha256"],
        "cells": {
            cell: {"source": parent["roots"][cell]["checkpoint"]}
            for cell in active
        },
    })
    bank(evidence)
    monkeypatch.setattr(
        repair.selected_lineage,
        "selected_lineage_deployment",
        lambda request_path, receipt_paths: copy.deepcopy(deployment),
    )

    request = {
        "kind": "p5-protocol-repair-request-v1",
        "parent_event_sha256": parent["event_sha256"],
        "parent_plan": str(tmp_path / "parent-plan.json"),
        "active_cells": active,
        "new_bindings": str(bindings_path),
        "registry": parent_request["registry"],
        "runtime_template": str(template_path),
        "broker_release": str(broker_release_path),
        "broker_validation": str(validation_path),
        "selected_lineage_evidence": str(evidence),
        "selected_readmission": str(
            Path(bindings["root"]) / "p5" / "protocol-repair"
            / "selected-readmission.json"
        ),
        "readmission_durable_root": str(
            Path(bindings["durable_root"]) / "selected-readmission-v1"
        ),
        "proofs": parent_request["proofs"],
        "lease_ids": {cell: uuid.uuid4().hex for cell in active},
    }
    request_path = tmp_path / "repair-request.json"
    write(Path(request["parent_plan"]), parent)
    write(request_path, request)
    return v2, request_path, request, parent, xroot


def build_readmission(repair, request_path, request, draft, tmp_path):
    authority = {
        "kind": "p5-selected-lineage-readmission-v1",
        "status": "verified",
        "parent_event_sha256": draft["parent_event_sha256"],
        "selected_lineage_deployment_sha256":
            draft["selected_lineage_deployment_sha256"],
        "cells": {},
    }
    for cell in draft["active_cells"]:
        lease = draft["owners"][cell]
        if lease["owner_node"] == draft["bindings"]["writer_host"]:
            continue
        selected = draft["selected"][cell]
        package = (
            Path(request["selected_readmission"]).parent / "packages" / cell
        )
        checkpoint = Path(draft["roots"][cell]["checkpoint"])
        (package / "checkpoint").mkdir(parents=True)
        for name in ("model.safetensors", "training-state.pt", "p5-seal.json"):
            shutil.copyfile(checkpoint / name, package / "checkpoint" / name)
        shutil.copyfile(
            draft["roots"][cell]["contract"],
            package / "scientific-contract.json",
        )
        attestation = {
            "kind": "p5-selected-lineage-readmission-attestation-v1",
            "historically_accepted": False,
            "source_event_sha256": draft["parent_event_sha256"],
            "selected_lineage_deployment_sha256":
                draft["selected_lineage_deployment_sha256"],
            "lease_sha256": repair.digest(lease),
            "root_checkpoint_seal_sha256": lease["root"]["seal_sha256"],
            "selected_checkpoint_seal_sha256": selected["seal_sha256"],
            "scientific_contract_rewritten": False,
        }
        write(package / "execution-attestation.json", attestation)
        write(package / "envelope.json", {
            "kind": "p5-xnode-selected-readmission-v1",
            "purpose": "production",
            **{
                key: lease[key]
                for key in ("cell", "owner_node", "gpu", "lease_id", "ownership_epoch")
            },
            "lease_sha256": repair.digest(lease),
            "wire_generation": 0,
            "root_seal_sha256": lease["root"]["seal_sha256"],
            "checkpoint": selected,
            "selected": selected,
            "execution_attestation_sha256": sha(
                package / "execution-attestation.json"
            ),
        })
        write(package / "package-seal.json", {
            "kind": "p5-xnode-package-v1",
            "status": "sealed",
            "files": repair.v1._file_manifest(package, {"package-seal.json"}),
        })
        destination = Path(request["readmission_durable_root"]) / cell
        shutil.copytree(package, destination)
        bank(destination)
        authority["cells"][cell] = {
            "historically_accepted": False,
            "lease_sha256": repair.digest(lease),
            "root_seal_sha256": lease["root"]["seal_sha256"],
            "selected": selected,
            "package": str(package.resolve()),
            "package_seal_sha256": sha(package / "package-seal.json"),
            "destination": str(destination.resolve()),
            "bank_receipt_sha256": sha(destination / "bank-receipt.json"),
        }
    path = Path(request["selected_readmission"])
    write(path, authority)
    request["selected_readmission"] = str(path)
    write(request_path, request)
    return path


def test_protocol_repair_draft_changes_runtime_and_epoch_not_scientific_root(
    tmp_path, monkeypatch,
):
    repair = module()
    _, request_path, _, parent, _ = repair_fixture(tmp_path, monkeypatch, repair)

    draft = repair.draft_repair(request_path)

    assert draft["parent_event_sha256"] == parent["event_sha256"]
    assert draft["active_cells"] == parent["event"]["transition_cells"]
    assert draft["runtime_manifest_sha256"] != parent["runtime_manifest_sha256"]
    assert draft["runtime_manifest"]["ownership_registry"] == json.loads(
        request_path.read_text()
    )["registry"]
    for cell in draft["active_cells"]:
        old = parent["event"]["owners"][cell]
        new = draft["owners"][cell]
        assert new["root"] == old["root"]
        assert new["gpu_uuid"] == old["gpu_uuid"]
        assert new["ownership_epoch"] == old["ownership_epoch"] + 1
        assert new["lease_id"] != old["lease_id"]


def test_protocol_repair_rejects_substituted_parent_proof(tmp_path, monkeypatch):
    repair = module()
    _, request_path, request, _, _ = repair_fixture(tmp_path, monkeypatch, repair)
    cell = request["active_cells"][0]
    proof_path = Path(request["proofs"][cell])
    proof = json.loads(proof_path.read_text())
    proof["untrusted_annotation"] = "same claims, different document"
    write(proof_path, proof)

    with pytest.raises(ValueError, match="proof|parent|authenticated"):
        repair.draft_repair(request_path)


def test_protocol_repair_rejects_relabelled_selected_deployment(
    tmp_path, monkeypatch,
):
    repair = module()
    _, request_path, request, _, _ = repair_fixture(
        tmp_path, monkeypatch, repair,
    )
    evidence = Path(request["selected_lineage_evidence"])
    deployment = json.loads((evidence / "deployment.json").read_text())
    cell = request["active_cells"][0]
    recomputed = copy.deepcopy(deployment)
    recomputed["cells"][cell]["selected"]["artifact"] = "foreign-selected"
    monkeypatch.setattr(
        repair.selected_lineage,
        "selected_lineage_deployment",
        lambda request_path, receipt_paths: recomputed,
    )

    with pytest.raises(ValueError, match="selected|deployment|parent"):
        repair.draft_repair(request_path)


def test_protocol_repair_finalizes_readmission_bound_sequence(tmp_path, monkeypatch):
    repair = module()
    _, request_path, request, parent, _ = repair_fixture(tmp_path, monkeypatch, repair)
    draft = repair.draft_repair(request_path)
    readmission = build_readmission(
        repair, request_path, request, draft, tmp_path,
    )

    plan = repair.prepare_repair(request_path)

    assert plan["event"]["sequence"] == parent["event"]["sequence"] + 1
    assert plan["event"]["previous_event_sha256"] == parent["event_sha256"]
    assert plan["event"]["active_cells"] == draft["active_cells"]
    assert plan["event"]["owners"] == draft["owners"]
    assert plan["event"]["selected_lineage_readmission_sha256"] == (
        repair.document_sha(json.loads(readmission.read_text()))
    )
    assert plan["event"]["selected_lineage_deployment"]["cells"] == draft["active_cells"]


def test_protocol_repair_builds_nonhistorical_worker_readmission(
    tmp_path, monkeypatch,
):
    repair = module()
    _, request_path, request, _, _ = repair_fixture(tmp_path, monkeypatch, repair)

    authority = repair.build_readmission(request_path)

    assert authority["kind"] == "p5-selected-lineage-readmission-v1"
    assert authority["status"] == "verified"
    for cell, entry in authority["cells"].items():
        assert entry["historically_accepted"] is False
        package = Path(entry["package"])
        envelope = json.loads((package / "envelope.json").read_text())
        assert envelope["wire_generation"] == 0
        assert "generation" not in envelope
        assert "parent_seal_sha256" not in envelope
        assert (
            Path(entry["destination"]) / "bank-receipt.json"
        ).is_file()
    assert json.loads(
        Path(request["selected_readmission"]).read_text()
    ) == authority


def test_protocol_repair_commit_activates_new_runtime_without_recopying_roots(
    tmp_path, monkeypatch,
):
    repair = module()
    _, request_path, request, _, xroot = repair_fixture(tmp_path, monkeypatch, repair)
    draft = repair.draft_repair(request_path)
    build_readmission(repair, request_path, request, draft, tmp_path)
    plan = repair.prepare_repair(request_path)
    old_root_hashes = {
        path.relative_to(xroot).as_posix(): sha(path)
        for path in xroot.joinpath("roots").rglob("*")
        if path.is_file()
    }

    result = repair.commit_repair(request_path, plan)

    assert result["status"] == "activated"
    activation = json.loads(
        (xroot / "activation" / "last-published.json").read_text()
    )
    assert activation["kind"] == "p5-xnode-activation-v2"
    assert activation["event_sha256"] == plan["event_sha256"]
    assert json.loads(
        (xroot / "current" / "selected-lineage-readmission.json").read_text()
    ) == plan["selected_readmission"]
    assert old_root_hashes == {
        path.relative_to(xroot).as_posix(): sha(path)
        for path in xroot.joinpath("roots").rglob("*")
        if path.is_file()
    }
    for cell in plan["active_cells"]:
        assert json.loads(
            (xroot / "leases" / f"{cell}.json").read_text()
        ) == plan["owners"][cell]
        if plan["bindings"]["cells"][cell]["role"] == "worker":
            registration = json.loads(
                (xroot / "worker-bootstrap" / cell / ".xnode"
                 / "registration.json").read_text()
            )
            assert registration["event_sha256"] == plan["event_sha256"]
            assert registration["lease_sha256"] == repair.digest(
                plan["owners"][cell]
            )
    assert repair.verify_activation(
        xroot, plan["broker_release_sha256"]
    )["event_sha256"] == plan["event_sha256"]


def test_interrupted_protocol_repair_requires_exact_activation_recovery(
    tmp_path, monkeypatch,
):
    repair = module()
    _, request_path, request, _, xroot = repair_fixture(tmp_path, monkeypatch, repair)
    draft = repair.draft_repair(request_path)
    build_readmission(repair, request_path, request, draft, tmp_path)
    plan = repair.prepare_repair(request_path)

    with pytest.raises(RuntimeError, match="activation"):
        repair.commit_repair(
            request_path, plan, fail_before_activation=True,
        )
    with pytest.raises(ValueError, match="activation|authority"):
        repair.verify_activation(xroot, plan["broker_release_sha256"])

    modified = copy.deepcopy(plan)
    modified["bindings"]["legacy_device_guard"] = "tampered guard"
    with pytest.raises(ValueError, match="plan|intent|committed"):
        repair.activate_repair(request_path, modified)

    result = repair.activate_repair(request_path, plan)
    assert result["status"] == "activated"
    assert repair.verify_activation(
        xroot, plan["broker_release_sha256"]
    )["event_sha256"] == plan["event_sha256"]
    assert repair.activate_repair(request_path, plan)["status"] == "activated"


@pytest.mark.parametrize("failure", [
    "after-intent", "durable-partial", "after-certificate",
    "registration-partial",
])
def test_protocol_repair_commit_resumes_early_crash_windows(
    tmp_path, monkeypatch, failure,
):
    repair = module()
    _, request_path, request, _, xroot = repair_fixture(tmp_path, monkeypatch, repair)
    draft = repair.draft_repair(request_path)
    build_readmission(repair, request_path, request, draft, tmp_path)
    plan = repair.prepare_repair(request_path)

    injected = (
        "after-intent"
        if failure == "durable-partial"
        else "after-certificate"
        if failure == "registration-partial"
        else failure
    )
    with pytest.raises(RuntimeError, match="Injected"):
        repair.commit_repair(request_path, plan, fail_at=injected)
    if failure == "durable-partial":
        durable = (
            Path(plan["bindings"]["durable_root"])
            / "ownership-protocol-repair-v1" / plan["event_sha256"]
        )
        durable.mkdir(parents=True)
        (durable / "ownership.json.partial-dead").write_bytes(b"partial")
    elif failure == "registration-partial":
        for cell in plan["active_cells"]:
            if plan["bindings"]["cells"][cell]["role"] == "worker":
                partial = (
                    xroot / "worker-bootstrap" / cell / ".xnode"
                    / "registration.json.partial-dead"
                )
                partial.parent.mkdir(parents=True, exist_ok=True)
                partial.write_bytes(b"partial")

    result = repair.commit_repair(request_path, plan)
    assert result["status"] == "activated"
    assert repair.verify_activation(
        xroot, plan["broker_release_sha256"]
    )["event_sha256"] == plan["event_sha256"]
