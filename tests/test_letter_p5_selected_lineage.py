import importlib.util
import json
import shutil
import socket
import sys
from pathlib import Path

import torch
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "letter"
sys.path.insert(0, str(SCRIPTS))


def module():
    path = SCRIPTS / "selected_lineage_materialize.py"
    assert path.exists(), "A selected-lineage materialization tool is required"
    spec = importlib.util.spec_from_file_location("selected_lineage_materialize_test", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def checkpoint(path, contract, state):
    path = Path(path)
    path.mkdir(parents=True)
    (path / "model.safetensors").write_bytes(("model-" + path.name).encode())
    torch.save({
        "schema_version": 1,
        "contract": contract,
        "optimizer": {},
        "scheduler": {},
        "sampler": {},
        "rng": {},
        "state": state,
        "checkpoint_boundary": "after_complete_optimizer_update_no_partial_gradients",
    }, path / "training-state.pt")
    import p5_source

    write(path / "p5-seal.json", {
        "kind": "p5-full-state-v1",
        "status": "sealed",
        "contract_sha256": p5_source.digest_json(contract),
        "global_step": state["global_step"],
        "epoch": state["epoch"],
        "phase": state["phase"],
        "files_sha256": {
            name: p5_source.sha256(path / name)
            for name in ("model.safetensors", "training-state.pt")
        },
    })
    return path


def bank_receipt(path):
    import p5_source

    path = Path(path)
    files = {
        member.relative_to(path).as_posix(): {
            "sha256": p5_source.sha256(member),
            "bytes": member.stat().st_size,
        }
        for member in path.rglob("*")
        if member.is_file() and member.name != "bank-receipt.json"
    }
    write(path / "bank-receipt.json", {"files": files})


def fixture(tmp_path, role="writer"):
    import p5_source
    import p5_xnode_reregister_v2 as v2

    host = "writer-host" if role == "writer" else "worker-host"
    cell = f"{role}-cell"
    p5_root = tmp_path / "p5"
    xroot = p5_root / "xnode-v1"
    contract = {"cell": cell, "recipe": "fixture"}
    selected_name = "p5-state-000000010-001-selected"
    history = [{"primary": {"recall@10": 0.5}}]
    selected_state = {
        "global_step": 10,
        "epoch": 1,
        "phase": "train",
        "selected_artifact": selected_name,
        "best_metric": 0.5,
        "validation_history": history,
    }
    root_state = {
        "global_step": 20,
        "epoch": 2,
        "phase": "train",
        "selected_artifact": selected_name,
        "best_metric": 0.5,
        "validation_history": history,
    }
    durable = tmp_path / "durable"
    if role == "writer":
        selected = checkpoint(durable / selected_name, contract, selected_state)
        bank_receipt(selected)
        source = selected
    else:
        source = durable / "g00000001-selected"
        selected = checkpoint(source / "checkpoint", contract, selected_state)
        write(source / "scientific-contract.json", contract)
        write(source / "execution-attestation.json", {"purpose": "production"})
        selected_seal = json.loads((selected / "p5-seal.json").read_text())
        write(source / "envelope.json", {
            "generation": 1,
            "checkpoint": {
                "artifact": selected_name,
                "seal_sha256": p5_source.sha256(selected / "p5-seal.json"),
                "contract_sha256": selected_seal["contract_sha256"],
                "step": selected_seal["global_step"],
                "epoch": selected_seal["epoch"],
                "phase": selected_seal["phase"],
                "model_sha256": selected_seal["files_sha256"]["model.safetensors"],
                "training_state_sha256": selected_seal["files_sha256"]["training-state.pt"],
            },
        })
        package_files = {
            member.relative_to(source).as_posix(): {
                "sha256": p5_source.sha256(member),
                "bytes": member.stat().st_size,
            }
            for member in source.rglob("*")
            if member.is_file()
        }
        write(source / "package-seal.json", {
            "kind": "p5-xnode-package-v1",
            "status": "sealed",
            "files": package_files,
        })
        bank_receipt(source)
    root = checkpoint(tmp_path / "root-checkpoint", contract, root_state)
    contract_path = tmp_path / "p5-run-config.json"
    write(contract_path, contract)

    runtime = {}
    (xroot / "current").mkdir(parents=True)
    for name in ("p5_xnode_control.py", "p5_xnode_state.py"):
        shutil.copyfile(SCRIPTS / name, xroot / "current" / name)
        runtime[name] = p5_source.sha256(xroot / "current" / name)
    runtime_manifest = {"files_sha256": runtime}
    write(xroot / "current" / "runtime-manifest.json", runtime_manifest)
    proof = {"runtime_manifest_sha256": v2.document_sha(runtime_manifest)}
    write(xroot / "current" / "crossnode-proof.json", proof)
    bindings = {
        "control_root": str(xroot),
        "p5_root": str(p5_root),
        "writer_host": "writer-host",
        "worker_host": "worker-host",
        "cells": {cell: {"role": role}},
    }
    write(xroot / "current" / "bindings.json", bindings)
    root_seal = json.loads((root / "p5-seal.json").read_text())
    root_header = {
        "artifact": root.name,
        "seal_sha256": p5_source.sha256(root / "p5-seal.json"),
        "contract_sha256": root_seal["contract_sha256"],
        "step": root_seal["global_step"],
        "epoch": root_seal["epoch"],
        "phase": root_seal["phase"],
        "model_sha256": root_seal["files_sha256"]["model.safetensors"],
        "training_state_sha256": root_seal["files_sha256"]["training-state.pt"],
    }
    lease = {
        "cell": cell,
        "owner_node": host,
        "control_manifest_sha256": v2.document_sha(runtime_manifest),
        "contract_file_sha256": p5_source.sha256(contract_path),
        "root": root_header,
    }
    cell_metadata = (
        {
            "evidence_kind": "writer-durable-recovery",
            "verified_checkpoints": [{
                "path": str(source.resolve()),
                "checkpoint_seal_sha256": p5_source.sha256(
                    selected / "p5-seal.json"
                ),
                "bank_receipt_sha256": p5_source.sha256(
                    source / "bank-receipt.json"
                ),
            }],
        } if role == "writer" else {
            "evidence_kind": "worker-durable-recovery",
            "verified_packages": [{
                "path": str(source.resolve()),
                "checkpoint_seal_sha256": p5_source.sha256(
                    selected / "p5-seal.json"
                ),
                "package_seal_sha256": p5_source.sha256(
                    source / "package-seal.json"
                ),
                "bank_receipt_sha256": p5_source.sha256(
                    source / "bank-receipt.json"
                ),
            }],
        }
    )
    event = {
        "sequence": 4,
        "transition_cells": [cell],
        "owners": {cell: lease},
        "durable_recovery": {cell: cell_metadata},
        "broker_release_sha256": "a" * 64,
        "crossnode_proof_sha256": v2.document_sha(proof),
    }
    event_sha = v2.event_sha(event)
    certificate = {
        "event": event,
        "event_sha256": event_sha,
        "writer_host": "writer-host",
    }
    write(xroot / "current" / "ownership-certificate.json", certificate)
    write(xroot / "leases" / f"{cell}.json", lease)
    canonical_root = xroot / "roots" / cell
    (canonical_root / "checkpoint").mkdir(parents=True)
    for name in ("model.safetensors", "training-state.pt", "p5-seal.json"):
        shutil.copyfile(root / name, canonical_root / "checkpoint" / name)
    shutil.copyfile(contract_path, canonical_root / "p5-run-config.json")
    latest = {
        "artifact": root.name,
        "seal_sha256": root_header["seal_sha256"],
        "global_step": root_header["step"],
        "phase": root_header["phase"],
    }
    write(canonical_root / "latest.json", latest)
    if role == "worker":
        write(xroot / "worker-bootstrap" / cell / ".xnode" / "registration.json", {
            "kind": "p5-xnode-registration-v1",
            "writer_host": "writer-host",
            "lease_sha256": v2.digest(lease),
            "event_sha256": event_sha,
            "input_manifest_sha256": "b" * 64,
        })
    activation_files = {}
    activation_paths = [
        "current/ownership-certificate.json",
        "current/runtime-manifest.json",
        "current/crossnode-proof.json",
        "current/bindings.json",
        "current/p5_xnode_control.py",
        "current/p5_xnode_state.py",
        f"leases/{cell}.json",
        f"roots/{cell}/checkpoint/model.safetensors",
        f"roots/{cell}/checkpoint/training-state.pt",
        f"roots/{cell}/checkpoint/p5-seal.json",
        f"roots/{cell}/p5-run-config.json",
        f"roots/{cell}/latest.json",
    ]
    if role == "worker":
        activation_paths.append(
            f"worker-bootstrap/{cell}/.xnode/registration.json"
        )
    for relative in activation_paths:
        activation_files[relative] = p5_source.sha256(xroot / relative)
    write(xroot / "activation" / "last-published.json", {
        "kind": "p5-xnode-activation-v1",
        "status": "activated",
        "sequence": 4,
        "event_sha256": event_sha,
        "broker_release_sha256": event["broker_release_sha256"],
        "files_sha256": activation_files,
    })
    plan = {
        "kind": "p5-uuid-reregistration-plan-v2",
        "status": "prepared",
        "event": event,
        "event_sha256": event_sha,
        "bindings": bindings,
        "cells": {cell: cell_metadata},
    }
    plan_path = tmp_path / "plan.json"
    write(plan_path, plan)
    certificate["transition_plan_sha256"] = v2.digest(plan)
    write(xroot / "current" / "ownership-certificate.json", certificate)
    activation_files["current/ownership-certificate.json"] = p5_source.sha256(
        xroot / "current" / "ownership-certificate.json"
    )
    write(xroot / "activation" / "last-published.json", {
        "kind": "p5-xnode-activation-v1",
        "status": "activated",
        "sequence": 4,
        "event_sha256": event_sha,
        "broker_release_sha256": event["broker_release_sha256"],
        "files_sha256": activation_files,
    })
    request = {
        "kind": "p5-selected-lineage-materialization-request-v1",
        "transition_plan": str(plan_path),
        "event_sha256": event_sha,
        "broker_release_sha256": event["broker_release_sha256"],
        "cells": {cell: {"source": str(source.resolve())}},
    }
    request_path = tmp_path / "request.json"
    write(request_path, request)
    return request_path, p5_root / "runs" / cell, cell, host


def test_materializes_selected_lineage_bound_to_activated_transition(tmp_path, monkeypatch):
    selected = module()
    request_path, run, cell, host = fixture(tmp_path)
    monkeypatch.setattr(socket, "gethostname", lambda: host)

    result = selected.materialize_selected_lineage(request_path)

    assert result["status"] == "materialized"
    assert result["cells"] == [cell]
    assert json.loads((run / "selected.json").read_text())["artifact"] == (
        "p5-state-000000010-001-selected"
    )
    receipt = json.loads((run / "selected-lineage-receipt.json").read_text())
    assert receipt["kind"] == "p5-selected-lineage-materialization-v1"
    assert receipt["cell"] == cell
    assert receipt["selected"]["artifact"] == "p5-state-000000010-001-selected"
    assert selected.verify_selected_lineage(request_path)["status"] == "verified"


def test_materializes_worker_selected_package_and_clean_registration(tmp_path, monkeypatch):
    selected = module()
    request_path, run, cell, host = fixture(tmp_path, role="worker")
    monkeypatch.setattr(socket, "gethostname", lambda: host)

    selected.materialize_selected_lineage(request_path)

    assert json.loads((run / "selected.json").read_text())["artifact"] == (
        "p5-state-000000010-001-selected"
    )
    assert {path.name for path in (run / ".xnode").iterdir()} == {
        "registration.json"
    }
    assert json.loads(
        (run / "selected-lineage-receipt.json").read_text()
    )["source"]["kind"] == "worker-package"


def test_aggregates_exact_per_cell_receipts_for_deployment(tmp_path, monkeypatch):
    selected = module()
    request_path, run, cell, host = fixture(tmp_path)
    monkeypatch.setattr(socket, "gethostname", lambda: host)
    selected.materialize_selected_lineage(request_path)
    output = tmp_path / "selected-lineage-deployment.json"

    result = selected.aggregate_selected_lineage(
        request_path,
        [run / "selected-lineage-receipt.json"],
        output,
    )

    assert result["status"] == "verified"
    deployment = json.loads(output.read_text())
    assert deployment["kind"] == "p5-selected-lineage-deployment-v1"
    assert set(deployment["cells"]) == {cell}


def test_verify_rejects_receipt_with_unbound_selected_source(tmp_path, monkeypatch):
    selected = module()
    request_path, run, _, host = fixture(tmp_path)
    monkeypatch.setattr(socket, "gethostname", lambda: host)
    selected.materialize_selected_lineage(request_path)
    receipt_path = run / "selected-lineage-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["source"]["path"] = str(tmp_path / "foreign-checkpoint")
    write(receipt_path, receipt)

    with pytest.raises(ValueError, match="source|receipt|authority"):
        selected.verify_selected_lineage(request_path)


def test_rejects_plan_fields_not_authenticated_by_certificate(tmp_path, monkeypatch):
    selected = module()
    request_path, _, _, host = fixture(tmp_path)
    monkeypatch.setattr(socket, "gethostname", lambda: host)
    request = json.loads(request_path.read_text())
    plan_path = Path(request["transition_plan"])
    plan = json.loads(plan_path.read_text())
    plan["bindings"]["p5_root"] = str(tmp_path / "redirected-p5")
    write(plan_path, plan)

    with pytest.raises(ValueError, match="bound|plan|certificate|binding|authority"):
        selected.materialize_selected_lineage(request_path)


def test_verify_rejects_internally_valid_selected_checkpoint_with_foreign_seal(
    tmp_path, monkeypatch,
):
    selected = module()
    request_path, run, _, host = fixture(tmp_path)
    monkeypatch.setattr(socket, "gethostname", lambda: host)
    selected.materialize_selected_lineage(request_path)
    pointer_path = run / "selected.json"
    pointer = json.loads(pointer_path.read_text())
    checkpoint_path = run / "states" / pointer["artifact"]
    (checkpoint_path / "model.safetensors").write_bytes(b"foreign-model")
    seal_path = checkpoint_path / "p5-seal.json"
    seal = json.loads(seal_path.read_text())
    import p5_source

    seal["files_sha256"]["model.safetensors"] = p5_source.sha256(
        checkpoint_path / "model.safetensors"
    )
    write(seal_path, seal)
    pointer["seal_sha256"] = p5_source.sha256(seal_path)
    write(pointer_path, pointer)
    receipt_path = run / "selected-lineage-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["selected"]["seal_sha256"] = pointer["seal_sha256"]
    receipt["files"] = {
        member.relative_to(run).as_posix(): {
            "sha256": p5_source.sha256(member),
            "bytes": member.stat().st_size,
        }
        for member in run.rglob("*")
        if member.is_file() and member.name != "selected-lineage-receipt.json"
    }
    write(receipt_path, receipt)

    with pytest.raises(ValueError, match="seal|source|checkpoint|pointer"):
        selected.verify_selected_lineage(request_path)


def test_aggregate_rejects_tampered_selected_receipt_fields(tmp_path, monkeypatch):
    selected = module()
    request_path, run, _, host = fixture(tmp_path)
    monkeypatch.setattr(socket, "gethostname", lambda: host)
    selected.materialize_selected_lineage(request_path)
    receipt_path = run / "selected-lineage-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["selected"]["artifact"] = "p5-state-000000999-999-foreign"
    write(receipt_path, receipt)

    with pytest.raises(ValueError, match="selected|receipt|source|authority"):
        selected.aggregate_selected_lineage(
            request_path,
            [receipt_path],
            tmp_path / "deployment.json",
        )


def test_fsyncs_staged_directory_tree_before_atomic_publication(tmp_path, monkeypatch):
    selected = module()
    request_path, run, _, host = fixture(tmp_path)
    monkeypatch.setattr(socket, "gethostname", lambda: host)
    calls = []
    monkeypatch.setattr(
        selected.p5_checkpoint,
        "fsync_directory",
        lambda path: calls.append(Path(path)),
    )

    selected.materialize_selected_lineage(request_path)

    staging_calls = [
        path for path in calls
        if path.name.startswith(run.name + ".partial-")
        or any(part.startswith(run.name + ".partial-") for part in path.parts)
    ]
    assert staging_calls
    assert any(path.name == "states" for path in staging_calls)
    assert calls.index(staging_calls[-1]) < calls.index(run.parent)
