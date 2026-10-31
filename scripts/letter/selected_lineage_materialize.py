"""Materialize validation-selected P5 state for an activated ownership event."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import uuid
from pathlib import Path

import p5_checkpoint
import p5_source
import p5_xnode_reregister as v1
import p5_xnode_reregister_v2 as v2


REQUEST_FIELDS = {
    "kind", "transition_plan", "event_sha256", "broker_release_sha256", "cells",
}
RECEIPT_FIELDS = {
    "kind", "status", "cell", "host", "role", "event_sha256",
    "activation_sha256", "request_sha256", "transition_plan_sha256",
    "root", "selected", "source", "files",
}


def _read(path):
    return p5_source.read_json(path)


def _write(path, value):
    p5_source.write_json(path, value)


def _manifest(directory, exclude=()):
    directory = Path(directory)
    excluded = set(exclude)
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Selected-lineage materialization may not contain symlinks")
        if path.is_file():
            name = path.relative_to(directory).as_posix()
            if name in excluded:
                continue
            if ".partial" in path.name:
                raise ValueError("Selected-lineage materialization contains a partial file")
            files[name] = {
                "sha256": p5_source.sha256(path),
                "bytes": path.stat().st_size,
            }
    return files


def _copy(source, destination):
    source, destination = Path(source), Path(destination)
    if source.is_symlink() or not source.is_file():
        raise ValueError("Selected-lineage source is missing or unsafe")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    with destination.open("r+b") as stream:
        os.fsync(stream.fileno())


def _fsync_tree(root):
    root = Path(root)
    directories = [root] + [
        path for path in root.rglob("*") if path.is_dir() and not path.is_symlink()
    ]
    for directory in sorted(
        directories, key=lambda path: len(path.relative_to(root).parts), reverse=True,
    ):
        p5_checkpoint.fsync_directory(directory)


def _file_entry(path):
    path = Path(path)
    return {"sha256": p5_source.sha256(path), "bytes": path.stat().st_size}


def _document_entry(value):
    encoded = v2.canonical(value)
    return {"sha256": v2.document_sha(value), "bytes": len(encoded)}


def _expected_files(
    plan, xroot, cell, selected_path, source_identity, root_state, selected_state, history,
):
    lease = plan["event"]["owners"][cell]
    root_artifact = lease["root"]["artifact"]
    selected_artifact = source_identity["artifact"]
    canonical = Path(xroot) / "roots" / cell
    files = {
        f"states/{root_artifact}/{name}": _file_entry(canonical / "checkpoint" / name)
        for name in ("model.safetensors", "training-state.pt", "p5-seal.json")
    }
    files.update({
        f"states/{selected_artifact}/{name}": _file_entry(Path(selected_path) / name)
        for name in ("model.safetensors", "training-state.pt", "p5-seal.json")
    })
    selected_pointer = {
        "artifact": selected_artifact,
        "seal_sha256": source_identity["checkpoint_seal_sha256"],
        "global_step": selected_state["global_step"],
        "phase": selected_state["phase"],
    }
    files.update({
        "p5-run-config.json": _file_entry(canonical / "p5-run-config.json"),
        "latest.json": _file_entry(canonical / "latest.json"),
        "selected.json": _document_entry(selected_pointer),
    })
    if "validation_history" in root_state:
        files["validation-history.json"] = _document_entry(history)
    if plan["bindings"]["cells"][cell]["role"] == "worker":
        files[".xnode/registration.json"] = _file_entry(
            Path(xroot) / "worker-bootstrap" / cell / ".xnode" / "registration.json"
        )
    return files


def _authority(request_path):
    request_path = Path(request_path).resolve()
    request = _read(request_path)
    if (set(request) != REQUEST_FIELDS
            or request["kind"] != "p5-selected-lineage-materialization-request-v1"
            or not isinstance(request["cells"], dict)
            or not request["cells"]):
        raise ValueError("Selected-lineage materialization request is incomplete")
    plan_path = Path(request["transition_plan"]).resolve()
    plan = _read(plan_path)
    if (plan.get("kind") != "p5-uuid-reregistration-plan-v2"
            or plan.get("status") != "prepared"
            or plan.get("event_sha256") != request["event_sha256"]
            or v2.event_sha(plan.get("event", {})) != request["event_sha256"]
            or plan["event"].get("broker_release_sha256")
            != request["broker_release_sha256"]):
        raise ValueError("Selected-lineage transition plan differs from the request")
    transitioned = set(plan["event"].get("transition_cells", []))
    if set(request["cells"]) != transitioned:
        raise ValueError("Selected-lineage request must cover every transitioned cell")
    xroot = Path(plan["bindings"]["control_root"]).resolve()
    certificate = v2.verify_activation(xroot, request["broker_release_sha256"])
    if (certificate.get("event_sha256") != request["event_sha256"]
            or certificate.get("event") != plan["event"]
            or certificate.get("transition_plan_sha256") != v2.digest(plan)
            or _read(xroot / "current" / "bindings.json") != plan["bindings"]
            or any(
                plan["cells"].get(cell)
                != plan["event"].get("durable_recovery", {}).get(cell)
                for cell in transitioned
            )):
        raise ValueError("Selected-lineage request is not bound to the activated event")
    return request_path, request, plan_path, plan, xroot


def _selection(root, selected, contract, selected_artifact=None):
    root_state, root_contract = p5_checkpoint._metadata(root, contract)
    selected_name = root_state.get("selected_artifact")
    selected_artifact = selected.name if selected_artifact is None else selected_artifact
    metric = root_state.get("best_metric")
    history = root_state.get("validation_history", [])
    if (not isinstance(selected_name, str)
            or Path(selected_name).name != selected_name
            or not selected_name.startswith("p5-state-")
            or selected_artifact != selected_name
            or not isinstance(history, list)
            or not isinstance(metric, (int, float))
            or not math.isfinite(metric)):
        raise ValueError("Activated root has an invalid selected-model identity")
    selected_state, selected_contract = p5_checkpoint._metadata(selected, root_contract)
    if (selected_contract != contract
            or selected_state.get("selected_artifact") != selected_name
            or selected_state.get("best_metric") != metric):
        raise ValueError("Selected checkpoint does not carry the activated validation decision")
    native_history = history and all(
        isinstance(entry, dict) and "recall@10" in entry.get("primary", {})
        for entry in history
    )
    if native_history:
        values = [entry["primary"]["recall@10"] for entry in history]
        selected_history = selected_state.get("validation_history")
        if (not all(isinstance(value, (int, float)) and math.isfinite(value)
                    for value in values)
                or max(values) != metric
                or not isinstance(selected_history, list)
                or selected_history != history[:len(selected_history)]
                or len(selected_history) != values.index(metric) + 1):
            raise ValueError("Selected checkpoint violates earliest-best validation lineage")
    return root_state, selected_state, history


def _verify_writer_source(source, metadata):
    source = Path(source).resolve()
    matches = [
        item for item in metadata.get("verified_checkpoints", [])
        if Path(item.get("path", "")).resolve() == source
    ]
    if len(matches) != 1:
        raise ValueError("Selected writer checkpoint is absent or ambiguous in the transition")
    entry = matches[0]
    receipt_path = source / "bank-receipt.json"
    if (p5_source.sha256(source / "p5-seal.json")
            != entry.get("checkpoint_seal_sha256")
            or p5_source.sha256(receipt_path) != entry.get("bank_receipt_sha256")
            or _read(receipt_path) != {"files": _manifest(source, {"bank-receipt.json"})}):
        raise ValueError("Selected writer checkpoint receipt or seal differs")
    return source, {
        "kind": "writer-checkpoint",
        "path": str(source),
        "artifact": source.name,
        "checkpoint_seal_sha256": entry["checkpoint_seal_sha256"],
        "bank_receipt_sha256": entry["bank_receipt_sha256"],
    }


def _verify_worker_source(source, metadata):
    source = Path(source).resolve()
    matches = [
        item for item in metadata.get("verified_packages", [])
        if Path(item.get("path", "")).resolve() == source
    ]
    if len(matches) != 1:
        raise ValueError("Selected worker package is absent or ambiguous in the transition")
    entry = matches[0]
    marker_path = source / "package-seal.json"
    receipt_path = source / "bank-receipt.json"
    package_files = v1._file_manifest(
        source, {"package-seal.json", "bank-receipt.json"},
    )
    marker = _read(marker_path)
    receipt = _read(receipt_path)
    checkpoint = source / "checkpoint"
    header = v1._checkpoint_header(checkpoint)
    envelope = _read(source / "envelope.json")
    if (marker.get("kind") != "p5-xnode-package-v1"
            or marker.get("status") != "sealed"
            or set(package_files) != v1.PACKAGE_FILES
            or marker.get("files") != package_files
            or receipt != {"files": _manifest(source, {"bank-receipt.json"})}
            or p5_source.sha256(checkpoint / "p5-seal.json")
            != entry.get("checkpoint_seal_sha256")
            or p5_source.sha256(marker_path) != entry.get("package_seal_sha256")
            or p5_source.sha256(receipt_path) != entry.get("bank_receipt_sha256")
            or any(
                envelope.get("checkpoint", {}).get(key) != value
                for key, value in header.items() if key != "artifact"
            )):
        raise ValueError("Selected worker package receipt, seal, or envelope differs")
    artifact = envelope.get("checkpoint", {}).get("artifact")
    if (not isinstance(artifact, str) or Path(artifact).name != artifact
            or not artifact.startswith("p5-state-")):
        raise ValueError("Selected worker package artifact identity is unsafe")
    return checkpoint, {
        "kind": "worker-package",
        "path": str(source),
        "artifact": artifact,
        "checkpoint_seal_sha256": entry["checkpoint_seal_sha256"],
        "package_seal_sha256": entry["package_seal_sha256"],
        "bank_receipt_sha256": entry["bank_receipt_sha256"],
    }


def _source(request, plan, cell):
    cell_request = request["cells"][cell]
    if set(cell_request) != {"source"}:
        raise ValueError("Selected-lineage cell request has unknown fields")
    metadata = plan["cells"].get(cell, {})
    if metadata.get("evidence_kind") == "writer-durable-recovery":
        return _verify_writer_source(cell_request["source"], metadata)
    if metadata.get("evidence_kind") == "worker-durable-recovery":
        return _verify_worker_source(cell_request["source"], metadata)
    raise ValueError("Selected-lineage source kind is not receipt-based durable recovery")


def _expected_run(plan, cell):
    return Path(plan["bindings"]["p5_root"]).resolve() / "runs" / cell


def _verify_run(request_path, request, plan_path, plan, xroot, cell):
    run = _expected_run(plan, cell)
    receipt_path = run / "selected-lineage-receipt.json"
    if not receipt_path.is_file():
        raise ValueError("Selected-lineage materialization receipt is missing")
    receipt = _read(receipt_path)
    files = _manifest(run, {"selected-lineage-receipt.json"})
    _, source_identity = _source(request, plan, cell)
    if (set(receipt) != RECEIPT_FIELDS
            or receipt.get("kind") != "p5-selected-lineage-materialization-v1"
            or receipt.get("status") != "materialized"
            or receipt.get("cell") != cell
            or receipt.get("host") != socket.gethostname()
            or receipt.get("event_sha256") != request["event_sha256"]
            or receipt.get("activation_sha256")
            != p5_source.sha256(xroot / "activation" / "last-published.json")
            or receipt.get("request_sha256") != p5_source.sha256(request_path)
            or receipt.get("transition_plan_sha256") != p5_source.sha256(plan_path)
            or receipt.get("source") != source_identity
            or receipt.get("files") != files):
        raise ValueError("Selected-lineage materialization receipt differs")
    lease = plan["event"]["owners"][cell]
    latest = _read(run / "latest.json")
    selected = _read(run / "selected.json")
    root = run / "states" / latest["artifact"]
    selected_path = run / "states" / selected["artifact"]
    contract = _read(run / "p5-run-config.json")
    root_seal_sha256 = p5_source.sha256(root / "p5-seal.json")
    selected_seal_sha256 = p5_source.sha256(selected_path / "p5-seal.json")
    if (latest != _read(xroot / "roots" / cell / "latest.json")
            or p5_source.sha256(run / "p5-run-config.json")
            != lease["contract_file_sha256"]
            or latest["seal_sha256"] != lease["root"]["seal_sha256"]
            or root_seal_sha256 != lease["root"]["seal_sha256"]
            or selected["artifact"] != receipt["selected"]["artifact"]
            or selected["seal_sha256"] != receipt["selected"]["seal_sha256"]
            or selected_seal_sha256 != source_identity["checkpoint_seal_sha256"]):
        raise ValueError("Selected-lineage pointers or contract differ")
    root_state, selected_state, history = _selection(root, selected_path, contract)
    if selected != {
        "artifact": selected_path.name,
        "seal_sha256": p5_source.sha256(selected_path / "p5-seal.json"),
        "global_step": selected_state["global_step"],
        "phase": selected_state["phase"],
    }:
        raise ValueError("Selected-lineage pointer differs from selected full state")
    if "validation_history" in p5_checkpoint._metadata(root, contract)[0]:
        if _read(run / "validation-history.json") != history:
            raise ValueError("Selected-lineage validation history differs")
    if receipt["files"] != _expected_files(
        plan, xroot, cell, selected_path, source_identity,
        root_state, selected_state, history,
    ):
        raise ValueError("Selected-lineage receipt file authority differs")
    if plan["bindings"]["cells"][cell]["role"] == "worker":
        transport = run / ".xnode"
        expected = xroot / "worker-bootstrap" / cell / ".xnode" / "registration.json"
        if ({path.name for path in transport.iterdir()} != {"registration.json"}
                or p5_source.sha256(transport / "registration.json")
                != p5_source.sha256(expected)):
            raise ValueError("Selected-lineage worker registration differs")
    return receipt


def materialize_selected_lineage(request_path):
    request_path, request, plan_path, plan, xroot = _authority(request_path)
    host = socket.gethostname()
    local_cells = [
        cell for cell in plan["event"]["transition_cells"]
        if plan["event"]["owners"][cell]["owner_node"] == host
    ]
    if not local_cells:
        raise ValueError("This host owns no transitioned P5 cell")
    completed = []
    for cell in local_cells:
        run = _expected_run(plan, cell)
        if run.exists():
            _verify_run(request_path, request, plan_path, plan, xroot, cell)
            completed.append(cell)
            continue
        lease = plan["event"]["owners"][cell]
        source, source_identity = _source(request, plan, cell)
        canonical = xroot / "roots" / cell
        contract = _read(canonical / "p5-run-config.json")
        root = canonical / "checkpoint"
        root_state, selected_state, history = _selection(
            root, source, contract, source_identity["artifact"],
        )
        staging = run.with_name(run.name + ".partial-" + uuid.uuid4().hex)
        if staging.exists():
            raise ValueError("Selected-lineage staging path already exists")
        for checkpoint, artifact in (
            (root, lease["root"]["artifact"]),
            (source, source_identity["artifact"]),
        ):
            destination = staging / "states" / artifact
            for name in ("model.safetensors", "training-state.pt", "p5-seal.json"):
                _copy(checkpoint / name, destination / name)
        _copy(canonical / "p5-run-config.json", staging / "p5-run-config.json")
        _copy(canonical / "latest.json", staging / "latest.json")
        selected_pointer = {
            "artifact": source_identity["artifact"],
            "seal_sha256": p5_source.sha256(source / "p5-seal.json"),
            "global_step": selected_state["global_step"],
            "phase": selected_state["phase"],
        }
        _write(staging / "selected.json", selected_pointer)
        if "validation_history" in root_state:
            _write(staging / "validation-history.json", history)
        if plan["bindings"]["cells"][cell]["role"] == "worker":
            _copy(
                xroot / "worker-bootstrap" / cell / ".xnode" / "registration.json",
                staging / ".xnode" / "registration.json",
            )
        receipt = {
            "kind": "p5-selected-lineage-materialization-v1",
            "status": "materialized",
            "cell": cell,
            "host": host,
            "role": plan["bindings"]["cells"][cell]["role"],
            "event_sha256": request["event_sha256"],
            "activation_sha256": p5_source.sha256(
                xroot / "activation" / "last-published.json"
            ),
            "request_sha256": p5_source.sha256(request_path),
            "transition_plan_sha256": p5_source.sha256(plan_path),
            "root": {
                "artifact": lease["root"]["artifact"],
                "seal_sha256": lease["root"]["seal_sha256"],
            },
            "selected": {
                "artifact": source_identity["artifact"],
                "seal_sha256": selected_pointer["seal_sha256"],
                "best_metric": root_state["best_metric"],
            },
            "source": source_identity,
            "files": _manifest(staging),
        }
        _write(staging / "selected-lineage-receipt.json", receipt)
        _fsync_tree(staging)
        run.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(run)
        p5_checkpoint.fsync_directory(run.parent)
        _verify_run(request_path, request, plan_path, plan, xroot, cell)
        completed.append(cell)
    return {"status": "materialized", "host": host, "cells": completed}


def verify_selected_lineage(request_path):
    request_path, request, plan_path, plan, xroot = _authority(request_path)
    host = socket.gethostname()
    local_cells = [
        cell for cell in plan["event"]["transition_cells"]
        if plan["event"]["owners"][cell]["owner_node"] == host
    ]
    if not local_cells:
        raise ValueError("This host owns no transitioned P5 cell")
    for cell in local_cells:
        _verify_run(request_path, request, plan_path, plan, xroot, cell)
    return {"status": "verified", "host": host, "cells": local_cells}


def selected_lineage_deployment(request_path, receipt_paths):
    request_path, request, plan_path, plan, xroot = _authority(request_path)
    expected_cells = set(plan["event"]["transition_cells"])
    receipts = {}
    for path in receipt_paths:
        path = Path(path).resolve()
        receipt = _read(path)
        cell = receipt.get("cell")
        if cell in receipts or cell not in expected_cells:
            raise ValueError("Selected-lineage deployment receipt set differs")
        owner = plan["event"]["owners"][cell]
        selected_path, source_identity = _source(request, plan, cell)
        canonical = xroot / "roots" / cell
        contract = _read(canonical / "p5-run-config.json")
        root_state, selected_state, history = _selection(
            canonical / "checkpoint",
            selected_path,
            contract,
            source_identity["artifact"],
        )
        expected_selected = {
            "artifact": source_identity["artifact"],
            "seal_sha256": source_identity["checkpoint_seal_sha256"],
            "best_metric": root_state["best_metric"],
        }
        if (set(receipt) != RECEIPT_FIELDS
                or receipt.get("kind") != "p5-selected-lineage-materialization-v1"
                or receipt.get("status") != "materialized"
                or receipt.get("host") != owner["owner_node"]
                or receipt.get("role") != plan["bindings"]["cells"][cell]["role"]
                or receipt.get("event_sha256") != request["event_sha256"]
                or receipt.get("activation_sha256")
                != p5_source.sha256(xroot / "activation" / "last-published.json")
                or receipt.get("request_sha256") != p5_source.sha256(request_path)
                or receipt.get("transition_plan_sha256") != p5_source.sha256(plan_path)
                or receipt.get("root", {}).get("artifact") != owner["root"]["artifact"]
                or receipt.get("root", {}).get("seal_sha256")
                != owner["root"]["seal_sha256"]
                or receipt.get("selected") != expected_selected
                or receipt.get("source") != source_identity
                or receipt.get("files") != _expected_files(
                    plan, xroot, cell, selected_path, source_identity,
                    root_state, selected_state, history,
                )):
            raise ValueError("Selected-lineage deployment receipt authority differs")
        receipts[cell] = {
            "host": receipt["host"],
            "role": receipt["role"],
            "receipt_sha256": p5_source.sha256(path),
            "root": receipt["root"],
            "selected": receipt["selected"],
        }
    if set(receipts) != expected_cells:
        raise ValueError("Selected-lineage deployment requires one receipt per cell")
    deployment = {
        "kind": "p5-selected-lineage-deployment-v1",
        "status": "verified",
        "event_sha256": request["event_sha256"],
        "activation_sha256": p5_source.sha256(
            xroot / "activation" / "last-published.json"
        ),
        "request_sha256": p5_source.sha256(request_path),
        "transition_plan_sha256": p5_source.sha256(plan_path),
        "cells": {cell: receipts[cell] for cell in sorted(receipts)},
    }
    return deployment


def aggregate_selected_lineage(request_path, receipt_paths, output):
    deployment = selected_lineage_deployment(request_path, receipt_paths)
    _write(output, deployment)
    return deployment


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="mode", required=True)
    for name in ("materialize", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--request", type=Path, required=True)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--request", type=Path, required=True)
    aggregate.add_argument("--receipt", type=Path, action="append", required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "materialize":
        result = materialize_selected_lineage(args.request)
    elif args.mode == "verify":
        result = verify_selected_lineage(args.request)
    else:
        result = aggregate_selected_lineage(args.request, args.receipt, args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
