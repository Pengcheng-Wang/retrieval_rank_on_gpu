"""Protocol/runtime repair transition for activated P5 ownership."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import socket
from pathlib import Path

import bank as bank_tool
import p5_xnode_reregister as v1
import p5_xnode_reregister_v2 as v2
import selected_lineage_materialize as selected_lineage


read = v1.read
sha = v1.sha
digest = v1.digest
canonical = v1.canonical
event_sha = v1.event_sha
document_sha = v2.document_sha


REQUEST_FIELDS = {
    "kind", "parent_event_sha256", "parent_plan", "active_cells", "new_bindings",
    "registry", "runtime_template", "broker_release", "broker_validation",
    "selected_lineage_evidence", "selected_readmission",
    "readmission_durable_root", "proofs", "lease_ids",
}


def _request(path):
    value = read(path)
    if (set(value) != REQUEST_FIELDS
            or value["kind"] != "p5-protocol-repair-request-v1"
            or not isinstance(value["active_cells"], list)
            or not value["active_cells"]
            or value["active_cells"] != sorted(value["active_cells"])
            or len(value["active_cells"]) != len(set(value["active_cells"]))):
        raise ValueError("Protocol-repair request is incomplete or ambiguous")
    active = set(value["active_cells"])
    if (set(value["proofs"]) != active
            or set(value["lease_ids"]) != active
            or any(not re.fullmatch(r"[0-9a-f]{32}", item)
                   for item in value["lease_ids"].values())
            or len(set(value["lease_ids"].values())) != len(active)):
        raise ValueError("Protocol-repair proofs and lease IDs must cover every active cell")
    return value


def _parent(request, pending_event_sha=None):
    parent_plan = read(request["parent_plan"])
    if (parent_plan.get("kind") != "p5-uuid-reregistration-plan-v2"
            or parent_plan.get("status") != "prepared"
            or parent_plan.get("event_sha256") != request["parent_event_sha256"]
            or event_sha(parent_plan.get("event", {}))
            != request["parent_event_sha256"]):
        raise ValueError("Protocol-repair parent plan is invalid or event-stale")
    parent_bindings = v1._validate_bindings(parent_plan["bindings"])
    xroot = Path(parent_bindings["control_root"])
    canonical_certificate = read(xroot / "current" / "ownership-certificate.json")
    parent = canonical_certificate["event"]
    if (canonical_certificate.get("event_sha256") != request["parent_event_sha256"]
            or event_sha(parent) != request["parent_event_sha256"]
            or canonical_certificate.get("transition_plan_sha256")
            != digest(parent_plan)
            or parent_plan["event"] != parent
            or read(xroot / "current" / "bindings.json") != parent_bindings):
        raise ValueError("Protocol-repair parent event differs from canonical ownership")
    v2.verify_activation(xroot, parent["broker_release_sha256"])
    current = v2._load_registry(
        request["registry"], parent_bindings["cells"],
        request["parent_event_sha256"] if pending_event_sha is None else pending_event_sha,
    )
    if current != canonical_certificate:
        raise ValueError("Protocol-repair registry and activated certificate differ")
    return parent_bindings, xroot, canonical_certificate, parent_plan


def _validate_bindings(parent_bindings, new_bindings, parent_event, active):
    if set(parent_bindings) != set(new_bindings):
        raise ValueError("Protocol repair cannot change binding document shape")
    for key in parent_bindings:
        if key != "cells" and parent_bindings[key] != new_bindings[key]:
            raise ValueError("Protocol repair changed a non-cell private binding")
    if set(parent_bindings["cells"]) != set(new_bindings["cells"]):
        raise ValueError("Protocol repair must preserve the complete cell set")
    for cell in parent_bindings["cells"]:
        before = parent_bindings["cells"][cell]
        after = new_bindings["cells"][cell]
        if cell not in active:
            if after != before:
                raise ValueError("Inactive P5 binding changed during protocol repair")
            continue
        expected = copy.deepcopy(before)
        expected["ownership_epoch"] = parent_event["owners"][cell]["ownership_epoch"] + 1
        if after != expected:
            raise ValueError("Active P5 binding changed beyond ownership epoch")
    return new_bindings


def _broker_release(request, runtime_manifest_sha256):
    path = Path(request["broker_release"])
    release = read(path)
    if (set(release) != {"kind", "files_sha256", "runtime_manifest_sha256"}
            or release.get("kind") != "p5-xnode-broker-v4"
            or set(release.get("files_sha256", {})) != {
                "p5_xnode_broker.py", "p5_xnode_state.py", "p5_xnode_adopt.py",
            }
            or release.get("runtime_manifest_sha256") != runtime_manifest_sha256
            or release["files_sha256"]["p5_xnode_state.py"]
            != read(request["runtime_template"])["files_sha256"]["p5_xnode_state.py"]):
        raise ValueError("Broker-v4 release authority is incomplete or runtime-stale")
    for name, expected in release["files_sha256"].items():
        if sha(path.parent / name) != expected:
            raise ValueError("Broker-v4 release source changed")
    validation_path = Path(request["broker_validation"])
    validation = read(validation_path)
    required_checks = {
        "dynamic_binding_api", "legacy_sequence_root",
        "selected_readmission", "active_cell_fence",
    }
    if (validation.get("kind") != "p5-broker-v4-validation-v1"
            or validation.get("status") != "pass"
            or validation.get("broker_release_sha256") != sha(path)
            or validation.get("runtime_manifest_sha256") != runtime_manifest_sha256
            or set(validation.get("checks", {})) != required_checks
            or not all(validation["checks"].values())):
        raise ValueError("Broker-v4 validation receipt is incomplete or stale")
    return path, validation_path


def _selected_evidence(
    request, bindings, certificate, parent_plan, active, xroot,
):
    evidence = Path(request["selected_lineage_evidence"]).resolve()
    if (evidence.is_symlink()
            or not evidence.is_relative_to(
                Path(bindings["durable_root"]).resolve()
            )):
        raise ValueError("Selected-lineage evidence is outside the durable root")
    receipt_path = evidence / "bank-receipt.json"
    receipt = read(receipt_path)
    actual = v1._file_manifest(evidence, {"bank-receipt.json"})
    if receipt != {"files": actual}:
        raise ValueError("Selected-lineage evidence packet receipt differs")
    deployment_path = evidence / "deployment.json"
    deployment = read(deployment_path)
    packet_plan_path = evidence / "plan-v2.json"
    packet_request_path = evidence / "request.json"
    packet_plan = read(packet_plan_path)
    packet_request = read(packet_request_path)
    if (canonical(packet_plan) != canonical(parent_plan)
            or certificate.get("transition_plan_sha256") != digest(packet_plan)
            or packet_plan.get("event") != certificate["event"]
            or packet_request.get("kind")
            != "p5-selected-lineage-materialization-request-v1"
            or packet_request.get("event_sha256")
            != certificate["event_sha256"]
            or canonical(read(packet_request["transition_plan"]))
            != canonical(parent_plan)):
        raise ValueError("Selected-lineage packet is not bound to the parent plan")
    receipt_paths = sorted((evidence / "receipts").glob("*.json"))
    recomputed = selected_lineage.selected_lineage_deployment(
        packet_request_path, receipt_paths,
    )
    if (deployment.get("kind") != "p5-selected-lineage-deployment-v1"
            or deployment.get("status") != "verified"
            or deployment.get("event_sha256") != request["parent_event_sha256"]
            or deployment.get("activation_sha256")
            != sha(xroot / "activation" / "last-published.json")
            or set(deployment.get("cells", {})) != active
            or canonical(deployment) != canonical(recomputed)):
        raise ValueError("Selected-lineage deployment is parent-stale or partial")
    selected = {}
    sources = {}
    for cell in sorted(active):
        owner = certificate["event"]["owners"][cell]
        item = deployment["cells"][cell]
        receipt_file = evidence / "receipts" / f"{cell}.json"
        if (sha(receipt_file) != item.get("receipt_sha256")
                or item.get("root") != {
                    "artifact": owner["root"]["artifact"],
                    "seal_sha256": owner["root"]["seal_sha256"],
                }):
            raise ValueError("Selected-lineage cell receipt or root differs")
        materialization = read(receipt_file)
        source = materialization.get("source", {})
        source_path = Path(source.get("path", "")).resolve()
        if source_path.is_symlink() or not source_path.is_absolute() or not (
            source_path.is_relative_to(Path(bindings["root"]).resolve())
            or source_path.is_relative_to(Path(bindings["durable_root"]).resolve())
        ):
            raise ValueError("Selected-lineage source path is outside bound roots")
        checkpoint = (
            source_path / "checkpoint"
            if source.get("kind") == "worker-package"
            else source_path
        )
        header = v1._checkpoint_header(checkpoint)
        header["artifact"] = source.get("artifact")
        if (source.get("kind") not in {"writer-checkpoint", "worker-package"}
                or header["artifact"] != item["selected"]["artifact"]
                or header["seal_sha256"] != item["selected"]["seal_sha256"]
                or header["contract_sha256"] != owner["root"]["contract_sha256"]
                or source.get("checkpoint_seal_sha256") != header["seal_sha256"]):
            raise ValueError("Selected-lineage source checkpoint differs")
        selected[cell] = header
        sources[cell] = {
            "checkpoint": str(checkpoint),
            "contract": str(
                Path(bindings["control_root"]) / "roots" / cell
                / "p5-run-config.json"
            ),
        }
    return deployment_path, selected, sources


def _owners(
    parent, parent_plan, bindings, runtime_manifest_sha256, request, active,
):
    owners = copy.deepcopy(parent["owners"])
    cells = {}
    roots = {}
    for cell in sorted(active):
        old = parent["owners"][cell]
        assignment = bindings["cells"][cell]
        expected_host = (
            bindings["writer_host"]
            if assignment["role"] == "writer"
            else bindings["worker_host"]
        )
        if (old.get("kind") != "p5-xnode-lease-v1"
                or old.get("owner_node") != expected_host
                or any(old.get(key) != assignment[key] for key in (
                    "gpu", "gpu_uuid", "microbatch", "accumulation",
                ))
                or old.get("root", {}).get("step") != assignment["root_step"]):
            raise ValueError("Protocol repair parent lease and binding differ")
        lease = copy.deepcopy(old)
        lease.update(
            lease_id=request["lease_ids"][cell],
            ownership_epoch=assignment["ownership_epoch"],
            control_manifest_sha256=runtime_manifest_sha256,
        )
        proof = v1._validate_proof(
            request["proofs"][cell], cell, old, assignment, old["root"],
        )
        proof_sha256 = sha(request["proofs"][cell])
        if (proof_sha256
                != parent.get("replacement_proofs_sha256", {}).get(cell)
                or proof_sha256
                != parent_plan.get("cells", {}).get(cell, {}).get("proof_sha256")):
            raise ValueError("Repair proof differs from parent-authenticated proof")
        owners[cell] = lease
        cells[cell] = {
            "parent_lease_sha256": digest(old),
            "new_lease_sha256": digest(lease),
            "proof_sha256": proof_sha256,
            "source_checkpoint_seal_sha256":
                proof["source_checkpoint_seal_sha256"],
        }
        roots[cell] = {
            "checkpoint": str(
                Path(bindings["control_root"]) / "roots" / cell / "checkpoint"
            ),
            "contract": str(
                Path(bindings["control_root"]) / "roots" / cell / "p5-run-config.json"
            ),
        }
    return owners, cells, roots


def draft_repair(request_path, pending_event_sha=None):
    request_path = Path(request_path)
    request = _request(request_path)
    parent_bindings, xroot, certificate, parent_plan = _parent(
        request, pending_event_sha,
    )
    new_bindings = v1._validate_bindings(read(request["new_bindings"]))
    parent = certificate["event"]
    active = set(request["active_cells"])
    if (active != set(parent.get("transition_cells", []))
            or not active.issubset(parent["owners"])):
        raise ValueError("Protocol repair must cover the exact activated transition cells")
    _validate_bindings(parent_bindings, new_bindings, parent, active)
    runtime_manifest = v2._runtime_manifest(request, new_bindings)
    runtime_manifest["ownership_registry"] = str(Path(request["registry"]).resolve())
    runtime_manifest_sha256 = document_sha(runtime_manifest)
    parent_runtime = read(xroot / "current" / "runtime-manifest.json")
    if (runtime_manifest.get("input_manifest_sha256")
            != parent_runtime.get("input_manifest_sha256")
            or any(
                parent["owners"][cell].get("input_manifest_sha256")
                != runtime_manifest.get("input_manifest_sha256")
                for cell in active
            )):
        raise ValueError("Protocol repair cannot change the scientific input manifest")
    if runtime_manifest_sha256 == sha(xroot / "current" / "runtime-manifest.json"):
        raise ValueError("Protocol repair must publish a distinct worker runtime")
    broker_release, broker_validation = _broker_release(
        request, runtime_manifest_sha256,
    )
    deployment, selected, selected_sources = _selected_evidence(
        request, new_bindings, certificate, parent_plan, active, xroot,
    )
    owners, cells, roots = _owners(
        parent, parent_plan, new_bindings, runtime_manifest_sha256,
        request, active,
    )
    proof = v2._aggregate_proof(
        {**request, "cells": sorted(active)}, new_bindings, owners,
        {cell: owners[cell]["root"] for cell in active},
        runtime_manifest_sha256,
    )
    return {
        "kind": "p5-protocol-repair-draft-v1",
        "status": "drafted",
        "request_sha256": sha(request_path),
        "parent_event_sha256": request["parent_event_sha256"],
        "parent_activation_sha256": sha(
            xroot / "activation" / "last-published.json"
        ),
        "active_cells": sorted(active),
        "bindings": new_bindings,
        "runtime_manifest": runtime_manifest,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "runtime_source_directory": str(Path(request["runtime_template"]).parent),
        "broker_release": str(broker_release),
        "broker_release_sha256": sha(broker_release),
        "broker_validation": str(broker_validation),
        "broker_validation_sha256": sha(broker_validation),
        "selected_lineage_deployment": str(deployment),
        "selected_lineage_deployment_sha256": sha(deployment),
        "selected": selected,
        "selected_sources": selected_sources,
        "owners": owners,
        "cells": cells,
        "roots": roots,
        "aggregate_proof": proof,
    }


def build_readmission(request_path):
    request_path = Path(request_path)
    request = _request(request_path)
    draft = draft_repair(request_path)
    bindings = draft["bindings"]
    if socket.gethostname() != bindings["writer_host"]:
        raise RuntimeError("Only the configured writer may build selected readmission")
    output = Path(request["selected_readmission"]).resolve()
    durable_root = Path(request["readmission_durable_root"]).resolve()
    if (not output.is_relative_to(Path(bindings["root"]).resolve())
            or not durable_root.is_relative_to(
                Path(bindings["durable_root"]).resolve()
            )
            or output.exists()
            or durable_root.exists()):
        raise ValueError("Selected readmission output or durable root is unsafe or reused")
    workers = [
        cell for cell in draft["active_cells"]
        if draft["owners"][cell]["owner_node"] == bindings["worker_host"]
    ]
    authority = {
        "kind": "p5-selected-lineage-readmission-v1",
        "status": "verified",
        "parent_event_sha256": draft["parent_event_sha256"],
        "selected_lineage_deployment_sha256":
            draft["selected_lineage_deployment_sha256"],
        "cells": {},
    }
    with v1._lock(Path(bindings["root"]) / "bank.lock"):
        durable_root.mkdir(parents=True)
        (durable_root / "README.md").write_text(
            "P5 selected-lineage readmission packages. "
            "Each child is immutable after bank-receipt.json publication.\n",
            encoding="utf-8",
        )
        for cell in workers:
            lease = draft["owners"][cell]
            selected = draft["selected"][cell]
            package = output.parent / "packages" / cell
            if package.exists():
                raise FileExistsError("Selected readmission package already exists")
            (package / "checkpoint").mkdir(parents=True)
            source = Path(draft["selected_sources"][cell]["checkpoint"])
            for name in ("model.safetensors", "training-state.pt", "p5-seal.json"):
                _copy_file(source / name, package / "checkpoint" / name)
            _copy_file(
                draft["selected_sources"][cell]["contract"],
                package / "scientific-contract.json",
            )
            attestation = {
                "kind": "p5-selected-lineage-readmission-attestation-v1",
                "historically_accepted": False,
                "source_event_sha256": draft["parent_event_sha256"],
                "selected_lineage_deployment_sha256":
                    draft["selected_lineage_deployment_sha256"],
                "lease_sha256": digest(lease),
                "root_checkpoint_seal_sha256": lease["root"]["seal_sha256"],
                "selected_checkpoint_seal_sha256": selected["seal_sha256"],
                "scientific_contract_rewritten": False,
            }
            v1._atomic_write(
                package / "execution-attestation.json", attestation,
            )
            v1._atomic_write(package / "envelope.json", {
                "kind": "p5-xnode-selected-readmission-v1",
                "purpose": "production",
                **{
                    key: lease[key]
                    for key in (
                        "cell", "owner_node", "gpu", "lease_id",
                        "ownership_epoch",
                    )
                },
                "lease_sha256": digest(lease),
                "wire_generation": 0,
                "root_seal_sha256": lease["root"]["seal_sha256"],
                "checkpoint": selected,
                "selected": selected,
                "execution_attestation_sha256": sha(
                    package / "execution-attestation.json"
                ),
            })
            files = v1._file_manifest(package, {"package-seal.json"})
            v1._atomic_write(package / "package-seal.json", {
                "kind": "p5-xnode-package-v1",
                "status": "sealed",
                "files": files,
            })
            destination = durable_root / cell
            receipt = bank_tool.copy_artifact(package, destination)
            if read(destination / "bank-receipt.json") != receipt:
                raise ValueError("Selected readmission bank receipt was not exact")
            authority["cells"][cell] = {
                "historically_accepted": False,
                "lease_sha256": digest(lease),
                "root_seal_sha256": lease["root"]["seal_sha256"],
                "selected": selected,
                "package": str(package),
                "package_seal_sha256": sha(package / "package-seal.json"),
                "destination": str(destination),
                "bank_receipt_sha256": sha(
                    destination / "bank-receipt.json"
                ),
            }
        v1._atomic_write(output, authority)
    _readmission(output, draft)
    return authority


def _readmission(path, draft):
    path = Path(path)
    value = read(path)
    workers = {
        cell for cell in draft["active_cells"]
        if draft["owners"][cell]["owner_node"] == draft["bindings"]["worker_host"]
    }
    if (value.get("kind") != "p5-selected-lineage-readmission-v1"
            or value.get("status") != "verified"
            or value.get("parent_event_sha256") != draft["parent_event_sha256"]
            or value.get("selected_lineage_deployment_sha256")
            != draft["selected_lineage_deployment_sha256"]
            or set(value.get("cells", {})) != workers):
        raise ValueError("Selected-lineage readmission authority is stale or partial")
    for cell in sorted(workers):
        lease = draft["owners"][cell]
        selected = draft["selected"][cell]
        entry = value["cells"][cell]
        package = Path(entry["package"]).resolve()
        destination = Path(entry["destination"]).resolve()
        if (not package.is_relative_to(Path(draft["bindings"]["root"]).resolve())
                or not destination.is_relative_to(
                    Path(draft["bindings"]["durable_root"]).resolve()
                )
                or package.is_symlink()
                or destination.is_symlink()
                or entry.get("historically_accepted") is not False
                or entry.get("lease_sha256") != digest(lease)
                or entry.get("root_seal_sha256") != lease["root"]["seal_sha256"]
                or entry.get("selected") != selected
                or sha(package / "package-seal.json")
                != entry.get("package_seal_sha256")
                or sha(destination / "bank-receipt.json")
                != entry.get("bank_receipt_sha256")):
            raise ValueError("Selected-lineage readmission cell authority differs")
        marker = read(package / "package-seal.json")
        files = v1._file_manifest(package, {"package-seal.json"})
        receipt = read(destination / "bank-receipt.json")
        durable = v1._file_manifest(destination, {"bank-receipt.json"})
        envelope = read(package / "envelope.json")
        attestation = read(package / "execution-attestation.json")
        header = v1._checkpoint_header(package / "checkpoint")
        header["artifact"] = selected["artifact"]
        if (marker != {
                "kind": "p5-xnode-package-v1", "status": "sealed", "files": files,
        } or set(files) != v1.PACKAGE_FILES
                or receipt != {"files": durable}
                or durable != v1._file_manifest(package)
                or header != selected
                or digest(read(package / "scientific-contract.json"))
                != selected["contract_sha256"]
                or envelope.get("kind") != "p5-xnode-selected-readmission-v1"
                or envelope.get("purpose") != "production"
                or envelope.get("wire_generation") != 0
                or "generation" in envelope
                or "parent_seal_sha256" in envelope
                or any(
                    envelope.get(key) != lease[key]
                    for key in (
                        "cell", "owner_node", "gpu", "lease_id",
                        "ownership_epoch",
                    )
                )
                or envelope.get("lease_sha256") != digest(lease)
                or envelope.get("checkpoint") != selected
                or envelope.get("selected") != selected
                or envelope.get("root_seal_sha256") != lease["root"]["seal_sha256"]
                or sha(package / "execution-attestation.json")
                != envelope.get("execution_attestation_sha256")
                or attestation.get("kind")
                != "p5-selected-lineage-readmission-attestation-v1"
                or attestation.get("historically_accepted") is not False
                or attestation.get("source_event_sha256")
                != draft["parent_event_sha256"]
                or attestation.get("selected_lineage_deployment_sha256")
                != draft["selected_lineage_deployment_sha256"]
                or attestation.get("lease_sha256") != digest(lease)
                or attestation.get("root_checkpoint_seal_sha256")
                != lease["root"]["seal_sha256"]
                or attestation.get("selected_checkpoint_seal_sha256")
                != selected["seal_sha256"]
                or attestation.get("scientific_contract_rewritten") is not False):
            raise ValueError("Selected-lineage readmission package differs")
    return value


def prepare_repair(request_path, pending_event_sha=None):
    request_path = Path(request_path)
    request = _request(request_path)
    draft = draft_repair(request_path, pending_event_sha)
    if not Path(request["selected_readmission"]).is_file():
        raise ValueError("Final protocol repair requires selected readmission evidence")
    readmission = _readmission(request["selected_readmission"], draft)
    parent = read(Path(draft["bindings"]["control_root"])
                  / "current" / "ownership-certificate.json")["event"]
    event = {
        "kind": "p5-xnode-ownership-v1",
        "status": "committed",
        "sequence": parent["sequence"] + 1,
        "previous_event_sha256": draft["parent_event_sha256"],
        "transition_kind": "p5-protocol-repair-v1",
        "transition_cells": draft["active_cells"],
        "active_cells": draft["active_cells"],
        "owners": draft["owners"],
        "broker_release_sha256": draft["broker_release_sha256"],
        "broker_validation_sha256": draft["broker_validation_sha256"],
        "crossnode_proof_sha256": document_sha(draft["aggregate_proof"]),
        "replacement_proofs_sha256": {
            cell: draft["cells"][cell]["proof_sha256"]
            for cell in draft["active_cells"]
        },
        "selected_lineage_deployment": {
            "sha256": draft["selected_lineage_deployment_sha256"],
            "cells": draft["active_cells"],
        },
        "selected_lineage_readmission_sha256":
            document_sha(readmission),
        "parent_runtime_manifest_sha256":
            sha(Path(draft["bindings"]["control_root"])
                / "current" / "runtime-manifest.json"),
        "parent_activation_sha256": draft["parent_activation_sha256"],
    }
    plan = {
        **draft,
        "kind": "p5-protocol-repair-plan-v1",
        "status": "prepared",
        "selected_readmission": readmission,
        "selected_readmission_path": request["selected_readmission"],
        "event": event,
        "event_sha256": event_sha(event),
    }
    return plan


def verify_repair(request_path, plan):
    prepared = prepare_repair(request_path, plan.get("event_sha256"))
    if canonical(prepared) != canonical(plan):
        raise ValueError("Protocol-repair plan differs from current verified evidence")
    return plan


def _verify_external_evidence(request, plan):
    if (sha(request["broker_release"]) != plan["broker_release_sha256"]
            or sha(request["broker_validation"])
            != plan["broker_validation_sha256"]
            or sha(
                Path(request["selected_lineage_evidence"]) / "deployment.json"
            ) != plan["selected_lineage_deployment_sha256"]
            or any(
                sha(request["proofs"][cell])
                != plan["cells"][cell]["proof_sha256"]
                for cell in plan["active_cells"]
            )):
        raise ValueError("Protocol-repair external evidence changed")
    return True


def worker_registration(plan, cell):
    lease = plan["owners"][cell]
    if plan["bindings"]["cells"][cell]["role"] != "worker":
        raise ValueError("Protocol-repair registration requested for writer cell")
    return {
        "kind": "p5-xnode-registration-v1",
        "writer_host": plan["bindings"]["writer_host"],
        "lease_sha256": digest(lease),
        "event_sha256": plan["event_sha256"],
        "input_manifest_sha256": lease["input_manifest_sha256"],
    }


def _copy_file(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    with destination.open("r+b") as stream:
        os.fsync(stream.fileno())


def _activation_files(xroot, plan):
    relative = {
        "current/ownership-certificate.json", "current/runtime-manifest.json",
        "current/crossnode-proof.json", "current/bindings.json",
        "current/p5_xnode_control.py", "current/p5_xnode_state.py",
        "current/selected-lineage-readmission.json",
    }
    for cell in plan["active_cells"]:
        relative.update({
            f"leases/{cell}.json",
            f"roots/{cell}/checkpoint/model.safetensors",
            f"roots/{cell}/checkpoint/training-state.pt",
            f"roots/{cell}/checkpoint/p5-seal.json",
            f"roots/{cell}/p5-run-config.json",
            f"roots/{cell}/latest.json",
        })
        if plan["bindings"]["cells"][cell]["role"] == "worker":
            relative.add(
                f"worker-bootstrap/{cell}/.xnode/registration.json"
            )
    return {name: sha(Path(xroot) / name) for name in sorted(relative)}


def _require_document(path, expected):
    path = Path(path)
    if (not path.is_file()
            or sha(path) != document_sha(expected)
            or read(path) != expected):
        raise ValueError("Canonical protocol-repair document differs")


def _verify_stage(xroot, plan, certificate):
    xroot = Path(xroot)
    _require_document(
        xroot / "current" / "ownership-certificate.json", certificate,
    )
    _require_document(
        xroot / "current" / "runtime-manifest.json", plan["runtime_manifest"],
    )
    _require_document(
        xroot / "current" / "crossnode-proof.json", plan["aggregate_proof"],
    )
    _require_document(xroot / "current" / "bindings.json", plan["bindings"])
    _require_document(
        xroot / "current" / "selected-lineage-readmission.json",
        plan["selected_readmission"],
    )
    source = Path(plan["runtime_source_directory"])
    for name, expected in plan["runtime_manifest"]["files_sha256"].items():
        if (sha(source / name) != expected
                or sha(xroot / "current" / name) != expected):
            raise ValueError("Canonical protocol-repair runtime source differs")
    for cell in plan["active_cells"]:
        lease = plan["owners"][cell]
        _require_document(xroot / "leases" / f"{cell}.json", lease)
        root = xroot / "roots" / cell
        for name, key in (
            ("p5-seal.json", "seal_sha256"),
            ("model.safetensors", "model_sha256"),
            ("training-state.pt", "training_state_sha256"),
        ):
            if sha(root / "checkpoint" / name) != lease["root"][key]:
                raise ValueError("Protocol repair changed a scientific root")
        if sha(root / "p5-run-config.json") != lease["contract_file_sha256"]:
            raise ValueError("Protocol repair changed a scientific contract")
        _require_document(root / "latest.json", {
            "artifact": lease["root"]["artifact"],
            "seal_sha256": lease["root"]["seal_sha256"],
            "global_step": lease["root"]["step"],
            "phase": lease["root"]["phase"],
        })
        if plan["bindings"]["cells"][cell]["role"] == "worker":
            transport = xroot / "worker-bootstrap" / cell / ".xnode"
            if ({path.name for path in transport.iterdir()}
                    != {"registration.json"}):
                raise ValueError("Protocol-repair worker bootstrap is not clean")
            _require_document(
                transport / "registration.json",
                worker_registration(plan, cell),
            )
    return True


def _stage(plan, certificate):
    xroot = Path(plan["bindings"]["control_root"])
    v1._atomic_write(
        xroot / "current" / "ownership-certificate.json", certificate,
    )
    v1._atomic_write(
        xroot / "current" / "runtime-manifest.json", plan["runtime_manifest"],
    )
    v1._atomic_write(
        xroot / "current" / "crossnode-proof.json", plan["aggregate_proof"],
    )
    v1._atomic_write(xroot / "current" / "bindings.json", plan["bindings"])
    v1._atomic_write(
        xroot / "current" / "selected-lineage-readmission.json",
        plan["selected_readmission"],
    )
    source = Path(plan["runtime_source_directory"])
    for name in ("p5_xnode_control.py", "p5_xnode_state.py"):
        _copy_file(source / name, xroot / "current" / name)
    for cell in plan["active_cells"]:
        v1._atomic_write(xroot / "leases" / f"{cell}.json", plan["owners"][cell])
        if plan["bindings"]["cells"][cell]["role"] == "worker":
            transport = xroot / "worker-bootstrap" / cell / ".xnode"
            if transport.exists():
                for partial in transport.glob("registration.json.partial-*"):
                    if not partial.is_file() and not partial.is_symlink():
                        raise ValueError("Worker registration partial path is unsafe")
                    partial.unlink()
                if ({path.name for path in transport.iterdir()}
                        != {"registration.json"}):
                    raise ValueError("Existing worker bootstrap is not clean")
            else:
                transport.mkdir(parents=True)
            v1._atomic_write(
                transport / "registration.json",
                worker_registration(plan, cell),
            )
    _verify_stage(xroot, plan, certificate)
    return xroot


def _activation_document(xroot, plan):
    return {
        "kind": "p5-xnode-activation-v2",
        "status": "activated",
        "sequence": plan["event"]["sequence"],
        "event_sha256": plan["event_sha256"],
        "broker_release_sha256": plan["broker_release_sha256"],
        "parent_activation_sha256": plan["parent_activation_sha256"],
        "files_sha256": _activation_files(xroot, plan),
    }


def verify_activation(xroot, broker_release_sha256):
    xroot = Path(xroot)
    certificate = read(xroot / "current" / "ownership-certificate.json")
    event = certificate["event"]
    receipt = certificate.get("bank_receipt", {})
    activation = read(xroot / "activation" / "last-published.json")
    active = event.get("active_cells")
    expected_plan = {
        "sha256": event.get("selected_lineage_deployment", {}).get("sha256"),
        "cells": active,
    }
    if (event.get("transition_kind") != "p5-protocol-repair-v1"
            or event_sha(event) != certificate.get("event_sha256")
            or certificate.get("writer_host") is None
            or digest(receipt) != certificate.get("bank_receipt_sha256")
            or receipt.get("files", {}).get("ownership.json", {}).get("sha256")
            != certificate.get("event_sha256")
            or not isinstance(active, list)
            or active != sorted(active)
            or active != event.get("transition_cells")
            or event.get("selected_lineage_deployment") != expected_plan
            or activation.get("kind") != "p5-xnode-activation-v2"
            or activation.get("status") != "activated"
            or activation.get("sequence") != event.get("sequence")
            or activation.get("event_sha256") != certificate["event_sha256"]
            or activation.get("broker_release_sha256") != broker_release_sha256
            or activation.get("parent_activation_sha256")
            != event.get("parent_activation_sha256")
            or event.get("broker_release_sha256") != broker_release_sha256):
        raise ValueError("Protocol-repair activation authority differs")
    expected = {
        "current/ownership-certificate.json", "current/runtime-manifest.json",
        "current/crossnode-proof.json", "current/bindings.json",
        "current/p5_xnode_control.py", "current/p5_xnode_state.py",
        "current/selected-lineage-readmission.json",
    }
    for cell in active:
        expected.update({
            f"leases/{cell}.json",
            f"roots/{cell}/checkpoint/model.safetensors",
            f"roots/{cell}/checkpoint/training-state.pt",
            f"roots/{cell}/checkpoint/p5-seal.json",
            f"roots/{cell}/p5-run-config.json",
            f"roots/{cell}/latest.json",
        })
        if event["owners"][cell]["owner_node"] != certificate["writer_host"]:
            expected.add(
                f"worker-bootstrap/{cell}/.xnode/registration.json"
            )
    if set(activation.get("files_sha256", {})) != expected:
        raise ValueError("Protocol-repair activation inventory is incomplete")
    for relative, expected_sha in activation["files_sha256"].items():
        if sha(xroot / relative) != expected_sha:
            raise ValueError("Protocol-repair activated file changed")
    runtime_path = xroot / "current" / "runtime-manifest.json"
    proof_path = xroot / "current" / "crossnode-proof.json"
    proof = read(proof_path)
    if (sha(runtime_path) != proof.get("runtime_manifest_sha256")
            or sha(proof_path) != event.get("crossnode_proof_sha256")
            or sha(xroot / "current" / "selected-lineage-readmission.json")
            != event.get("selected_lineage_readmission_sha256")):
        raise ValueError("Protocol-repair runtime, proof or readmission differs")
    for cell in active:
        lease = read(xroot / "leases" / f"{cell}.json")
        if (lease != event["owners"][cell]
                or lease["control_manifest_sha256"] != sha(runtime_path)):
            raise ValueError("Protocol-repair activated lease differs")
    return certificate


def _validate_plan_static(request_path, plan):
    request_path = Path(request_path)
    request = _request(request_path)
    if (plan.get("kind") != "p5-protocol-repair-plan-v1"
            or plan.get("status") != "prepared"
            or plan.get("request_sha256") != sha(request_path)
            or plan.get("event_sha256") != event_sha(plan.get("event", {}))
            or plan.get("parent_event_sha256") != request["parent_event_sha256"]
            or plan.get("bindings") != read(request["new_bindings"])
            or plan.get("active_cells") != request["active_cells"]
            or plan.get("event", {}).get("owners") != plan.get("owners")
            or plan.get("event", {}).get("active_cells") != plan.get("active_cells")
            or plan.get("event", {}).get("selected_lineage_readmission_sha256")
            != document_sha(plan.get("selected_readmission", {}))):
        raise ValueError("Protocol-repair plan identity differs")
    _verify_external_evidence(request, plan)
    if (_readmission(
            request["selected_readmission"], plan,
    ) != plan["selected_readmission"]):
        raise ValueError("Selected readmission changed after preparation")
    source = Path(plan["runtime_source_directory"])
    if any(
        sha(source / name) != expected
        for name, expected in plan["runtime_manifest"]["files_sha256"].items()
    ):
        raise ValueError("Protocol-repair runtime source changed")
    return request


def _durable_certificate(plan, bindings, durable):
    ownership_path = durable / "ownership.json"
    receipt_path = durable / "bank-receipt.json"
    durable.mkdir(parents=True, exist_ok=True)
    for name in ("ownership.json", "bank-receipt.json"):
        for partial in durable.glob(name + ".partial-*"):
            if not partial.is_file() and not partial.is_symlink():
                raise ValueError("Durable repair partial path is unsafe")
            partial.unlink()
    if any(
        path.name not in {"ownership.json", "bank-receipt.json"}
        for path in durable.iterdir()
    ):
        raise ValueError("Durable repair directory has foreign members")
    if ownership_path.exists():
        if read(ownership_path) != plan["event"]:
            raise ValueError("Existing durable repair event differs")
    else:
        v1._atomic_write(ownership_path, plan["event"])
    expected_receipt = {"files": {"ownership.json": {
        "sha256": sha(ownership_path),
        "bytes": ownership_path.stat().st_size,
    }}}
    if receipt_path.exists():
        if read(receipt_path) != expected_receipt:
            raise ValueError("Existing durable repair receipt differs")
    else:
        v1._atomic_write(receipt_path, expected_receipt)
    receipt = read(receipt_path)
    return {
        "event": plan["event"],
        "event_sha256": plan["event_sha256"],
        "writer_host": bindings["writer_host"],
        "bank_receipt": receipt,
        "bank_receipt_sha256": digest(receipt),
        "protocol_repair_plan_sha256": digest(plan),
        "selected_lineage_deployment_sha256":
            plan["selected_lineage_deployment_sha256"],
        "selected_lineage_readmission_sha256":
            plan["event"]["selected_lineage_readmission_sha256"],
    }


def _certificate_path(registry, event_sha256):
    matches = [
        path for path in (Path(registry) / "committed").glob("*.json")
        if read(path).get("event_sha256") == event_sha256
    ]
    if len(matches) > 1:
        raise ValueError("Protocol-repair certificate is ambiguous")
    return matches[0] if matches else None


def commit_repair(
    request_path, plan, fail_before_activation=False, fail_at=None,
):
    request = _validate_plan_static(request_path, plan)
    bindings = plan["bindings"]
    if socket.gethostname() != bindings["writer_host"]:
        raise RuntimeError("Only the configured writer may commit protocol repair")
    registry = Path(request["registry"])
    intent = {
        "kind": "p5-protocol-repair-intent-v1",
        "status": "prepared",
        "event_sha256": plan["event_sha256"],
        "previous_event_sha256": plan["parent_event_sha256"],
        "sequence": plan["event"]["sequence"],
        "plan_sha256": digest(plan),
    }
    with v1._lock(registry / "registry.lock"):
        current = v2._load_registry(
            registry, bindings["cells"], plan["event_sha256"],
        )
        if current["event_sha256"] not in {
                plan["parent_event_sha256"], plan["event_sha256"],
        }:
            raise ValueError("Ownership changed after protocol-repair preparation")
        intent_path = registry / "transition-intent.json"
        existing_intent = read(intent_path) if intent_path.exists() else None
        if (existing_intent is not None
                and existing_intent.get("event_sha256")
                not in {current["event_sha256"], plan["event_sha256"]}):
            raise ValueError("A foreign ownership transition is pending")
        if current["event_sha256"] == plan["parent_event_sha256"]:
            verify_repair(request_path, plan)
            v1._atomic_write(intent_path, intent)
            if fail_at == "after-intent":
                raise RuntimeError("Injected failure after protocol-repair intent")
        else:
            if (current.get("event") != plan["event"]
                    or current.get("protocol_repair_plan_sha256")
                    != digest(plan)):
                raise ValueError("Committed protocol repair differs from supplied plan")
        with v1._lock(Path(bindings["root"]) / "bank.lock"):
            _verify_external_evidence(request, plan)
            if (_readmission(
                    request["selected_readmission"], plan,
            ) != plan["selected_readmission"]):
                raise ValueError("Selected readmission changed before commit")
            durable = (
                Path(bindings["durable_root"]) / "ownership-protocol-repair-v1"
                / plan["event_sha256"]
            )
            certificate = _durable_certificate(plan, bindings, durable)
            certificate_path = _certificate_path(
                registry, plan["event_sha256"],
            )
            if certificate_path is None:
                commits = sorted((registry / "committed").glob("*.json"))
                certificate_path = (
                    registry / "committed"
                    / f"{len(commits) + 1:04d}-{plan['event_sha256'][:16]}.json"
                )
                v1._atomic_write(certificate_path, certificate)
            elif read(certificate_path) != certificate:
                raise ValueError("Existing protocol-repair certificate differs")
            if fail_at == "after-certificate":
                raise RuntimeError("Injected failure after protocol-repair certificate")
            xroot = _stage(plan, certificate)
            v1._atomic_write(intent_path, {
                **intent,
                "status": "committed",
                "certificate_sha256": sha(certificate_path),
            })
            if fail_before_activation:
                raise RuntimeError(
                    "Injected interruption before protocol-repair activation"
                )
            activation = _activation_document(xroot, plan)
            v1._atomic_write(
                xroot / "activation" / "last-published.json", activation,
            )
            v1._atomic_write(intent_path, {
                **intent,
                "status": "activated",
                "certificate_sha256": sha(certificate_path),
                "activation_sha256": sha(
                    xroot / "activation" / "last-published.json"
                ),
            })
    verify_activation(xroot, plan["broker_release_sha256"])
    return {
        "status": "activated",
        "event_sha256": plan["event_sha256"],
        "certificate": str(certificate_path),
        "activation": str(xroot / "activation" / "last-published.json"),
    }


def activate_repair(request_path, plan):
    request_path = Path(request_path)
    request = _request(request_path)
    if (plan.get("kind") != "p5-protocol-repair-plan-v1"
            or plan.get("status") != "prepared"
            or plan.get("request_sha256") != sha(request_path)
            or plan.get("event_sha256") != event_sha(plan.get("event", {}))):
        raise ValueError("Activation requires the exact prepared repair plan")
    bindings = plan["bindings"]
    if socket.gethostname() != bindings["writer_host"]:
        raise RuntimeError("Only the configured writer may activate protocol repair")
    registry = Path(request["registry"])
    xroot = Path(bindings["control_root"])
    with v1._lock(registry / "registry.lock"), v1._lock(
        Path(bindings["root"]) / "bank.lock"
    ):
        current = v2._load_registry(
            registry, bindings["cells"], plan["event_sha256"],
        )
        if current["event_sha256"] != plan["event_sha256"]:
            raise ValueError("Only the current committed repair may be activated")
        certificate_path = _certificate_path(registry, plan["event_sha256"])
        intent_path = registry / "transition-intent.json"
        committed_intent = read(intent_path) if intent_path.exists() else {}
        if (certificate_path is None
                or current.get("protocol_repair_plan_sha256") != digest(plan)
                or committed_intent.get("event_sha256") != plan["event_sha256"]
                or committed_intent.get("plan_sha256") != digest(plan)
                or committed_intent.get("certificate_sha256")
                != sha(certificate_path)):
            raise ValueError("Committed repair plan or intent differs")
        canonical_certificate = read(
            xroot / "current" / "ownership-certificate.json"
        )
        if canonical_certificate != current:
            raise ValueError("Canonical repair certificate differs from registry")
        _verify_external_evidence(request, plan)
        if (_readmission(
                request["selected_readmission"], plan,
        ) != plan["selected_readmission"]):
            raise ValueError("Selected readmission changed before activation")
        _verify_stage(xroot, plan, current)
        v1._atomic_write(
            xroot / "activation" / "last-published.json",
            _activation_document(xroot, plan),
        )
        v1._atomic_write(intent_path, {
            "kind": "p5-protocol-repair-intent-v1",
            "status": "activated",
            "event_sha256": plan["event_sha256"],
            "previous_event_sha256": plan["parent_event_sha256"],
            "sequence": plan["event"]["sequence"],
            "plan_sha256": digest(plan),
            "certificate_sha256": sha(certificate_path),
            "activation_sha256": sha(
                xroot / "activation" / "last-published.json"
            ),
        })
    verify_activation(xroot, plan["broker_release_sha256"])
    return {
        "status": "activated",
        "event_sha256": plan["event_sha256"],
        "activation": str(xroot / "activation" / "last-published.json"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=(
        "draft", "build-readmission", "prepare", "verify", "commit", "activate",
        "verify-activation",
    ))
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--broker-release-sha256")
    args = parser.parse_args()
    if args.mode in {"draft", "prepare"}:
        if args.output is None:
            raise ValueError("Draft and prepare require --output")
        value = (
            draft_repair(args.request)
            if args.mode == "draft"
            else prepare_repair(args.request)
        )
        v1._atomic_write(args.output, value)
    elif args.mode == "build-readmission":
        print(json.dumps(build_readmission(args.request), sort_keys=True))
    elif args.mode == "verify":
        verify_repair(args.request, read(args.plan))
    elif args.mode == "commit":
        print(json.dumps(
            commit_repair(args.request, read(args.plan)), sort_keys=True,
        ))
    elif args.mode == "activate":
        print(json.dumps(
            activate_repair(args.request, read(args.plan)), sort_keys=True,
        ))
    else:
        request = _request(args.request)
        bindings = read(request["new_bindings"])
        verify_activation(
            bindings["control_root"], args.broker_release_sha256,
        )


if __name__ == "__main__":
    main()
