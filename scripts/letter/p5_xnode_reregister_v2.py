"""Deployable offline P5 ownership re-registration after job-wide GPU replacement."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import socket
from pathlib import Path

import p5_xnode_reregister as v1


VERSION = "p5-xnode-uuid-reregistration-v2"
COMPACT_WRITER_OWNER_FIELDS = {
    "cell", "gpu", "lease_id", "owner_node", "ownership_epoch",
}
FULL_WRITER_OWNER_FIELDS = COMPACT_WRITER_OWNER_FIELDS | {
    "kind", "gpu_uuid", "root", "contract_file_sha256",
    "input_manifest_sha256", "control_manifest_sha256",
    "microbatch", "accumulation",
}
REGISTRATION_FIELDS = {
    "kind", "writer_host", "lease_sha256", "event_sha256",
    "input_manifest_sha256",
}

read = v1.read
canonical = v1.canonical
digest = v1.digest
sha = v1.sha
event_sha = v1.event_sha
_atomic_write = v1._atomic_write
_lock = v1._lock


def document_sha(value):
    import hashlib
    return hashlib.sha256(canonical(value)).hexdigest()


def _request(path):
    value = read(path)
    required = {
        "kind", "cells", "old_bindings", "new_bindings", "registry", "evidence",
        "proofs", "broker_release_sha256", "runtime_template",
    }
    if set(value) != required or value["kind"] != "p5-uuid-reregistration-request-v2":
        raise ValueError("Unknown v2 re-registration request")
    if not re.fullmatch(r"[0-9a-f]{64}", value["broker_release_sha256"]):
        raise ValueError("Exact broker-v3 release SHA-256 is required")
    return value


def _runtime_manifest(request, bindings):
    template_path = Path(request["runtime_template"])
    template = read(template_path)
    if (template.get("kind") != "p5-xnode-runtime-v1"
            or set(template.get("files_sha256", {}))
            != {"p5_xnode_control.py", "p5_xnode_state.py"}):
        raise ValueError("The unchanged worker runtime template is invalid")
    for name, expected in template["files_sha256"].items():
        if sha(template_path.parent / name) != expected:
            raise ValueError("Worker control/state source changed")
    if sha(template["input_manifest"]) != template.get("input_manifest_sha256"):
        raise ValueError("Unchanged input manifest differs from the runtime template")
    xroot = Path(bindings["control_root"])
    manifest = copy.deepcopy(template)
    manifest.update({
        "root": bindings["root"], "p5_root": bindings["p5_root"],
        "control_root": bindings["control_root"],
        "durable_root": bindings["durable_root"], "python": bindings["python"],
        "ownership_certificate": str(xroot / "current" / "ownership-certificate.json"),
        "crossnode_proof": str(xroot / "current" / "crossnode-proof.json"),
    })
    return manifest


def _writer_owner_shape(owner):
    if set(owner) == COMPACT_WRITER_OWNER_FIELDS:
        return "compact"
    if (owner.get("kind") == "p5-xnode-lease-v1"
            and FULL_WRITER_OWNER_FIELDS <= set(owner)
            and isinstance(owner.get("root"), dict)):
        return "full"
    raise ValueError("Historical writer owner shape is ambiguous")


def _load_registry(directory, expected_cells, pending_event_sha=None):
    directory = Path(directory)
    commits = sorted((directory / "committed").glob("*.json"))
    if not commits:
        raise ValueError("No committed ownership ancestry")
    previous = None
    previous_sequence = None
    sequenced = False
    latest = None
    for index, path in enumerate(commits):
        certificate = read(path)
        event, receipt = certificate["event"], certificate["bank_receipt"]
        encoded = event_sha(event)
        sequence = event.get("sequence")
        if sequence is None:
            valid_sequence = not sequenced
        elif type(sequence) is int and sequence > 0:
            valid_sequence = (
                index == 0 and previous is None and event.get("previous_event_sha256") is None
                or not sequenced and sequence == 1
                or sequenced and sequence == previous_sequence + 1
            )
        else:
            valid_sequence = False
        if (event.get("kind") != "p5-xnode-ownership-v1"
                or event.get("status") != "committed"
                or set(event.get("owners", {})) != set(expected_cells)
                or event.get("previous_event_sha256") != previous
                or certificate.get("event_sha256") != encoded
                or certificate.get("bank_receipt_sha256") != digest(receipt)
                or receipt.get("files", {}).get("ownership.json", {}).get("sha256") != encoded
                or not valid_sequence):
            raise ValueError("Invalid committed ownership ancestry or certificate receipt")
        if sequence is not None:
            sequenced = True
            previous_sequence = sequence
        previous, latest = encoded, certificate
    intent = directory / "transition-intent.json"
    if intent.exists():
        pending = read(intent).get("event_sha256")
        if pending != latest["event_sha256"] and pending != pending_event_sha:
            raise ValueError("A pending ownership transition must be recovered first")
    return latest


def _writer_root(evidence, old_owner, writer_host):
    shape = _writer_owner_shape(old_owner)
    if evidence.get("kind") == "p5-uuid-reregistration-writer-durable-root-v2":
        raise AssertionError("Durable writer recovery requires event and destination binding")
    if (evidence.get("kind") != "p5-uuid-reregistration-writer-root-v2"
            or evidence.get("cell") != old_owner["cell"]):
        raise ValueError("Writer cells require ordinary receipt-verified bank evidence")
    checkpoint = Path(evidence["checkpoint"])
    header = v1._checkpoint_header(checkpoint)
    ack = read(evidence["ack"])
    receipt = read(checkpoint / "bank-receipt.json")
    actual = v1._file_manifest(checkpoint, {"bank-receipt.json"})
    if (ack.get("kind") != "p5-full-state-v1" or ack.get("status") != "verified"
            or ack.get("artifact") != checkpoint.name
            or ack.get("seal_sha256") != header["seal_sha256"]
            or ack.get("contract_sha256") != header["contract_sha256"]
            or Path(ack.get("destination", "")).resolve() != checkpoint.resolve()
            or ack.get("bank_receipt_sha256") != digest(receipt)
            or ack.get("writer_host") != writer_host
            or receipt != {"files": actual}):
        raise ValueError("Writer full-state bank acknowledgement or receipt differs")
    contract = Path(evidence["contract"])
    contract_file_sha256 = sha(contract)
    if digest(read(contract)) != header["contract_sha256"]:
        raise ValueError("Writer scientific contract changed")
    if shape == "full":
        if (contract_file_sha256 != old_owner.get("contract_file_sha256")
                or header["contract_sha256"] != old_owner["root"]["contract_sha256"]):
            raise ValueError("Writer banked full state changed scientific contract")
        v1._progress(old_owner["root"], header)
    evidence_identity = {
        "checkpoint": sha(checkpoint / "p5-seal.json"),
        "ack": sha(evidence["ack"]),
        "contract": contract_file_sha256,
    }
    return header, {
        "evidence_kind": "writer-bank",
        "bank_ack_sha256": sha(evidence["ack"]),
        "bank_receipt_sha256": sha(checkpoint / "bank-receipt.json"),
        "writer_evidence_sha256": digest(evidence_identity),
        "contract_file_sha256": contract_file_sha256,
    }, checkpoint, contract


def _writer_durable_root(
    evidence, old_owner, writer_host, registry_event_sha256, durable_root,
):
    _writer_owner_shape(old_owner)
    required = {
        "kind", "cell", "checkpoints", "contract", "registry_event_sha256",
        "destination_root", "selection",
    }
    if (set(evidence) != required
            or evidence["kind"] != "p5-uuid-reregistration-writer-durable-root-v2"
            or evidence["cell"] != old_owner["cell"]
            or evidence["registry_event_sha256"] != registry_event_sha256
            or not isinstance(evidence["checkpoints"], list)
            or not evidence["checkpoints"]):
        raise ValueError("Durable writer recovery evidence is incomplete or event-stale")
    destination = Path(evidence["destination_root"]).resolve()
    durable_root = Path(durable_root).resolve()
    checkpoints = [Path(path).resolve() for path in evidence["checkpoints"]]
    if (not destination.is_absolute() or destination == durable_root
            or not destination.is_relative_to(durable_root)
            or len(checkpoints) != len(set(checkpoints))
            or any(path.parent != destination for path in checkpoints)):
        raise ValueError("Durable writer recovery destination binding differs")
    entries = list(destination.iterdir())
    if (sorted(path.resolve() for path in entries) != sorted(checkpoints)
            or any(path.is_symlink() or not path.is_dir() for path in entries)):
        raise ValueError("Durable writer recovery inventory is incomplete or unlisted")
    contract = Path(evidence["contract"])
    contract_value = read(contract)
    contract_file_sha256 = sha(contract)
    contract_sha256 = digest(contract_value)
    verified = []
    previous = old_owner.get("root")
    for checkpoint in checkpoints:
        receipt_path = checkpoint / "bank-receipt.json"
        if not receipt_path.is_file():
            raise ValueError("Durable writer checkpoint has no bank receipt")
        receipt = read(receipt_path)
        actual = v1._file_manifest(checkpoint, {"bank-receipt.json"})
        if receipt != {"files": actual}:
            raise ValueError("Durable writer checkpoint receipt inventory differs")
        header = v1._checkpoint_header(checkpoint)
        if header["contract_sha256"] != contract_sha256:
            raise ValueError("Durable writer checkpoint contract seal differs")
        if previous is not None:
            v1._progress(previous, header)
        previous = header
        try:
            import torch
            payload = torch.load(
                checkpoint / "training-state.pt", map_location="cpu", weights_only=False,
            )
        except Exception as error:
            raise ValueError("Durable writer training state cannot be verified") from error
        state = payload.get("state", {})
        if (payload.get("contract") != contract_value
                or payload.get("checkpoint_boundary")
                != "after_complete_optimizer_update_no_partial_gradients"
                or state.get("global_step") != header["step"]
                or state.get("epoch") != header["epoch"]
                or state.get("phase") != header["phase"]):
            raise ValueError("Durable writer embedded contract or full state differs")
        verified.append({
            "path": str(checkpoint),
            "generation": len(verified) + 1,
            "checkpoint_seal_sha256": sha(checkpoint / "p5-seal.json"),
            "bank_receipt_sha256": sha(receipt_path),
            "step": header["step"],
            "epoch": header["epoch"],
            "phase": header["phase"],
        })
    selection = evidence["selection"]
    if not isinstance(selection, dict) or selection.get("policy") not in {
        "latest", "intentional_rollback",
    }:
        raise ValueError("Durable writer recovery selection policy is invalid")
    highest = len(verified)
    chosen_generation = selection.get("generation", highest)
    if type(chosen_generation) is not int or not 1 <= chosen_generation <= highest:
        raise ValueError("Durable writer recovery generation selection is invalid")
    if selection["policy"] == "latest" and chosen_generation != highest:
        raise ValueError("A lower durable writer root requires explicit intentional rollback")
    if (selection["policy"] == "intentional_rollback"
            and (chosen_generation == highest
                 or not isinstance(selection.get("reason"), str)
                 or not selection["reason"].strip())):
        raise ValueError("Intentional writer rollback requires a lower generation and reason")
    chosen_checkpoint = checkpoints[chosen_generation - 1]
    head = v1._checkpoint_header(chosen_checkpoint)
    evidence_identity = {
        "destination_root": str(destination),
        "checkpoints": [{
            "path": item["path"],
            "checkpoint_seal_sha256": item["checkpoint_seal_sha256"],
            "bank_receipt_sha256": item["bank_receipt_sha256"],
        } for item in verified],
        "contract": contract_file_sha256,
        "registry_event_sha256": registry_event_sha256,
    }
    metadata = {
        "evidence_kind": "writer-durable-recovery",
        "recovery_kind": "p5-writer-durable-recovery-v1",
        "authority": "receipt-verified-durable-full-state-not-historical-acceptance",
        "historically_accepted": False,
        "registry_event_sha256": registry_event_sha256,
        "writer_host": writer_host,
        "destination_root": str(destination),
        "selection_policy": selection["policy"],
        "chosen_generation": chosen_generation,
        "chosen_checkpoint": str(chosen_checkpoint),
        "chosen_checkpoint_seal_sha256": head["seal_sha256"],
        "verified_checkpoints": verified,
        "abandoned_verified_checkpoints": verified[chosen_generation:],
        "writer_evidence_sha256": digest(evidence_identity),
        "contract_file_sha256": contract_file_sha256,
    }
    return head, metadata, chosen_checkpoint, contract


def _writer_lease(
    old_owner, old_assignment, new_assignment, writer_host, runtime, runtime_sha,
    head, metadata, lease_id, epoch,
):
    if (old_owner.get("cell") is None
            or old_owner.get("owner_node") != writer_host
            or old_owner.get("gpu") != old_assignment["gpu"]
            or old_owner.get("ownership_epoch") != old_assignment["ownership_epoch"]
            or new_assignment["ownership_epoch"] != old_assignment["ownership_epoch"]):
        raise ValueError("Historical writer identity differs from exact old/new bindings")
    shape = _writer_owner_shape(old_owner)
    compact, full = shape == "compact", shape == "full"
    if full:
        if (old_owner.get("gpu_uuid") != old_assignment["gpu_uuid"]
                or old_owner.get("microbatch") != old_assignment["microbatch"]
                or old_owner.get("accumulation") != old_assignment["accumulation"]
                or old_owner.get("input_manifest_sha256") != runtime["input_manifest_sha256"]):
            raise ValueError("Historical writer lease differs from exact bindings or inputs")
    lease = ({
        key: copy.deepcopy(value) for key, value in old_owner.items()
        if key not in {
            "kind", "cell", "owner_node", "gpu", "gpu_uuid", "lease_id",
            "ownership_epoch", "root", "contract_file_sha256",
            "input_manifest_sha256", "control_manifest_sha256",
            "microbatch", "accumulation",
        }
    } if full else {})
    lease.update({
        "kind": "p5-xnode-lease-v1",
        "cell": old_owner["cell"],
        "owner_node": writer_host,
        "gpu": new_assignment["gpu"],
        "gpu_uuid": new_assignment["gpu_uuid"],
        "lease_id": lease_id,
        "ownership_epoch": epoch,
        "root": {
            key: head[key] for key in (
                "artifact", "seal_sha256", "contract_sha256", "step", "epoch",
                "phase", "model_sha256", "training_state_sha256",
            )
        },
        "contract_file_sha256": metadata["contract_file_sha256"],
        "input_manifest_sha256": runtime["input_manifest_sha256"],
        "control_manifest_sha256": runtime_sha,
        "microbatch": new_assignment["microbatch"],
        "accumulation": new_assignment["accumulation"],
    })
    reconstruction = {
        "kind": "p5-writer-lease-reconstruction-v1",
        "historical_owner_shape": "compact" if compact else "full",
        "historical_owner_sha256": digest(old_owner),
        "old_assignment_sha256": digest(old_assignment),
        "new_assignment_sha256": digest(new_assignment),
        "writer_evidence_sha256": metadata["writer_evidence_sha256"],
        "contract_file_sha256": metadata["contract_file_sha256"],
        "input_manifest_sha256": runtime["input_manifest_sha256"],
        "runtime_manifest_sha256": runtime_sha,
    }
    return lease, reconstruction


def _worker_root(evidence, old_lease, registry_event_sha256, durable_root):
    if evidence.get("kind") == "p5-uuid-reregistration-worker-durable-root-v2":
        return _worker_durable_root(
            evidence, old_lease, registry_event_sha256, durable_root,
        )
    if (evidence.get("kind") != "p5-uuid-reregistration-worker-root-v2"
            or evidence.get("cell") != old_lease["cell"]):
        raise ValueError("Worker cells require accepted xnode ledger/package evidence")
    translated = {**evidence, "kind": "p5-uuid-reregistration-root-v1"}
    head, record, package, contract = v1._accepted_root(
        translated, old_lease, registry_event_sha256,
    )
    return head, {
        "evidence_kind": "worker-xnode",
        "accepted_record_sha256": digest(record),
        "package_seal_sha256": sha(package / "package-seal.json"),
    }, package / "checkpoint", contract


def _worker_durable_root(evidence, old_lease, registry_event_sha256, durable_root):
    required = {
        "kind", "cell", "packages", "contract", "registry_event_sha256",
        "destination_root", "selection", "successor",
    }
    if (set(evidence) != required
            or evidence["cell"] != old_lease["cell"]
            or evidence["registry_event_sha256"] != registry_event_sha256
            or not isinstance(evidence["packages"], list)
            or not evidence["packages"]):
        raise ValueError("Durable worker recovery evidence is incomplete or event-stale")
    destination = Path(evidence["destination_root"]).resolve()
    durable_root = Path(durable_root).resolve()
    if (not destination.is_absolute() or destination == durable_root
            or not destination.is_relative_to(durable_root)):
        raise ValueError("Durable recovery destination must be explicit")
    packages = [Path(path).resolve() for path in evidence["packages"]]
    if (len(packages) != len(set(packages))
            or any(path.parent != destination for path in packages)):
        raise ValueError("Durable recovery package destination binding differs")
    entries = list(destination.iterdir())
    inventory = sorted(path.resolve() for path in entries)
    if (inventory != sorted(packages)
            or any(path.is_symlink() or not path.is_dir() for path in entries)):
        raise ValueError("Durable recovery destination inventory is incomplete or unlisted")

    head = {**old_lease["root"], "generation": 0, "envelope_sha256": None}
    verified = []
    contract = Path(evidence["contract"])
    if (sha(contract) != old_lease["contract_file_sha256"]
            or digest(read(contract)) != old_lease["root"]["contract_sha256"]):
        raise ValueError("Durable recovery scientific contract changed")
    for package in packages:
        if not (package / "bank-receipt.json").is_file():
            raise ValueError("Durable recovery package has no bank receipt")
        marker = read(package / "package-seal.json")
        package_files = v1._file_manifest(
            package, {"package-seal.json", "bank-receipt.json"},
        )
        if (marker.get("kind") != "p5-xnode-package-v1"
                or marker.get("status") != "sealed"
                or set(package_files) != v1.PACKAGE_FILES
                or marker.get("files") != package_files):
            raise ValueError("Durable recovery package is incomplete or changed")
        header = v1._checkpoint_header(package / "checkpoint")
        envelope = read(package / "envelope.json")
        if any(
            envelope.get("checkpoint", {}).get(key) != value
            for key, value in header.items() if key != "artifact"
        ):
            raise ValueError("Durable recovery envelope differs from full state")
        if (digest(read(package / "scientific-contract.json"))
                != header["contract_sha256"]
                or sha(package / "execution-attestation.json")
                != envelope.get("execution_attestation_sha256")):
            raise ValueError("Durable recovery contract or attestation differs")
        attestation = read(package / "execution-attestation.json")
        if (attestation.get("lease_sha256") != digest(old_lease)
                or attestation.get("host") != old_lease["owner_node"]
                or attestation.get("gpu") != old_lease["gpu"]
                or attestation.get("purpose") != "production"
                or attestation.get("root_checkpoint_seal_sha256")
                != old_lease["root"]["seal_sha256"]
                or attestation.get("scientific_contract_rewritten") is not False):
            raise ValueError("Durable package execution provenance differs")
        receipt = read(package / "bank-receipt.json")
        actual = v1._file_manifest(package, {"bank-receipt.json"})
        if receipt != {"files": actual}:
            raise ValueError("Durable package receipt inventory differs")
        head = v1._validate_envelope(envelope, old_lease, head)
        if envelope["generation"] != len(verified) + 1:
            raise ValueError("Durable recovery chain has a generation gap")
        verified.append({
            "path": str(package), "generation": envelope["generation"],
            "checkpoint_seal_sha256": envelope["checkpoint"]["seal_sha256"],
            "package_seal_sha256": sha(package / "package-seal.json"),
            "bank_receipt_sha256": sha(package / "bank-receipt.json"),
        })

    selection = evidence["selection"]
    if not isinstance(selection, dict) or selection.get("policy") not in {
        "latest", "intentional_rollback",
    }:
        raise ValueError("Durable recovery selection policy is invalid")
    highest = verified[-1]["generation"]
    chosen_generation = selection.get("generation", highest)
    if type(chosen_generation) is not int or chosen_generation < 1 or chosen_generation > highest:
        raise ValueError("Durable recovery generation selection is invalid")
    if selection["policy"] == "latest" and chosen_generation != highest:
        raise ValueError("A lower durable root requires explicit intentional rollback")
    if (selection["policy"] == "intentional_rollback"
            and (chosen_generation == highest
                 or not isinstance(selection.get("reason"), str)
                 or not selection["reason"].strip())):
        raise ValueError("Intentional rollback requires a lower generation and reason")
    chosen_package = packages[chosen_generation - 1]
    chosen = read(chosen_package / "envelope.json")["checkpoint"]
    abandoned = verified[chosen_generation:]
    successor = evidence["successor"]
    if successor is not None and not isinstance(successor, dict):
        raise ValueError("Optional successor observation must be an object")
    metadata = {
        "evidence_kind": "worker-durable-recovery",
        "recovery_kind": "p5-xnode-durable-recovery-v1",
        "authority": "receipt-verified-durable-packages-not-historical-acceptance",
        "historically_accepted": False,
        "registry_event_sha256": registry_event_sha256,
        "destination_root": str(destination),
        "selection_policy": selection["policy"],
        "chosen_generation": chosen_generation,
        "chosen_checkpoint_seal_sha256": chosen["seal_sha256"],
        "verified_packages": verified,
        "abandoned_verified_packages": abandoned,
        "successor_observation": successor,
    }
    return chosen, metadata, chosen_package / "checkpoint", contract


def _aggregate_proof(request, bindings, owners, heads, runtime_sha):
    cells = {}
    for cell in request["cells"]:
        assignment = bindings["cells"][cell]
        if assignment["role"] != "worker":
            continue
        proof = v1._validate_proof(
            request["proofs"][cell], cell, owners[cell], assignment,
            heads[cell],
        )
        cells[cell] = {
            "source_checkpoint_seal_sha256": proof["source_checkpoint_seal_sha256"],
            "bitwise_state_match": proof["cold_save_reload_bitwise_equal"],
            "sampled_rows_losses_match": proof["next_sampled_rows_and_losses_equal"],
            "cold_save_reload": proof["cold_save_reload_bitwise_equal"],
            "populated_optimizer_memory_pass": proof["populated_optimizer_memory_pass"],
            "complete_20beam_geometry_pass": proof["complete_20beam_geometry_pass"],
        }
    return {
        "kind": "p5-xnode-crossnode-proof-v1", "status": "pass",
        "source_node": bindings["writer_host"],
        "destination_node": bindings["worker_host"],
        "runtime_manifest_sha256": runtime_sha, "cells": cells,
    }


def prepare_transition(request_path):
    request_path = Path(request_path)
    request = _request(request_path)
    old_binding = v1._validate_bindings(read(request["old_bindings"]))
    new_binding = v1._validate_bindings(read(request["new_bindings"]))
    certificate = _load_registry(request["registry"], old_binding["cells"])
    old_event = certificate["event"]
    if set(old_binding["cells"]) != set(new_binding["cells"]):
        raise ValueError("New bindings must preserve the complete cell set")
    changed = {
        cell for cell in old_binding["cells"]
        if old_binding["cells"][cell]["gpu_uuid"] != new_binding["cells"][cell]["gpu_uuid"]
    }
    if (set(request["cells"]) != changed or set(request["evidence"]) != changed
            or set(request["proofs"]) != changed):
        raise ValueError("The exact all-cell UUID transition set is mandatory")
    if (new_binding["writer_host"] != old_binding["writer_host"]
            or new_binding["worker_host"] != old_binding["worker_host"]):
        raise ValueError("Host assignments cannot change")

    runtime = _runtime_manifest(request, new_binding)
    runtime_sha = document_sha(runtime)
    if any(
        old_binding["cells"][cell]["role"] == "worker"
        and old_event["owners"][cell].get("input_manifest_sha256")
        != runtime["input_manifest_sha256"]
        for cell in changed
    ):
        raise ValueError("Unchanged input manifest differs from an accepted lease")
    owners = copy.deepcopy(old_event["owners"])
    effective = copy.deepcopy(new_binding)
    cells = {}
    roots = {}
    heads = {}
    for cell in sorted(changed):
        old_assignment = old_binding["cells"][cell]
        assignment = new_binding["cells"][cell]
        old_owner = old_event["owners"][cell]
        expected_host = (old_binding["writer_host"] if old_assignment["role"] == "writer"
                         else old_binding["worker_host"])
        if (old_owner["owner_node"] != expected_host
                or old_owner.get("cell") != cell
                or old_owner.get("gpu") != old_assignment["gpu"]
                or old_owner.get("ownership_epoch") != old_assignment["ownership_epoch"]
                or assignment["role"] != old_assignment["role"]
                or assignment["gpu"] != old_assignment["gpu"]
                or assignment["microbatch"] != old_assignment["microbatch"]
                or assignment["accumulation"] != old_assignment["accumulation"]):
            raise ValueError("Role, host, index or batch geometry changed")
        if assignment["role"] == "writer":
            evidence = request["evidence"][cell]
            if evidence.get("kind") == "p5-uuid-reregistration-writer-durable-root-v2":
                head, metadata, checkpoint, contract = _writer_durable_root(
                    evidence, old_owner, old_binding["writer_host"],
                    certificate["event_sha256"], new_binding["durable_root"],
                )
            else:
                head, metadata, checkpoint, contract = _writer_root(
                    evidence, old_owner, old_binding["writer_host"],
                )
        else:
            head, metadata, checkpoint, contract = _worker_root(
                request["evidence"][cell], old_owner, certificate["event_sha256"],
                new_binding["durable_root"],
            )
        proof = v1._validate_proof(
            request["proofs"][cell], cell, old_owner, assignment, head,
        )
        epoch = old_owner["ownership_epoch"] + 1
        effective["cells"][cell]["ownership_epoch"] = epoch
        effective["cells"][cell]["root_step"] = head["step"]
        lease_id = (
            f"{cell.lower()}-epoch-{epoch}-"
            f"{digest({'parent': certificate['event_sha256'], 'cell': cell, 'uuid': assignment['gpu_uuid'], 'proof': sha(request['proofs'][cell])})[:12]}"
        )
        if assignment["role"] == "writer":
            lease, reconstruction = _writer_lease(
                old_owner, old_assignment, assignment, old_binding["writer_host"],
                runtime, runtime_sha, head, metadata, lease_id, epoch,
            )
            metadata["writer_lease_reconstruction"] = reconstruction
        else:
            lease = {
                **old_owner, "lease_id": lease_id, "ownership_epoch": epoch,
                "gpu_uuid": assignment["gpu_uuid"], "root": {
                    key: head[key] for key in (
                        "artifact", "seal_sha256", "contract_sha256", "step", "epoch",
                        "phase", "model_sha256", "training_state_sha256",
                    )
                },
                "control_manifest_sha256": runtime_sha,
            }
        owners[cell] = lease
        cells[cell] = {
            **metadata, "old_lease_sha256": digest(old_owner),
            "new_lease_sha256": digest(lease), "proof_sha256": sha(request["proofs"][cell]),
            "root": lease["root"],
        }
        roots[cell] = {"checkpoint": str(checkpoint), "contract": str(contract)}
        heads[cell] = head

    proof = _aggregate_proof(
        request, new_binding, old_event["owners"], heads, runtime_sha,
    )
    parent_sequence = old_event.get("sequence", 0)
    if type(parent_sequence) is not int or parent_sequence < 0:
        raise ValueError("Invalid parent ownership sequence")
    event = {
        "kind": "p5-xnode-ownership-v1", "status": "committed",
        "sequence": parent_sequence + 1, "owners": owners,
        "previous_event_sha256": certificate["event_sha256"],
        "broker_release_sha256": request["broker_release_sha256"],
        "crossnode_proof_sha256": document_sha(proof),
        "replacement_proofs_sha256": {
            cell: sha(request["proofs"][cell]) for cell in sorted(changed)
        },
        "durable_recovery": {
            cell: cells[cell] for cell in sorted(changed)
            if cells[cell].get("historically_accepted") is False
        },
        "transition_kind": VERSION, "transition_cells": sorted(changed),
    }
    return {
        "kind": "p5-uuid-reregistration-plan-v2", "status": "prepared",
        "request_sha256": sha(request_path), "cells": cells, "roots": roots,
        "bindings": effective, "runtime_manifest": runtime,
        "runtime_manifest_sha256": runtime_sha, "aggregate_proof": proof,
        "event": event, "event_sha256": event_sha(event),
        "runtime_source_directory": str(Path(request["runtime_template"]).parent),
        "publication_generation_policy": "reset-to-zero-at-verified-selected-root",
    }


def verify_transition(request_path, plan):
    expected = prepare_transition(request_path)
    if plan != expected:
        raise ValueError("V2 transition plan differs from current evidence")
    return plan


def worker_registration(plan, cell):
    lease = plan["event"]["owners"].get(cell)
    if lease is None or plan["bindings"]["cells"].get(cell, {}).get("role") != "worker":
        raise ValueError("Registration requested for a non-worker cell")
    value = {
        "kind": "p5-xnode-registration-v1",
        "writer_host": plan["bindings"]["writer_host"],
        "lease_sha256": digest(lease), "event_sha256": plan["event_sha256"],
        "input_manifest_sha256": lease["input_manifest_sha256"],
    }
    if set(value) != REGISTRATION_FIELDS:
        raise AssertionError("Registration core changed")
    return value


def _copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    with destination.open("r+b") as stream:
        os.fsync(stream.fileno())


def _stage_canonical(plan, certificate):
    xroot = Path(plan["bindings"]["control_root"])
    current = xroot / "current"
    _atomic_write(current / "ownership-certificate.json", certificate)
    _atomic_write(current / "runtime-manifest.json", plan["runtime_manifest"])
    _atomic_write(current / "crossnode-proof.json", plan["aggregate_proof"])
    _atomic_write(current / "bindings.json", plan["bindings"])
    source_runtime = Path(plan["runtime_source_directory"])
    for name in ("p5_xnode_control.py", "p5_xnode_state.py"):
        _copy_file(source_runtime / name, current / name)
    for cell in plan["event"]["transition_cells"]:
        _atomic_write(xroot / "leases" / f"{cell}.json", plan["event"]["owners"][cell])
        root = xroot / "roots" / cell
        checkpoint = Path(plan["roots"][cell]["checkpoint"])
        for name in ("model.safetensors", "training-state.pt", "p5-seal.json"):
            _copy_file(checkpoint / name, root / "checkpoint" / name)
        _copy_file(plan["roots"][cell]["contract"], root / "p5-run-config.json")
        _atomic_write(root / "latest.json", {
            "artifact": plan["event"]["owners"][cell]["root"]["artifact"],
            "seal_sha256": plan["event"]["owners"][cell]["root"]["seal_sha256"],
            "global_step": plan["event"]["owners"][cell]["root"]["step"],
            "phase": plan["event"]["owners"][cell]["root"]["phase"],
        })
        if plan["bindings"]["cells"][cell]["role"] == "worker":
            transport = xroot / "worker-bootstrap" / cell / ".xnode"
            transport.mkdir(parents=True, exist_ok=False)
            _atomic_write(transport / "registration.json", worker_registration(plan, cell))
    return xroot


def _activation_files(xroot, plan):
    relative = [
        "current/ownership-certificate.json", "current/runtime-manifest.json",
        "current/crossnode-proof.json", "current/bindings.json",
        "current/p5_xnode_control.py", "current/p5_xnode_state.py",
    ]
    for cell in plan["event"]["transition_cells"]:
        relative.extend([
            "/".join(("leases", cell + ".json")),
            "/".join(("roots", cell, "checkpoint", "model.safetensors")),
            "/".join(("roots", cell, "checkpoint", "training-state.pt")),
            "/".join(("roots", cell, "checkpoint", "p5-seal.json")),
            "/".join(("roots", cell, "p5-run-config.json")),
            "/".join(("roots", cell, "latest.json")),
        ])
        if plan["bindings"]["cells"][cell]["role"] == "worker":
            relative.append("/".join(
                ("worker-bootstrap", cell, ".xnode", "registration.json"),
            ))
    return {name: sha(xroot / name) for name in sorted(relative)}


def _require_document(path, expected):
    path = Path(path)
    if (not path.is_file() or sha(path) != document_sha(expected)
            or read(path) != expected):
        raise ValueError("Canonical staged JSON document differs")


def _verify_canonical_stage(xroot, plan, certificate):
    xroot = Path(xroot)
    _require_document(xroot / "current" / "ownership-certificate.json", certificate)
    _require_document(xroot / "current" / "runtime-manifest.json", plan["runtime_manifest"])
    _require_document(xroot / "current" / "crossnode-proof.json", plan["aggregate_proof"])
    _require_document(xroot / "current" / "bindings.json", plan["bindings"])
    source_runtime = Path(plan["runtime_source_directory"])
    for name, expected in plan["runtime_manifest"]["files_sha256"].items():
        if (sha(source_runtime / name) != expected
                or sha(xroot / "current" / name) != expected):
            raise ValueError("Canonical staged runtime source hash differs")
    for cell in plan["event"]["transition_cells"]:
        lease = plan["event"]["owners"][cell]
        _require_document(xroot / "leases" / f"{cell}.json", lease)
        checkpoint = xroot / "roots" / cell / "checkpoint"
        for name, key in (
            ("p5-seal.json", "seal_sha256"),
            ("model.safetensors", "model_sha256"),
            ("training-state.pt", "training_state_sha256"),
        ):
            if sha(checkpoint / name) != lease["root"][key]:
                raise ValueError("Canonical staged root checkpoint hash differs")
        if sha(xroot / "roots" / cell / "p5-run-config.json") != lease["contract_file_sha256"]:
            raise ValueError("Canonical staged scientific contract differs")
        _require_document(xroot / "roots" / cell / "latest.json", {
            "artifact": lease["root"]["artifact"],
            "seal_sha256": lease["root"]["seal_sha256"],
            "global_step": lease["root"]["step"], "phase": lease["root"]["phase"],
        })
        if plan["bindings"]["cells"][cell]["role"] == "worker":
            transport = xroot / "worker-bootstrap" / cell / ".xnode"
            if (not transport.is_dir()
                    or {path.name for path in transport.iterdir()} != {"registration.json"}):
                raise ValueError("Canonical staged worker transport is not clean")
            _require_document(
                transport / "registration.json", worker_registration(plan, cell),
            )
    return True


def _activation_document(xroot, plan):
    return {
        "kind": "p5-xnode-activation-v1", "status": "activated",
        "sequence": plan["event"]["sequence"],
        "event_sha256": plan["event_sha256"],
        "broker_release_sha256": plan["event"]["broker_release_sha256"],
        "files_sha256": _activation_files(xroot, plan),
    }


def activate_transition(request_path, plan):
    if (plan.get("kind") != "p5-uuid-reregistration-plan-v2"
            or plan.get("status") != "prepared"
            or plan.get("request_sha256") != sha(request_path)
            or plan.get("event_sha256") != event_sha(plan.get("event", {}))):
        raise ValueError("Activation requires the exact prepared v2 plan and request")
    bindings = plan["bindings"]
    if socket.gethostname() != bindings["writer_host"]:
        raise RuntimeError("Only the configured writer may activate ownership")
    registry = Path(_request(request_path)["registry"])
    xroot = Path(bindings["control_root"])
    with _lock(registry / "registry.lock"), _lock(Path(bindings["root"]) / "bank.lock"):
        current = _load_registry(
            registry, bindings["cells"], plan["event_sha256"],
        )
        if current["event_sha256"] != plan["event_sha256"]:
            raise ValueError("Only the current committed v2 event may be activated")
        canonical = read(xroot / "current" / "ownership-certificate.json")
        if canonical != current:
            raise ValueError("Canonical certificate differs from current committed ownership")
        _verify_canonical_stage(xroot, plan, current)
        activation = _activation_document(xroot, plan)
        _atomic_write(xroot / "activation" / "last-published.json", activation)
        _atomic_write(registry / "transition-intent.json", {
            "kind": "p5-xnode-transition-intent-v2", "status": "activated",
            "event_sha256": plan["event_sha256"],
            "previous_event_sha256": plan["event"]["previous_event_sha256"],
            "sequence": plan["event"]["sequence"], "plan_sha256": digest(plan),
            "certificate_sha256": sha(
                sorted((registry / "committed").glob("*.json"))[-1]
            ),
            "activation_sha256": sha(xroot / "activation" / "last-published.json"),
        })
    verify_activation(xroot, plan["event"]["broker_release_sha256"])
    return {
        "status": "activated", "event_sha256": plan["event_sha256"],
        "activation": str(xroot / "activation" / "last-published.json"),
    }


def commit_transition(request_path, plan, fail_before_activation=False):
    plan = verify_transition(request_path, plan)
    bindings = plan["bindings"]
    if socket.gethostname() != bindings["writer_host"]:
        raise RuntimeError("Only the configured writer may commit ownership")
    registry = Path(_request(request_path)["registry"])
    intent = {
        "kind": "p5-xnode-transition-intent-v2", "status": "prepared",
        "event_sha256": plan["event_sha256"],
        "previous_event_sha256": plan["event"]["previous_event_sha256"],
        "sequence": plan["event"]["sequence"], "plan_sha256": digest(plan),
    }
    with _lock(registry / "registry.lock"):
        current = _load_registry(registry, bindings["cells"])
        if current["event_sha256"] != plan["event"]["previous_event_sha256"]:
            raise ValueError("Ownership changed after v2 preparation")
        _atomic_write(registry / "transition-intent.json", intent)
        with _lock(Path(bindings["root"]) / "bank.lock"):
            durable = (Path(bindings["durable_root"]) / "ownership-reregistration-v2"
                       / plan["event_sha256"])
            if durable.exists():
                raise FileExistsError("Durable v2 ownership event already exists")
            durable.mkdir(parents=True)
            _atomic_write(durable / "ownership.json", plan["event"])
            receipt = {"files": {"ownership.json": {
                "sha256": sha(durable / "ownership.json"),
                "bytes": (durable / "ownership.json").stat().st_size,
            }}}
            _atomic_write(durable / "bank-receipt.json", receipt)
            certificate = {
                "event": plan["event"], "event_sha256": plan["event_sha256"],
                "writer_host": bindings["writer_host"], "bank_receipt": receipt,
                "bank_receipt_sha256": digest(receipt),
                "transition_plan_sha256": digest(plan),
            }
            if plan["event"]["durable_recovery"]:
                certificate["recovery"] = {
                    "kind": "p5-ownership-durable-recovery-v1",
                    "cells": plan["event"]["durable_recovery"],
                }
            commits = sorted((registry / "committed").glob("*.json"))
            certificate_path = (
                registry / "committed"
                / f"{len(commits) + 1:04d}-{plan['event_sha256'][:16]}.json"
            )
            _atomic_write(certificate_path, certificate)
            xroot = _stage_canonical(plan, certificate)
            _atomic_write(registry / "transition-intent.json", {
                **intent, "status": "committed",
                "certificate_sha256": sha(certificate_path),
            })
            if fail_before_activation:
                raise RuntimeError("Injected interruption before activation publication")
            _verify_canonical_stage(xroot, plan, certificate)
            activation = _activation_document(xroot, plan)
            _atomic_write(xroot / "activation" / "last-published.json", activation)
            _atomic_write(registry / "transition-intent.json", {
                **intent, "status": "activated",
                "certificate_sha256": sha(certificate_path),
                "activation_sha256": sha(xroot / "activation" / "last-published.json"),
            })
    verify_activation(xroot, plan["event"]["broker_release_sha256"])
    return {
        "status": "activated", "event_sha256": plan["event_sha256"],
        "certificate": str(certificate_path),
        "activation": str(xroot / "activation" / "last-published.json"),
    }


def verify_activation(xroot, broker_release_sha256):
    xroot = Path(xroot)
    path = xroot / "activation" / "last-published.json"
    if not path.is_file():
        raise ValueError("Committed transition is not activated")
    activation = read(path)
    certificate = read(xroot / "current" / "ownership-certificate.json")
    event = certificate["event"]
    expected = {
        "current/ownership-certificate.json", "current/runtime-manifest.json",
        "current/crossnode-proof.json", "current/bindings.json",
        "current/p5_xnode_control.py", "current/p5_xnode_state.py",
    }
    for cell in event.get("transition_cells", []):
        expected.update({
            "/".join(("leases", cell + ".json")),
            "/".join(("roots", cell, "checkpoint", "model.safetensors")),
            "/".join(("roots", cell, "checkpoint", "training-state.pt")),
            "/".join(("roots", cell, "checkpoint", "p5-seal.json")),
            "/".join(("roots", cell, "p5-run-config.json")),
            "/".join(("roots", cell, "latest.json")),
        })
        if event["owners"][cell]["owner_node"] != certificate["writer_host"]:
            expected.add("/".join(
                ("worker-bootstrap", cell, ".xnode", "registration.json"),
            ))
    if (activation.get("kind") != "p5-xnode-activation-v1"
            or activation.get("status") != "activated"
            or activation.get("sequence") != event.get("sequence")
            or activation.get("event_sha256") != certificate.get("event_sha256")
            or activation.get("broker_release_sha256") != broker_release_sha256
            or event.get("broker_release_sha256") != broker_release_sha256
            or set(activation.get("files_sha256", {})) != expected):
        raise ValueError("Activation does not authorize the current certificate")
    for relative, expected in activation.get("files_sha256", {}).items():
        path = xroot / relative
        if not path.is_file() or sha(path) != expected:
            raise ValueError("Canonical activation file hash differs")
    manifest = read(xroot / "current" / "runtime-manifest.json")
    proof = read(xroot / "current" / "crossnode-proof.json")
    if (document_sha(manifest) != proof.get("runtime_manifest_sha256")
            or document_sha(proof) != event.get("crossnode_proof_sha256")):
        raise ValueError("Activation runtime/proof hashes differ")
    for cell in event["transition_cells"]:
        lease = read(xroot / "leases" / f"{cell}.json")
        if (lease != event["owners"][cell]
                or lease["control_manifest_sha256"] != document_sha(manifest)):
            raise ValueError("Canonical activated lease differs")
    return certificate


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="mode", required=True)
    for name in ("prepare", "verify", "commit", "activate"):
        command = commands.add_parser(name)
        command.add_argument("--request", type=Path, required=True)
        command.add_argument("--plan", type=Path, required=name != "prepare")
        if name == "prepare":
            command.add_argument("--output", type=Path, required=True)
    activation = commands.add_parser("verify-activation")
    activation.add_argument("--xroot", type=Path, required=True)
    activation.add_argument("--broker-release-sha256", required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        _atomic_write(args.output, prepare_transition(args.request))
    elif args.mode == "verify":
        verify_transition(args.request, read(args.plan))
    elif args.mode == "commit":
        print(json.dumps(commit_transition(args.request, read(args.plan)), sort_keys=True))
    elif args.mode == "activate":
        print(json.dumps(activate_transition(args.request, read(args.plan)), sort_keys=True))
    else:
        verify_activation(args.xroot, args.broker_release_sha256)


if __name__ == "__main__":
    main()
