"""Offline, receipt-bound P5 ownership re-registration after GPU replacement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
from contextlib import contextmanager
from pathlib import Path


VERSION = "p5-xnode-uuid-reregistration-v1"
PACKAGE_FILES = {
    "checkpoint/model.safetensors",
    "checkpoint/training-state.pt",
    "checkpoint/p5-seal.json",
    "scientific-contract.json",
    "execution-attestation.json",
    "envelope.json",
}
PROOF_CHECKS = (
    "cold_save_reload_bitwise_equal",
    "next_sampled_rows_and_losses_equal",
    "populated_optimizer_memory_pass",
    "complete_20beam_geometry_pass",
)


def read(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate JSON key")
            value[key] = item
        return value

    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def event_sha(event):
    return hashlib.sha256(canonical(event)).hexdigest()


def _plain(value):
    if (not isinstance(value, str) or not value or Path(value).name != value
            or "/" in value or "\\" in value or value in {".", ".."}):
        raise ValueError("Unsafe protocol name")
    return value


def _absolute(value):
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or path == Path(path.anchor):
        raise ValueError("Explicit non-root absolute path required")
    return path.resolve()


def _validate_bindings(value):
    fields = {
        "kind", "root", "p5_root", "control_root", "durable_root", "python",
        "writer_host", "worker_host", "cells", "adopted_cell", "learner_sha256",
        "legacy_device_guard", "legacy_transport_guard",
    }
    if not isinstance(value, dict) or set(value) != fields or value["kind"] != "p5-xnode-bindings-v1":
        raise ValueError("Unknown private bindings")
    paths = {key: _absolute(value[key]) for key in
             ("root", "p5_root", "control_root", "durable_root", "python")}
    if (not paths["p5_root"].is_relative_to(paths["root"])
            or not paths["control_root"].is_relative_to(paths["root"])
            or paths["durable_root"].is_relative_to(paths["root"])
            or paths["root"].is_relative_to(paths["durable_root"])):
        raise ValueError("Private binding roots overlap or escape")
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value["writer_host"])
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value["worker_host"])
            or value["writer_host"] == value["worker_host"]):
        raise ValueError("Invalid private host aliases")
    if not isinstance(value["cells"], dict) or not value["cells"]:
        raise ValueError("Private bindings require explicit cells")
    expected = {"role", "gpu", "gpu_uuid", "microbatch", "accumulation",
                "root_step", "ownership_epoch"}
    for cell, assignment in value["cells"].items():
        _plain(cell)
        if (not isinstance(assignment, dict) or set(assignment) != expected
                or assignment["role"] not in ("writer", "worker")
                or type(assignment["gpu"]) is not int or assignment["gpu"] < 0
                or not isinstance(assignment["gpu_uuid"], str) or not assignment["gpu_uuid"]
                or type(assignment["microbatch"]) is not int or assignment["microbatch"] < 1
                or type(assignment["accumulation"]) is not int or assignment["accumulation"] < 1
                or type(assignment["root_step"]) is not int or assignment["root_step"] < 0
                or type(assignment["ownership_epoch"]) is not int
                or assignment["ownership_epoch"] < 0
                or assignment["microbatch"] * assignment["accumulation"] != 128):
            raise ValueError("Invalid private cell assignment")
    return value


def _load_registry(directory, expected_cells, pending_event_sha=None):
    directory = Path(directory)
    commits = sorted((directory / "committed").glob("*.json"))
    if not commits:
        raise ValueError("No committed ownership ancestry")
    previous = None
    previous_sequence = 0
    sequenced = False
    latest = None
    for path in commits:
        certificate = read(path)
        event, receipt = certificate["event"], certificate["bank_receipt"]
        encoded = event_sha(event)
        sequence = event.get("sequence")
        valid_sequence = (
            sequence is None and not sequenced
            or type(sequence) is int and sequence == previous_sequence + 1
        )
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


def _file_manifest(directory, exclude=()):
    result = {}
    for path in sorted(Path(directory).rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlinks are forbidden in transition evidence")
        if path.is_file():
            name = path.relative_to(directory).as_posix()
            if name not in exclude:
                result[name] = {"sha256": sha(path), "bytes": path.stat().st_size}
    return result


def _checkpoint_header(path):
    path = Path(path)
    seal = read(path / "p5-seal.json")
    if (seal.get("kind") != "p5-full-state-v1" or seal.get("status") != "sealed"
            or set(seal.get("files_sha256", {})) != {"model.safetensors", "training-state.pt"}):
        raise ValueError("Missing or partial full-state checkpoint")
    for name, expected in seal["files_sha256"].items():
        member = path / name
        if not member.is_file() or member.is_symlink() or sha(member) != expected:
            raise ValueError("Missing, redirected or corrupt full-state member")
    return {
        "artifact": path.name, "seal_sha256": sha(path / "p5-seal.json"),
        "contract_sha256": seal["contract_sha256"], "step": seal["global_step"],
        "epoch": seal["epoch"], "phase": seal["phase"],
        "model_sha256": seal["files_sha256"]["model.safetensors"],
        "training_state_sha256": seal["files_sha256"]["training-state.pt"],
    }


def _verify_package(path, lease):
    path = Path(path)
    marker = read(path / "package-seal.json")
    if marker.get("kind") != "p5-xnode-package-v1" or marker.get("status") != "sealed":
        raise ValueError("Missing package seal")
    actual = _file_manifest(path, {"package-seal.json"})
    if set(actual) != PACKAGE_FILES or marker.get("files") != actual:
        raise ValueError("Missing, partial or changed full-state package")
    header = _checkpoint_header(path / "checkpoint")
    envelope = read(path / "envelope.json")
    checkpoint = envelope.get("checkpoint", {})
    if any(checkpoint.get(key) != value for key, value in header.items() if key != "artifact"):
        raise ValueError("Package envelope differs from full state")
    contract = read(path / "scientific-contract.json")
    if digest(contract) != header["contract_sha256"]:
        raise ValueError("Package scientific contract differs")
    if sha(path / "execution-attestation.json") != envelope.get("execution_attestation_sha256"):
        raise ValueError("Package execution attestation differs")
    attestation = read(path / "execution-attestation.json")
    if (attestation.get("lease_sha256") != digest(lease)
            or attestation.get("host") != lease["owner_node"]
            or attestation.get("gpu") != lease["gpu"]
            or attestation.get("purpose") != "production"
            or attestation.get("root_checkpoint_seal_sha256") != lease["root"]["seal_sha256"]
            or attestation.get("scientific_contract_rewritten") is not False):
        raise ValueError("Package execution provenance differs from the accepted owner")
    return envelope


def _progress(parent, child):
    if child["step"] > parent["step"]:
        valid = (parent["phase"] == "train" and child["phase"] in ("train", "validate")
                 and child["epoch"] == parent["epoch"])
    elif child["step"] == parent["step"]:
        valid = (
            parent["phase"] == "train" and child["phase"] == "validate"
            and child["epoch"] == parent["epoch"]
            or parent["phase"] == "validate" and child["phase"] in ("train", "frozen")
            and child["epoch"] == parent["epoch"] + 1
            or parent["phase"] == child["phase"] and child["epoch"] == parent["epoch"]
            and child["model_sha256"] == parent["model_sha256"]
            and child["training_state_sha256"] == parent["training_state_sha256"]
        )
    else:
        valid = False
    if not valid:
        raise ValueError("Accepted checkpoint ancestry regressed or diverged")


def _validate_envelope(envelope, lease, head):
    if (envelope.get("kind") != "p5-xnode-checkpoint-v1"
            or envelope.get("purpose") != "production"):
        raise ValueError("Only production full-state records may be transition roots")
    for key in ("cell", "owner_node", "gpu", "lease_id", "ownership_epoch"):
        if envelope.get(key) != lease.get(key):
            raise ValueError("Accepted record belongs to a stale owner")
    if (envelope.get("lease_sha256") != digest(lease)
            or envelope.get("root_seal_sha256") != lease["root"]["seal_sha256"]
            or envelope.get("generation") != head["generation"] + 1
            or envelope.get("parent_seal_sha256") != head["seal_sha256"]):
        raise ValueError("Accepted record has broken lease or generation ancestry")
    checkpoint = envelope["checkpoint"]
    if checkpoint.get("contract_sha256") != lease["root"]["contract_sha256"]:
        raise ValueError("Scientific contract changed in accepted lineage")
    _progress(head, checkpoint)
    return {**checkpoint, "generation": envelope["generation"],
            "envelope_sha256": digest(envelope)}


def _validate_ack(ack, lease, envelope, package, registry_event_sha256):
    manifest = _file_manifest(package)
    if (ack.get("kind") != "p5-xnode-bank-ack-v1" or ack.get("status") != "verified"
            or ack.get("cell") != lease["cell"] or ack.get("lease_id") != lease["lease_id"]
            or ack.get("lease_sha256") != digest(lease)
            or ack.get("ownership_epoch") != lease["ownership_epoch"]
            or ack.get("registry_event_sha256") != registry_event_sha256
            or ack.get("generation") != envelope["generation"]
            or ack.get("envelope") != envelope or ack.get("checkpoint") != envelope["checkpoint"]
            or ack.get("bank_receipt") != {"files": manifest}
            or ack.get("bank_receipt_sha256") != digest(ack["bank_receipt"])):
        raise ValueError("Accepted acknowledgement or durable package receipt differs")


def _accepted_root(evidence, old_lease, registry_event_sha256):
    ledger = Path(evidence["ledger"])
    head = {**old_lease["root"], "generation": 0, "envelope_sha256": None}
    records = {}
    for path in sorted((ledger / "accepted").glob("g*.json")):
        record = read(path)
        if record.get("lease_sha256") != digest(old_lease):
            raise ValueError("Accepted record uses an old or foreign lease")
        head = _validate_envelope(record["envelope"], old_lease, head)
        records[record["envelope"]["generation"]] = record
    if not records:
        raise ValueError("No durable accepted full-state continuation head")
    pointer = read(ledger / "head.json")
    if pointer != head:
        raise ValueError("Accepted lineage head pointer differs")
    package = Path(evidence["package"])
    envelope = _verify_package(package, old_lease)
    record = records[head["generation"]]
    if record["envelope"] != envelope:
        raise ValueError("Durable package is not the accepted lineage head")
    _validate_ack(record["ack"], old_lease, envelope, package, registry_event_sha256)
    contract = Path(evidence["contract"])
    if (sha(contract) != old_lease["contract_file_sha256"]
            or digest(read(contract)) != head["contract_sha256"]):
        raise ValueError("Accepted root scientific contract differs")
    return head, record, package, contract


def _validate_proof(path, cell, owner, assignment, head):
    proof = read(path)
    expected_host = owner["owner_node"]
    if (proof.get("kind") != "p5-replacement-gpu-proof-v1" or proof.get("status") != "pass"
            or proof.get("cell") != cell or proof.get("host") != expected_host
            or proof.get("gpu") != assignment["gpu"]
            or proof.get("gpu_uuid") != assignment["gpu_uuid"]
            or proof.get("source_checkpoint_seal_sha256") != head["seal_sha256"]
            or proof.get("contract_sha256") != head["contract_sha256"]
            or proof.get("model_sha256") != head["model_sha256"]
            or proof.get("training_state_sha256") != head["training_state_sha256"]
            or not all(proof.get(check) is True for check in PROOF_CHECKS)):
        raise ValueError("Replacement-GPU continuation proof is incomplete, stale or failed")
    return proof


def _request(path):
    value = read(path)
    required = {
        "kind", "cells", "old_bindings", "new_bindings", "registry", "evidence", "proofs",
        "broker_release_sha256",
    }
    if set(value) != required or value["kind"] != "p5-uuid-reregistration-request-v1":
        raise ValueError("Unknown re-registration request")
    if not re.fullmatch(r"[0-9a-f]{64}", value["broker_release_sha256"]):
        raise ValueError("New broker release identity is required")
    if (not isinstance(value["cells"], list) or not value["cells"]
            or len(value["cells"]) != len(set(value["cells"]))):
        raise ValueError("Transition cells must be explicit and unique")
    return value


def prepare_transition(request_path):
    request_path = Path(request_path)
    request = _request(request_path)
    old_binding = _validate_bindings(read(request["old_bindings"]))
    new_binding = _validate_bindings(read(request["new_bindings"]))
    if set(old_binding["cells"]) != set(new_binding["cells"]):
        raise ValueError("New bindings must preserve the complete cell set")
    certificate = _load_registry(request["registry"], old_binding["cells"])
    if certificate.get("writer_host") != old_binding["writer_host"]:
        raise ValueError("Old ownership certificate writer differs")
    old_event = certificate["event"]
    changed = {
        cell for cell in old_binding["cells"]
        if old_binding["cells"][cell]["gpu_uuid"] != new_binding["cells"][cell]["gpu_uuid"]
    }
    if set(request["cells"]) != changed:
        raise ValueError("Request must name the exact transition set implied by changed UUIDs")
    if set(request["evidence"]) != changed or set(request["proofs"]) != changed:
        raise ValueError("Cross-cell partial transition evidence is forbidden")
    effective_binding = json.loads(json.dumps(new_binding))
    owners = json.loads(json.dumps(old_event["owners"]))
    cells = {}
    for cell in sorted(changed):
        old_assignment = old_binding["cells"][cell]
        new_assignment = new_binding["cells"][cell]
        old_owner = old_event["owners"][cell]
        expected_owner = (old_binding["writer_host"]
                          if old_assignment["role"] == "writer" else old_binding["worker_host"])
        if (old_owner.get("owner_node") != expected_owner
                or new_assignment["role"] != old_assignment["role"]
                or new_assignment["gpu"] != old_assignment["gpu"]
                or new_assignment["microbatch"] != old_assignment["microbatch"]
                or new_assignment["accumulation"] != old_assignment["accumulation"]
                or new_binding["writer_host"] != old_binding["writer_host"]
                or new_binding["worker_host"] != old_binding["worker_host"]):
            raise ValueError("Host/index/batch/scientific assignment changed during UUID replacement")
        evidence = request["evidence"][cell]
        if evidence.get("kind") != "p5-uuid-reregistration-root-v1" or evidence.get("cell") != cell:
            raise ValueError("Unknown transition root evidence")
        head, record, package, contract = _accepted_root(
            evidence, old_owner, certificate["event_sha256"],
        )
        proof = _validate_proof(request["proofs"][cell], cell, old_owner, new_assignment, head)
        epoch = old_owner["ownership_epoch"] + 1
        effective_binding["cells"][cell]["ownership_epoch"] = epoch
        effective_binding["cells"][cell]["root_step"] = head["step"]
        lease_seed = digest({
            "cell": cell, "previous": certificate["event_sha256"],
            "gpu_uuid": new_assignment["gpu_uuid"], "head": head["seal_sha256"],
            "proof_sha256": sha(request["proofs"][cell]),
        })
        lease = {
            **{key: old_owner[key] for key in old_owner
               if key not in {"lease_id", "ownership_epoch", "gpu_uuid", "root"}},
            "kind": "p5-xnode-lease-v1", "cell": cell,
            "lease_id": f"{cell.lower()}-epoch-{epoch}-{lease_seed[:12]}",
            "ownership_epoch": epoch, "gpu_uuid": new_assignment["gpu_uuid"],
            "root": {key: head[key] for key in (
                "artifact", "seal_sha256", "contract_sha256", "step", "epoch", "phase",
                "model_sha256", "training_state_sha256",
            )},
        }
        owners[cell] = lease
        cells[cell] = {
            "old_lease_sha256": digest(old_owner), "new_lease_sha256": digest(lease),
            "old_accepted_head": lease["root"], "accepted_record_sha256": digest(record),
            "package_seal_sha256": sha(package / "package-seal.json"),
            "contract_file_sha256": sha(contract),
            "proof_sha256": sha(request["proofs"][cell]),
        }
    event = {
        "kind": "p5-xnode-ownership-v1", "status": "committed", "owners": owners,
        "previous_event_sha256": certificate["event_sha256"],
        "broker_release_sha256": request["broker_release_sha256"],
        "transition_kind": VERSION, "transition_cells": sorted(changed),
    }
    return {
        "kind": "p5-uuid-reregistration-plan-v1", "status": "prepared",
        "request_sha256": sha(request_path),
        "old_bindings_sha256": sha(request["old_bindings"]),
        "proposed_bindings_sha256": sha(request["new_bindings"]),
        "previous_certificate_sha256": sha(
            sorted((Path(request["registry"]) / "committed").glob("*.json"))[-1]
        ),
        "cells": cells, "bindings": effective_binding, "event": event,
        "event_sha256": event_sha(event),
        "publication_generation_policy": "reset-to-zero-at-verified-old-accepted-head",
    }


def verify_transition(request_path, plan):
    expected = prepare_transition(request_path)
    if plan != expected:
        raise ValueError("Transition plan differs from current registry, evidence, proofs or bindings")
    for cell in plan["event"]["transition_cells"]:
        lease = plan["event"]["owners"][cell]
        if lease["root"] != plan["cells"][cell]["old_accepted_head"]:
            raise ValueError("New publication root differs from the old accepted head")
    return plan


def worker_registration(plan, cell):
    _plain(cell)
    lease = plan["event"]["owners"].get(cell)
    assignment = plan["bindings"]["cells"].get(cell, {})
    if lease is None or assignment.get("role") != "worker":
        raise ValueError("Worker registration requested for a non-worker cell")
    return {
        "kind": "p5-xnode-registration-v1",
        "writer_host": plan["bindings"]["writer_host"], "cell": cell,
        "lease_sha256": digest(lease), "event_sha256": plan["event_sha256"],
        "input_manifest_sha256": lease["input_manifest_sha256"],
        "registration_anchor_sha256": plan["event_sha256"],
    }


def verify_worker_registration(registration, plan, cell):
    expected = worker_registration(plan, cell)
    if registration != expected:
        raise ValueError("Old or stale worker registration cannot authorize the new lease")
    return registration


def verify_worker_bootstrap(transport, plan, cell):
    transport = Path(transport)
    registration = read(transport / "registration.json")
    verify_worker_registration(registration, plan, cell)
    forbidden_files = ("head.json", "results-ready.json", "guardian-status.json")
    if any((transport / name).exists() for name in forbidden_files):
        raise ValueError("New worker transport contains an old publication generation")
    for name in ("outbox", "acks", "requests", "responses"):
        directory = transport / name
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("New worker transport reuses an old outbox or acknowledgement")
    return True


def _atomic_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    with pending.open("wb") as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


@contextmanager
def _lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if stream.seek(0, os.SEEK_END) == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)


def commit_transition(request_path, plan, output):
    raise RuntimeError(
        "p5 UUID re-registration v1 is historical nondeployable evidence; use the v2 protocol"
    )


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="mode", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--request", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--request", type=Path, required=True)
    verify.add_argument("--plan", type=Path, required=True)
    commit = commands.add_parser("commit")
    commit.add_argument("--request", type=Path, required=True)
    commit.add_argument("--plan", type=Path, required=True)
    commit.add_argument("--output", type=Path, required=True)
    registrations = commands.add_parser("registrations")
    registrations.add_argument("--plan", type=Path, required=True)
    registrations.add_argument("--output", type=Path, required=True)
    worker = commands.add_parser("verify-worker")
    worker.add_argument("--plan", type=Path, required=True)
    worker.add_argument("--cell", required=True)
    worker.add_argument("--transport", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        value = prepare_transition(args.request)
        _atomic_write(args.output, value)
    elif args.mode == "verify":
        verify_transition(args.request, read(args.plan))
    elif args.mode == "commit":
        print(json.dumps(commit_transition(args.request, read(args.plan), args.output), sort_keys=True))
    elif args.mode == "registrations":
        plan = read(args.plan)
        for cell in plan["event"]["transition_cells"]:
            if plan["bindings"]["cells"][cell]["role"] == "worker":
                _atomic_write(args.output / f"{cell}.json", worker_registration(plan, cell))
    else:
        verify_worker_bootstrap(args.transport, read(args.plan), args.cell)


if __name__ == "__main__":
    main()
