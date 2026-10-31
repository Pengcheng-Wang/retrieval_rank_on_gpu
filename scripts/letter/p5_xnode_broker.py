"""Explicitly bound writer-host broker, authoritative leases and immutable import ledger."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import p5_xnode_state as state


def event_sha(event):
    return hashlib.sha256((json.dumps(event, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()).hexdigest()


def protocol_compatibility(runtime_manifest, updated_protocol):
    import ast

    state.configuration()
    state.require_path(runtime_manifest)
    state.require_path(updated_protocol)
    runtime_manifest, updated_protocol = Path(runtime_manifest), Path(updated_protocol)
    runtime = state.read(runtime_manifest)
    if (runtime.get("kind") != "p5-xnode-runtime-v1"
            or set(runtime["files_sha256"]) != {"p5_xnode_state.py", "p5_xnode_control.py"}):
        raise ValueError("Result compatibility requires the unchanged original runtime")
    for name, digest in runtime["files_sha256"].items():
        state.require_path(runtime_manifest.parent / name)
        if state.sha(runtime_manifest.parent / name) != digest:
            raise ValueError("Original runtime source changed; compatibility is not established")
    before = ast.parse((runtime_manifest.parent / "p5_xnode_state.py").read_text(encoding="utf-8"))
    after = ast.parse(updated_protocol.read_text(encoding="utf-8"))
    allowed = {"validate_result_ack", "validate_result_binding", "local_result_context"}

    def unchanged(tree):
        return ast.dump(ast.Module(body=[
            node for node in tree.body
            if not isinstance(node, ast.FunctionDef) or node.name not in allowed
        ], type_ignores=[]), include_attributes=False)

    original, updated = unchanged(before), unchanged(after)
    if original != updated:
        raise ValueError("Result compatibility cannot change any checkpoint protocol statement")
    before_functions = {node.name: ast.dump(node) for node in before.body if isinstance(node, ast.FunctionDef)}
    after_functions = {node.name: ast.dump(node) for node in after.body if isinstance(node, ast.FunctionDef)}
    changed = sorted(name for name in before_functions
                     if before_functions[name] != after_functions.get(name))
    added = sorted(set(after_functions) - set(before_functions))
    if changed != ["validate_result_ack"] or added != ["local_result_context", "validate_result_binding"]:
        raise ValueError("Result compatibility requires the exact declared result-only function boundary")
    return {
        "kind": "p5-xnode-result-only-compatibility-v1",
        "runtime_manifest_sha256": state.sha(runtime_manifest),
        "original_protocol_sha256": runtime["files_sha256"]["p5_xnode_state.py"],
        "updated_protocol_sha256": state.sha(updated_protocol),
        "learner_control_sha256": runtime["files_sha256"]["p5_xnode_control.py"],
        "checkpoint_protocol_ast_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "checkpoint_protocol_unchanged": True, "learner_control_unchanged": True,
        "changed_existing_functions": changed, "added_result_functions": added,
    }


def registration_message(config, certificate, lease, anchor_sha256):
    state.configuration()
    state.validate_runtime_paths(config)
    path = Path(config["ownership_certificate"])
    if state.sha(path) != anchor_sha256:
        raise ValueError("The immutable original registration certificate anchor changed")
    original = state.read(path)
    event, receipt = original["event"], original["bank_receipt"]
    if (original.get("writer_host") != state.WRITER_HOST
            or event_sha(event) != original["event_sha256"]
            or state.digest(receipt) != original["bank_receipt_sha256"]
            or receipt["files"]["ownership.json"]["sha256"] != original["event_sha256"]
            or event["owners"] != certificate["event"]["owners"]
            or state.require_owner(certificate["event"], lease["cell"], state.WORKER_HOST, lease["lease_id"]) != lease):
        raise ValueError("Original registration anchor does not authorize the current owner")
    return {
        "kind": "p5-xnode-registration-v1", "writer_host": state.WRITER_HOST,
        "lease_sha256": state.digest(lease), "event_sha256": original["event_sha256"],
        "input_manifest_sha256": config["input_manifest_sha256"],
    }


def validate_broker_v3_release(release):
    if (not isinstance(release, dict)
            or set(release) != {"kind", "files_sha256", "runtime_manifest_sha256"}
            or release.get("kind") != "p5-xnode-broker-v3"
            or set(release.get("files_sha256", {})) != {
                "p5_xnode_broker.py", "p5_xnode_state.py", "p5_xnode_adopt.py",
            }):
        raise ValueError(
            "Broker v3 release must avoid certificate, registration-anchor and activation hash cycles"
        )
    return release


def load_registry(directory):
    state.configuration()
    state.require_path(directory)
    directory = Path(directory)
    commits = sorted((directory / "committed").glob("*.json"))
    if not commits:
        raise ValueError("No committed ownership; pending handoff is fenced, never defaulted to the writer")
    previous, latest = None, None
    previous_sequence, sequenced = 0, False
    for path in commits:
        certificate = state.read(path)
        event = certificate["event"]
        receipt = certificate["bank_receipt"]
        sequence = event.get("sequence")
        valid_sequence = (
            sequence is None and not sequenced
            or type(sequence) is int and sequence == previous_sequence + 1
        )
        if (event.get("kind") != "p5-xnode-ownership-v1" or event.get("status") != "committed"
                or set(event.get("owners", {})) != state.ALL_CELLS
                or event.get("previous_event_sha256") != previous
                or event_sha(event) != certificate["event_sha256"]
                or state.digest(receipt) != certificate["bank_receipt_sha256"]
                or receipt["files"]["ownership.json"]["sha256"] != certificate["event_sha256"]
                or certificate.get("writer_host") != state.WRITER_HOST
                or not valid_sequence):
            raise ValueError("Invalid committed ownership ancestry/receipt")
        if sequence is not None:
            sequenced = True
            previous_sequence = sequence
        previous, latest = certificate["event_sha256"], certificate
    intent = directory / "transition-intent.json"
    if intent.exists() and state.read(intent)["event_sha256"] != latest["event_sha256"]:
        raise ValueError("A newer pending ownership transition is fenced until recovered")
    return latest


def activated_current(directory, broker_release_sha256):
    state.configuration()
    directory = Path(directory)
    certificate = load_registry(directory / "ownership")
    activation_path = directory / "activation" / "last-published.json"
    if not activation_path.is_file():
        raise ValueError("Committed ownership is not activated")
    activation = state.read(activation_path)
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
        if event["owners"][cell]["owner_node"] != state.WRITER_HOST:
            expected.add("/".join(("worker-bootstrap", cell, ".xnode", "registration.json")))
    if (activation.get("kind") != "p5-xnode-activation-v1"
            or activation.get("status") != "activated"
            or activation.get("sequence") != event.get("sequence")
            or activation.get("event_sha256") != certificate["event_sha256"]
            or activation.get("broker_release_sha256") != broker_release_sha256
            or event.get("broker_release_sha256") != broker_release_sha256
            or set(activation.get("files_sha256", {})) != expected):
        raise ValueError("activation does not authorize the current committed certificate")
    for relative, expected in activation.get("files_sha256", {}).items():
        path = directory / relative
        if not path.is_file() or state.sha(path) != expected:
            raise ValueError("Activated canonical file hash differs")
    canonical = state.read(directory / "current" / "ownership-certificate.json")
    if canonical != certificate:
        raise ValueError("Activated current certificate differs from the registry")
    runtime = state.read(directory / "current" / "runtime-manifest.json")
    proof_path = directory / "current" / "crossnode-proof.json"
    proof = state.read(proof_path)
    if (state.sha(directory / "current" / "runtime-manifest.json")
            != proof.get("runtime_manifest_sha256")
            or state.sha(proof_path) != event.get("crossnode_proof_sha256")):
        raise ValueError("Activated runtime/proof binding differs")
    for cell in event.get("transition_cells", []):
        lease = state.read(directory / "leases" / f"{cell}.json")
        if (lease != event["owners"][cell]
                or lease["control_manifest_sha256"] != state.sha(
                    directory / "current" / "runtime-manifest.json"
                )):
            raise ValueError("Activated lease differs from the current certificate")
    return certificate


def accepted_chain(directory, lease):
    directory = Path(directory)
    head, records = state.root_head(lease), {}
    for path in sorted((directory / "accepted").glob("g*.json")):
        record = state.read(path)
        envelope, ack = record["envelope"], record["ack"]
        if record["lease_sha256"] != state.digest(lease):
            raise ValueError("Accepted record belongs to a stale owner")
        if state.validate_envelope(envelope, lease, head) != "next":
            raise ValueError("Invalid accepted ancestry")
        state.validate_ack(ack, lease, envelope["checkpoint"])
        if ack["envelope"] != envelope:
            raise ValueError("Acceptance receipt differs from its envelope")
        records[envelope["generation"]] = record
        head = state.envelope_head(envelope)
    pointer = directory / "head.json"
    if pointer.exists():
        old = state.read(pointer)
        if old["generation"] > head["generation"] or (
            old["generation"] == head["generation"] and old != head
        ):
            raise ValueError("Accepted lineage head is corrupt")
    return head, records


def record_acceptance(directory, lease, envelope, ack):
    directory = Path(directory)
    with state.locked(directory / "ledger.lock"):
        head, records = accepted_chain(directory, lease)
        generation = envelope["generation"]
        if generation in records:
            if records[generation]["envelope"] != envelope:
                raise ValueError("A divergent state cannot reuse an accepted generation")
            return records[generation]["ack"]
        state.validate_envelope(envelope, lease, head)
        state.validate_ack(ack, lease, envelope["checkpoint"])
        if ack["envelope"] != envelope:
            raise ValueError("Acceptance and receipt differ")
        state.write(directory / "accepted" / f"g{generation:08d}.json", {
            "lease_sha256": state.digest(lease), "envelope": envelope, "ack": ack,
        })
        state.write(directory / "head.json", state.envelope_head(envelope))
        return ack


def find_accepted_ack(directory, lease, seal_sha256):
    _, records = accepted_chain(directory, lease)
    for record in records.values():
        if record["envelope"]["checkpoint"]["seal_sha256"] == seal_sha256:
            return record["ack"]
    raise ValueError("Requested checkpoint has not been durably accepted in this owner lineage")


def remote_inventory(lease_path):
    state.configuration()
    state.require_path(lease_path)
    if socket.gethostname() != state.WORKER_HOST:
        raise RuntimeError("Inventory is worker-local only")
    lease = state.read(lease_path)
    run = state.P5 / "runs" / lease["cell"]
    state.require_assignment(lease, state.WORKER_HOST, lease["gpu"])
    packages = []
    for path in sorted((run / ".xnode/outbox").glob("g*")):
        envelope = state.verify_package(path, full=False)
        packages.append({"name": path.name, "envelope": envelope,
                         "package_seal_sha256": state.sha(path / "package-seal.json")})
    requests = []
    for path in sorted((run / ".xnode/requests").glob("*.json")):
        if not (run / ".xnode/responses" / path.name).exists():
            requests.append(state.read(path))
    ready = state.read(run / ".xnode/results-ready.json") if (run / ".xnode/results-ready.json").exists() else None
    if ready:
        head, _ = state.local_result_context(run / ready["snapshot"], lease)
        if (ready.get("lease_sha256") != state.digest(lease) or ready.get("head") != head
                or ready.get("result_seal_sha256") != state.sha(run / ready["snapshot"] / "p5-result-seal.json")):
            raise ValueError("Producer result advertisement differs from its frozen head")
    result = {
        "lease_sha256": state.digest(lease), "packages": packages, "requests": requests,
        "guardian": state.read(run / ".xnode/guardian-status.json")
        if (run / ".xnode/guardian-status.json").exists() else None,
        "results_ready": ready,
        "results_banked": (run / "results-banked.json").exists(),
        "registration_sha256": state.digest(state.read(run / ".xnode/registration.json"))
        if (run / ".xnode/registration.json").exists() else None,
        "acknowledged": {
            str(value["generation"]): state.digest(value)
            for value in (state.read(p) for p in (run / ".xnode/acks").glob("*.json"))
        },
    }
    return result


def prepare_message(lease_path):
    state.configuration()
    state.require_path(lease_path)
    if socket.gethostname() != state.WORKER_HOST:
        raise RuntimeError("Message staging is worker-local only")
    lease = state.read(lease_path)
    state.require_assignment(lease, state.WORKER_HOST, lease["gpu"])
    incoming = state.P5 / "runs" / lease["cell"] / ".xnode/incoming"
    state.require_path(incoming, writable=True)
    incoming.mkdir(parents=True, exist_ok=True)
    return {"directory": str(incoming), "bindings_sha256": state.BINDINGS_SHA256}


def receive_message(lease_path, message_path):
    state.configuration()
    state.require_path(lease_path)
    state.require_path(message_path)
    if socket.gethostname() != state.WORKER_HOST:
        raise RuntimeError("Receipt delivery is worker-local only")
    lease, message = state.read(lease_path), state.read(message_path)
    state.require_assignment(lease, state.WORKER_HOST, lease["gpu"])
    run = state.P5 / "runs" / lease["cell"]
    if message.get("lease_sha256") != state.digest(lease):
        raise ValueError("Transport delivery belongs to another owner")
    kind = message["kind"]
    if kind == "p5-xnode-registration-v1":
        state.write(run / ".xnode/registration.json", message)
    elif kind == "p5-xnode-bank-ack-v1":
        artifact = state.plain_name(message["checkpoint"]["artifact"])
        header = state.checkpoint_header(run / "states" / artifact, lease["root"]["contract_sha256"])
        state.validate_ack(message, lease, header)
        state.write(run / ".xnode/acks" / (artifact + ".json"), message)
    elif kind == "p5-xnode-checkpoint-response-v1":
        nonce = state.plain_name(message["nonce"])
        request = state.read(run / ".xnode/requests" / (nonce + ".json"))
        if request["checkpoint_seal_sha256"] != message["ack"]["checkpoint"]["seal_sha256"]:
            raise ValueError("An old receipt cannot satisfy a different request")
        header = state.checkpoint_header(
            run / "states" / state.plain_name(message["ack"]["checkpoint"]["artifact"]),
            lease["root"]["contract_sha256"],
        )
        state.validate_ack(message["ack"], lease, header)
        state.write(run / ".xnode/responses" / (nonce + ".json"), message)
    elif kind == "p5-xnode-result-ack-v1":
        with state.locked(run / ".p5-pointer.lock"), state.locked(run / ".xnode/publish.lock"):
            ready = state.read(run / ".xnode/results-ready.json")
            state.validate_result_ack(message, lease, run / ready["snapshot"], publication_locked=True)
            state.write(run / "results-banked.json", message)
    else:
        raise ValueError("Unknown owner-bound transport message")


class Broker:
    def __init__(self, runtime_manifest, directory):
        state.configuration()
        state.require_path(runtime_manifest)
        state.require_path(directory, writable=True)
        if socket.gethostname() != state.WRITER_HOST:
            raise RuntimeError("Only the configured writer may run the durable broker")
        if Path(directory).resolve() != state.XROOT:
            raise ValueError("Broker directory differs from private control-root bindings")
        self.directory = Path(directory)
        self.registry = self.directory / "ownership"
        self.runtime_manifest = Path(runtime_manifest)
        self.config = state.read(runtime_manifest)
        state.validate_runtime_paths(self.config)
        release_path = Path(__file__).parent / "broker-release.json"
        release = state.read(release_path)
        if (release.get("kind") not in {
                "p5-xnode-broker-v1", "p5-xnode-broker-v2", "p5-xnode-broker-v3",
        }
                or set(release["files_sha256"]) != {
                    "p5_xnode_broker.py", "p5_xnode_state.py", "p5_xnode_adopt.py"}):
            raise ValueError("Unapproved broker release")
        for name, digest in release["files_sha256"].items():
            state.require_path(release_path.parent / name)
            if state.sha(release_path.parent / name) != digest:
                raise ValueError("Broker source changed")
        if release.get("runtime_manifest_sha256") != state.sha(self.runtime_manifest):
            raise ValueError("Broker is bound to another unchanged learner runtime")
        self.registration_anchor_sha256 = None
        if release["kind"] == "p5-xnode-broker-v2":
            proof = protocol_compatibility(self.runtime_manifest, release_path.parent / "p5_xnode_state.py")
            if proof != release.get("protocol_compatibility"):
                raise ValueError("Result-only runtime compatibility proof differs")
            self.registration_anchor_sha256 = release["registration_anchor_sha256"]
            if state.sha(self.config["ownership_certificate"]) != self.registration_anchor_sha256:
                raise ValueError("Original worker registration anchor changed")
        elif release["kind"] == "p5-xnode-broker-v3":
            validate_broker_v3_release(release)
            if release["files_sha256"]["p5_xnode_state.py"] != self.config["files_sha256"]["p5_xnode_state.py"]:
                raise ValueError("Broker and worker must share the exact protocol")
        elif release["files_sha256"]["p5_xnode_state.py"] != self.config["files_sha256"]["p5_xnode_state.py"]:
            raise ValueError("Broker and worker must share the exact protocol")
        self.release_sha256 = state.sha(release_path)
        maintenance_path = Path(self.config["maintenance_manifest"]).parent / "p5_maintenance.py"
        state.require_path(maintenance_path)
        spec = importlib.util.spec_from_file_location("p5_xnode_broker_maintenance", maintenance_path)
        maintenance = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(maintenance)
        self.context = maintenance.install(self.config["base"], self.config["maintenance_manifest"])
        self.copy_artifact = __import__("bank").copy_artifact
        self.last_local_result_ack = {}

    def current(self):
        release = state.read(Path(__file__).parent / "broker-release.json")
        certificate = (
            activated_current(self.directory, self.release_sha256)
            if release["kind"] == "p5-xnode-broker-v3"
            else load_registry(self.registry)
        )
        if certificate["event"]["broker_release_sha256"] != self.release_sha256:
            raise ValueError("The authoritative owner event names a different broker")
        return certificate

    def rpc(self, arguments):
        state.configuration()
        command = shlex.join([str(state.PYTHON), "-B", str(Path(__file__).resolve()),
                             "--bindings", str(state.BINDINGS_PATH),
                             "--bindings-sha256", state.BINDINGS_SHA256] + arguments)
        result = subprocess.run(
            ["ssh", "-n", "-o", "BatchMode=yes", state.WORKER_HOST, command],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120, check=True,
        )
        rows = [line[len("P5_XNODE_RPC:"):] for line in result.stdout.splitlines()
                if line.startswith("P5_XNODE_RPC:")]
        if len(rows) != 1:
            raise RuntimeError("Remote response framing failed")
        return json.loads(rows[0])

    def send(self, lease, message):
        state.configuration()
        state.require_assignment(lease, state.WORKER_HOST, lease["gpu"])
        name = uuid.uuid4().hex + ".json"
        source = self.directory / "messages" / name
        state.write(source, message)
        remote = state.P5 / "runs" / lease["cell"] / ".xnode/incoming" / name
        prepared = self.rpc(["prepare-message", "--lease",
                             str(state.XROOT / "leases" / (lease["cell"] + ".json"))])
        if prepared != {"directory": str(remote.parent), "bindings_sha256": state.BINDINGS_SHA256}:
            raise ValueError("Remote message staging differs from the pinned private bindings")
        state.require_path(source, writable=True)
        subprocess.run(["scp", "-q", str(source), state.WORKER_HOST + ":" + str(remote)], check=True, timeout=120)
        self.rpc(["receive", "--lease", str(state.XROOT / "leases" / (lease["cell"] + ".json")),
                  "--message", str(remote)])

    def fetch_package(self, lease, item):
        state.configuration()
        state.require_assignment(lease, state.WORKER_HOST, lease["gpu"])
        state.require_path(self.directory, writable=True)
        name = state.plain_name(item["name"])
        if not name.startswith("g") or not name[1:9].isdigit():
            raise ValueError("Unsafe generation path")
        incoming = self.directory / "imports" / lease["cell"] / lease["lease_id"]
        final = incoming / name
        if final.exists():
            state.verify_package(final)
            return final
        pending = incoming / ("receive.partial-" + uuid.uuid4().hex)
        pending.mkdir(parents=True)
        remote = state.P5 / "runs" / lease["cell"] / ".xnode/outbox" / name
        subprocess.run(["scp", "-q", "-r", state.WORKER_HOST + ":" + str(remote), str(pending)], check=True, timeout=600)
        transferred = pending / name
        if state.sha(transferred / "package-seal.json") != item["package_seal_sha256"]:
            raise ValueError("Received package identity differs")
        if state.verify_package(transferred) != item["envelope"]:
            raise ValueError("Received package envelope differs")
        transferred.rename(final)
        state.sync_directory(incoming)
        pending.rmdir()
        return final

    def validate_native_payload(self, package, lease, envelope):
        import torch
        state.configuration()
        state.require_path(package)
        original_contract = state.read(state.P5 / "runs" / lease["cell"] / "p5-run-config.json")
        if state.digest(original_contract) != lease["root"]["contract_sha256"]:
            raise ValueError("Source scientific contract changed")
        self.context.checkpoint.verify_checkpoint(package / "checkpoint", original_contract)
        payload = torch.load(package / "checkpoint/training-state.pt", map_location="cpu", weights_only=False)
        native = payload["state"]
        if (payload["contract"] != original_contract
                or payload["checkpoint_boundary"] != "after_complete_optimizer_update_no_partial_gradients"
                or any(native[k] != envelope["checkpoint"][mapped] for k, mapped in
                       (("global_step", "step"), ("epoch", "epoch"), ("phase", "phase")))):
            raise ValueError("Imported native state differs from its ownership envelope")
        selected = envelope["selected"]
        if native.get("selected_artifact") != (selected["artifact"] if selected else None):
            raise ValueError("Imported selected identity differs from sealed state")
        attestation = state.read(package / "execution-attestation.json")
        if (attestation.get("lease_sha256") != state.digest(lease) or attestation.get("host") != state.WORKER_HOST
                or attestation.get("gpu") != lease["gpu"] or attestation.get("purpose") != "production"
                or attestation.get("runtime_manifest_sha256") != lease["control_manifest_sha256"]
                or attestation.get("scientific_contract_rewritten") is not False):
            raise ValueError("Imported execution provenance differs from the admitted owner")

    def import_one(self, lease, item):
        state.configuration()
        package = self.fetch_package(lease, item)
        envelope = state.verify_package(package)
        ledger = self.directory / "ledger" / lease["cell"] / lease["lease_id"]
        self.validate_native_payload(package, lease, envelope)
        with state.locked(self.registry / "registry.lock"):
            certificate = self.current()
            if state.require_owner(certificate["event"], lease["cell"], state.WORKER_HOST, lease["lease_id"]) != lease:
                raise ValueError("Owner changed while receiving state")
            head, records = accepted_chain(ledger, lease)
            if envelope["generation"] in records:
                if records[envelope["generation"]]["envelope"] != envelope:
                    raise ValueError("A divergent state reused an accepted generation")
                return records[envelope["generation"]]["ack"]
            state.validate_envelope(envelope, lease, head)
            selected = envelope["selected"]
            if selected and selected["seal_sha256"] != envelope["checkpoint"]["seal_sha256"]:
                if find_accepted_ack(ledger, lease, selected["seal_sha256"])["checkpoint"] != selected:
                    raise ValueError("Selected checkpoint metadata differs from its accepted ancestor")
            elif selected and selected != envelope["checkpoint"]:
                raise ValueError("Self-selected checkpoint metadata differs")
            destination = state.DURABLE / "xnode" / lease["cell"] / lease["lease_id"] / package.name
            state.require_path(destination, writable=True, durable=True)
            with state.locked(state.ROOT / "bank.lock"):
                # All SSH/scp and native payload verification happened before taking bank.lock.
                if destination.exists() and not (destination / "bank-receipt.json").exists():
                    destination = destination.with_name(destination.name + "-retry-" + uuid.uuid4().hex[:12])
                receipt = self.copy_artifact(package, destination)
                if state.read(destination / "bank-receipt.json") != receipt:
                    raise ValueError("Durable transfer receipt was not published exactly")
            ack = state.make_ack(package, lease, receipt, destination, certificate["event_sha256"])
            return record_acceptance(ledger, lease, envelope, ack)

    def local(self, cell, owner):
        state.configuration()
        run = state.P5 / "runs" / cell
        state.write(run / "bank-watcher-ready.json", {
            "status": "ready", "pid": os.getpid(), "host": state.WRITER_HOST,
            "common_lock_file": str(state.ROOT / "bank.lock"),
            "owner_epoch": owner["ownership_epoch"], "heartbeat_unix": time.time(),
        })
        if not (run / "latest.json").exists():
            return
        self.context.fixed_checkpoint.recover_selection(run)
        artifacts = [self.context.checkpoint.pointer_path(run)]
        if (run / "selected.json").exists():
            artifacts.append(self.context.checkpoint.pointer_path(run, "selected"))
        for artifact in dict.fromkeys(artifacts):
            if not (run / "bank-acks" / (artifact.name + ".json")).exists():
                with state.locked(self.registry / "registry.lock"):
                    state.require_owner(self.current()["event"], cell, state.WRITER_HOST)
                    self.context.bank.bank_checkpoint(
                        run, artifact, state.DURABLE / "runs" / cell / artifact.name,
                        state.ROOT / "bank.lock", state.WRITER_HOST,
                    )
        if (run / "completed.json").exists():
            ack = run / "results-banked.json"
            if not ack.exists() or self.last_local_result_ack.get(cell) != state.sha(ack):
                with state.locked(self.registry / "registry.lock"):
                    state.require_owner(self.current()["event"], cell, state.WRITER_HOST)
                    self.context.bank.bank_results(run, state.DURABLE / "runs" / cell, state.ROOT / "bank.lock")
                self.last_local_result_ack[cell] = state.sha(ack)

    def remote_results(self, lease, ready):
        state.configuration()
        state.require_assignment(lease, state.WORKER_HOST, lease["gpu"])
        state.require_path(self.directory, writable=True)
        if ready["lease_sha256"] != state.digest(lease):
            raise ValueError("Foreign result owner")
        ledger = self.directory / "ledger" / lease["cell"] / lease["lease_id"]
        head, records = accepted_chain(ledger, lease)
        if head["phase"] != "frozen" or head["epoch"] != 20 or ready["head"] != head:
            raise ValueError("Results precede the accepted full20-epoch frozen state")
        frozen_record = records[head["generation"]]
        relative = Path(ready["snapshot"])
        if len(relative.parts) != 2 or relative.parts[0] != "result-artifacts":
            raise ValueError("Unsafe result snapshot path")
        state.plain_name(relative.name)
        local = self.directory / "results" / lease["lease_id"] / relative.name
        if not local.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            pending = local.parent / ("receive.partial-" + uuid.uuid4().hex)
            pending.mkdir()
            remote = state.P5 / "runs" / lease["cell"] / relative
            subprocess.run(["scp", "-q", "-r", state.WORKER_HOST + ":" + str(remote), str(pending)], check=True, timeout=900)
            snapshot = pending / relative.name
            self.context.bank.verify_result_snapshot(snapshot)
            snapshot.rename(local)
            pending.rmdir()
        self.context.bank.verify_result_snapshot(local)
        if state.sha(local / "p5-result-seal.json") != ready["result_seal_sha256"]:
            raise ValueError("Result seal changed")
        state.validate_result_binding(local, lease, frozen_record["envelope"], head)
        with state.locked(self.registry / "registry.lock"):
            certificate = self.current()
            state.require_owner(certificate["event"], lease["cell"], state.WORKER_HOST, lease["lease_id"])
            current_head, current_records = accepted_chain(ledger, lease)
            if current_head != head or current_records[head["generation"]] != frozen_record:
                raise ValueError("Accepted frozen head changed before result publication")
            state.validate_result_binding(local, lease, frozen_record["envelope"], current_head)
            destination = state.DURABLE / "xnode" / lease["cell"] / lease["lease_id"] / relative.name
            state.require_path(destination, writable=True, durable=True)
            with state.locked(state.ROOT / "bank.lock"):
                if destination.exists() and not (destination / "bank-receipt.json").exists():
                    destination = destination.with_name(destination.name + "-retry-" + uuid.uuid4().hex[:12])
                receipt = self.copy_artifact(local, destination)
                self.context.bank.verify_result_snapshot(destination, durable=True)
        return {
            "kind": "p5-xnode-result-ack-v1", "status": "verified", "writer_host": state.WRITER_HOST,
            "lease_id": lease["lease_id"], "lease_sha256": state.digest(lease),
            "result_seal_sha256": ready["result_seal_sha256"], "destination": str(destination),
            "bank_receipt": receipt, "bank_receipt_sha256": state.digest(receipt),
            "registry_event_sha256": certificate["event_sha256"],
            "binding_version": 2, "frozen_head": head, "frozen_checkpoint_ack": frozen_record["ack"],
        }

    def remote(self, lease):
        state.configuration()
        state.require_assignment(lease, state.WORKER_HOST, lease["gpu"])
        run = state.P5 / "runs" / lease["cell"]
        state.write(run / "bank-watcher-ready.json", {
            "status": "fenced", "pid": os.getpid(), "host": state.WRITER_HOST,
            "owner_node": state.WORKER_HOST, "lease_sha256": state.digest(lease),
        })
        certificate = self.current()
        expected_registration = {
            "kind": "p5-xnode-registration-v1", "writer_host": state.WRITER_HOST,
            "lease_sha256": state.digest(lease), "event_sha256": certificate["event_sha256"],
            "input_manifest_sha256": self.config["input_manifest_sha256"],
        }
        if self.registration_anchor_sha256 is not None:
            expected_registration = registration_message(
                self.config, certificate, lease, self.registration_anchor_sha256,
            )
        inventory = self.rpc(["inventory", "--lease", str(state.XROOT / "leases" / (lease["cell"] + ".json"))])
        if inventory["lease_sha256"] != state.digest(lease):
            raise ValueError("Remote inventory belongs to a stale lease")
        if inventory["registration_sha256"] != state.digest(expected_registration):
            self.send(lease, expected_registration)
        ledger = self.directory / "ledger" / lease["cell"] / lease["lease_id"]
        for item in inventory["packages"]:
            head, records = accepted_chain(ledger, lease)
            generation = item["envelope"]["generation"]
            if generation in records:
                if records[generation]["envelope"] != item["envelope"]:
                    raise ValueError("Remote producer rewrote an accepted generation")
                ack = records[generation]["ack"]
            else:
                ack = self.import_one(lease, item)
                print("XNODE_BANKED", lease["cell"], generation, ack["checkpoint"]["step"], flush=True)
            if inventory["acknowledged"].get(str(generation)) != state.digest(ack):
                self.send(lease, ack)
        for request in inventory["requests"]:
            with state.locked(self.registry / "registry.lock"):
                current = self.current()
                state.require_owner(current["event"], lease["cell"], state.WORKER_HOST, lease["lease_id"])
                if request["lease_sha256"] != state.digest(lease):
                    raise ValueError("Stale final-test checkpoint request")
                frozen_head, frozen_records = accepted_chain(ledger, lease)
                if frozen_head["phase"] != "frozen" or frozen_head["epoch"] != 20:
                    continue
                frozen_envelope = frozen_records[frozen_head["generation"]]["envelope"]
                try:
                    ack = find_accepted_ack(ledger, lease, request["checkpoint_seal_sha256"])
                except ValueError:
                    continue
                if ack["checkpoint"] not in (frozen_envelope["checkpoint"], frozen_envelope["selected"]):
                    raise ValueError("Final-test request differs from the accepted frozen selected/latest state")
                self.send(lease, {
                    "kind": "p5-xnode-checkpoint-response-v1", "nonce": request["nonce"],
                    "lease_sha256": state.digest(lease), "ack": ack,
                    "registry_event_sha256": current["event_sha256"],
                })
        if inventory["results_ready"] and not inventory["results_banked"]:
            self.send(lease, self.remote_results(lease, inventory["results_ready"]))
        state.write(self.directory / "status" / (lease["cell"] + ".json"), {
            "owner_node": state.WORKER_HOST, "lease_sha256": state.digest(lease),
            "head": accepted_chain(ledger, lease)[0], "guardian": inventory["guardian"],
            "observed_at": time.time(),
        })

    def loop(self):
        state.configuration()
        with state.locked(self.directory / "broker.lock", nonblocking=True):
            while True:
                certificate = self.current()
                for cell, owner in sorted(certificate["event"]["owners"].items()):
                    try:
                        if owner["owner_node"] == state.WRITER_HOST:
                            self.local(cell, owner)
                        else:
                            self.remote(owner)
                    except Exception as error:
                        state.write(self.directory / "errors" / (cell + ".json"), {
                            "type": type(error).__name__, "error": str(error), "time": time.time(),
                        })
                        print("XNODE_BROKER_ERROR", cell, type(error).__name__, str(error), flush=True)
                time.sleep(20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--bindings-sha256")
    parser.add_argument("mode", choices=("inventory", "receive", "prepare-message", "serve"))
    parser.add_argument("--lease", type=Path)
    parser.add_argument("--message", type=Path)
    parser.add_argument("--runtime-manifest", type=Path)
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    state.configure_bindings(args.bindings, args.bindings_sha256)
    if args.directory is None:
        args.directory = state.XROOT
    if args.mode == "serve":
        Broker(args.runtime_manifest, args.directory).loop()
    else:
        if args.mode == "inventory":
            result = remote_inventory(args.lease)
        elif args.mode == "prepare-message":
            result = prepare_message(args.lease)
        else:
            result = receive_message(args.lease, args.message) or {"status": "received"}
        print("P5_XNODE_RPC:" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
