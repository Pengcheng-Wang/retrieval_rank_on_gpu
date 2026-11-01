"""Exact, non-mutating replay recovery for missing P5 validation files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import p5_xnode_reregister as xreg
from p5_checkpoint import fsync_directory, pointer_lock
from p5_source import digest_json, read_json, sha256, write_json


REQUEST_FIELDS = {
    "kind", "cell", "source_event_sha256", "lease_sha256",
    "root_seal_sha256", "runtime_identity_sha256", "input_identity_sha256",
    "binding", "run", "contract", "runtime_request", "output", "epochs",
    "checkpoint_certificate", "ownership_certificate",
    "lease", "history_checkpoint", "history_bank_ack",
    "history_bank_receipt", "history_stage_receipt",
    "publication_fence", "bank_lock", "broker_lock",
    "writer_host", "durable_evidence_destination", "result_bank_ack",
    "result_binding_destination", "result_destination_root",
    "bindings", "activation",
    "replay_worker", "replay_worker_sha256", "proof_runner",
    "proof_runner_sha256", "frozen_runtime_files_sha256",
    "replay_pythonpath", "determinism_environment",
}
PROTECTED_FILES = (
    "latest.json", "selected.json", "frozen.json", "test-started.json",
    "test-result.json", "completed.json", "validation-history.json",
    "p5-run-config.json", "updates.jsonl",
)


def _request(path):
    path = Path(path).resolve()
    value = read_json(path)
    if (set(value) != REQUEST_FIELDS
            or value["kind"] != "p5-validation-history-recovery-request-v1"
            or not isinstance(value["epochs"], dict)
            or not value["epochs"]):
        raise ValueError("Validation-history recovery request is incomplete")
    for name in (
        "source_event_sha256", "lease_sha256", "root_seal_sha256",
        "runtime_identity_sha256", "input_identity_sha256",
        "replay_worker_sha256", "proof_runner_sha256",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", value[name]):
            raise ValueError("Validation-history recovery hash identity is invalid")
    runtime_files = value["frozen_runtime_files_sha256"]
    if (set(runtime_files) != {
            "p5_run.py", "p5_checkpoint.py", "p5_source.py",
            }
            or any(
                not re.fullmatch(r"[0-9a-f]{64}", digest)
                for digest in runtime_files.values()
            )
            or sha256(value["replay_worker"])
            != value["replay_worker_sha256"]
            or sha256(value["proof_runner"])
            != value["proof_runner_sha256"]):
        raise ValueError("Validation replay worker or frozen runtime changed")
    if not isinstance(value["replay_pythonpath"], list):
        raise ValueError("Validation replay dependency roots are invalid")
    for item in value["replay_pythonpath"]:
        if set(item) != {"path", "anchor", "anchor_sha256"}:
            raise ValueError("Validation replay dependency binding is invalid")
        root = Path(item["path"]).resolve()
        anchor = Path(item["anchor"]).resolve()
        if (not root.is_absolute()
                or not anchor.is_file()
                or not anchor.is_relative_to(root)
                or sha256(anchor) != item["anchor_sha256"]):
            raise ValueError("Validation replay dependency changed")
    if value["determinism_environment"] != {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            }:
        raise ValueError("Validation replay determinism environment differs")
    binding = value["binding"]
    if (set(binding) != {"host", "gpu", "gpu_uuid"}
            or not isinstance(binding["host"], str) or not binding["host"]
            or type(binding["gpu"]) is not int or binding["gpu"] < 0
            or not isinstance(binding["gpu_uuid"], str)
            or not binding["gpu_uuid"]):
        raise ValueError("Validation-history GPU binding is invalid")
    for epoch, item in value["epochs"].items():
        if (not epoch.isdigit() or not 1 <= int(epoch) <= 20
                or set(item) != {
                    "checkpoint", "checkpoint_seal_sha256",
                    "stage_receipt_sha256", "authority", "bank_ack",
                    "bank_receipt",
                }
                or item["authority"] not in {
                    "parent-event-inventory", "current-run-bank-ack",
                }
                or (
                    item["authority"] == "parent-event-inventory"
                    and (
                        item["bank_ack"] is not None
                        or item["bank_receipt"] is not None
                    )
                )
                or (
                    item["authority"] == "current-run-bank-ack"
                    and (
                        not isinstance(item["bank_ack"], str)
                        or not Path(item["bank_ack"]).is_absolute()
                        or not isinstance(item["bank_receipt"], str)
                        or not Path(item["bank_receipt"]).is_absolute()
                    )
                )
                or not re.fullmatch(
                    r"[0-9a-f]{64}", item["checkpoint_seal_sha256"],
                )
                or not re.fullmatch(
                    r"[0-9a-f]{64}", item["stage_receipt_sha256"],
                )):
            raise ValueError("Validation-history epoch authority is invalid")
    run = Path(value["run"]).resolve()
    output = Path(value["output"]).resolve()
    durable = Path(value["durable_evidence_destination"])
    result_binding = Path(value["result_binding_destination"])
    if (not run.is_absolute() or not output.is_absolute()
            or run == output or run.is_relative_to(output)
            or output.is_relative_to(run)
            or not durable.is_absolute() or not result_binding.is_absolute()
            or not isinstance(value["writer_host"], str)
            or not value["writer_host"]):
        raise ValueError("Validation recovery run and evidence roots must be separate")
    _deployment_authority(value)
    return path, value


def _deployment_authority(request):
    bindings_path = Path(request["bindings"]).resolve()
    activation_path = Path(request["activation"]).resolve()
    bindings = read_json(bindings_path)
    activation = read_json(activation_path)
    lease = read_json(request["lease"])
    files = activation.get("files_sha256", {})
    control_root = Path(bindings.get("control_root", "")).resolve()
    durable_root = Path(bindings.get("durable_root", "")).resolve()
    root = Path(bindings.get("root", "")).resolve()
    cell = bindings.get("cells", {}).get(request["cell"], {})
    if (bindings.get("kind") != "p5-xnode-bindings-v1"
            or activation.get("kind") != "p5-xnode-activation-v2"
            or activation.get("status") != "activated"
            or activation.get("event_sha256")
            != request["source_event_sha256"]
            or files.get("current/bindings.json") != sha256(bindings_path)
            or files.get("current/ownership-certificate.json")
            != sha256(request["ownership_certificate"])
            or files.get(f"leases/{request['cell']}.json")
            != sha256(request["lease"])
            or request["writer_host"] != bindings.get("writer_host")
            or request["writer_host"] != lease.get("owner_node")
            or cell.get("role") != "writer"
            or cell.get("gpu") != lease.get("gpu")
            or cell.get("gpu_uuid") != lease.get("gpu_uuid")
            or Path(request["bank_lock"]).resolve() != root / "bank.lock"
            or Path(request["broker_lock"]).resolve()
            != control_root / "broker.lock"
            or Path(request["run"]).resolve()
            != Path(bindings.get("p5_root", "")).resolve()
            / "runs" / request["cell"]
            or bindings_path != control_root / "current" / "bindings.json"
            or activation_path
            != control_root / "activation" / "last-published.json"
            or Path(request["ownership_certificate"]).resolve()
            != control_root / "current" / "ownership-certificate.json"
            or Path(request["lease"]).resolve()
            != control_root / "leases" / f"{request['cell']}.json"):
        raise ValueError("Validation recovery deployment authority differs")
    for name in (
        "durable_evidence_destination", "result_binding_destination",
        "result_destination_root",
    ):
        destination = Path(request[name]).resolve()
        if (destination == durable_root
                or not destination.is_relative_to(durable_root)):
            raise ValueError("Recovery durable destination is outside authority")
    if Path(request["result_destination_root"]).resolve() != (
            durable_root / "runs" / request["cell"]):
        raise ValueError("Recovery result destination root differs")


def _protected(run):
    result = {}
    for name in PROTECTED_FILES:
        path = Path(run) / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Protected P5 result file is absent or unsafe: {name}")
        result[name] = sha256(path)
    return result


def _file_manifest(directory, exclude=()):
    directory = Path(directory)
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Recovery authority may not contain symlinks")
        if path.is_file():
            name = path.relative_to(directory).as_posix()
            if name not in exclude:
                result[name] = {
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
    return result


def _checkpoint_header(path):
    path = Path(path)
    seal = read_json(path / "p5-seal.json")
    files = seal.get("files_sha256", {})
    if (seal.get("kind") != "p5-full-state-v1"
            or seal.get("status") != "sealed"
            or set(files) != {"model.safetensors", "training-state.pt"}):
        raise ValueError("Recovery checkpoint is not sealed full state")
    for name, expected in files.items():
        member = path / name
        if member.is_symlink() or not member.is_file() or sha256(member) != expected:
            raise ValueError("Recovery checkpoint member differs from its seal")
    return {
        "artifact": path.name,
        "seal_sha256": sha256(path / "p5-seal.json"),
        "contract_sha256": seal["contract_sha256"],
        "step": seal["global_step"],
        "epoch": seal["epoch"],
        "phase": seal["phase"],
        "model_sha256": files["model.safetensors"],
        "training_state_sha256": files["training-state.pt"],
    }


def _verify_local_stage(checkpoint, receipt, expected_names):
    if set(receipt.get("files", {})) != expected_names:
        raise ValueError("Checkpoint stage receipt has an incomplete inventory")
    for name, metadata in receipt["files"].items():
        path = checkpoint / name
        if (path.is_symlink() or not path.is_file()
                or set(metadata) != {"sha256", "bytes"}
                or path.stat().st_size != metadata["bytes"]
                or sha256(path) != metadata["sha256"]):
            raise ValueError("Staged checkpoint file differs from its receipt")


def _history_authority(request):
    import torch

    run = Path(request["run"]).resolve()
    certificate = read_json(request["ownership_certificate"])
    lease = read_json(request["lease"])
    checkpoint_certificate = read_json(request["checkpoint_certificate"])
    event = certificate["event"]
    receipt = certificate["bank_receipt"]
    if (certificate.get("event_sha256") != request["source_event_sha256"]
            or xreg.event_sha(event) != request["source_event_sha256"]
            or xreg.digest(receipt) != certificate.get("bank_receipt_sha256")
            or receipt.get("files", {}).get("ownership.json", {}).get("sha256")
            != request["source_event_sha256"]
            or xreg.digest(lease) != request["lease_sha256"]
            or event.get("owners", {}).get(request["cell"]) != lease
            or lease.get("root", {}).get("seal_sha256")
            != request["root_seal_sha256"]
            or event.get("previous_event_sha256")
            != checkpoint_certificate.get("event_sha256")
            or checkpoint_certificate.get(
                "event", {},
            ).get("owners", {}).get(
                request["cell"], {},
            ).get("root") != lease.get("root")):
        raise ValueError("Frozen history ownership authority differs")
    checkpoint = Path(request["history_checkpoint"]).resolve()
    header = _checkpoint_header(checkpoint)
    ack = read_json(request["history_bank_ack"])
    bank_receipt_path = Path(request["history_bank_receipt"])
    bank_receipt = read_json(bank_receipt_path)
    stage_receipt_path = Path(request["history_stage_receipt"]).resolve()
    stage_receipt = read_json(stage_receipt_path)
    if (ack.get("kind") != "p5-full-state-v1"
            or ack.get("status") != "verified"
            or ack.get("artifact") != checkpoint.name
            or ack.get("seal_sha256") != header["seal_sha256"]
            or ack.get("contract_sha256") != header["contract_sha256"]
            or stage_receipt.get("kind")
            != "p5-validation-checkpoint-stage-v1"
            or stage_receipt.get("status") != "verified"
            or stage_receipt.get("purpose") != "history-authority"
            or stage_receipt.get("source") != ack.get("destination")
            or stage_receipt.get("source_bank_ack_sha256")
            != sha256(request["history_bank_ack"])
            or stage_receipt.get("source_bank_receipt_sha256")
            != ack.get("bank_receipt_sha256")
            or ack.get("bank_receipt_sha256")
            != digest_json(bank_receipt)
            or bank_receipt != {"files": stage_receipt.get("files")}
            or stage_receipt.get("source_bank_receipt_file_sha256")
            != sha256(bank_receipt_path)
            or stage_receipt.get("checkpoint_seal_sha256")
            != header["seal_sha256"]):
        raise ValueError("Frozen history checkpoint bank authority differs")
    _verify_local_stage(
        checkpoint, stage_receipt,
        {"model.safetensors", "training-state.pt", "p5-seal.json"},
    )
    root = lease["root"]
    if (header["contract_sha256"] != root["contract_sha256"]
            or header["step"] < root["step"]
            or header["epoch"] < root["epoch"]):
        raise ValueError("Frozen history checkpoint regresses from the owned root")
    contract = read_json(request["contract"])
    payload = torch.load(
        checkpoint / "training-state.pt", map_location="cpu", weights_only=False,
    )
    state = payload.get("state", {})
    history = read_json(run / "validation-history.json")
    latest = read_json(run / "latest.json")
    frozen = read_json(run / "frozen.json")
    selected = read_json(run / "selected.json")
    if (payload.get("contract") != contract
            or payload.get("checkpoint_boundary")
            != "after_complete_optimizer_update_no_partial_gradients"
            or header["phase"] != "frozen" or header["epoch"] != 20
            or state.get("phase") != "frozen" or state.get("epoch") != 20
            or state.get("validation_history") != history
            or len(history) != 20
            or latest.get("artifact") != checkpoint.name
            or latest.get("seal_sha256") != header["seal_sha256"]
            or latest.get("global_step") != header["step"]
            or latest.get("phase") != "frozen"
            or frozen.get("status") != "frozen-before-test"
            or frozen.get("epochs") != 20
            or frozen.get("selected") != selected.get("artifact")
            or frozen.get("selected_seal_sha256")
            != selected.get("seal_sha256")):
        raise ValueError("Frozen checkpoint embedded validation history differs")
    return history, header


def _checkpoint_inventory(certificate, cell):
    event = certificate.get("event", {})
    receipt = certificate.get("bank_receipt", {})
    recovery = event.get("durable_recovery", {}).get(cell, {})
    entries = recovery.get("verified_checkpoints", [])
    owner = event.get("owners", {}).get(cell, {})
    if (certificate.get("event_sha256") != xreg.event_sha(event)
            or xreg.digest(receipt) != certificate.get("bank_receipt_sha256")
            or receipt.get("files", {}).get(
                "ownership.json", {},
            ).get("sha256") != certificate.get("event_sha256")
            or event.get("status") != "committed"
            or recovery.get("historically_accepted") is not False
            or recovery.get("authority")
            != "receipt-verified-durable-full-state-not-historical-acceptance"
            or not isinstance(entries, list) or not entries
            or owner.get("root") != recovery.get("root")
            or recovery.get("chosen_checkpoint_seal_sha256")
            != recovery.get("root", {}).get("seal_sha256")):
        raise ValueError("Historical checkpoint inventory is not event-authoritative")
    previous = None
    paths = set()
    for generation, entry in enumerate(entries, 1):
        path = entry.get("path")
        if (entry.get("generation") != generation
                or not isinstance(path, str) or not path
                or path in paths
                or not re.fullmatch(
                    r"[0-9a-f]{64}",
                    entry.get("bank_receipt_sha256", ""),
                )
                or not re.fullmatch(
                    r"[0-9a-f]{64}",
                    entry.get("checkpoint_seal_sha256", ""),
                )
                or type(entry.get("step")) is not int
                or type(entry.get("epoch")) is not int
                or entry.get("phase") not in {"train", "validate", "frozen"}):
            raise ValueError("Historical checkpoint inventory is malformed")
        if previous is not None:
            if entry["step"] > previous["step"]:
                valid = (
                    previous["phase"] == "train"
                    and entry["phase"] in {"train", "validate"}
                    and entry["epoch"] == previous["epoch"]
                )
            elif entry["step"] == previous["step"]:
                valid = (
                    previous["phase"] == "train"
                    and entry["phase"] == "validate"
                    and entry["epoch"] == previous["epoch"]
                    or previous["phase"] == "validate"
                    and entry["phase"] in {"train", "frozen"}
                    and entry["epoch"] == previous["epoch"] + 1
                )
            else:
                valid = False
            if not valid:
                raise ValueError("Historical checkpoint inventory ancestry differs")
        paths.add(path)
        previous = entry
    if (entries[-1].get("path") != recovery.get("chosen_checkpoint")
            or entries[-1].get("checkpoint_seal_sha256")
            != recovery.get("chosen_checkpoint_seal_sha256")):
        raise ValueError("Historical checkpoint inventory omits its chosen root")
    return entries


def _exact_state_equal(left, right):
    import numpy as np
    import torch

    if type(left) is not type(right):
        return False
    if torch.is_tensor(left):
        left = left.detach().cpu().contiguous()
        right = right.detach().cpu().contiguous()
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and left.reshape(-1).view(torch.uint8).numpy().tobytes()
            == right.reshape(-1).view(torch.uint8).numpy().tobytes()
        )
    if isinstance(left, np.ndarray):
        left = np.ascontiguousarray(left)
        right = np.ascontiguousarray(right)
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and left.tobytes() == right.tobytes()
        )
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _exact_state_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _exact_state_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, float):
        return struct.pack(">d", left) == struct.pack(">d", right)
    return left == right


def _verify_validation_frozen_parent(validation, frozen, history):
    import torch
    from p5_checkpoint import IndexSampler

    validation_payload = torch.load(
        Path(validation) / "training-state.pt",
        map_location="cpu", weights_only=False,
    )
    frozen_payload = torch.load(
        Path(frozen) / "training-state.pt",
        map_location="cpu", weights_only=False,
    )
    if set(validation_payload) != set(frozen_payload):
        raise ValueError("Validation/frozen checkpoint schemas differ")
    for name in set(validation_payload) - {"state", "sampler"}:
        if not _exact_state_equal(
            validation_payload[name], frozen_payload[name],
        ):
            raise ValueError("Frozen checkpoint changed non-sampler full state")
    before = validation_payload["state"]
    after = frozen_payload["state"]
    excluded = {
        "epoch", "phase", "validation_history",
        "best_metric", "selected_artifact",
    }
    final_metric = history[-1]["primary"]["recall@10"]
    improved = (
        before.get("best_metric") is None
        or final_metric > before["best_metric"]
    )
    expected_best = final_metric if improved else before.get("best_metric")
    expected_selected = (
        Path(frozen).name if improved else before.get("selected_artifact")
    )
    if (set(before) != set(after)
            or any(
                not _exact_state_equal(before[name], after[name])
                for name in set(before) - excluded
            )
            or before.get("phase") != "validate"
            or after.get("phase") != "frozen"
            or after.get("epoch") != before.get("epoch") + 1
            or after.get("validation_history") != history
            or before.get("validation_history") != history[:-1]
            or after.get("best_metric") != expected_best
            or after.get("selected_artifact") != expected_selected):
        raise ValueError("Frozen state is not the exact validation successor")
    sampler_state = validation_payload["sampler"]
    sampler = IndexSampler(sampler_state["size"], 0)
    sampler.load_state_dict(sampler_state)
    sampler.advance()
    if not _exact_state_equal(
            sampler.state_dict(), frozen_payload["sampler"]):
        raise ValueError("Frozen sampler is not the exact epoch successor")


def _stage_authority(request, epoch):
    item = request["epochs"][str(epoch)]
    checkpoint = Path(item["checkpoint"]).resolve()
    receipt_path = checkpoint.parent / "stage-receipt.json"
    receipt = read_json(receipt_path)
    source = Path(receipt.get("source", "")).resolve()
    expected_names = {
        "model.safetensors", "training-state.pt", "p5-seal.json",
    }
    if (sha256(receipt_path) != item["stage_receipt_sha256"]
            or receipt.get("kind") != "p5-validation-checkpoint-stage-v1"
            or receipt.get("status") != "verified"
            or receipt.get("epoch") != epoch
            or receipt.get("checkpoint_seal_sha256")
            != item["checkpoint_seal_sha256"]):
        raise ValueError("Validation checkpoint stage receipt differs")
    _verify_local_stage(checkpoint, receipt, expected_names)
    seal = read_json(checkpoint / "p5-seal.json")
    contract = read_json(request["contract"])
    if (seal.get("kind") != "p5-full-state-v1"
            or seal.get("status") != "sealed"
            or seal.get("phase") != "validate"
            or seal.get("epoch") != epoch - 1
            or seal.get("contract_sha256") != digest_json(contract)
            or sha256(checkpoint / "p5-seal.json")
            != item["checkpoint_seal_sha256"]):
        raise ValueError("Validation checkpoint maps to another epoch or contract")
    header = _checkpoint_header(checkpoint)
    if item["authority"] == "parent-event-inventory":
        certificate = read_json(request["checkpoint_certificate"])
        entries = _checkpoint_inventory(certificate, request["cell"])
        matching = [
            entry for entry in entries
            if Path(entry.get("path", "")).resolve() == source
        ]
        if (len(matching) != 1
                or matching[0].get("checkpoint_seal_sha256")
                != item["checkpoint_seal_sha256"]
                or matching[0].get("bank_receipt_sha256")
                != receipt.get("source_bank_receipt_sha256")
                or header["seal_sha256"]
                != matching[0]["checkpoint_seal_sha256"]
                or header["step"] != matching[0]["step"]
                or header["epoch"] != matching[0]["epoch"]
                or header["phase"] != matching[0]["phase"]):
            raise ValueError(
                "Staged validation checkpoint differs from parent event"
            )
    else:
        ack_path = Path(item["bank_ack"])
        ack = read_json(ack_path)
        bank_receipt_path = Path(item["bank_receipt"])
        bank_receipt = read_json(bank_receipt_path)
        final_header = _checkpoint_header(request["history_checkpoint"])
        if (ack.get("kind") != "p5-full-state-v1"
                or ack.get("status") != "verified"
                or ack.get("artifact") != source.name
                or ack.get("seal_sha256") != header["seal_sha256"]
                or ack.get("contract_sha256") != header["contract_sha256"]
                or ack.get("destination") != str(source)
                or ack.get("bank_receipt_sha256")
                != receipt.get("source_bank_receipt_sha256")
                or ack.get("bank_receipt_sha256")
                != digest_json(bank_receipt)
                or bank_receipt != {"files": receipt.get("files")}
                or receipt.get("source_bank_receipt_file_sha256")
                != sha256(bank_receipt_path)
                or receipt.get("source_bank_ack_sha256")
                != sha256(ack_path)):
            raise ValueError("Current-run checkpoint bank authority differs")
        xreg._progress(header, final_header)
        if header["model_sha256"] != final_header["model_sha256"]:
            raise ValueError("Frozen model differs from validation parent")
        _verify_validation_frozen_parent(
            checkpoint, request["history_checkpoint"],
            read_json(Path(request["run"]) / "validation-history.json"),
        )
    return checkpoint, receipt


class NativeValidationBackend:
    def replay(self, request, epoch, checkpoint, output):
        runtime_request_path = Path(request["runtime_request"])
        runtime_request = read_json(runtime_request_path)
        if (sha256(runtime_request_path) != request["runtime_identity_sha256"]
                or digest_json(runtime_request["inputs"])
                != request["input_identity_sha256"]):
            raise ValueError("Validation replay runtime or input identity changed")
        runtime_request["checkpoint"] = str(checkpoint)
        runtime_request["contract"] = request["contract"]
        runtime_request["output"] = str(
            Path(runtime_request["binding"]["control_root"])
            / "gates"
            / (
                "validation-history-recovery-"
                + request["cell"] + "-"
                + f"epoch-{epoch:02d}-"
                + uuid.uuid4().hex
            )
        )
        evidence = Path(output).parent
        worker_request = evidence / (
            "runtime-request-" + uuid.uuid4().hex + ".json"
        )
        response = evidence / (
            "runtime-response-" + uuid.uuid4().hex + ".json"
        )
        write_json(worker_request, runtime_request)
        environment = os.environ.copy()
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PYTHONPATH"] = os.pathsep.join(
            item["path"] for item in request["replay_pythonpath"]
        )
        environment.update(request["determinism_environment"])
        subprocess.run([
            sys.executable,
            str(Path(request["replay_worker"])),
            "--request", str(worker_request),
            "--proof-runner", str(Path(request["proof_runner"])),
            "--output", str(output),
            "--response", str(response),
        ], check=True, env=environment)
        result = read_json(response)
        runtime_root = Path(
            runtime_request["inputs"]["runtime"]["path"]
        ).resolve()
        expected_origins = {
            name: str(runtime_root / name)
            for name in request["frozen_runtime_files_sha256"]
        }
        if result.get("module_origins") != expected_origins:
            raise ValueError("Validation replay imported non-frozen helpers")
        for name, expected in request[
                "frozen_runtime_files_sha256"].items():
            if sha256(runtime_root / name) != expected:
                raise ValueError("Frozen validation runtime helper changed")
        return result


def recover_epoch(request_path, epoch, backend=None):
    request_path, request = _request(request_path)
    if str(epoch) not in request["epochs"]:
        raise ValueError("Requested validation epoch is not authorized")
    evidence = Path(request["output"]) / f"epoch-{epoch:02d}"
    evidence.mkdir(parents=True, exist_ok=True)
    with _exclusive_file_lock(evidence / "recovery.lock"):
        return _recover_epoch_locked(
            request_path, request, epoch, backend, evidence,
        )


def _recover_epoch_locked(
        request_path, request, epoch, backend, evidence):
    receipt_path = (
        evidence / "recovery-receipt.json"
    )
    if receipt_path.exists():
        verify_epoch(request_path, epoch)
        return read_json(receipt_path)
    if socket.gethostname() == "":
        raise RuntimeError("Validation recovery requires an identified host")
    run = Path(request["run"]).resolve()
    history, history_header = _history_authority(request)
    expected = history[epoch - 1]
    if (expected.get("split") != "valid"
            or expected.get("users") != expected.get("expected_users")
            or expected.get("beams_per_user") != 20
            or not re.fullmatch(
                r"[0-9a-f]{64}", expected.get("predictions_sha256", ""),
            )):
        raise ValueError("Historical validation entry is incomplete")
    checkpoint, stage_receipt = _stage_authority(request, epoch)
    canonical = run / "validation" / f"epoch-{epoch:02d}.jsonl"
    if canonical.exists() and (
            canonical.is_symlink()
            or sha256(canonical) != expected["predictions_sha256"]
    ):
        raise ValueError("Canonical validation file conflicts with sealed history")
    attempt = evidence / ("attempt-" + uuid.uuid4().hex + ".jsonl")
    before = _protected(run)
    result = (NativeValidationBackend() if backend is None else backend).replay(
        request, epoch, checkpoint, attempt,
    )
    after = _protected(run)
    state = result.get("state", {})
    metrics = result.get("metrics")
    identity = result.get("identity", request["binding"])
    if (before != after
            or state.get("phase") != "validate"
            or state.get("epoch") != epoch - 1
            or state.get("validation_history") != history[:epoch - 1]
            or metrics != expected
            or sha256(attempt) != expected["predictions_sha256"]
            or any(
                identity.get(name) != request["binding"][name]
                for name in ("host", "gpu", "gpu_uuid")
            )):
        raise ValueError("Validation replay differs from sealed history or mutated state")
    status = "existing_exact_verified" if canonical.exists() else "replayed_exact"
    receipt = {
        "kind": "p5-validation-history-recovery-v1",
        "status": status,
        "cell": request["cell"],
        "epoch": epoch,
        "source_event_sha256": request["source_event_sha256"],
        "lease_sha256": request["lease_sha256"],
        "root_seal_sha256": request["root_seal_sha256"],
        "runtime_identity_sha256": request["runtime_identity_sha256"],
        "input_identity_sha256": request["input_identity_sha256"],
        "host": identity["host"],
        "gpu": identity["gpu"],
        "gpu_uuid": identity["gpu_uuid"],
        "history_checkpoint_seal_sha256": history_header["seal_sha256"],
        "checkpoint_seal_sha256": sha256(checkpoint / "p5-seal.json"),
        "stage_receipt_sha256": sha256(
            checkpoint.parent / "stage-receipt.json"
        ),
        "predictions_sha256": expected["predictions_sha256"],
        "history_sha256_matches": True,
        "reconstructed_by_replay": not canonical.exists(),
        "historical_execution_claimed": False,
        "metrics": metrics,
        "protected_files_sha256": after,
        "staged_output": str(attempt),
        "staged_output_sha256": sha256(attempt),
    }
    if canonical.exists():
        receipt["predictions_sha256"] = sha256(canonical)
    _write_json_immutable(receipt_path, receipt)
    return receipt


def verify_epoch(request_path, epoch):
    _, request = _request(request_path)
    run = Path(request["run"]).resolve()
    history, history_header = _history_authority(request)
    expected = history[epoch - 1]
    canonical = run / "validation" / f"epoch-{epoch:02d}.jsonl"
    receipt_path = (
        Path(request["output"]) / f"epoch-{epoch:02d}"
        / "recovery-receipt.json"
    )
    receipt = read_json(receipt_path)
    checkpoint, stage_receipt = _stage_authority(request, epoch)
    staged = Path(receipt.get("staged_output", ""))
    status = receipt.get("status")
    if (receipt.get("kind") != "p5-validation-history-recovery-v1"
            or status not in {"replayed_exact", "existing_exact_verified"}
            or receipt.get("cell") != request["cell"]
            or receipt.get("epoch") != epoch
            or receipt.get("source_event_sha256")
            != request["source_event_sha256"]
            or receipt.get("lease_sha256") != request["lease_sha256"]
            or receipt.get("root_seal_sha256") != request["root_seal_sha256"]
            or receipt.get("runtime_identity_sha256")
            != request["runtime_identity_sha256"]
            or receipt.get("input_identity_sha256")
            != request["input_identity_sha256"]
            or {
                "host": receipt.get("host"),
                "gpu": receipt.get("gpu"),
                "gpu_uuid": receipt.get("gpu_uuid"),
            } != request["binding"]
            or receipt.get("history_checkpoint_seal_sha256")
            != history_header["seal_sha256"]
            or receipt.get("history_sha256_matches") is not True
            or receipt.get("historical_execution_claimed") is not False
            or receipt.get("metrics") != expected
            or receipt.get("checkpoint_seal_sha256")
            != sha256(checkpoint / "p5-seal.json")
            or receipt.get("stage_receipt_sha256")
            != sha256(checkpoint.parent / "stage-receipt.json")
            or receipt.get("predictions_sha256")
            != expected["predictions_sha256"]
            or not staged.is_file()
            or sha256(staged) != expected["predictions_sha256"]
            or receipt.get("staged_output_sha256")
            != expected["predictions_sha256"]
            or receipt.get("reconstructed_by_replay")
            is not (status == "replayed_exact")
            or receipt.get("protected_files_sha256") != _protected(run)):
        raise ValueError("Validation recovery receipt differs")
    if canonical.exists() and sha256(canonical) != expected["predictions_sha256"]:
        raise ValueError("Canonical validation file differs")
    return {"status": "verified", "cell": request["cell"], "epoch": epoch}


def _publication_fence(request):
    fence = read_json(request["publication_fence"])
    if (fence.get("kind") != "p5-validation-history-recovery-fence-v1"
            or fence.get("status") != "active"
            or fence.get("cell") != request["cell"]
            or fence.get("source_event_sha256")
            != request["source_event_sha256"]
            or fence.get("broker_stopped") is not True
            or type(fence.get("broker_pid")) is not int
            or fence["broker_pid"] <= 0
            or not isinstance(fence.get("broker_stopped_at_unix"), (int, float))):
        raise ValueError("Validation recovery publication is not fenced")
    if Path(f"/proc/{fence['broker_pid']}").exists():
        raise ValueError("The fenced broker PID is still live")
    return fence


def _reject_live_broker():
    proc = Path("/proc")
    if not proc.is_dir():
        return
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"p5_xnode_broker.py" in command:
            raise ValueError(
                f"P5 broker process {entry.name} is live during publication"
            )


@contextmanager
def _exclusive_file_lock(path, nonblocking=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            if lock.seek(0, os.SEEK_END) == 0:
                lock.write(b"\0")
                lock.flush()
            lock.seek(0)
            mode = msvcrt.LK_NBLCK if nonblocking else msvcrt.LK_LOCK
            msvcrt.locking(lock.fileno(), mode, 1)
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0)
            fcntl.flock(lock, flags)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


def _json_bytes(value):
    return (
        json.dumps(
            value, indent=2, sort_keys=True, ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
    ).encode()


def _write_json_immutable(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _json_bytes(value)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != encoded:
            raise ValueError(f"Immutable recovery evidence differs: {path.name}")
        return
    staging = _evidence_staging(path)
    with staging.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(staging, path)
    except FileExistsError:
        if path.is_symlink() or path.read_bytes() != encoded:
            raise ValueError(
                f"Immutable recovery evidence differs: {path.name}"
            )
    finally:
        staging.unlink(missing_ok=True)
    fsync_directory(path.parent)


def _copy_immutable(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink() or not source.is_file():
        raise ValueError("Recovery evidence source is absent or unsafe")
    if destination.exists():
        if destination.is_symlink() or sha256(destination) != sha256(source):
            raise ValueError(
                f"Immutable recovery evidence differs: {destination.name}"
            )
        return
    staging = _evidence_staging(destination)
    with source.open("rb") as incoming, staging.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, 8 << 20)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    try:
        os.link(staging, destination)
    except FileExistsError:
        if destination.is_symlink() or sha256(destination) != sha256(source):
            raise ValueError(
                f"Immutable recovery evidence differs: {destination.name}"
            )
    finally:
        staging.unlink(missing_ok=True)
    fsync_directory(destination.parent)


def _evidence_staging(destination):
    destination = Path(destination)
    packet = next(
        (
            parent for parent in destination.parents
            if parent.name in {"publication-evidence", "result-binding"}
        ),
        None,
    )
    root = packet.parent if packet is not None else destination.parent
    staging = root / ".recovery-staging"
    staging.mkdir(parents=True, exist_ok=True)
    return staging / (
        destination.name + "." + uuid.uuid4().hex + ".partial"
    )


def _publication_packet(request_path, request, history):
    packet = Path(request["output"]) / "publication-evidence"
    packet.mkdir(parents=True, exist_ok=True)
    _copy_immutable(request_path, packet / "request.json")
    _copy_immutable(Path(__file__), packet / Path(__file__).name)
    _copy_immutable(
        request["replay_worker"],
        packet / Path(request["replay_worker"]).name,
    )
    _copy_immutable(
        request["proof_runner"],
        packet / Path(request["proof_runner"]).name,
    )
    authority = {
        "checkpoint-certificate.json": request["checkpoint_certificate"],
        "ownership-certificate.json": request["ownership_certificate"],
        "lease.json": request["lease"],
        "history-bank-ack.json": request["history_bank_ack"],
        "history-bank-receipt.json": request["history_bank_receipt"],
        "history-stage-receipt.json": request["history_stage_receipt"],
        "bindings.json": request["bindings"],
        "activation.json": request["activation"],
    }
    for name, source in authority.items():
        _copy_immutable(source, packet / "authority" / name)
    epochs = {}
    for epoch in range(1, 21):
        receipt_path = (
            Path(request["output"]) / f"epoch-{epoch:02d}"
            / "recovery-receipt.json"
        )
        checkpoint = Path(
            request["epochs"][str(epoch)]["checkpoint"]
        )
        stage_path = checkpoint.parent / "stage-receipt.json"
        receipt = read_json(receipt_path)
        stage = read_json(stage_path)
        _copy_immutable(
            receipt_path,
            packet / "epochs" / f"epoch-{epoch:02d}-recovery.json",
        )
        _copy_immutable(
            stage_path,
            packet / "epochs" / f"epoch-{epoch:02d}-stage.json",
        )
        bank_ack = None
        bank_receipt = None
        if request["epochs"][str(epoch)]["bank_ack"] is not None:
            bank_ack_path = Path(
                request["epochs"][str(epoch)]["bank_ack"]
            )
            bank_receipt_path = Path(
                request["epochs"][str(epoch)]["bank_receipt"]
            )
            bank_ack = read_json(bank_ack_path)
            bank_receipt = read_json(bank_receipt_path)
            _copy_immutable(
                bank_ack_path,
                packet / "epochs" / f"epoch-{epoch:02d}-bank-ack.json",
            )
            _copy_immutable(
                bank_receipt_path,
                packet / "epochs"
                / f"epoch-{epoch:02d}-bank-receipt.json",
            )
        epochs[str(epoch)] = {
            "predictions_sha256": history[epoch - 1]["predictions_sha256"],
            "history_entry": history[epoch - 1],
            "recovery_receipt": receipt,
            "recovery_receipt_digest": digest_json(receipt),
            "recovery_receipt_sha256": sha256(receipt_path),
            "stage_receipt": stage,
            "stage_receipt_digest": digest_json(stage),
            "stage_receipt_sha256": sha256(stage_path),
            "bank_ack": bank_ack,
            "bank_ack_sha256": (
                sha256(bank_ack_path) if bank_ack is not None else None
            ),
            "bank_receipt": bank_receipt,
            "bank_receipt_digest": (
                digest_json(bank_receipt)
                if bank_receipt is not None else None
            ),
        }
    plan = {
        "kind": "p5-validation-history-recovery-publication-plan-v1",
        "status": "validated-before-publication",
        "cell": request["cell"],
        "source_event_sha256": request["source_event_sha256"],
        "lease_sha256": request["lease_sha256"],
        "root_seal_sha256": request["root_seal_sha256"],
        "runtime_identity_sha256": request["runtime_identity_sha256"],
        "input_identity_sha256": request["input_identity_sha256"],
        "binding": request["binding"],
        "request_sha256": sha256(request_path),
        "tool_sha256": sha256(__file__),
        "validation_history_sha256": sha256(
            Path(request["run"]) / "validation-history.json"
        ),
        "epochs": epochs,
        "protected_files_sha256": _protected(request["run"]),
    }
    _write_json_immutable(packet / "publication-plan.json", plan)
    return packet, plan


def _bank_packet(packet, base):
    from bank import copy_artifact

    packet, base = Path(packet), Path(base)
    candidates = [base] + sorted(base.parent.glob(base.name + "-retry-*"))
    for candidate in candidates:
        if (candidate / "bank-receipt.json").is_file():
            try:
                return candidate, copy_artifact(packet, candidate)
            except ValueError:
                if candidate == base:
                    raise
    target = base
    if target.exists():
        target = base.with_name(
            base.name + "-retry-" + uuid.uuid4().hex[:12]
        )
    return target, copy_artifact(packet, target)


def _bank_result_packet(packet, base):
    from bank import copy_artifact

    packet, base = Path(packet), Path(base)
    marker = base / "bank-receipt.json"
    if base.exists() and not marker.is_file():
        if base.is_symlink() or not base.is_dir():
            raise ValueError("Unsafe incomplete durable result destination")
        quarantine = base.parent / "quarantine" / (
            base.name + "-incomplete-" + uuid.uuid4().hex[:12]
        )
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        base.rename(quarantine)
        write_json(
            quarantine.parent / (quarantine.name + ".receipt.json"),
            {
                "kind": "p5-incomplete-result-quarantine-v1",
                "status": "preserved",
                "original_path": str(base),
                "preserved_path": str(quarantine),
            },
        )
    return base, copy_artifact(packet, base)


def _bank_staged_result_locked(request, run, final_manifest):
    import p5_bank

    p5_bank._recover_selection_locked(run)
    source = Path(request["output"]) / "result-source"
    source.mkdir(parents=True, exist_ok=True)
    result_files = list(p5_bank.RESULT_FILES)
    if (run / "maintenance-attestation.json").is_file():
        result_files.append("maintenance-attestation.json")
    for name in result_files:
        _copy_immutable(run / name, source / name)
    for epoch in range(1, 21):
        receipt = read_json(
            Path(request["output"]) / f"epoch-{epoch:02d}"
            / "recovery-receipt.json"
        )
        _copy_immutable(
            receipt["staged_output"],
            source / "validation" / f"epoch-{epoch:02d}.jsonl",
        )
    snapshot = p5_bank._build_result_snapshot_locked(source)
    result_destination, result_receipt = _bank_result_packet(
        snapshot,
        Path(request["result_destination_root"]) / snapshot.name,
    )
    result_seal = p5_bank.verify_result_snapshot(
        result_destination, durable=True,
    )
    if read_json(
            result_destination / "bank-receipt.json"
            ) != result_receipt:
        raise ValueError("Durable P5 result receipt differs")
    if (result_seal["files_sha256"]["validation-history.json"]
            != final_manifest["validation_history_sha256"]):
        raise ValueError("Staged result names another validation history")
    for epoch in range(1, 21):
        name = f"validation/epoch-{epoch:02d}.jsonl"
        if (result_seal["files_sha256"][name]
                != final_manifest["epochs"][str(epoch)][
                    "predictions_sha256"
                ]):
            raise ValueError("Staged result differs from recovered predictions")
    for name, expected in final_manifest["protected_files_sha256"].items():
        if (name in result_seal["files_sha256"]
                and result_seal["files_sha256"][name] != expected):
            raise ValueError("Staged result changed protected P5 metadata")
    result_ack = {
        "status": "verified",
        "destination": str(result_destination),
        "bank_receipt_sha256": digest_json(result_receipt),
        "result_seal_sha256": sha256(
            result_destination / "p5-result-seal.json"
        ),
        "test_predictions_sha256": result_seal[
            "files_sha256"
        ]["test-predictions.jsonl"],
    }
    evidence_destination = Path(
        final_manifest["durable_evidence_destination"]
    )
    evidence_receipt = read_json(
        evidence_destination / "bank-receipt.json"
    )
    if (digest_json(evidence_receipt)
            != final_manifest["durable_evidence_bank_receipt_sha256"]
            or evidence_receipt != {
                "files": _file_manifest(
                    evidence_destination, {"bank-receipt.json"},
                ),
            }):
        raise ValueError("Durable recovery evidence receipt differs")
    packet = Path(request["output"]) / "result-binding"
    packet.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(request["output"]) / "recovery-manifest.json"
    _copy_immutable(manifest_path, packet / "recovery-manifest.json")
    _write_json_immutable(packet / "results-banked.json", result_ack)
    _copy_immutable(
        result_destination / "bank-receipt.json",
        packet / "result-bank-receipt.json",
    )
    _copy_immutable(
        evidence_destination / "bank-receipt.json",
        packet / "recovery-bank-receipt.json",
    )
    binding = {
        "kind": "p5-validation-history-result-binding-v1",
        "status": "verified",
        "cell": request["cell"],
        "source_event_sha256": request["source_event_sha256"],
        "lease_sha256": request["lease_sha256"],
        "recovery_manifest_sha256": sha256(manifest_path),
        "recovery_evidence_destination": str(evidence_destination),
        "recovery_bank_receipt_sha256": digest_json(evidence_receipt),
        "result_destination": str(result_destination),
        "result_bank_ack_sha256": hashlib.sha256(
            _json_bytes(result_ack)
        ).hexdigest(),
        "result_bank_receipt_sha256": digest_json(result_receipt),
        "result_seal_sha256": result_ack["result_seal_sha256"],
        "result_inventory_sha256": digest_json(
            result_seal["files_sha256"]
        ),
    }
    _write_json_immutable(packet / "binding.json", binding)
    destination, binding_receipt = _bank_packet(
        packet, request["result_binding_destination"],
    )
    if read_json(destination / "bank-receipt.json") != binding_receipt:
        raise ValueError("Durable recovery/result binding receipt differs")
    complete = {
        **binding,
        "binding_destination": str(destination),
        "binding_bank_receipt_sha256": digest_json(binding_receipt),
    }
    _write_json_immutable(
        run / "validation-history-recovery-complete.json", complete,
    )
    return complete, result_ack, result_seal


def _publish_all(request_path):
    request_path, request = _request(request_path)
    complete_path = (
        Path(request["run"])
        / "validation-history-recovery-complete.json"
    )
    result_ack_path = Path(request["result_bank_ack"])
    if result_ack_path.exists():
        if complete_path.is_file():
            return verify_complete(request_path)
        raise ValueError("An unbound result acknowledgement already exists")
    resume_bound_publication = complete_path.is_file()
    _publication_fence(request)
    history, history_header = _history_authority(request)
    run = Path(request["run"]).resolve()
    epochs = sorted(map(int, request["epochs"]))
    if epochs != list(range(1, 21)):
        raise ValueError("Canonical publication requires all 20 validation epochs")
    receipts = {}
    for epoch in epochs:
        verify_epoch(request_path, epoch)
        receipt_path = (
            Path(request["output"]) / f"epoch-{epoch:02d}"
            / "recovery-receipt.json"
        )
        recovery_receipt = read_json(receipt_path)
        stage_path = (
            Path(request["epochs"][str(epoch)]["checkpoint"]).parent
            / "stage-receipt.json"
        )
        receipts[str(epoch)] = {
            "recovery_receipt_digest": digest_json(recovery_receipt),
            "recovery_receipt_sha256": sha256(receipt_path),
            "stage_receipt_digest": digest_json(read_json(stage_path)),
            "stage_receipt_sha256": sha256(stage_path),
            "predictions_sha256": history[epoch - 1]["predictions_sha256"],
        }
    packet, _ = _publication_packet(
        request_path, request, history,
    )
    manifest_path = Path(request["output"]) / "recovery-manifest.json"
    prepared = {
        "kind": "p5-validation-history-recovery-manifest-v1",
        "status": "prepared",
        "cell": request["cell"],
        "source_event_sha256": request["source_event_sha256"],
        "lease_sha256": request["lease_sha256"],
        "root_seal_sha256": request["root_seal_sha256"],
        "history_checkpoint_seal_sha256": history_header["seal_sha256"],
        "binding": request["binding"],
        "epochs": receipts,
        "protected_files_sha256": _protected(run),
        "validation_history_sha256": sha256(run / "validation-history.json"),
        "publication_plan_sha256": sha256(
            packet / "publication-plan.json"
        ),
    }
    with pointer_lock(run), _exclusive_file_lock(request["bank_lock"]):
        _publication_fence(request)
        _reject_live_broker()
        if socket.gethostname() != request["writer_host"]:
            raise RuntimeError("Only the declared node0 writer may publish recovery")
        durable, evidence_receipt = _bank_packet(
            packet, request["durable_evidence_destination"],
        )
        durable_receipt_path = durable / "bank-receipt.json"
        if read_json(durable_receipt_path) != evidence_receipt:
            raise ValueError("Durable recovery evidence receipt differs")
        prepared.update({
            "durable_evidence_destination": str(durable),
            "durable_evidence_bank_receipt_sha256": digest_json(
                evidence_receipt
            ),
            "durable_evidence_bank_receipt_file_sha256": sha256(
                durable_receipt_path
            ),
        })
        final_manifest = {
            **prepared,
            "status": "published-and-result-bound",
        }
        if manifest_path.exists():
            existing = read_json(manifest_path)
            if existing not in (
                final_manifest,
            ):
                raise ValueError("Validation recovery manifest differs")
        else:
            write_json(manifest_path, final_manifest)
        run_manifest_path = run / "validation-history-recovery.json"
        if resume_bound_publication:
            if read_json(run_manifest_path) not in (
                    prepared, final_manifest):
                raise ValueError("Bound recovery manifest changed on retry")
        else:
            write_json(run_manifest_path, prepared)
        for epoch in epochs:
            receipt = read_json(
                Path(request["output"]) / f"epoch-{epoch:02d}"
                / "recovery-receipt.json"
            )
            if sha256(receipt["staged_output"]) != receipt[
                    "predictions_sha256"]:
                raise ValueError(
                    "Staged validation bytes changed before publication"
                )
        complete, result_ack, result_seal = _bank_staged_result_locked(
            request, run, final_manifest,
        )
        write_json(
            run_manifest_path, final_manifest,
        )
        fsync_directory(run)
        for epoch in epochs:
            receipt = read_json(
                Path(request["output"]) / f"epoch-{epoch:02d}"
                / "recovery-receipt.json"
            )
            canonical = run / "validation" / f"epoch-{epoch:02d}.jsonl"
            staged = Path(receipt["staged_output"])
            if sha256(staged) != receipt["predictions_sha256"]:
                raise ValueError("Staged validation bytes changed before publication")
            if canonical.exists():
                if sha256(canonical) != receipt["predictions_sha256"]:
                    raise ValueError("Canonical validation publication conflicts")
                continue
            _reject_live_broker()
            canonical.parent.mkdir(parents=True, exist_ok=True)
            staging = canonical.with_name(
                canonical.name + ".recovery-" + uuid.uuid4().hex
            )
            with staged.open("rb") as source, staging.open("xb") as target:
                for chunk in iter(lambda: source.read(8 << 20), b""):
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
            if sha256(staging) != receipt["predictions_sha256"]:
                staging.unlink(missing_ok=True)
                raise ValueError("Copied validation bytes changed before linking")
            try:
                os.link(staging, canonical)
            except FileExistsError:
                if sha256(canonical) != receipt["predictions_sha256"]:
                    raise ValueError("Concurrent validation publication differs")
            finally:
                staging.unlink(missing_ok=True)
            fsync_directory(canonical.parent)
        _reject_live_broker()
        import p5_bank

        if p5_bank._result_inventory(run) != result_seal["files_sha256"]:
            raise ValueError("Canonical recovered result differs from bound result")
        _write_json_immutable(
            Path(request["result_bank_ack"]), result_ack,
        )
    return complete


def publish_all(request_path):
    _, request = _request(request_path)
    _reject_live_broker()
    with _exclusive_file_lock(request["broker_lock"], nonblocking=True):
        _publication_fence(request)
        _reject_live_broker()
        return _publish_all(request_path)


def bank_result_and_bind(request_path, _broker_locked=False):
    if not _broker_locked:
        _, request = _request(request_path)
        _reject_live_broker()
        with _exclusive_file_lock(
                request["broker_lock"], nonblocking=True):
            return bank_result_and_bind(
                request_path, _broker_locked=True,
            )
    request_path, request = _request(request_path)
    run = Path(request["run"])
    manifest = read_json(run / "validation-history-recovery.json")
    if (manifest.get("kind")
            != "p5-validation-history-recovery-manifest-v1"
            or manifest.get("status")
            != "published-awaiting-result-binding"
            or sorted(map(int, manifest.get("epochs", {})))
            != list(range(1, 21))):
        raise ValueError("Published validation recovery manifest is incomplete")
    result_ack_path = Path(request["result_bank_ack"])
    if (result_ack_path.resolve() != (run / "results-banked.json").resolve()
            or socket.gethostname() != request["writer_host"]):
        raise ValueError("Result binding must run on the authoritative writer")
    if result_ack_path.exists():
        if (run / "validation-history-recovery-complete.json").is_file():
            return verify_complete(request_path)
        raise ValueError("An unbound ordinary result acknowledgement already exists")
    _publication_fence(request)
    _reject_live_broker()
    evidence_destination = Path(
        manifest["durable_evidence_destination"]
    )
    with pointer_lock(run), _exclusive_file_lock(request["bank_lock"]):
        _publication_fence(request)
        _reject_live_broker()
        import p5_bank

        p5_bank._recover_selection_locked(run)
        snapshot = p5_bank._build_result_snapshot_locked(run)
        result_destination, result_receipt = _bank_packet(
            snapshot,
            Path(request["result_destination_root"]) / snapshot.name,
        )
        result_seal = p5_bank.verify_result_snapshot(
            result_destination, durable=True,
        )
        if read_json(
                result_destination / "bank-receipt.json"
                ) != result_receipt:
            raise ValueError("Durable P5 result receipt differs")
        result_ack = {
            "status": "verified",
            "destination": str(result_destination),
            "bank_receipt_sha256": digest_json(result_receipt),
            "result_seal_sha256": sha256(
                result_destination / "p5-result-seal.json"
            ),
            "test_predictions_sha256": result_seal[
                "files_sha256"
            ]["test-predictions.jsonl"],
        }
        evidence_receipt = read_json(
            evidence_destination / "bank-receipt.json"
        )
        if (result_receipt != {
                    "files": _file_manifest(
                        result_destination, {"bank-receipt.json"},
                    ),
                }
                or result_seal.get("files_sha256", {}).get(
                    "validation-history.json"
                ) != manifest["validation_history_sha256"]
                or digest_json(evidence_receipt)
                != manifest["durable_evidence_bank_receipt_sha256"]
                or evidence_receipt != {
                    "files": _file_manifest(
                        evidence_destination, {"bank-receipt.json"},
                    ),
                }):
            raise ValueError("Recovery provenance or result bank receipt differs")
        for epoch in range(1, 21):
            name = f"validation/epoch-{epoch:02d}.jsonl"
            if (result_seal["files_sha256"].get(name)
                    != manifest["epochs"][str(epoch)]["predictions_sha256"]):
                raise ValueError("Banked result differs from recovered history")
        for name, expected in manifest["protected_files_sha256"].items():
            if (name in result_seal["files_sha256"]
                    and result_seal["files_sha256"][name] != expected):
                raise ValueError("Banked result changed protected P5 metadata")
        packet = Path(request["output"]) / "result-binding"
        packet.mkdir(parents=True, exist_ok=True)
        _copy_immutable(
            run / "validation-history-recovery.json",
            packet / "recovery-manifest.json",
        )
        _write_json_immutable(
            packet / "results-banked.json", result_ack,
        )
        _copy_immutable(
            result_destination / "bank-receipt.json",
            packet / "result-bank-receipt.json",
        )
        _copy_immutable(
            evidence_destination / "bank-receipt.json",
            packet / "recovery-bank-receipt.json",
        )
        binding = {
            "kind": "p5-validation-history-result-binding-v1",
            "status": "verified",
            "cell": request["cell"],
            "source_event_sha256": request["source_event_sha256"],
            "lease_sha256": request["lease_sha256"],
            "recovery_manifest_sha256": sha256(
                run / "validation-history-recovery.json"
            ),
            "recovery_evidence_destination": str(evidence_destination),
            "recovery_bank_receipt_sha256": digest_json(evidence_receipt),
            "result_destination": str(result_destination),
            "result_bank_ack_sha256": hashlib.sha256(
                _json_bytes(result_ack)
            ).hexdigest(),
            "result_bank_receipt_sha256": digest_json(result_receipt),
            "result_seal_sha256": result_ack["result_seal_sha256"],
            "result_inventory_sha256": digest_json(
                result_seal["files_sha256"]
            ),
        }
        _write_json_immutable(packet / "binding.json", binding)
        destination, binding_receipt = _bank_packet(
            packet, request["result_binding_destination"],
        )
        if read_json(destination / "bank-receipt.json") != binding_receipt:
            raise ValueError("Durable recovery/result binding receipt differs")
        complete = {
            **binding,
            "binding_destination": str(destination),
            "binding_bank_receipt_sha256": digest_json(binding_receipt),
        }
        _write_json_immutable(
            run / "validation-history-recovery-complete.json", complete,
        )
        _write_json_immutable(result_ack_path, result_ack)
    return verify_complete(request_path)


def verify_complete(request_path):
    _, request = _request(request_path)
    run = Path(request["run"])
    manifest_path = run / "validation-history-recovery.json"
    manifest = read_json(manifest_path)
    result_ack_path = Path(request["result_bank_ack"])
    result_ack = read_json(result_ack_path)
    complete_path = run / "validation-history-recovery-complete.json"
    complete = read_json(complete_path)
    destination = Path(complete.get("binding_destination", ""))
    binding = read_json(destination / "binding.json")
    receipt = read_json(destination / "bank-receipt.json")
    if (complete.get("kind") != "p5-validation-history-result-binding-v1"
            or complete.get("status") != "verified"
            or complete.get("cell") != request["cell"]
            or complete.get("source_event_sha256")
            != request["source_event_sha256"]
            or complete.get("lease_sha256") != request["lease_sha256"]
            or complete.get("recovery_manifest_sha256")
            != sha256(manifest_path)
            or complete.get("recovery_evidence_destination")
            != manifest.get("durable_evidence_destination")
            or complete.get("recovery_bank_receipt_sha256")
            != manifest.get("durable_evidence_bank_receipt_sha256")
            or complete.get("result_destination")
            != result_ack.get("destination")
            or complete.get("result_bank_ack_sha256")
            != sha256(result_ack_path)
            or complete.get("result_bank_receipt_sha256")
            != result_ack.get("bank_receipt_sha256")
            or complete.get("result_seal_sha256")
            != result_ack.get("result_seal_sha256")
            or binding != {
                key: value for key, value in complete.items()
                if key not in {
                    "binding_destination", "binding_bank_receipt_sha256",
                }
            }
            or digest_json(receipt)
            != complete.get("binding_bank_receipt_sha256")
            or receipt != {
                "files": _file_manifest(
                    destination, {"bank-receipt.json"},
                ),
            }):
        raise ValueError("Recovered P5 result lacks its required binding gate")
    with _exclusive_file_lock(request["bank_lock"]):
        from p5_bank import verify_result_snapshot

        evidence_destination = Path(
            complete["recovery_evidence_destination"]
        )
        evidence_receipt = read_json(
            evidence_destination / "bank-receipt.json"
        )
        if (digest_json(evidence_receipt)
                != complete["recovery_bank_receipt_sha256"]
                or evidence_receipt != {
                    "files": _file_manifest(
                        evidence_destination, {"bank-receipt.json"},
                    ),
                }):
            raise ValueError("Durable recovery evidence changed after binding")
        result_destination = Path(complete["result_destination"])
        result_receipt = read_json(
            result_destination / "bank-receipt.json"
        )
        if (digest_json(result_receipt)
                != complete["result_bank_receipt_sha256"]
                or result_receipt != {
                    "files": _file_manifest(
                        result_destination, {"bank-receipt.json"},
                    ),
                }):
            raise ValueError("Durable recovered result receipt changed")
        seal = verify_result_snapshot(
            result_destination, durable=True,
        )
        if (sha256(
                result_destination / "p5-result-seal.json"
                ) != complete["result_seal_sha256"]
                or digest_json(seal["files_sha256"])
                != complete["result_inventory_sha256"]):
            raise ValueError("Bound recovered result changed after completion")
    return complete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=(
            "recover", "verify", "publish",
            "verify-complete",
        ),
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--epoch", type=int)
    args = parser.parse_args()
    if args.mode == "publish":
        result = publish_all(args.request)
    elif args.mode == "verify-complete":
        result = verify_complete(args.request)
    else:
        if args.epoch is None:
            parser.error("recover and verify require --epoch")
        result = (
            recover_epoch(args.request, args.epoch)
            if args.mode == "recover"
            else verify_epoch(args.request, args.epoch)
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
