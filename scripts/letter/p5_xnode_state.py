"""Checkpoint-bound owner fencing and lineage with explicit private execution bindings."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import socket
import struct
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType


_BINDINGS = None
_BINDINGS_LOCK = threading.Lock()
_BINDING_NAMES = frozenset((
    "ROOT", "P5", "XROOT", "DURABLE", "PYTHON", "BINDINGS_PATH", "BINDINGS_SHA256",
    "WRITER_HOST", "WORKER_HOST", "DESTINATIONS", "ALL_CELLS",
))
CHECKPOINT_FILES = ("model.safetensors", "training-state.pt", "p5-seal.json")


def __getattr__(name):
    if name not in _BINDING_NAMES:
        raise AttributeError(name)
    if _BINDINGS is None:
        if name == "DESTINATIONS":
            return MappingProxyType({})
        if name == "ALL_CELLS":
            return frozenset()
        return None
    return _BINDINGS[name]


def _absolute_path(value):
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise ValueError("An explicit absolute path is required")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or path == Path(path.anchor):
        raise ValueError("Relative, traversal or filesystem-root paths are not allowed")
    return path.resolve()


def configure_bindings(path, expected_sha256=None):
    with _BINDINGS_LOCK:
        _configure_bindings(path, expected_sha256)


def _configure_bindings(path, expected_sha256):
    global _BINDINGS

    if not isinstance(path, (str, os.PathLike)) or not str(path).strip() or ".." in Path(path).parts:
        raise ValueError("An explicit private bindings file is required")
    path = Path(path).resolve()
    raw = path.read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
        or checksum != expected_sha256
    ):
        raise ValueError("Private bindings checksum differs from the parent pin")
    if _BINDINGS is not None:
        if path != _BINDINGS["BINDINGS_PATH"] or checksum != _BINDINGS["BINDINGS_SHA256"]:
            raise ValueError("This process is already configured with different bindings")
        return

    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate bindings key")
            value[key] = item
        return value

    value = json.loads(raw, object_pairs_hook=unique)
    fields = {
        "kind", "root", "p5_root", "control_root", "durable_root", "python",
        "writer_host", "worker_host", "cells", "adopted_cell", "learner_sha256",
        "legacy_device_guard", "legacy_transport_guard",
    }
    if not isinstance(value, dict) or set(value) != fields or value["kind"] != "p5-xnode-bindings-v1":
        raise ValueError("Unknown or incomplete private bindings")
    paths = {name: _absolute_path(value[name]) for name in
             ("root", "p5_root", "control_root", "durable_root", "python")}
    if (not path.is_relative_to(paths["root"])
            or any(not paths[name].is_relative_to(paths["root"]) for name in ("p5_root", "control_root"))
            or paths["durable_root"] == paths["root"]
            or any(paths[name].is_relative_to(paths["durable_root"])
                   or paths["durable_root"].is_relative_to(paths[name])
                   for name in ("p5_root", "control_root"))):
        raise ValueError("Bindings paths must stay within their separate declared roots")
    hosts = (value["writer_host"], value["worker_host"])
    if (any(not isinstance(host, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", host)
            for host in hosts) or hosts[0] == hosts[1]):
        raise ValueError("Distinct explicit writer and worker host aliases are required")
    cells = value["cells"]
    if not isinstance(cells, dict) or not cells:
        raise ValueError("Explicit cell assignments are required")
    assignment_fields = {"role", "gpu", "gpu_uuid", "microbatch", "accumulation", "root_step", "ownership_epoch"}
    for cell, assignment in cells.items():
        plain_name(cell)
        if (not isinstance(assignment, dict) or set(assignment) != assignment_fields
                or assignment["role"] not in ("writer", "worker")
                or any(type(assignment[name]) is not int or assignment[name] < minimum for name, minimum in
                       (("gpu", 0), ("microbatch", 1), ("accumulation", 1), ("root_step", 0), ("ownership_epoch", 1)))
                or not isinstance(assignment["gpu_uuid"], str) or not assignment["gpu_uuid"]
                or assignment["microbatch"] * assignment["accumulation"] != 128):
            raise ValueError("Invalid private cell/device/batch assignment")
    adopted = value["adopted_cell"]
    if adopted is not None and (adopted not in cells or cells[adopted]["role"] != "writer"):
        raise ValueError("The adopted cell must be an explicitly retained writer-host cell")
    if (not isinstance(value["learner_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", value["learner_sha256"])
            or any(not isinstance(value[name], str) or not value[name].strip()
                   for name in ("legacy_device_guard", "legacy_transport_guard"))):
        raise ValueError("Exact private learner identity and operational guard patterns are required")
    def freeze(item):
        if isinstance(item, dict):
            return MappingProxyType({key: freeze(child) for key, child in item.items()})
        if isinstance(item, list):
            return tuple(freeze(child) for child in item)
        return item

    _BINDINGS = freeze({
        "document": value, "ROOT": paths["root"], "P5": paths["p5_root"],
        "XROOT": paths["control_root"], "DURABLE": paths["durable_root"], "PYTHON": paths["python"],
        "WRITER_HOST": hosts[0], "WORKER_HOST": hosts[1], "BINDINGS_PATH": path,
        "BINDINGS_SHA256": checksum, "ALL_CELLS": frozenset(cells),
        "DESTINATIONS": {name: item["gpu"] for name, item in cells.items() if item["role"] == "worker"},
    })


def configuration():
    if _BINDINGS is None:
        raise ValueError("Private bindings have not been configured")
    return _BINDINGS["document"]


def require_path(path, writable=False, durable=False):
    configuration()
    resolved = _absolute_path(path)
    under_durable = resolved.is_relative_to(_BINDINGS["DURABLE"])
    if durable:
        allowed = under_durable
    else:
        allowed = resolved.is_relative_to(_BINDINGS["ROOT"]) and not under_durable
    if not allowed:
        raise ValueError("Path escapes its configured local or durable root")
    if writable:
        try:
            unchanged = sha(_BINDINGS["BINDINGS_PATH"]) == _BINDINGS["BINDINGS_SHA256"]
        except OSError:
            unchanged = False
        if not unchanged:
            raise ValueError("Private bindings changed after configuration")
        host = socket.gethostname()
        if host not in (_BINDINGS["WRITER_HOST"], _BINDINGS["WORKER_HOST"]):
            raise ValueError("This host has no configured write authority")
        if under_durable and host != _BINDINGS["WRITER_HOST"]:
            raise ValueError("Only the configured writer may write durable artifacts")
    return resolved


def validate_runtime_paths(value):
    configuration()
    for field, expected in (("root", _BINDINGS["ROOT"]), ("p5_root", _BINDINGS["P5"]),
                            ("control_root", _BINDINGS["XROOT"]),
                            ("durable_root", _BINDINGS["DURABLE"]), ("python", _BINDINGS["PYTHON"])):
        if field in value and _absolute_path(value[field]) != expected:
            raise ValueError("Runtime paths differ from explicit private bindings")
    for field in ("base", "input_manifest", "maintenance_manifest", "maintenance_proof",
                  "crossnode_proof", "ownership_certificate", "ownership_registry"):
        if field in value:
            require_path(value[field])


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8 << 20), b""):
            value.update(data)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate protocol key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)


def sync_directory(path):
    configuration()
    require_path(path, writable=True)
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def write(path, value):
    configuration()
    require_path(path, writable=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    with pending.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)
    sync_directory(path.parent)


@contextmanager
def locked(path, nonblocking=False):
    configuration()
    require_path(path, writable=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            if lock.seek(0, os.SEEK_END) == 0:
                lock.write(b"\0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK if nonblocking else msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


def plain_name(value):
    if (not isinstance(value, str) or not value or Path(value).name != value
            or "/" in value or "\\" in value or value in {".", ".."}):
        raise ValueError("Unsafe protocol artifact name")
    return value


def require_assignment(lease, host, gpu, gpu_uuid=None):
    bindings = configuration()
    cell = lease.get("cell")
    assignment = bindings["cells"].get(cell, {})
    if (lease.get("kind") != "p5-xnode-lease-v1" or cell not in _BINDINGS["DESTINATIONS"]
            or host != _BINDINGS["WORKER_HOST"] or lease.get("owner_node") != host
            or type(gpu) is not int or _BINDINGS["DESTINATIONS"][cell] != gpu or lease.get("gpu") != gpu
            or gpu_uuid is not None and lease.get("gpu_uuid") != gpu_uuid
            or lease.get("gpu_uuid") != assignment.get("gpu_uuid")
            or lease.get("ownership_epoch") != assignment.get("ownership_epoch") or not lease.get("lease_id")
            or lease["root"].get("step") != assignment.get("root_step")
            or lease.get("microbatch") != assignment.get("microbatch")
            or lease.get("accumulation") != assignment.get("accumulation")
            or lease.get("microbatch", 0) * lease.get("accumulation", 0) != 128):
        raise ValueError("Unapproved P5 cross-node assignment or scientific operating batch")
    plain_name(lease["lease_id"])
    return lease


def require_owner(registry, cell, host, lease_id=None):
    if registry.get("kind") != "p5-xnode-ownership-v1" or registry.get("status") != "committed":
        raise ValueError("Ownership is not durably committed")
    owner = registry.get("owners", {}).get(cell)
    if (owner is None or owner["owner_node"] != host
            or lease_id is not None and owner.get("lease_id") != lease_id):
        raise ValueError("Producer is fenced by the committed owner")
    return owner


def root_head(lease):
    return {**lease["root"], "generation": 0, "envelope_sha256": None}


def envelope_head(envelope):
    return {**envelope["checkpoint"], "generation": envelope["generation"],
            "envelope_sha256": digest(envelope)}


def _progress(parent, child):
    for key in ("step", "epoch"):
        if type(child.get(key)) is not int or child[key] < 0:
            raise ValueError("Invalid checkpoint progress")
    p, c = parent["phase"], child.get("phase")
    if child["step"] > parent["step"]:
        valid = p == "train" and c in ("train", "validate") and child["epoch"] == parent["epoch"]
    elif child["step"] == parent["step"]:
        valid = (
            p == "train" and c == "validate" and child["epoch"] == parent["epoch"]
            or p == "validate" and c in ("train", "frozen") and child["epoch"] == parent["epoch"] + 1
            or p == c and child["epoch"] == parent["epoch"]
            and all(child[key] == parent[key] for key in ("model_sha256", "training_state_sha256"))
        )
    else:
        valid = False
    if not valid or c == "frozen" and child["epoch"] != 20:
        raise ValueError("Backward or divergent checkpoint progress")


def validate_envelope(envelope, lease, head, allow_probe=False):
    require_assignment(lease, lease["owner_node"], lease["gpu"])
    if (envelope.get("kind") != "p5-xnode-checkpoint-v1"
            or envelope.get("purpose") not in (("production", "probe") if allow_probe else ("production",))):
        raise ValueError("Only production checkpoints may enter the durable lineage")
    for key in ("cell", "owner_node", "gpu", "lease_id", "ownership_epoch"):
        if envelope.get(key) != lease.get(key):
            raise ValueError("Checkpoint producer owner/lease is fenced")
    if (envelope.get("lease_sha256") != digest(lease)
            or envelope.get("root_seal_sha256") != lease["root"]["seal_sha256"]):
        raise ValueError("Checkpoint lease/root binding changed")
    checkpoint = envelope["checkpoint"]
    plain_name(checkpoint["artifact"])
    if checkpoint.get("contract_sha256") != lease["root"]["contract_sha256"]:
        raise ValueError("Scientific contract changed")
    generation = envelope.get("generation")
    if type(generation) is not int or generation < 1:
        raise ValueError("Invalid monotonic generation")
    if generation == head["generation"]:
        if digest(envelope) == head["envelope_sha256"]:
            return "replay"
        raise ValueError("Same-generation divergent state is forbidden")
    if generation != head["generation"] + 1 or envelope.get("parent_seal_sha256") != head["seal_sha256"]:
        raise ValueError("Broken or skipped checkpoint ancestry")
    _progress(head, checkpoint)
    return "next"


def checkpoint_header(path, contract_sha256=None):
    path = Path(path)
    plain_name(path.name)
    seal = read(path / "p5-seal.json")
    if (seal.get("kind") != "p5-full-state-v1" or seal.get("status") != "sealed"
            or set(seal.get("files_sha256", {})) != {"model.safetensors", "training-state.pt"}
            or contract_sha256 is not None and seal.get("contract_sha256") != contract_sha256):
        raise ValueError("Not a matching sealed native P5 state")
    for name in CHECKPOINT_FILES:
        file = path / name
        if file.is_symlink() or not file.is_file():
            raise ValueError("Checkpoint member is unsafe or missing")
        if name != "p5-seal.json" and sha(file) != seal["files_sha256"][name]:
            raise ValueError("Native checkpoint checksum mismatch")
    return {
        "artifact": path.name, "seal_sha256": sha(path / "p5-seal.json"),
        "contract_sha256": seal["contract_sha256"], "step": seal["global_step"],
        "epoch": seal["epoch"], "phase": seal["phase"],
        "model_sha256": seal["files_sha256"]["model.safetensors"],
        "training_state_sha256": seal["files_sha256"]["training-state.pt"],
    }


def tree_digest(value):
    """Hash every tensor/array bit and scalar/container type, not pickle zip metadata."""
    import numpy as np
    import torch

    result = hashlib.sha256()

    def atom(tag, data):
        result.update(tag.encode() + struct.pack(">Q", len(data)) + data)

    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            atom("torch-type", str(tensor.dtype).encode())
            atom("shape", str(tuple(tensor.shape)).encode())
            atom("tensor", tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            atom("numpy-type", item.dtype.str.encode())
            atom("shape", str(item.shape).encode())
            atom("array", item.tobytes(order="C"))
        elif isinstance(item, np.generic):
            atom("numpy-scalar", item.dtype.str.encode() + item.tobytes())
        elif isinstance(item, dict):
            atom("dict", str(len(item)).encode())
            for key in sorted(item, key=lambda x: (type(x).__name__, repr(x))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            atom(type(item).__name__, str(len(item)).encode())
            for child in item:
                visit(child)
        elif item is None:
            atom("none", b"")
        elif type(item) is bool:
            atom("bool", bytes([item]))
        elif type(item) is int:
            atom("int", str(item).encode())
        elif type(item) is float:
            atom("float64", struct.pack(">d", item))
        elif isinstance(item, str):
            atom("str", item.encode())
        elif isinstance(item, bytes):
            atom("bytes", item)
        else:
            raise TypeError("Unsupported scientific state type: " + type(item).__name__)

    visit(value)
    return result.hexdigest()


def file_manifest(directory, exclude=()):
    directory = Path(directory)
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlinks are not allowed in transfer artifacts")
        if path.is_file():
            name = path.relative_to(directory).as_posix()
            if name not in exclude:
                if ".partial" in path.name:
                    raise ValueError("Incomplete transfer member")
                result[name] = {"sha256": sha(path), "bytes": path.stat().st_size}
    return result


PACKAGE_FILES = {
    "checkpoint/model.safetensors", "checkpoint/training-state.pt", "checkpoint/p5-seal.json",
    "scientific-contract.json", "execution-attestation.json", "envelope.json",
}


def verify_package(path, full=True):
    path = Path(path)
    marker = read(path / "package-seal.json")
    if marker.get("kind") != "p5-xnode-package-v1" or marker.get("status") != "sealed":
        raise ValueError("Missing complete transfer seal")
    files = marker.get("files", {})
    if set(files) != PACKAGE_FILES:
        raise ValueError("Incomplete transfer inventory")
    actual = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()}
    if actual != PACKAGE_FILES | {"package-seal.json"}:
        raise ValueError("Unexpected transfer inventory")
    for name, metadata in files.items():
        file = path / name
        if file.is_symlink():
            raise ValueError("Unsafe transfer member")
        if file.stat().st_size != metadata["bytes"]:
            raise ValueError("Transfer checksum/size mismatch: " + name)
        if full or name not in ("checkpoint/model.safetensors", "checkpoint/training-state.pt"):
            if sha(file) != metadata["sha256"]:
                raise ValueError("Transfer checksum mismatch: " + name)
    envelope = read(path / "envelope.json")
    checkpoint = envelope["checkpoint"]
    seal = read(path / "checkpoint/p5-seal.json")
    if seal.get("kind") != "p5-full-state-v1" or seal.get("status") != "sealed":
        raise ValueError("Transfer does not contain a sealed native checkpoint")
    checks = {
        "seal_sha256": files["checkpoint/p5-seal.json"]["sha256"],
        "model_sha256": files["checkpoint/model.safetensors"]["sha256"],
        "training_state_sha256": files["checkpoint/training-state.pt"]["sha256"],
        "contract_sha256": seal["contract_sha256"],
        "step": seal["global_step"], "epoch": seal["epoch"], "phase": seal["phase"],
    }
    if any(checkpoint[k] != value for k, value in checks.items()):
        raise ValueError("Envelope differs from the native checkpoint seal")
    if (digest(read(path / "scientific-contract.json")) != checkpoint["contract_sha256"]
            or sha(path / "execution-attestation.json") != envelope["execution_attestation_sha256"]):
        raise ValueError("Scientific contract or execution attestation differs")
    return envelope


def recover_published(run, lease, allow_probe=False):
    transport = Path(run) / ".xnode"
    head = root_head(lease)
    for package in sorted((transport / "outbox").glob("g*")):
        if not package.is_dir():
            raise ValueError("Invalid outbox package")
        envelope = verify_package(package, full=False)
        if validate_envelope(envelope, lease, head, allow_probe) != "next":
            raise ValueError("Duplicate publication generation")
        head = envelope_head(envelope)
    current = transport / "head.json"
    if current.exists():
        previous = read(current)
        if previous["generation"] > head["generation"]:
            raise ValueError("Outbox lost a committed generation")
        if previous["generation"] == head["generation"] and previous != head:
            raise ValueError("Divergent publication head")
    return head


def publish_checkpoint(run, checkpoint, lease, purpose="production", selected=None):
    configuration()
    require_path(run, writable=True)
    require_path(checkpoint)
    run, checkpoint = Path(run), Path(checkpoint)
    require_assignment(lease, lease["owner_node"], lease["gpu"])
    if checkpoint.parent.resolve() != (run / "states").resolve():
        raise ValueError("Checkpoint is outside this owned run")
    if sha(run / "p5-run-config.json") != lease["contract_file_sha256"]:
        raise ValueError("Scientific contract file changed")
    header = checkpoint_header(checkpoint, lease["root"]["contract_sha256"])
    transport = run / ".xnode"
    with locked(transport / "publish.lock"):
        head = recover_published(run, lease, allow_probe=purpose == "probe")
        if header["seal_sha256"] == head["seal_sha256"]:
            if head["generation"] == 0:
                return None
            write(transport / "head.json", head)
            return next((transport / "outbox").glob(f"g{head['generation']:08d}-*"))
        envelope = {
            "kind": "p5-xnode-checkpoint-v1", "purpose": purpose,
            **{key: lease[key] for key in ("cell", "owner_node", "gpu", "lease_id", "ownership_epoch")},
            "lease_sha256": digest(lease), "generation": head["generation"] + 1,
            "parent_seal_sha256": head["seal_sha256"], "root_seal_sha256": lease["root"]["seal_sha256"],
            "checkpoint": header, "selected": selected,
            "execution_attestation_sha256": sha(run / "crossnode-attestation.json"),
        }
        validate_envelope(envelope, lease, head, allow_probe=purpose == "probe")
        outbox = transport / "outbox"
        outbox.mkdir(parents=True, exist_ok=True)
        name = f"g{envelope['generation']:08d}-{header['seal_sha256'][:16]}"
        final = outbox / name
        if final.exists():
            raise ValueError("A divergent package occupies this generation")
        pending = transport / ("package.partial-" + uuid.uuid4().hex)
        (pending / "checkpoint").mkdir(parents=True)
        for filename in CHECKPOINT_FILES:
            os.link(checkpoint / filename, pending / "checkpoint" / filename)
        shutil.copyfile(run / "p5-run-config.json", pending / "scientific-contract.json")
        shutil.copyfile(run / "crossnode-attestation.json", pending / "execution-attestation.json")
        write(pending / "envelope.json", envelope)
        write(pending / "package-seal.json", {
            "kind": "p5-xnode-package-v1", "status": "sealed", "files": file_manifest(pending),
        })
        verify_package(pending)
        for file in pending.rglob("*"):
            if file.is_file():
                with file.open("r+b") as stream:
                    os.fsync(stream.fileno())
        sync_directory(pending / "checkpoint")
        sync_directory(pending)
        pending.rename(final)
        sync_directory(outbox)
        write(transport / "head.json", envelope_head(envelope))
        return final


def make_ack(package, lease, receipt, destination, registry_event_sha256):
    configuration()
    envelope = verify_package(package)
    if envelope["purpose"] != "production" or envelope["lease_sha256"] != digest(lease):
        raise ValueError("Cannot acknowledge a probe or foreign producer")
    if receipt != {"files": file_manifest(package)}:
        raise ValueError("Durable receipt differs from the verified complete package")
    return {
        "kind": "p5-xnode-bank-ack-v1", "status": "verified", "writer_host": _BINDINGS["WRITER_HOST"],
        "cell": lease["cell"], "lease_id": lease["lease_id"], "lease_sha256": digest(lease),
        "ownership_epoch": lease["ownership_epoch"], "registry_event_sha256": registry_event_sha256,
        "generation": envelope["generation"], "envelope": envelope,
        "checkpoint": envelope["checkpoint"], "destination": str(destination),
        "bank_receipt": receipt, "bank_receipt_sha256": digest(receipt),
    }


def validate_selected_readmission_ack(ack, lease, checkpoint):
    configuration()
    readmission = ack.get("readmission")
    if (ack.get("kind") != "p5-xnode-selected-readmission-ack-v1"
            or ack.get("status") != "verified"
            or ack.get("writer_host") != _BINDINGS["WRITER_HOST"]
            or ack.get("cell") != lease["cell"]
            or ack.get("lease_id") != lease["lease_id"]
            or ack.get("lease_sha256") != digest(lease)
            or ack.get("ownership_epoch") != lease["ownership_epoch"]
            or ack.get("checkpoint") != checkpoint
            or not isinstance(ack.get("destination"), str)
            or not ack["destination"]
            or digest(ack.get("bank_receipt")) != ack.get("bank_receipt_sha256")
            or not isinstance(readmission, dict)
            or set(readmission) != {
                "kind", "historically_accepted", "wire_generation",
                "root_seal_sha256", "source_event_sha256",
                "selected_lineage_deployment_sha256", "package_seal_sha256",
            }
            or readmission.get("kind") != "p5-selected-lineage-readmission-v1"
            or readmission.get("historically_accepted") is not False
            or readmission.get("wire_generation") != 0
            or readmission.get("root_seal_sha256") != lease["root"]["seal_sha256"]
            or any(
                not isinstance(readmission.get(name), str)
                or not re.fullmatch(r"[0-9a-f]{64}", readmission[name])
                for name in (
                    "source_event_sha256", "selected_lineage_deployment_sha256",
                    "package_seal_sha256",
                )
            )):
        raise ValueError("Selected checkpoint readmission is not exact or explicitly nonhistorical")
    receipt = ack["bank_receipt"]
    files = receipt.get("files", {})
    if set(files) != PACKAGE_FILES | {"package-seal.json"}:
        raise ValueError("Selected checkpoint readmission omits native state files")
    expected = {
        "checkpoint/model.safetensors": checkpoint["model_sha256"],
        "checkpoint/training-state.pt": checkpoint["training_state_sha256"],
        "checkpoint/p5-seal.json": checkpoint["seal_sha256"],
        "package-seal.json": readmission["package_seal_sha256"],
    }
    if any(files[name]["sha256"] != value for name, value in expected.items()):
        raise ValueError("Selected checkpoint readmission contains different state")
    return True


def validate_ack(ack, lease, checkpoint):
    configuration()
    if ack.get("kind") == "p5-xnode-selected-readmission-ack-v1":
        return validate_selected_readmission_ack(ack, lease, checkpoint)
    if (ack.get("kind") != "p5-xnode-bank-ack-v1" or ack.get("status") != "verified"
            or ack.get("writer_host") != _BINDINGS["WRITER_HOST"] or ack.get("cell") != lease["cell"]
            or ack.get("lease_id") != lease["lease_id"] or ack.get("lease_sha256") != digest(lease)
            or ack.get("ownership_epoch") != lease["ownership_epoch"]
            or ack.get("checkpoint") != checkpoint
            or digest(ack.get("bank_receipt")) != ack.get("bank_receipt_sha256")):
        raise ValueError("Acknowledgement is not bound to this owner and exact checkpoint")
    receipt = ack["bank_receipt"]
    files = receipt.get("files", {})
    if set(files) != PACKAGE_FILES | {"package-seal.json"}:
        raise ValueError("Acknowledgement omits native state files")
    expected = {
        "checkpoint/model.safetensors": checkpoint["model_sha256"],
        "checkpoint/training-state.pt": checkpoint["training_state_sha256"],
        "checkpoint/p5-seal.json": checkpoint["seal_sha256"],
    }
    if any(files[name]["sha256"] != value for name, value in expected.items()):
        raise ValueError("Acknowledgement contains different model/optimizer state")
    envelope = ack["envelope"]
    if (envelope["checkpoint"] != checkpoint or envelope["lease_sha256"] != digest(lease)
            or envelope["generation"] != ack["generation"] or envelope["purpose"] != "production"
            or envelope["owner_node"] != _BINDINGS["WORKER_HOST"] or envelope["lease_id"] != lease["lease_id"]):
        raise ValueError("Acknowledgement lineage metadata changed")
    encoded = (json.dumps(envelope, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if (files["envelope.json"]["sha256"] != hashlib.sha256(encoded).hexdigest()
            or envelope["root_seal_sha256"] != lease["root"]["seal_sha256"]):
        raise ValueError("Acknowledgement envelope checksum or root changed")
    return True


def validate_result_binding(snapshot, lease, frozen_envelope, head):
    snapshot = Path(snapshot)
    if (head.get("phase") != "frozen" or head.get("epoch") != 20
            or envelope_head(frozen_envelope) != head
            or validate_envelope(frozen_envelope, lease, head) != "replay"):
        raise ValueError("Result does not name the actual full20-epoch frozen head")
    selected = frozen_envelope.get("selected")
    frozen_checkpoint = frozen_envelope["checkpoint"]
    if (not selected or selected.get("contract_sha256") != lease["root"]["contract_sha256"]
            or frozen_checkpoint["contract_sha256"] != lease["root"]["contract_sha256"]):
        raise ValueError("Frozen selected checkpoint or scientific contract is absent or different")
    frozen = read(snapshot / "frozen.json")
    pointer = read(snapshot / "selected.json")
    latest = read(snapshot / "latest.json")
    if (frozen.get("status") != "frozen-before-test" or frozen.get("epochs") != 20
            or frozen.get("selected") != selected["artifact"]
            or frozen.get("selected_seal_sha256") != selected["seal_sha256"]
            or any(pointer.get(key) != selected[key] for key in ("artifact", "seal_sha256"))
            or any(read(snapshot / name).get("selected_seal_sha256") != selected["seal_sha256"]
                   for name in ("test-result.json", "test-started.json"))):
        raise ValueError("Result selected artifact/seal differs from the accepted frozen selection")
    if any(latest.get(key) != frozen_checkpoint[key] for key in ("artifact", "seal_sha256")):
        raise ValueError("Result latest pointer differs from the actual frozen head")
    contract = snapshot / "p5-run-config.json"
    if (digest(read(contract)) != frozen_checkpoint["contract_sha256"]
            or sha(contract) != lease["contract_file_sha256"]):
        raise ValueError("Result scientific contract differs from the frozen head and lease")
    return True


def local_result_context(snapshot, lease, publication_locked=False):
    from contextlib import nullcontext

    configuration()
    require_path(snapshot)
    snapshot = Path(snapshot)
    run = _BINDINGS["P5"] / "runs" / lease["cell"]
    if snapshot.parent != run / "result-artifacts":
        raise ValueError("Result binding requires this owner's canonical local snapshot")
    with nullcontext() if publication_locked else locked(run / ".xnode/publish.lock"):
        head = recover_published(run, lease)
        if head["phase"] != "frozen" or head["epoch"] != 20:
            raise ValueError("Local publication has no full20-epoch frozen head")
        packages = list((run / ".xnode/outbox").glob(f"g{head['generation']:08d}-*"))
        if len(packages) != 1:
            raise ValueError("Local frozen publication is absent or ambiguous")
        envelope = verify_package(packages[0], full=False)
        latest = read(run / "latest.json")
        if (envelope_head(envelope) != head
                or any(latest.get(key) != head[key] for key in ("artifact", "seal_sha256"))
                or checkpoint_header(run / "states" / head["artifact"], lease["root"]["contract_sha256"])
                != envelope["checkpoint"]):
            raise ValueError("Local frozen native state and publication differ")
        validate_result_binding(snapshot, lease, envelope, head)
        return head, envelope


def validate_result_ack(ack, lease, snapshot, publication_locked=False):
    configuration()
    snapshot = Path(snapshot)
    if (ack.get("kind") != "p5-xnode-result-ack-v1" or ack.get("status") != "verified"
            or ack.get("writer_host") != _BINDINGS["WRITER_HOST"] or ack.get("lease_id") != lease["lease_id"]
            or ack.get("lease_sha256") != digest(lease)
            or ack.get("result_seal_sha256") != sha(snapshot / "p5-result-seal.json")
            or digest(ack.get("bank_receipt")) != ack.get("bank_receipt_sha256")
            or ack["bank_receipt"] != {"files": file_manifest(snapshot)}):
        raise ValueError("A result acknowledgement must bind the complete owner snapshot and writer receipt")
    if ack.get("binding_version") != 2 or not isinstance(ack.get("frozen_checkpoint_ack"), dict):
        raise ValueError("Result acknowledgement has no accepted frozen-head binding")
    head, envelope = local_result_context(snapshot, lease, publication_locked)
    accepted = ack["frozen_checkpoint_ack"]
    validate_ack(accepted, lease, envelope["checkpoint"])
    local_ack = read(_BINDINGS["P5"] / "runs" / lease["cell"] / ".xnode/acks" / (head["artifact"] + ".json"))
    if (ack.get("frozen_head") != head or accepted["envelope"] != envelope or accepted != local_ack):
        raise ValueError("Result acknowledgement differs from the locally accepted frozen head")
    return True
