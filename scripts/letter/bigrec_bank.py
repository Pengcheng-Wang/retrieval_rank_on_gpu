"""BIGRec-native immutable outboxes and a node-zero-only serialized exporter."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import socket
import subprocess
import time
import uuid
from pathlib import Path

from bank import copy_artifact, source_manifest
from bigrec_checkpoint import verify as verify_checkpoint
from protocol import file_sha256, require_hash, write_json


README = """# Compatible BIGRec reference

Private artifacts for the separately labelled OpenLLaMA-7B-v2 BIGRec reference
on the frozen LETTER leave-two-out splits. This is not an original-LLaMA1
weight reproduction. Independently acquired BIGRec source has no confirmed
redistribution license; source, data, adapters, predictions and embeddings
must remain private. Only aggregate metrics, hashes and method documentation
may be published.

Only the designated node-zero exporter writes here, under the common
experiment bank.lock. Other nodes stage locally; the exporter rsyncs those
immutable payloads to node zero, verifies every file, serializes the durable
copy, and returns a checksum-bound acknowledgment. Objects retain their native
adapter/full-state or embedding formats; they are not T5 checkpoints.

Each objects/<identity>/bank-receipt.json binds all copied files.
Complete training states contain adapter safetensors, optimizer, scheduler,
FP16 scaler, RNG, native sampler permutations, position and early-stop state.
Selected adapters must have an acknowledged durable copy before test
generation. Base weights are redownloadable and are deliberately not copied.
"""


def seal_payload(path: Path, kind: str, binding: dict) -> dict:
    files = {name: meta["sha256"] for name, meta in source_manifest(path)["files"].items()
             if name != "complete.json"}
    if not files:
        raise ValueError("Cannot seal an empty BIGRec artifact")
    receipt = {"kind": kind, "binding": binding, "files": files}
    write_json(path / "complete.json", receipt)
    return receipt


def verify_payload(path: Path) -> dict:
    receipt = json.loads((path / "complete.json").read_text(encoding="utf-8"))
    if receipt["kind"] == "bigrec_complete_adapter_optimizer_scheduler_fp16_rng_sampler_state":
        return verify_checkpoint(path)
    actual = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()
              and p.relative_to(path).as_posix() not in ("complete.json", "bank-receipt.json")}
    if actual != set(receipt["files"]):
        raise ValueError("BIGRec payload checksum file-set mismatch")
    for name, digest in receipt["files"].items():
        target = path / name
        if target.is_symlink() or not target.resolve().is_relative_to(path.resolve()):
            raise ValueError("BIGRec artifact escapes its payload")
        require_hash(target, digest)
    return receipt


def export_payload(source: Path, destination: Path) -> dict:
    verify_payload(source)
    if destination.exists() and not (destination / "bank-receipt.json").exists():
        expected = source_manifest(source)
        actual = {p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()}
        if actual - set(expected["files"]) - {"bank-receipt.json.tmp"}:
            raise ValueError("Incomplete destination contains unrelated files")
        for name, metadata in expected["files"].items():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_file() or file_sha256(target) != metadata["sha256"]:
                shutil.copyfile(source / name, target)
            require_hash(target, metadata["sha256"])
        if source_manifest(source) != expected:
            raise ValueError("Source changed while recovering an interrupted durable export")
        write_json(destination / "bank-receipt.json", expected)
        return expected
    return copy_artifact(source, destination)


def stage_payload(source: Path, outbox: Path, name: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("Unsafe BIGRec artifact name")
    verify_payload(source)
    digest = file_sha256(source / "complete.json")
    identity = f"{name}-{digest[:16]}"
    target = outbox / identity
    if not target.exists():
        outbox.mkdir(parents=True, exist_ok=True)
        pending = outbox / f".partial-{identity}-{uuid.uuid4().hex}"
        shutil.copytree(source, pending / "payload")
        verify_payload(pending / "payload")
        write_json(pending / "ready.json", {
            "identity": identity, "name": name, "complete_sha256": digest,
        })
        pending.rename(target)
    request = json.loads((target / "ready.json").read_text(encoding="utf-8"))
    if request["complete_sha256"] != digest:
        raise ValueError("Immutable staged payload binding differs")
    return {**request, "payload": str(target / "payload")}


def stage_metadata(source: Path, staging: Path, outbox: Path, name: str, kind: str) -> dict:
    from bigrec_data import digest_json

    inventory = source_manifest(source)
    target = staging / f"{name}-{digest_json(inventory)[:16]}"
    if not target.exists():
        pending = staging / f".partial-{target.name}-{uuid.uuid4().hex}"
        shutil.copytree(source, pending)
        seal_payload(pending, kind, {"source_files": inventory})
        pending.rename(target)
    return stage_payload(target, outbox, name)


def pending_requests(outbox: Path) -> list[dict]:
    requests = []
    for path in sorted(outbox.glob("*/ready.json")):
        if (path.parent / "banked.json").exists():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if value["identity"] != path.parent.name or not re.fullmatch(r"[A-Za-z0-9_.-]+", value["identity"]):
            raise ValueError("Invalid outbox request identity")
        requests.append(value)
    def priority(request):
        name = request["name"]
        urgent = "-checkpoint-" in name or name.endswith(("-selected", "-exporter-handshake"))
        return (0 if urgent else 2 if "-catalog-unique-" in name else 1, request["identity"])
    return sorted(requests, key=priority)


def acknowledge(outbox: Path, identity: str, complete_sha256: str, bank_sha256: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", identity):
        raise ValueError("Invalid bank acknowledgment identity")
    request = json.loads((outbox / identity / "ready.json").read_text(encoding="utf-8"))
    if request["complete_sha256"] != complete_sha256:
        raise ValueError("Durable acknowledgment does not bind the staged source")
    require_hash(outbox / identity / "payload" / "complete.json", complete_sha256)
    write_json(outbox / identity / "banked.json", {
        "status": "banked", "identity": identity, "complete_sha256": complete_sha256,
        "bank_receipt_sha256": bank_sha256, "unix": time.time(),
    })


def require_ack(outbox: Path, name: str, source: Path) -> dict:
    digest = file_sha256(source / "complete.json")
    identity = f"{name}-{digest[:16]}"
    ack = json.loads((outbox / identity / "banked.json").read_text(encoding="utf-8"))
    if ack.get("status") != "banked" or ack["complete_sha256"] != digest or ack["identity"] != identity:
        raise ValueError("Selected artifact lacks a hash-bound durable acknowledgment")
    return ack


def remote_command(root: Path, *parts: str) -> list[str]:
    command = shlex.join([
        "python3", str(root / "project/scripts/letter/bigrec_bank.py"), *parts,
    ])
    return ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "node-2", command]


def watch(args) -> None:
    import fcntl

    if socket.gethostname() != args.writer_host:
        raise ValueError("Only the explicitly bound node-zero host can export BIGRec")
    if not args.destination.is_absolute() or args.destination.name != "bigrec-compatible":
        raise ValueError("Use the explicitly approved BIGRec durable destination")
    outbox = args.root / "bigrec/outbox"
    lock_path = args.root / "bank.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        args.destination.mkdir(parents=True, exist_ok=True)
        readme = args.destination / "README.md"
        if readme.exists() and readme.read_text(encoding="utf-8") != README:
            raise ValueError("Existing durable BIGRec README differs")
        if not readme.exists():
            readme.write_text(README, encoding="utf-8")
    print("BIGREC_BANKER_READY node0_single_writer_common_lock", flush=True)
    while True:
        cycle = {"unix": time.time(), "status": "alive", "banked": 0, "errors": []}
        backlog = False
        for host in ("node-0", "node-2"):
            try:
                if host == "node-0":
                    requests = pending_requests(outbox)
                else:
                    response = subprocess.run(
                        remote_command(args.root, "list", "--outbox", str(outbox)),
                        check=True, capture_output=True, text=True, timeout=120,
                    )
                    requests = json.loads(response.stdout)
                cycle[f"pending_{host}"] = len(requests)
                backlog = backlog or len(requests) > 8
                for request in requests[:8]:
                    identity = request["identity"]
                    if not re.fullmatch(r"[A-Za-z0-9_.-]+", identity):
                        raise ValueError("Remote outbox identity is unsafe")
                    local = outbox / identity
                    if host == "node-2":
                        local = args.root / "bigrec/incoming/node-2" / identity
                        local.mkdir(parents=True, exist_ok=True)
                        subprocess.run([
                            "rsync", "-a", "--quiet", "--exclude=banked.json",
                            f"node-2:{outbox / identity}/", f"{local}/",
                        ], check=True, timeout=600)
                    payload = local / "payload"
                    require_hash(payload / "complete.json", request["complete_sha256"])
                    verify_payload(payload)
                    destination = args.destination / "objects" / identity
                    cycle.update(status="waiting_for_common_lock", current=f"{host}:{identity}", unix=time.time())
                    write_json(args.root / "bigrec/banker-status.json", cycle)
                    with lock_path.open("a") as lock:
                        fcntl.flock(lock, fcntl.LOCK_EX)
                        export_payload(payload, destination)
                        bank_digest = file_sha256(destination / "bank-receipt.json")
                    if host == "node-0":
                        acknowledge(outbox, identity, request["complete_sha256"], bank_digest)
                    else:
                        subprocess.run(remote_command(
                            args.root, "ack", "--outbox", str(outbox), "--identity", identity,
                            "--complete-sha256", request["complete_sha256"], "--bank-sha256", bank_digest,
                        ), check=True, timeout=120)
                    cycle["banked"] += 1
                    cycle.update(status="exporting", unix=time.time())
                    write_json(args.root / "bigrec/banker-status.json", cycle)
                    print(f"BIGREC_BANKED {host} {identity}", flush=True)
            except Exception as error:
                cycle["errors"].append({"host": host, "error": str(error)})
                print(f"BIGREC_BANK_RETRY {host} {type(error).__name__}: {error}", flush=True)
        cycle.update(status="retrying" if cycle["errors"] else "alive", unix=time.time())
        write_json(args.root / "bigrec/banker-status.json", cycle)
        if args.once:
            if cycle["errors"]:
                raise RuntimeError("BIGRec bank cycle had errors")
            return
        time.sleep(1 if backlog and not cycle["errors"] else args.interval)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list")
    listing.add_argument("--outbox", type=Path, required=True)
    ack = commands.add_parser("ack")
    ack.add_argument("--outbox", type=Path, required=True)
    ack.add_argument("--identity", required=True)
    ack.add_argument("--complete-sha256", required=True)
    ack.add_argument("--bank-sha256", required=True)
    stage = commands.add_parser("stage")
    stage.add_argument("--source", type=Path, required=True)
    stage.add_argument("--outbox", type=Path, required=True)
    stage.add_argument("--name", required=True)
    stage.add_argument("--metadata-kind")
    watcher = commands.add_parser("watch")
    watcher.add_argument("--root", type=Path, required=True)
    watcher.add_argument("--destination", type=Path, required=True)
    watcher.add_argument("--writer-host", required=True)
    watcher.add_argument("--interval", type=int, default=45)
    watcher.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.command == "list":
        print(json.dumps(pending_requests(args.outbox)))
    elif args.command == "ack":
        acknowledge(args.outbox, args.identity, args.complete_sha256, args.bank_sha256)
    elif args.command == "stage":
        if args.metadata_kind:
            result = stage_metadata(args.source, args.outbox.parent / "metadata-staging",
                                    args.outbox, args.name, args.metadata_kind)
        else:
            result = stage_payload(args.source, args.outbox, args.name)
        print(json.dumps(result))
    else:
        watch(args)


if __name__ == "__main__":
    main()
