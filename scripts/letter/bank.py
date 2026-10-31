"""Serialized, verified exports from node-local immutable artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import socket
from pathlib import Path

from protocol import file_sha256, verify_checkpoint, write_json


def require_bankable(source: Path) -> None:
    if source.name.startswith("checkpoint-"):
        verify_checkpoint(source)
    if not source.is_dir():
        raise ValueError(f"Artifact source is not a directory: {source}")
    if (source / "run-config.json").is_file():
        marker = source / "completed.json"
        if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")).get("status") != "complete":
            raise ValueError("Training run is incomplete; bank a sealed checkpoint instead")


def source_manifest(source: Path) -> dict:
    files = {}
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Refusing a symlink in an artifact: {path}")
        if path.is_file():
            if path.name.endswith((".tmp", ".transfer", ".partial")):
                raise ValueError(f"Incomplete artifact file: {path}")
            files[path.relative_to(source).as_posix()] = {
                "sha256": file_sha256(path), "bytes": path.stat().st_size,
            }
    if not files:
        raise ValueError("An empty directory is not a bankable artifact")
    return {"files": files}


def snapshot_checkpoint(source: Path, destination: Path) -> None:
    verify_checkpoint(source)
    if destination.exists():
        raise FileExistsError("Checkpoint snapshots are immutable")
    shutil.copytree(source, destination)
    verify_checkpoint(destination)


def copy_artifact(source: Path, destination: Path) -> dict:
    require_bankable(source)
    receipt = source_manifest(source)
    marker = destination / "bank-receipt.json"
    if destination.exists():
        if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != receipt:
            raise ValueError("Destination is immutable or incomplete; choose a new artifact name")
        for name, metadata in receipt["files"].items():
            if file_sha256(destination / name) != metadata["sha256"]:
                raise ValueError(f"Durable artifact checksum mismatch: {name}")
        return receipt
    destination.mkdir(parents=True)
    for name, metadata in receipt["files"].items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
        if file_sha256(target) != metadata["sha256"]:
            raise ValueError(f"Durable transfer checksum mismatch: {name}")
    if source_manifest(source) != receipt:
        raise ValueError("Source changed during banking; no completion receipt published")
    write_json(marker, receipt)
    return receipt


def main() -> None:
    import fcntl

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument("--writer-host", required=True)
    args = parser.parse_args()
    if socket.gethostname() != args.writer_host:
        raise RuntimeError("Only the explicitly designated node may write to durable storage")
    args.lock_file.parent.mkdir(parents=True, exist_ok=True)
    with args.lock_file.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt = copy_artifact(args.source, args.destination)
        print(json.dumps({"status": "verified", "files": len(receipt["files"])}), flush=True)


if __name__ == "__main__":
    main()
