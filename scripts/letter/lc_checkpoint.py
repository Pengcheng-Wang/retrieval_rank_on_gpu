"""Complete LC adapter checkpoints and a node-zero-only immutable bank broker."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path

from protocol import file_sha256, require_hash, write_json


def inventory(directory):
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("LC checkpoints cannot contain symlinks")
        if path.is_file() and path.name not in {"complete.json", "bank-receipt.json"}:
            if path.name.endswith((".partial", ".transfer", ".tmp")):
                raise ValueError("Partial checkpoint file")
            files[path.relative_to(directory).as_posix()] = file_sha256(path)
    return files


def required_files(directory, world_size):
    if world_size < 1:
        raise ValueError("Invalid checkpoint world size")
    names = {p.name for p in directory.iterdir() if p.is_file()}
    required = {
        "adapter_config.json", "optimizer.pt", "scheduler.pt", "trainer_state.json",
        "callback-state.json", "run-config.json",
        *{f"rng_state_{rank}.pth" for rank in range(world_size)},
        *{f"rank-state-{rank}.pt" for rank in range(world_size)},
    }
    adapters = names & {"adapter_model.bin", "adapter_model.safetensors"}
    if len(adapters) != 1 or not required <= names:
        raise ValueError(f"Incomplete LC checkpoint, including every rank/RNG: {required - names}")
    if names & {"pytorch_model.bin", "model.safetensors"}:
        raise ValueError("LC checkpoint must contain an adapter, not a disguised base model")
    if {n for n in names if n.startswith("rng_state_")} != {
        f"rng_state_{rank}.pth" for rank in range(world_size)
    }:
        raise ValueError("Checkpoint RNG rank inventory mismatch")
    return next(iter(adapters))


def seal(directory, world_size):
    adapter = required_files(directory, world_size)
    marker = directory / "complete.json"
    if marker.exists():
        raise FileExistsError("LC checkpoint seals are immutable")
    state = json.loads((directory / "trainer_state.json").read_text())
    receipt = {
        "schema_version": 1, "kind": "lc-complete-adapter-continuation",
        "global_step": int(state["global_step"]), "world_size": world_size,
        "adapter_file": adapter, "files": inventory(directory),
    }
    write_json(marker, receipt)
    return receipt


def verify(directory):
    marker = directory / "complete.json"
    if not marker.is_file():
        raise ValueError("LC checkpoint has no complete seal")
    receipt = json.loads(marker.read_text())
    if receipt.get("kind") != "lc-complete-adapter-continuation":
        raise ValueError("Not a complete LC checkpoint")
    adapter = required_files(directory, receipt["world_size"])
    if adapter != receipt["adapter_file"] or inventory(directory) != receipt["files"]:
        raise ValueError("LC checkpoint checksum or file inventory mismatch")
    state = json.loads((directory / "trainer_state.json").read_text())
    if state["global_step"] != receipt["global_step"]:
        raise ValueError("LC checkpoint step mismatch")
    return receipt


def snapshot(source, destination):
    if destination.exists():
        raise FileExistsError("LC checkpoint snapshots are immutable")
    expected = verify(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    building = destination.with_name(destination.name + f".building-{os.getpid()}")
    shutil.copytree(source, building)
    if verify(building) != expected or verify(source) != expected:
        raise ValueError("Checkpoint changed during independent snapshot")
    building.rename(destination)


def snapshot_identity(directory):
    receipt = verify(directory)
    return {
        "global_step": receipt["global_step"],
        "run_config_sha256": receipt["files"]["run-config.json"],
        "adapter_sha256": receipt["files"][receipt["adapter_file"]],
        "complete_sha256": file_sha256(directory / "complete.json"),
    }


def verify_ack(directory, acknowledgment):
    expected = snapshot_identity(directory)
    if (acknowledgment.get("status") != "verified"
            or any(acknowledgment.get(key) != value for key, value in expected.items())):
        raise ValueError("Durable acknowledgment is not bound to this complete LC snapshot")


def wait_for_ack(directory, acknowledgments, timeout=900):
    target = acknowledgments / (directory.name + ".json")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if target.is_file():
            value = json.loads(target.read_text())
            verify_ack(directory, value)
            return value
        time.sleep(5)
    raise TimeoutError(f"No hash-bound durable acknowledgment for {directory.name}")


def assert_identical(left, right, path="state"):
    import numpy as np
    import torch

    if isinstance(left, torch.Tensor):
        if (not isinstance(right, torch.Tensor) or left.dtype != right.dtype
                or left.shape != right.shape or not torch.equal(left.cpu(), right.cpu())):
            raise ValueError(f"Non-bitwise continuation: {path}")
    elif isinstance(left, np.ndarray):
        if (not isinstance(right, np.ndarray) or left.dtype != right.dtype
                or left.shape != right.shape or not np.array_equal(left, right)):
            raise ValueError(f"Non-bitwise continuation: {path}")
    elif isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise ValueError(f"Different continuation keys: {path}")
        for key in left:
            assert_identical(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (tuple, list)):
        if type(left) is not type(right) or len(left) != len(right):
            raise ValueError(f"Different continuation sequence: {path}")
        for index, (a, b) in enumerate(zip(left, right)):
            assert_identical(a, b, f"{path}[{index}]")
    elif type(left) is not type(right) or left != right:
        raise ValueError(f"Different continuation value: {path}")


def compare(left, right):
    import torch
    from safetensors.torch import load_file

    a, b = verify(left), verify(right)
    if a["world_size"] != b["world_size"] or a["global_step"] != b["global_step"]:
        raise ValueError("Continuation endpoints differ")
    names = ["optimizer.pt", "scheduler.pt"] + [
        name for rank in range(a["world_size"])
        for name in (f"rng_state_{rank}.pth", f"rank-state-{rank}.pt")
    ]
    for name in names:
        assert_identical(torch.load(left / name, map_location="cpu"),
                         torch.load(right / name, map_location="cpu"), name)
    for directory, receipt in ((left, a), (right, b)):
        path = directory / receipt["adapter_file"]
        weights = load_file(str(path)) if path.suffix == ".safetensors" else torch.load(
            path, map_location="cpu",
        )
        if directory == left:
            expected = weights
        else:
            assert_identical(expected, weights, "adapter")
    assert_identical(
        json.loads((left / "callback-state.json").read_text()),
        json.loads((right / "callback-state.json").read_text()), "callbacks",
    )
    states = [json.loads((p / "trainer_state.json").read_text()) for p in (left, right)]
    for name in ("global_step", "epoch", "max_steps", "num_train_epochs", "best_metric"):
        assert_identical(states[0][name], states[1][name], f"trainer.{name}")
    if Path(states[0]["best_model_checkpoint"]).name != Path(states[1]["best_model_checkpoint"]).name:
        raise ValueError("Validation-selected checkpoint differs")
    return {"status": "pass", "comparison": "bitwise; no numerical tolerance",
            "global_step": a["global_step"], "world_size": a["world_size"],
            "left": snapshot_identity(left), "right": snapshot_identity(right)}


def bank_one(source, destination, lock_file, writer_host):
    import fcntl

    if socket.gethostname() != writer_host:
        raise RuntimeError("Only the explicitly designated node-zero writer may bank LC state")
    identity = snapshot_identity(source)
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with lock_file.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            marker = destination / "bank-receipt.json"
            if not marker.is_file():
                raise ValueError("Incomplete bank destination; preserve it for recovery")
            ack = json.loads(marker.read_text())
            verify_ack(source, ack)
            verify_ack(destination, ack)
            return ack
        destination.parent.mkdir(parents=True, exist_ok=True)
        building = destination.with_name(destination.name + f".building-{os.getpid()}")
        shutil.copytree(source, building)
        if snapshot_identity(building) != identity or snapshot_identity(source) != identity:
            raise ValueError("LC bank transfer failed checksum verification")
        ack = {**identity, "status": "verified", "writer_host": writer_host,
               "bank_directory": str(destination), "verified_unix": time.time(),
               "bank_helper_sha256": file_sha256(Path(__file__))}
        write_json(building / "bank-receipt.json", ack)
        building.rename(destination)
    return ack


def ssh(host, command):
    return subprocess.check_output(
        ["ssh", "-n", "-o", "BatchMode=yes", host, command], text=True,
    ).strip()


def remote_json(host, path, value):
    encoded = base64.b64encode((json.dumps(value) + "\n").encode()).decode()
    ssh(host, f"mkdir -p '{path.parent}'; printf '%s' '{encoded}' | base64 -d > "
        f"'{path}.transfer'; mv '{path}.transfer' '{path}'")


def published_snapshot_names(found):
    names = {Path(path).parent.name for path in found.splitlines() if path}
    return sorted(
        (name for name in names if re.fullmatch(r"checkpoint-[0-9]+", name)),
        key=lambda name: int(name.split("-")[-1]),
    )


def broker(args):
    if socket.gethostname() != args.writer_host or args.source_host != "node-2":
        raise RuntimeError("Broker must run on the designated node zero and pull only node two")
    if not re.fullmatch(r"/scratch/[A-Za-z0-9_./-]+", str(args.source)) or ".." in args.source.parts:
        raise ValueError("Broker source must be a node-local scratch path")
    args.inbox.mkdir(parents=True, exist_ok=True)
    while True:
        remote_json(args.source_host, args.source.parent / "exporter-heartbeat.json", {
            "writer_host": args.writer_host, "pid": os.getpid(), "unix": time.time(),
            "destination": str(args.destination), "source": str(args.source),
        })
        found = ssh(args.source_host, f"find '{args.source}' -mindepth 2 -maxdepth 2 "
                    "-type f -name complete.json")
        names = published_snapshot_names(found)
        for name in names:
            if not re.fullmatch(r"checkpoint-[0-9]+", name):
                raise ValueError("Unexpected LC snapshot name")
            local = args.inbox / name
            sent = args.inbox / (name + ".ack.json")
            if sent.is_file():
                continue
            if not local.exists():
                building = local.with_name(local.name + ".incoming")
                subprocess.run([
                    "rsync", "-a", "--", f"{args.source_host}:{args.source / name}/",
                    str(building) + "/",
                ], check=True)
                verify(building)
                building.rename(local)
            acknowledgment = bank_one(local, args.destination / name, args.lock_file, args.writer_host)
            remote_json(args.source_host, args.source.parent / "acks" / (name + ".json"), acknowledgment)
            write_json(sent, acknowledgment)
            print(json.dumps({"event": "banked", "snapshot": name, **acknowledgment}), flush=True)
        if args.once:
            return
        time.sleep(args.interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("verify")
    check.add_argument("directory", type=Path)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--left", type=Path, required=True)
    comparison.add_argument("--right", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    export = commands.add_parser("broker")
    for name in ("source", "inbox", "destination", "lock-file"):
        export.add_argument(f"--{name}", type=Path, required=True)
    export.add_argument("--source-host", required=True)
    export.add_argument("--writer-host", required=True)
    export.add_argument("--interval", type=int, default=15)
    export.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.command == "verify":
        print(json.dumps(snapshot_identity(args.directory)), flush=True)
    elif args.command == "compare":
        result = compare(args.left, args.right)
        write_json(args.output, result)
        print(json.dumps(result), flush=True)
    else:
        broker(args)


if __name__ == "__main__":
    main()
