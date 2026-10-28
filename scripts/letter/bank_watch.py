"""Bank complete local checkpoints through one designated node and one lock."""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

from bank import copy_artifact, snapshot_checkpoint
from protocol import file_sha256, verify_checkpoint, write_json


def main() -> None:
    import fcntl

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument("--writer-host", required=True)
    parser.add_argument("--interval", type=int, default=120)
    args = parser.parse_args()
    if socket.gethostname() != args.writer_host or args.interval < 30:
        raise ValueError("Invalid designated writer or polling interval")
    args.staging.mkdir(parents=True, exist_ok=True)
    print("SINGLE_WRITER_EXPORTER_READY", flush=True)
    banked = set()
    while True:
        for run in args.runs:
            latest = run / "latest-state.json"
            if not latest.is_file():
                continue
            state = json.loads(latest.read_text(encoding="utf-8"))
            name = state["checkpoint"]
            if Path(name).name != name or not name.startswith("checkpoint-"):
                raise ValueError("Unsafe checkpoint name in local run state")
            checkpoint = run / name
            marker = checkpoint / "complete.json"
            if not marker.is_file():
                print(f"CHECKPOINT_ROTATED_WAITING {run.name} {name}", flush=True)
                continue
            identity = f"{run.name}-{name}-{file_sha256(marker)[:12]}"
            if identity in banked:
                continue
            snapshot = args.staging / identity
            if not snapshot.exists():
                try:
                    snapshot_checkpoint(checkpoint, snapshot)
                except FileNotFoundError:
                    print(f"CHECKPOINT_ROTATED_DURING_SNAPSHOT {identity}", flush=True)
                    continue
            verify_checkpoint(snapshot)
            with args.lock_file.open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                receipt = copy_artifact(snapshot, args.destination / identity)
            write_json(args.staging / f"{identity}.banked.json", {
                "status": "verified", "files": len(receipt["files"]), "unix": time.time(),
            })
            banked.add(identity)
            print(f"BANKED {identity}", flush=True)
        if all((run / "completed.json").is_file() for run in args.runs):
            print("ALL_WATCHED_TRAINING_COMPLETE", flush=True)
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
