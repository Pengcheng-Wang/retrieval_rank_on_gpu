"""Bank complete local checkpoints through one designated node and one lock."""

from __future__ import annotations

import argparse
import socket
import time
from pathlib import Path

from bank import copy_artifact
from bank_remote import snapshot_run
from protocol import write_json


def checkpoint_artifacts(run: Path, staging: Path) -> list[dict]:
    return snapshot_run(run, staging, None, include_completed=False)["artifacts"]


def bank_checkpoint(snapshot: Path, destination: Path, marker: Path, lock_file: Path) -> dict:
    import fcntl

    with lock_file.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt = copy_artifact(snapshot, destination)
        write_json(marker, {
            "status": "verified", "files": len(receipt["files"]), "unix": time.time(),
        })
    return receipt


def main() -> None:
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
            for artifact in checkpoint_artifacts(run, args.staging):
                identity = artifact["identity"]
                if identity in banked:
                    continue
                snapshot = Path(artifact["path"])
                bank_checkpoint(
                    snapshot, args.destination / identity,
                    args.staging / f"{identity}.banked.json", args.lock_file,
                )
                banked.add(identity)
                print(f"BANKED {identity}", flush=True)
        if all((run / "completed.json").is_file() for run in args.runs):
            print("ALL_WATCHED_TRAINING_COMPLETE", flush=True)
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
