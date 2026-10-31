"""Snapshot on a worker; transfer and bank only on the designated writer node."""

from __future__ import annotations

import argparse
import errno
import json
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

from bank import copy_artifact, snapshot_checkpoint
from protocol import file_sha256, require_hash, verify_checkpoint, write_json


def verify_artifact(path: Path, kind: str) -> None:
    if kind == "checkpoint":
        verify_checkpoint(path)
        return
    receipt = json.loads((path / "completed.json").read_text(encoding="utf-8"))
    if receipt.get("status") != "complete":
        raise ValueError(f"Invalid {kind} completion receipt")
    if kind == "model":
        require_hash(path / "pytorch_model.bin", receipt["model_sha256"])
        require_hash(path / "run-config.json", receipt["run_config_sha256"])
    elif kind == "evaluation":
        require_hash(path / "predictions.jsonl", receipt["predictions_sha256"])
    else:
        raise ValueError("Unknown artifact kind")


def _inside(path: str | Path, root: Path) -> bool:
    try:
        Path(path).absolute().relative_to(root.absolute())
        return True
    except ValueError:
        return False


def _enoent(reason) -> bool:
    if isinstance(reason, OSError):
        return reason.errno == errno.ENOENT
    text = str(reason)
    return f"[Errno {errno.ENOENT}]" in text and "No such file or directory" in text


def _checkpoint_rotated(error: Exception, source: Path) -> bool:
    if isinstance(error, FileNotFoundError):
        return (
            error.errno == errno.ENOENT
            and error.filename is not None
            and _inside(error.filename, source)
        )
    if not isinstance(error, shutil.Error) or not error.args:
        return False
    problems = error.args[0]
    return (
        isinstance(problems, list)
        and bool(problems)
        and all(
            isinstance(problem, tuple)
            and len(problem) == 3
            and _inside(problem[0], source)
            and _enoent(problem[2])
            for problem in problems
        )
    )


def _quarantine_partial(temporary: Path, identity: str) -> None:
    destination = temporary.parent / f".quarantine-{identity}-{uuid.uuid4().hex}"
    temporary.rename(destination)
    print(
        f"QUARANTINED_PARTIAL {temporary} {destination}",
        file=sys.stderr,
        flush=True,
    )


def immutable_snapshot(source: Path, staging: Path, identity: str, kind: str) -> Path:
    destination = staging / identity
    if destination.exists():
        verify_artifact(destination, kind)
        return destination
    staging.mkdir(parents=True, exist_ok=True)
    temporary = staging / f".partial-{identity}-{uuid.uuid4().hex}"
    try:
        if kind == "checkpoint":
            snapshot_checkpoint(source, temporary)
        else:
            verify_artifact(source, kind)
            shutil.copytree(source, temporary)
            verify_artifact(temporary, kind)
        temporary.rename(destination)
    except Exception as error:
        if temporary.exists():
            try:
                _quarantine_partial(temporary, identity)
            except Exception as quarantine_error:
                error.add_note(f"Failed to quarantine {temporary}: {quarantine_error}")
        raise
    return destination


def snapshot_run(run: Path, staging: Path, evaluation: Path | None, include_completed: bool = True) -> dict:
    result = {"model_complete": False, "evaluation_complete": False, "artifacts": []}
    latest = run / "latest-state.json"
    if latest.is_file():
        state = json.loads(latest.read_text(encoding="utf-8"))
        name = state["checkpoint"]
        if not re.fullmatch(r"checkpoint-\d+", name):
            raise ValueError("Invalid checkpoint name in worker state")
        checkpoints = {name}
        best = state.get("best_model_checkpoint")
        if best:
            path = Path(best)
            if path.parent.resolve() != run.resolve() or not re.fullmatch(r"checkpoint-\d+", path.name):
                raise ValueError("Best checkpoint escapes the owned run directory")
            checkpoints.add(path.name)
        for name in sorted(checkpoints):
            source = run / name
            marker = source / "complete.json"
            if not marker.is_file():
                continue
            try:
                identity = f"{run.name}-{name}-{file_sha256(marker)[:12]}"
                path = immutable_snapshot(source, staging, identity, "checkpoint")
            except (FileNotFoundError, shutil.Error) as error:
                if not _checkpoint_rotated(error, source):
                    raise
                print(
                    f"CHECKPOINT_ROTATED_DURING_SNAPSHOT {run.name}-{name}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            result["artifacts"].append({
                "identity": identity, "kind": "checkpoint", "path": str(path),
                "destination_name": identity,
            })
    if not include_completed:
        return result
    for kind, source, field in (("model", run, "model_complete"),
                                ("evaluation", evaluation, "evaluation_complete")):
        if source is None:
            continue
        marker = source / "completed.json"
        if marker.is_file():
            identity = f"{run.name}-{kind}-{file_sha256(marker)[:12]}"
            path = immutable_snapshot(source, staging, identity, kind)
            result[field] = True
            result["artifacts"].append({
                "identity": identity, "kind": kind, "path": str(path),
                "destination_name": run.name,
            })
    return result


def acknowledge_model(run: Path, ack_root: Path, expected: str) -> None:
    require_hash(run / "completed.json", expected)
    verify_artifact(run, "model")
    write_json(ack_root / f"{run.name}.model.json", {
        "status": "banked", "completed_sha256": expected,
    })


def watch(args) -> None:
    import fcntl

    if socket.gethostname() != args.writer_host or not re.fullmatch(r"node-\d+", args.source_host):
        raise ValueError("Invalid writer or in-job worker host")
    if args.interval < 30:
        raise ValueError("Snapshot interval must be at least 30 seconds")
    banked = set()
    folders = {"checkpoint": "checkpoints", "model": "completed", "evaluation": "evaluations"}
    print(f"REMOTE_BANK_BROKER_READY {args.source_host}", flush=True)
    while True:
        all_done = True
        for run in args.runs:
            command = shlex.join([
                "python3", str(args.source_project / "scripts" / "letter" / "bank_remote.py"),
                "snapshot", "--run", str(run), "--staging", str(args.remote_staging),
                "--evaluation", str(args.evaluations_root / run.name),
            ])
            response = subprocess.run(
                ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", args.source_host, command],
                check=True, capture_output=True, text=True, timeout=180,
            )
            if response.stderr:
                print(response.stderr.strip(), file=sys.stderr, flush=True)
            state = json.loads(response.stdout)
            all_done = all_done and state["model_complete"] and state["evaluation_complete"]
            for artifact in state["artifacts"]:
                identity = artifact["identity"]
                if identity in banked:
                    continue
                remote = Path(artifact["path"])
                expected_name = identity if artifact["kind"] == "checkpoint" else run.name
                if (not re.fullmatch(r"[A-Za-z0-9_.-]+", identity)
                        or not identity.startswith(run.name + "-")
                        or remote.parent != args.remote_staging or remote.name != identity
                        or ".." in remote.parts or not re.fullmatch(r"/[A-Za-z0-9_./-]+", str(remote))
                        or artifact["destination_name"] != expected_name):
                    raise ValueError("Worker snapshot response escapes its declared staging area")
                local = args.local_staging / args.source_host / identity
                local.mkdir(parents=True, exist_ok=True)
                subprocess.run([
                    "rsync", "-a", "--quiet", f"{args.source_host}:{remote}/", f"{local}/",
                ], check=True, timeout=300)
                verify_artifact(local, artifact["kind"])
                destination = args.destination / folders[artifact["kind"]] / artifact["destination_name"]
                with args.lock_file.open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    copy_artifact(local, destination)
                if artifact["kind"] == "model":
                    acknowledgment = shlex.join([
                        "python3", str(args.source_project / "scripts" / "letter" / "bank_remote.py"),
                        "ack", "--run", str(run), "--ack-root", str(args.ack_root),
                        "--completed-sha256", file_sha256(local / "completed.json"),
                    ])
                    subprocess.run(
                        ["ssh", "-n", "-o", "BatchMode=yes", args.source_host, acknowledgment],
                        check=True, timeout=120,
                    )
                banked.add(identity)
                print(f"BANKED_REMOTE {args.source_host} {identity}", flush=True)
        if all_done:
            print("ALL_REMOTE_RUNS_AND_EVALUATIONS_BANKED", flush=True)
            return
        if args.once:
            print("REMOTE_BANK_CYCLE_COMPLETE_WITH_PENDING_WORK", flush=True)
            return
        time.sleep(args.interval)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser("snapshot")
    for name in ("run", "staging", "evaluation"):
        snapshot.add_argument(f"--{name}", type=Path, required=True)
    ack = commands.add_parser("ack")
    ack.add_argument("--run", type=Path, required=True)
    ack.add_argument("--ack-root", type=Path, required=True)
    ack.add_argument("--completed-sha256", required=True)
    watcher = commands.add_parser("watch")
    watcher.add_argument("--source-host", required=True)
    watcher.add_argument("--writer-host", required=True)
    watcher.add_argument("--runs", nargs="+", type=Path, required=True)
    for name in ("source-project", "remote-staging", "evaluations-root",
                 "local-staging", "destination", "lock-file", "ack-root"):
        watcher.add_argument(f"--{name}", type=Path, required=True)
    watcher.add_argument("--interval", type=int, default=120)
    watcher.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.command == "snapshot":
        print(json.dumps(snapshot_run(args.run, args.staging, args.evaluation)))
    elif args.command == "ack":
        acknowledge_model(args.run, args.ack_root, args.completed_sha256)
    else:
        watch(args)


if __name__ == "__main__":
    main()
