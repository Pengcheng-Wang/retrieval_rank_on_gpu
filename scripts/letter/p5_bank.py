"""One node0 P5 checkpoint watcher, sharing the experiment's common bank lock."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import time
import uuid
from pathlib import Path

from bank import copy_artifact, source_manifest
from p5_checkpoint import _recover_selection_locked
from p5_checkpoint import fsync_directory, pointer_lock, pointer_path, recover_selection, verify_checkpoint
from p5_source import digest_json, read_json, require_hash, sha256, write_json


RESULT_FILES = (
    "test-result.json", "test-predictions.jsonl", "completed.json", "test-started.json",
    "frozen.json", "p5-run-config.json", "validation-history.json", "selected.json",
    "latest.json", "updates.jsonl",
)


def _result_inventory(path):
    path = Path(path)
    frozen = read_json(path / "frozen.json")
    if frozen.get("status") != "frozen-before-test" or frozen.get("epochs") != 20:
        raise ValueError("A result requires the unchanged complete 20-epoch frozen budget")
    names = list(RESULT_FILES) + [f"validation/epoch-{i:02d}.jsonl" for i in range(1, 21)]
    if (path / "maintenance-attestation.json").exists():
        names.append("maintenance-attestation.json")
    for name in names:
        file = path / name
        if not file.is_file() or file.is_symlink() or any(
            (path / parent).is_symlink() for parent in Path(name).parents
        ):
            raise ValueError(f"Mandatory result artifact is missing or unsafe: {name}")
    return {name: sha256(path / name) for name in names}


def _prediction_users(path, expected_count):
    users = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            user = row.get("original_user")
            predictions, scores = row.get("predictions"), row.get("scores")
            if (not isinstance(user, str) or user in users
                    or not isinstance(predictions, list) or len(predictions) != 20
                    or not all(isinstance(p, str) and p for p in predictions)
                    or not isinstance(scores, list) or len(scores) != 20
                    or not all(isinstance(s, (float, int)) and math.isfinite(s) for s in scores)
                    or type(row.get("original_target_item")) is not int
                    or not isinstance(row.get("target"), str)):
                raise ValueError("Incomplete, duplicated or malformed prediction coverage")
            users.add(user)
    if len(users) != expected_count:
        raise ValueError("Prediction rows do not cover every declared user")
    return users


def _validate_result_contents(path, inventory, expected_result=None):
    result = read_json(path / "test-result.json")
    complete = read_json(path / "completed.json")
    frozen = read_json(path / "frozen.json")
    started = read_json(path / "test-started.json")
    selected = read_json(path / "selected.json")
    history = read_json(path / "validation-history.json")
    users = result.get("users")
    if (result.get("status") != "complete" or result.get("split") != "test"
            or type(users) is not int or users <= 0 or result.get("expected_users") != users
            or result.get("beams_per_user") != 20 or result.get("seen_item_exclusion") is not False
            or complete != {"status": "complete", "test_evaluations": 1}
            or started.get("started_after_bank_ack") is not True):
        raise ValueError("No genuine completed full-user P5 result is available")
    if expected_result is not None and result != expected_result:
        raise ValueError("Snapshot names a different result")
    selected_sha = result.get("selected_seal_sha256")
    if (not isinstance(selected_sha, str) or len(selected_sha) != 64
            or started.get("selected_seal_sha256") != selected_sha
            or frozen.get("selected_seal_sha256") != selected_sha
            or selected.get("seal_sha256") != selected_sha
            or frozen.get("selected") != selected.get("artifact")
            or result.get("predictions_sha256") != inventory["test-predictions.jsonl"]):
        raise ValueError("Result, frozen model, selection or predictions are not consistently bound")
    if not isinstance(history, list) or len(history) != frozen["epochs"]:
        raise ValueError("Incomplete validation history for the frozen generation")
    expected_users = _prediction_users(path / "test-predictions.jsonl", users)
    for epoch, metrics in enumerate(history, 1):
        name = f"validation/epoch-{epoch:02d}.jsonl"
        if (metrics.get("split") != "valid" or metrics.get("users") != users
                or metrics.get("predictions_sha256") != inventory[name]
                or _prediction_users(path / name, users) != expected_users):
            raise ValueError("Validation prediction/history coverage differs from the frozen result")
    read_json(path / "p5-run-config.json")
    read_json(path / "latest.json")
    return result


def verify_result_snapshot(snapshot, expected_result=None, durable=False):
    snapshot = Path(snapshot)
    seal = read_json(snapshot / "p5-result-seal.json")
    if seal.get("kind") != "p5-reference-results-v1" or seal.get("status") != "sealed":
        raise ValueError("Missing or malformed P5 result seal")
    inventory = _result_inventory(snapshot)
    if seal.get("files_sha256") != inventory:
        raise ValueError("Result seal has incomplete inventory or mismatched hashes")
    excluded = {"p5-result-seal.json"} | ({"bank-receipt.json"} if durable else set())
    actual = {p.relative_to(snapshot).as_posix() for p in snapshot.rglob("*") if p.is_file()}
    if actual != set(inventory) | excluded:
        raise ValueError("Unexpected or incomplete files in a P5 result snapshot")
    _validate_result_contents(snapshot, inventory, expected_result)
    return seal


def _quarantine(path, reason):
    target = path.parent / "quarantine" / (path.name + "-" + uuid.uuid4().hex)
    target.parent.mkdir(exist_ok=True)
    path.rename(target)
    fsync_directory(target.parent)
    fsync_directory(path.parent)
    write_json(target.parent / (target.name + ".receipt.json"), {
        "reason": reason, "original_path": str(path), "preserved_path": str(target),
    })
    return target


def _build_result_snapshot_locked(run):
    inventory = _result_inventory(run)
    result = _validate_result_contents(run, inventory)
    name = "p5-results-" + digest_json(result)[:16]
    parent = run / "result-artifacts"
    parent.mkdir(exist_ok=True)
    snapshot = parent / name
    for abandoned in list(parent.glob(name + ".partial-*")):
        _quarantine(abandoned, "Unpublished partial result attempt")
    if snapshot.exists():
        try:
            verify_result_snapshot(snapshot, result)
            if read_json(snapshot / "p5-result-seal.json")["files_sha256"] != inventory:
                raise ValueError("Snapshot differs from the frozen source generation")
            return snapshot
        except (ValueError, KeyError, OSError):
            _quarantine(snapshot, "Incomplete or inconsistent existing result snapshot")
    partial = parent / (name + ".partial-" + uuid.uuid4().hex)
    partial.mkdir()
    for filename, digest in inventory.items():
        destination = partial / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run / filename, destination)
        require_hash(destination, digest)
        with destination.open("r+b") as stream:
            os.fsync(stream.fileno())
    if _result_inventory(run) != inventory:
        raise ValueError("Result inputs changed during snapshot construction")
    write_json(partial / "p5-result-seal.json", {
        "kind": "p5-reference-results-v1", "status": "sealed", "files_sha256": inventory,
    })
    verify_result_snapshot(partial, result)
    fsync_directory(partial / "validation")
    fsync_directory(partial)
    if snapshot.exists():
        raise FileExistsError("A competing result snapshot must not be overwritten")
    partial.rename(snapshot)
    fsync_directory(parent)
    verify_result_snapshot(snapshot, result)
    return snapshot


def bank_checkpoint(run, checkpoint, destination, lock_file, writer_host):
    import fcntl

    if socket.gethostname() != writer_host:
        raise RuntimeError("Only the declared node0 writer may bank P5 artifacts")
    seal = verify_checkpoint(checkpoint)
    with Path(lock_file).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        destination = Path(destination)
        if destination.exists() and not (destination / "bank-receipt.json").exists():
            destination = destination.with_name(destination.name + "-retry-" + uuid.uuid4().hex[:12])
        receipt = copy_artifact(Path(checkpoint), destination)
    acknowledgement = {
        "status": "verified", "kind": "p5-full-state-v1", "artifact": Path(checkpoint).name,
        "seal_sha256": sha256(Path(checkpoint) / "p5-seal.json"),
        "contract_sha256": seal["contract_sha256"], "destination": str(destination),
        "bank_receipt_sha256": digest_json(receipt), "writer_host": writer_host,
    }
    write_json(Path(run) / "bank-acks" / (Path(checkpoint).name + ".json"), acknowledgement)
    return acknowledgement


def verify_ack(run, checkpoint):
    acknowledgement = read_json(Path(run) / "bank-acks" / (Path(checkpoint).name + ".json"))
    if (acknowledgement["status"] != "verified" or acknowledgement["artifact"] != Path(checkpoint).name
            or acknowledgement["seal_sha256"] != sha256(Path(checkpoint) / "p5-seal.json")):
        raise ValueError("P5 bank acknowledgement is stale or names different weights/state")
    destination = Path(acknowledgement["destination"])
    receipt = read_json(destination / "bank-receipt.json")
    if digest_json(receipt) != acknowledgement["bank_receipt_sha256"]:
        raise ValueError("Durable P5 receipt differs from the node0 acknowledgement")
    local_seal = read_json(Path(checkpoint) / "p5-seal.json")
    for name, digest in local_seal["files_sha256"].items():
        if receipt["files"][name]["sha256"] != digest:
            raise ValueError("Durable receipt names a different P5 model/optimizer state")
    require_hash(destination / "p5-seal.json", acknowledgement["seal_sha256"])
    return acknowledgement


def wait_for_bank(run, checkpoint, timeout_seconds):
    start = time.monotonic()
    while not (Path(run) / "bank-acks" / (Path(checkpoint).name + ".json")).exists():
        if time.monotonic() - start >= timeout_seconds:
            raise TimeoutError("Selected/latest P5 checkpoint has no durable bank acknowledgement; test blocked")
        time.sleep(5)
    return verify_ack(run, checkpoint)


def bank_results(run, destination, lock_file):
    import fcntl

    run, destination = Path(run), Path(destination)
    # Lock ordering is per-run metadata first, then the common durable-writer lock.
    # Checkpoint banking never holds the common lock while acquiring this run lock.
    with pointer_lock(run):
        if (run / "latest.json").exists() and (run / "states").exists():
            _recover_selection_locked(run)
        snapshot = _build_result_snapshot_locked(run)
        target = destination / snapshot.name
        result = read_json(snapshot / "test-result.json")
        with Path(lock_file).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if target.exists():
                try:
                    verify_result_snapshot(target, result, durable=True)
                    if read_json(target / "bank-receipt.json") != source_manifest(snapshot):
                        raise ValueError("A different immutable result generation occupies this name")
                except (ValueError, KeyError, OSError):
                    target = target.with_name(target.name + "-retry-" + uuid.uuid4().hex[:12])
            receipt = copy_artifact(snapshot, target)
            seal = verify_result_snapshot(target, result, durable=True)
            expected_files = set(seal["files_sha256"]) | {"p5-result-seal.json"}
            if (read_json(target / "bank-receipt.json") != receipt
                    or set(receipt["files"]) != expected_files):
                raise ValueError("Durable result receipt omits mandatory predictions or seal")
            for name in expected_files:
                if receipt["files"][name]["sha256"] != sha256(target / name):
                    raise ValueError("Durable result receipt checksum differs")
        acknowledgement = {
            "status": "verified", "destination": str(target),
            "bank_receipt_sha256": digest_json(receipt),
            "result_seal_sha256": sha256(target / "p5-result-seal.json"),
            "test_predictions_sha256": seal["files_sha256"]["test-predictions.jsonl"],
        }
        ack_path = run / "results-banked.json"
        if ack_path.exists():
            try:
                unchanged = read_json(ack_path) == acknowledgement
            except ValueError:
                unchanged = False
            if not unchanged:
                preserved = run / "result-artifacts" / "quarantine" / (
                    uuid.uuid4().hex + "-results-banked.json"
                )
                preserved.parent.mkdir(exist_ok=True)
                shutil.copyfile(ack_path, preserved)
                with preserved.open("r+b") as stream:
                    os.fsync(stream.fileno())
                fsync_directory(preserved.parent)
        write_json(ack_path, acknowledgement)
        fsync_directory(run)


def watch(args):
    if socket.gethostname() != args.writer_host:
        raise RuntimeError("Wrong P5 durable writer host")
    args.runs.mkdir(parents=True, exist_ok=True)
    verified_results = {}
    while True:
        for run in sorted(args.runs.iterdir()):
            if not run.is_dir() or run.is_symlink():
                continue
            write_json(run / "bank-watcher-ready.json", {
                "status": "ready", "pid": os.getpid(), "host": socket.gethostname(),
                "common_lock_file": str(args.lock_file), "heartbeat_unix": time.time(),
            })
            if not (run / "latest.json").exists():
                continue
            try:
                recover_selection(run)
                artifacts = [pointer_path(run)]
                if (run / "selected.json").exists():
                    artifacts.append(pointer_path(run, "selected"))
                for artifact in dict.fromkeys(artifacts):
                    ack = run / "bank-acks" / (artifact.name + ".json")
                    if not ack.exists():
                        bank_checkpoint(
                            run, artifact, args.durable / run.name / artifact.name,
                            args.lock_file, args.writer_host,
                        )
                        print(f"P5_BANKED {run.name} {artifact.name}", flush=True)
                if (run / "completed.json").exists():
                    ack = run / "results-banked.json"
                    if not ack.exists() or verified_results.get(run.name) != sha256(ack):
                        bank_results(run, args.durable / run.name, args.lock_file)
                        verified_results[run.name] = sha256(ack)
                        print(f"P5_RESULTS_BANKED {run.name}", flush=True)
            except Exception as error:
                write_json(run / "bank-error.json", {
                    "error": str(error), "type": type(error).__name__, "time": time.time(),
                })
                print(f"P5_BANK_ERROR {run.name} {type(error).__name__}: {error}", flush=True)
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("runs", "durable", "lock-file"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--writer-host", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    watch(parser.parse_args())
