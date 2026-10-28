"""Node-0 snapshots and serialized banking for the frozen conventional baseline runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import socket
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import bank
from protocol import file_sha256, rank_metrics, require_hash, verify_coverage, write_json


OWNER = "letter-baselines-snapshot-v1"
OWNER_FILE = "snapshot-owner.json"
PARTIAL = ".bank-baselines-partial-"
METRICS = {f"{metric}@{k}" for metric in ("recall", "ndcg") for k in (5, 10)}


def _log(event, **details):
    print(json.dumps({"event": event, **details}, allow_nan=False), file=sys.stderr, flush=True)


def _json(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"Duplicate metadata key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Nonfinite metadata: {value}")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("Invalid SHA256 metadata")
    return value


def _epoch(value):
    if type(value) is not int or value < 1:
        raise ValueError("Invalid checkpoint epoch")
    return value


def _path(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError("Artifact path escapes its run")
    path = root.joinpath(*relative.parts)
    for candidate in [root, *[root.joinpath(*relative.parts[:i])
                             for i in range(1, len(relative.parts) + 1)]]:
        if candidate.is_symlink():
            raise ValueError(f"Symlink is not an immutable artifact: {candidate}")
    return path


class _Rotated(Exception):
    pass


@contextmanager
def _source_read(path, rotatable):
    try:
        yield
    except FileNotFoundError as error:
        if rotatable and error.filename is not None and (
            Path(error.filename).absolute() == path.absolute()
        ):
            raise _Rotated(str(path)) from error
        raise


def _metrics(value):
    if not isinstance(value, dict) or set(value) != METRICS or any(
        type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
        for v in value.values()
    ):
        raise ValueError("Invalid original-item metric metadata")


def _prediction_metrics(rows, users, items):
    if not isinstance(rows, list):
        raise ValueError("Predictions must cover every original user")
    verify_coverage(users, [row["user"] for row in rows])
    mapping = {user: i for i, user in enumerate(users, 1)}
    totals = {metric: 0.0 for metric in METRICS}
    for row in rows:
        predictions, target = row["top10"], row["target"]
        if (type(row["model_user"]) is not int or row["model_user"] != mapping[row["user"]]
                or type(target) is not int or not 0 <= target < items
                or not isinstance(predictions, list) or len(predictions) != min(10, items)
                or any(type(i) is not int or not 0 <= i < items for i in predictions)
                or len(set(predictions)) != len(predictions)):
            raise ValueError("Invalid original-user/item prediction metadata")
        for metric, value in rank_metrics(predictions, target, (5, 10)).items():
            totals[metric] += value
    return {metric: value / len(users) for metric, value in totals.items()}


def _expect_metrics(actual, expected):
    _metrics(actual)
    _metrics(expected)
    if any(not math.isclose(actual[k], expected[k], rel_tol=0, abs_tol=1e-12) for k in METRICS):
        raise ValueError("Prediction metrics differ from selected/completed metadata")


def _expect(record, expected, label):
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{label} metadata/selection/config checksum mismatch")


def _inspect_run(run, *, allow_rotation):
    """Read completion first, then checkpoint markers; never mix a later freeze into a live view."""
    if not run.exists():
        return None
    if not run.is_dir() or run.is_symlink():
        raise ValueError("Baseline run must be a real directory")
    files, documents, checkpoints = {}, {}, {}
    rotatable = False

    def read(name, expected=None, *, rotating=False):
        path = _path(run, name)
        with _source_read(path, rotating):
            raw = path.read_bytes()
        metadata = {"sha256": _digest(raw), "bytes": len(raw)}
        if expected is not None and metadata["sha256"] != _hash(expected):
            raise ValueError(f"Source checksum mismatch: {path}")
        if name in files and files[name] != metadata:
            raise ValueError(f"Source metadata changed during snapshot: {path}")
        files[name] = metadata
        documents[name] = _json(raw)
        return documents[name]

    completed = read("completed.json") if _path(run, "completed.json").exists() else None
    frozen = read("train-completed.json") if _path(run, "train-completed.json").exists() else None
    if completed is not None and frozen is None:
        raise ValueError("Completed evaluation has no frozen training receipt")
    phase = "evaluation" if completed is not None else "model" if frozen is not None else "checkpoint"
    rotatable = allow_rotation and phase == "checkpoint"
    markers = sorted(_path(run, "checkpoints").glob("epoch-*.json"))
    if not markers:
        if frozen is not None:
            raise ValueError("Frozen baseline has no complete checkpoints")
        return None
    epochs = []
    for marker in markers:
        match = re.fullmatch(r"epoch-(\d{5,})\.json", marker.name)
        if not match or marker.name != f"epoch-{int(match[1]):05d}.json":
            raise ValueError("Invalid checkpoint marker name")
        epochs.append(_epoch(int(match[1])))
    latest_epoch = max(epochs)
    config = read("run-config.json")
    config_sha = files["run-config.json"]["sha256"]
    users = read("user-map.json")
    counts = config["counts"]
    if (type(config["schema_version"]) is not int or config["schema_version"] != 1
            or type(counts["users"]) is not int or counts["users"] < 1
            or type(counts["items"]) is not int or counts["items"] < 10
            or not isinstance(users, list) or len(users) != counts["users"]
            or any(not isinstance(u, str) or not u.isdecimal() for u in users)
            or len({int(u) for u in users}) != len(users)
            or users != sorted(users, key=int)):
        raise ValueError("Invalid run config or original user mapping")

    def checkpoint(epoch):
        if epoch in checkpoints:
            return checkpoints[epoch]
        _epoch(epoch)
        if epoch > config["training"]["max_epochs"]:
            raise ValueError("Checkpoint exceeds the frozen training budget")
        stem = f"checkpoints/epoch-{epoch:05d}"
        record = read(stem + ".json", rotating=rotatable)
        if type(record["epoch"]) is not int or record["epoch"] != epoch:
            raise ValueError("Checkpoint filename/metadata epoch mismatch")
        _expect(record, {"config_sha256": config_sha}, "Checkpoint config")
        log = record["log"]
        if (log["epoch"] != epoch or type(log["mean_loss"]) not in (int, float)
                or not math.isfinite(log["mean_loss"])
                or type(log["optimizer_steps"]) is not int or log["optimizer_steps"] < 1):
            raise ValueError("Invalid checkpoint training log")
        path = _path(run, stem + ".pt")
        with _source_read(path, rotatable):
            digest = file_sha256(path)
            size = path.stat().st_size
        if digest != _hash(record["sha256"]):
            raise ValueError(f"Checkpoint checksum mismatch: {path}")
        files[stem + ".pt"] = {"sha256": digest, "bytes": size}
        checkpoints[epoch] = record
        return record

    def selected(record):
        selection = record["selection"]
        if selection is None:
            return None
        best_epoch = _epoch(selection["epoch"])
        if best_epoch > record["epoch"] or selection["predictions"] != (
            f"validation-top10-{best_epoch:05d}.json"
        ):
            raise ValueError("Invalid selected checkpoint/prediction path")
        _metrics(selection["validation"])
        best = checkpoint(best_epoch)
        if best["selection"] != selection:
            raise ValueError("Selected checkpoint selection metadata mismatch")
        rows = read(selection["predictions"], selection["predictions_sha256"], rotating=rotatable)
        _expect_metrics(_prediction_metrics(rows, users, counts["items"]), selection["validation"])
        return best

    latest = checkpoint(latest_epoch)
    best = selected(latest)
    selection = latest["selection"]
    if frozen is not None:
        if best is None:
            raise ValueError("Frozen model has no selected validation checkpoint")
        _expect(frozen, {
            "status": "train-complete", "run_class": config["run_class"],
            "model": config["model"], "dataset": config["dataset"],
            "selected_epoch": selection["epoch"], "last_epoch": latest_epoch,
            "last_checkpoint": f"checkpoints/epoch-{latest_epoch:05d}.pt",
            "selected_checkpoint": f"checkpoints/epoch-{selection['epoch']:05d}.pt",
            "selected_checkpoint_sha256": best["sha256"], "validation": selection["validation"],
            "validation_predictions": selection["predictions"],
            "validation_predictions_sha256": selection["predictions_sha256"],
            "config_sha256": config_sha, "user_map_sha256": files["user-map.json"]["sha256"],
            "provenance": config["provenance"], "test_evaluated": False,
        }, "Frozen selected model")
        history = read("epochs.json")
        if (not isinstance(history, list) or [row["epoch"] for row in history]
                != list(range(1, latest_epoch + 1)) or history[-1] != latest["log"]):
            raise ValueError("Frozen epoch history differs from latest checkpoint metadata")
    if completed is not None:
        if (run / "train.lock").exists():
            raise ValueError("Completed evaluation still has a training lock; inspect the writer")
        frozen_sha = files["train-completed.json"]["sha256"]
        _expect(completed, {
            "status": "complete", "run_class": config["run_class"], "kind": config["kind"],
            "model": config["model"], "dataset": config["dataset"],
            "test_users": counts["users"], "items": counts["items"], "test_evaluated": True,
            "selected_epoch": selection["epoch"], "validation": selection["validation"],
            "config_sha256": config_sha, "frozen_train_receipt_sha256": frozen_sha,
            "selected_checkpoint_sha256": best["sha256"],
            "user_map_sha256": files["user-map.json"]["sha256"], "provenance": config["provenance"],
        }, "Completed evaluation/frozen model")
        _expect(read("test-claimed.json"), {
            "config_sha256": config_sha, "frozen_sha256": frozen_sha,
            "selected_checkpoint_sha256": best["sha256"],
        }, "Single-use test claim")
        predictions = read("test-top10.json", completed["test_predictions_sha256"])
        _expect_metrics(_prediction_metrics(predictions, users, counts["items"]), completed["test"])
        for epoch in epochs:
            selected(checkpoint(epoch))
        inventory = bank.source_manifest(run)["files"]
        if any(inventory.get(name) != metadata for name, metadata in files.items()):
            raise ValueError("Completed source changed during inspection")
        for name in inventory:
            if name.startswith("checkpoints/") and name.endswith(".pt") and (
                name[:-3] + ".json" not in inventory
            ):
                raise ValueError("Completed run contains an unsealed checkpoint")
        files = inventory
    return {
        "phase": phase, "latest_epoch": latest_epoch, "latest_checkpoint_sha256": latest["sha256"],
        "selection": selection, "config_sha256": config_sha,
        "train_completed_sha256": files["train-completed.json"]["sha256"] if frozen else None,
        "completed_sha256": files["completed.json"]["sha256"] if completed else None,
        "files": files,
    }


@contextmanager
def _locked(path):
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            if stream.seek(0, os.SEEK_END) == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)


def _overlap(left, right):
    left, right = left.resolve(), right.resolve()
    return left == right or left in right.parents or right in left.parents


def _quarantine(partial):
    destination = partial.parent / f".bank-baselines-quarantine-{uuid.uuid4().hex}"
    partial.rename(destination)
    _log("QUARANTINED_PARTIAL", source=str(partial), destination=str(destination))


def _recover_partials(staging):
    for path in sorted(staging.glob(PARTIAL + "*")):
        if (not re.fullmatch(re.escape(PARTIAL) + r"[a-f0-9]{32}", path.name)
                or path.is_symlink() or not path.is_dir()):
            continue
        owner = path / OWNER_FILE
        if owner.is_file() and not owner.is_symlink():
            if _json(owner.read_bytes()).get("owner") == OWNER:
                _quarantine(path)


def verify_snapshot(snapshot):
    if snapshot.is_symlink():
        raise ValueError("Snapshot must not be a symlink")
    receipt = _json((snapshot / "completed.json").read_bytes())
    if (receipt.get("status") != "complete" or receipt.get("kind") != OWNER
            or receipt.get("schema_version") != 1):
        raise ValueError("Invalid baseline snapshot completion receipt")
    owner = _json((snapshot / OWNER_FILE).read_bytes())
    _expect(owner, {"owner": OWNER, "identity": receipt["identity"],
                    "source_run": receipt["source_run"]}, "Snapshot ownership")
    if {p.name for p in snapshot.iterdir()} - {
        "completed.json", "payload", OWNER_FILE, "bank-receipt.json"
    }:
        raise ValueError("Unexpected file in immutable snapshot root")
    actual = bank.source_manifest(snapshot / "payload")["files"]
    if actual != receipt["files"]:
        raise ValueError("Snapshot payload checksum/manifest mismatch")
    plan = _inspect_run(snapshot / "payload", allow_rotation=False)
    if plan is None or any(receipt.get(key) != value for key, value in plan.items()):
        raise ValueError("Snapshot payload metadata differs from its receipt")
    return receipt


def create_snapshot(run, staging):
    run, staging = Path(run).absolute(), Path(staging).absolute()
    if _overlap(run, staging):
        raise ValueError("Snapshot staging must not overlap a source run")
    partial = None
    try:
        plan = _inspect_run(run, allow_rotation=True)
        if plan is None:
            _log("NOT_STARTED_OR_NO_COMPLETE_EPOCH", run=str(run))
            return {"status": "pending", "run": str(run)}
        parent = staging
        while not parent.exists():
            parent = parent.parent
        if parent.stat().st_dev != run.stat().st_dev:
            raise ValueError("Snapshot staging must be on the source-local filesystem")
        staging.mkdir(parents=True, exist_ok=True)
        content = {"schema_version": 1, "source_run": str(run), **plan}
        digest = _digest(json.dumps(content, sort_keys=True, separators=(",", ":")).encode())
        identity = f"baseline-{plan['phase']}-{plan['latest_checkpoint_sha256'][:12]}-{digest[:20]}"
        destination = staging / identity
        result = {"status": "ready", "run": str(run), "phase": plan["phase"],
                  "identity": identity, "path": str(destination)}
        with _locked(staging / ".bank-baselines.lock"):
            _recover_partials(staging)
            if destination.exists():
                receipt = verify_snapshot(destination)
                _expect(receipt, content, "Existing immutable snapshot")
                return result
            partial = staging / f"{PARTIAL}{uuid.uuid4().hex}"
            partial.mkdir()
            write_json(partial / OWNER_FILE,
                       {"owner": OWNER, "identity": identity, "source_run": str(run)})
            payload = partial / "payload"
            for name, metadata in sorted(plan["files"].items()):
                source, target = _path(run, name), _path(payload, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                rotating = plan["phase"] == "checkpoint" and (
                    re.fullmatch(r"checkpoints/epoch-\d{5,}\.(pt|json)", name) is not None
                    or re.fullmatch(r"validation-top10-\d{5,}\.json", name) is not None
                )
                with _source_read(source, rotating):
                    shutil.copyfile(source, target)
                if target.stat().st_size != metadata["bytes"]:
                    raise ValueError(f"Snapshot source size changed: {name}")
                require_hash(target, metadata["sha256"])
            write_json(partial / "completed.json",
                       {"status": "complete", "kind": OWNER, "identity": identity, **content})
            verify_snapshot(partial)
            partial.rename(destination)
            partial = None
            return result
    except _Rotated as error:
        if partial is not None and partial.exists():
            _quarantine(partial)
        _log("CHECKPOINT_ROTATED_DURING_SNAPSHOT", run=str(run), path=str(error))
        return {"status": "rotated", "run": str(run)}


def require_node0(writer_host="node-0"):
    if writer_host != "node-0" or socket.gethostname() != "node-0" or any(
        os.environ.get(key, "0") != "0" for key in ("NODE_RANK", "AZUREML_CR_NODE_RANK")
    ):
        raise RuntimeError("Only the designated node-0 may snapshot/watch/bank baselines")


def _watch_paths(args):
    require_node0(args.writer_host)
    if not args.runs or len({run.name for run in args.runs}) != len(args.runs):
        raise ValueError("Watch a nonempty list of runs with distinct acknowledgment names")
    if args.lock_file.name != "bank.lock" or not args.lock_file.parent.is_dir():
        raise ValueError("Supply the existing common bank.lock directory, not a per-run lock")
    paths = [args.staging, args.destination, args.ack_root]
    if any(_overlap(a, b) for i, a in enumerate(paths) for b in paths[i+1:]) or any(
        _overlap(run, path) for run in args.runs for path in paths
    ):
        raise ValueError("Run, staging, destination and acknowledgment directories must not overlap")


def _verify_durable(destination):
    marker = destination / "bank-receipt.json"
    expected = _json(marker.read_bytes())
    actual = bank.source_manifest(destination)
    actual["files"].pop("bank-receipt.json")
    if actual != expected:
        raise ValueError("Durable bank receipt checksum/manifest mismatch")
    if verify_snapshot(destination)["identity"] != destination.name:
        raise ValueError("Durable snapshot identity differs from its immutable directory name")
    return file_sha256(marker)


def _acknowledge(args, run, snapshot, banked):
    receipt = verify_snapshot(snapshot)
    if receipt["train_completed_sha256"] is None:
        return
    require_hash(run / "train-completed.json", receipt["train_completed_sha256"])
    args.ack_root.mkdir(parents=True, exist_ok=True)
    kinds = ["model", "evaluation"] if receipt["phase"] == "evaluation" else ["model"]
    for kind in kinds:
        binding = {
            "status": "banked", "kind": kind, "source_run": str(run.absolute()),
            "config_sha256": receipt["config_sha256"],
            "train_completed_sha256": receipt["train_completed_sha256"],
        }
        if kind == "evaluation":
            require_hash(run / "completed.json", receipt["completed_sha256"])
            binding["completed_sha256"] = receipt["completed_sha256"]
        ack = args.ack_root / f"{run.name}.{kind}.json"
        if ack.exists():
            existing = _json(ack.read_bytes())
            _expect(existing, binding, "Existing acknowledgment")
            recorded = Path(existing["destination"])
            if (recorded.parent.resolve() != args.destination.resolve()
                    or recorded.name != existing["snapshot_identity"]):
                raise ValueError("Existing acknowledgment escapes the supplied banking root")
            if _verify_durable(recorded) != existing["bank_receipt_sha256"]:
                raise ValueError("Existing acknowledgment bank receipt checksum mismatch")
            recorded_receipt = _json((recorded / "completed.json").read_bytes())
            expected = {key: binding[key] for key in
                        ("source_run", "config_sha256", "train_completed_sha256")}
            if kind == "evaluation":
                expected["completed_sha256"] = binding["completed_sha256"]
            _expect(recorded_receipt, expected, "Existing acknowledgment snapshot")
        else:
            write_json(ack, {**binding, **banked})


def _ack_files(args, run, phase):
    kinds = [] if phase == "checkpoint" else (
        ["model", "evaluation"] if phase == "evaluation" else ["model"]
    )
    return [args.ack_root / f"{run.name}.{kind}.json" for kind in kinds]


def watch_cycle(args, banked):
    _watch_paths(args)
    all_done = True
    final_destinations = []
    for run in args.runs:
        try:
            artifact = create_snapshot(run, args.staging)
            if artifact["status"] != "ready":
                all_done = False
                continue
            identity, snapshot = artifact["identity"], Path(artifact["path"])
            destination = args.destination / identity
            if identity not in banked:
                with _locked(args.lock_file):
                    bank.copy_artifact(snapshot, destination)
                    proof = {"snapshot_identity": identity, "destination": str(destination.absolute()),
                             "bank_receipt_sha256": _verify_durable(destination)}
                    _acknowledge(args, run, snapshot, proof)
                    acknowledgments = {
                        str(path): file_sha256(path)
                        for path in _ack_files(args, run, artifact["phase"])
                    }
                banked[identity] = {"proof": proof, "acknowledgments": acknowledgments}
                _log("BANKED_BASELINE", run=str(run), phase=artifact["phase"], identity=identity)
            else:
                for path, digest in banked[identity]["acknowledgments"].items():
                    require_hash(Path(path), digest)
                _log("BASELINE_UNCHANGED_ALREADY_BANKED", run=str(run), identity=identity)
            if artifact["phase"] == "evaluation":
                final_destinations.append(destination)
            else:
                all_done = False
        except Exception as error:
            all_done = False
            _log("BASELINE_BANK_ERROR", run=str(run), error_type=type(error).__name__,
                 error=str(error))
    if all_done:
        try:
            with _locked(args.lock_file):
                for destination in final_destinations:
                    _verify_durable(destination)
        except Exception as error:
            _log("BASELINE_FINAL_VERIFICATION_ERROR",
                 error_type=type(error).__name__, error=str(error))
            return False
    return all_done


def watch(args):
    _watch_paths(args)
    if args.interval < 30:
        raise ValueError("Watch interval must be at least 30 seconds")
    banked = {}
    _log("BASELINE_NODE0_WATCH_READY", runs=[str(run) for run in args.runs],
         lock_file=str(args.lock_file))
    while True:
        if watch_cycle(args, banked):
            _log("ALL_BASELINES_AND_EVALUATIONS_DURABLY_VERIFIED")
            return
        time.sleep(args.interval)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser("snapshot")
    snapshot.add_argument("--run", type=Path, required=True)
    snapshot.add_argument("--staging", type=Path, required=True)
    watcher = commands.add_parser("watch")
    watcher.add_argument("--runs", nargs="+", type=Path, required=True)
    for name in ("staging", "destination", "lock-file", "ack-root"):
        watcher.add_argument(f"--{name}", type=Path, required=True)
    watcher.add_argument("--writer-host", choices=("node-0",), required=True)
    watcher.add_argument("--interval", type=int, default=120)
    return parser


def main():
    args = make_parser().parse_args()
    require_node0()
    try:
        if args.command == "snapshot":
            print(json.dumps(create_snapshot(args.run, args.staging)), flush=True)
        else:
            watch(args)
    except Exception as error:
        _log("BASELINE_SNAPSHOT_ERROR", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
