from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from pathlib import PureWindowsPath
import shlex
import shutil
import socket
import subprocess
import time
import uuid
from typing import Any

from rrbench.io import sha256, write_json
from rrbench.semantic_encoding import validate_completed
from rrbench.semantic_models import configuration_digest, validate_encoding_contract


def safe_relative(filename: str) -> Path:
    path = Path(filename)
    windows = PureWindowsPath(filename)
    if (
        not filename or path.is_absolute() or windows.is_absolute() or windows.drive
        or filename.startswith(("/", "\\")) or ".." in path.parts or ".." in windows.parts
    ):
        raise ValueError("Artifact path must be an in-scope relative path without roots or traversal.")
    return path


def checked_artifact_path(path: Path, root: Path) -> Path:
    if not path.absolute().is_relative_to(root.absolute()):
        raise ValueError("Artifact write path is outside its explicitly granted root.")
    for candidate in (path, *path.parents):
        if candidate.is_symlink() or (hasattr(candidate, "is_junction") and candidate.is_junction()):
            raise ValueError("Artifact target/staging symlink or junction is not permitted.")
        if candidate == root:
            break
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Artifact write path resolves outside its explicitly granted root.")
    return path


def exclusive_file(path: Path, root: Path) -> int:
    checked_artifact_path(path, root)
    return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)


def write_artifact_receipt(path: Path, value: dict[str, Any], root: Path) -> None:
    checked_artifact_path(path, root)
    checked_artifact_path(path.with_suffix(path.suffix + ".tmp"), root)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    descriptor = exclusive_file(temporary, root)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    checked_artifact_path(path, root)
    temporary.replace(path)


def process_artifact_requests(
    requests: Path, bank: Path, verified_requests: dict[str, str] | None = None,
) -> None:
    if not requests.exists():
        return
    for request_file in sorted(requests.glob("*.json")):
        if request_file.name.endswith(".banked.json"):
            continue
        snapshot = request_file.read_bytes()
        request = json.loads(snapshot)
        request_digest = hashlib.sha256(snapshot).hexdigest()
        if verified_requests is not None and request_file.name in verified_requests:
            if verified_requests[request_file.name] != request_digest:
                raise ValueError("An already verified artifact request changed during publishing.")
            continue
        source = Path(request["source_root"]).resolve()
        bank_root = (bank / "artifacts").absolute()
        destination = bank_root / safe_relative(request["destination"])
        checked_artifact_path(destination, bank_root)
        if not source.is_dir():
            raise ValueError("Artifact request source or durable destination is outside its declared root.")
        destination.mkdir(parents=True, exist_ok=True)
        checked_artifact_path(destination, bank_root)
        files = request["files"]
        if not files:
            raise ValueError("Artifact request must include checked completed files.")
        durable_path = destination / "durable.json"
        checked_artifact_path(durable_path, destination)
        prior = json.loads(durable_path.read_text(encoding="utf-8")) if durable_path.exists() else None
        if prior is not None and (
            not prior["complete"] or prior["request_sha256"] != request_digest or prior["files"] != files
        ):
            raise ValueError("Durable artifact request is already bound to a different immutable file set.")
        for filename, expected in files.items():
            relative = safe_relative(filename)
            original, target = (source / relative).resolve(), destination / relative
            checked_artifact_path(target, destination)
            if (
                not original.is_relative_to(source) or not target.is_relative_to(destination)
                or not isinstance(expected, str) or len(expected) != 64
            ):
                raise ValueError("Artifact request file paths or expected hashes are invalid.")
            if sha256(original) != expected:
                raise ValueError("Artifact request source checksum differs from its immutable request.")
            if prior is not None:
                if sha256(target) != expected:
                    raise ValueError("Previously published durable artifact payload checksum changed.")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or sha256(target) != expected:
                temporary = target.with_suffix(target.suffix + ".partial")
                checked_artifact_path(temporary, destination)
                if temporary.exists():
                    temporary.unlink()
                descriptor = exclusive_file(temporary, destination)
                with original.open("rb") as input_file, os.fdopen(descriptor, "wb") as output_file:
                    shutil.copyfileobj(input_file, output_file, length=8 << 20)
                    output_file.flush()
                    os.fsync(output_file.fileno())
                if sha256(temporary) != expected:
                    raise ValueError("Copied durable artifact checksum differs from the frozen source.")
                checked_artifact_path(target, destination)
                temporary.replace(target)
        if hashlib.sha256(request_file.read_bytes()).hexdigest() != request_digest:
            raise ValueError("Artifact request changed during publishing; no durability acknowledgement is issued.")
        durable = {
            "complete": True, "request_sha256": request_digest, "files": files,
            "description": request["description"],
        }
        if prior is None:
            write_artifact_receipt(durable_path, durable, destination)
        if hashlib.sha256(request_file.read_bytes()).hexdigest() != request_digest:
            raise ValueError("Artifact request changed before the durability acknowledgement.")
        checked_artifact_path(durable_path, destination)
        write_artifact_receipt(request_file.with_suffix(".banked.json"), {
            "complete": True, "request_sha256": request_digest,
            "durable_receipt_sha256": sha256(durable_path),
        }, requests)
        if verified_requests is not None:
            verified_requests[request_file.name] = request_digest
        print(f"FINISHED_ARTIFACT_DURABLY_PUBLISHED {request_file.stem}", flush=True)


def representation_files(completed: dict[str, Any]) -> dict[str, str]:
    return {
        "vectors.npy": completed["vectors_sha256"], "ids.npy": completed["ids_sha256"],
        **{
            value["file"]: value["sha256"] for value in completed["mrl_representations"].values()
        },
    }


def validate_task_binding(
    completed: dict[str, Any], configuration: dict[str, Any],
    task: dict[str, Any], specification: dict[str, Any],
) -> None:
    validate_encoding_contract(specification, completed)
    if completed["rows"] != task["rows"]:
        raise ValueError("Complete embedding row count differs from its frozen task.")
    expected_source = task["source_sha256"]
    if completed.get("source_sha256") != expected_source or configuration.get("source_sha256") != expected_source:
        raise ValueError("Embedding receipt/configuration does not match the frozen source-text checksum.")
    if configuration.get("specification_sha256") != completed["specification_sha256"]:
        raise ValueError("Embedding configuration does not match the complete model specification.")
    if configuration.get("model") != specification["model"] or configuration.get("revision") != specification["revision"]:
        raise ValueError("Embedding configuration model/revision differs from the frozen task.")
    if configuration.get("batch_size") != task["batch_size"] or configuration["max_rows"] is not None:
        raise ValueError("Embedding configuration is not a full-shard run at the frozen batch size.")


def bank_shard(
    source: Path, destination: Path, task: dict[str, Any], specification: dict[str, Any],
) -> dict[str, Any]:
    completed = validate_completed(source, task["rows"], task["start"])
    configuration = json.loads((source / "run-config.json").read_text(encoding="utf-8"))
    validate_task_binding(completed, configuration, task, specification)
    if (destination / "completed.json").exists():
        existing = validate_completed(destination, task["rows"], task["start"])
        existing_config = json.loads((destination / "run-config.json").read_text(encoding="utf-8"))
        validate_task_binding(existing, existing_config, task, specification)
        if existing != completed:
            raise ValueError("Durable shard is already bound to a different complete artifact.")
        return existing
    destination.mkdir(parents=True, exist_ok=True)
    for filename, expected in representation_files(completed).items():
        target = destination / filename
        if target.exists() and sha256(target) == expected:
            continue
        temporary = target.with_suffix(target.suffix + ".partial")
        shutil.copyfile(source / filename, temporary)
        if sha256(temporary) != expected:
            raise ValueError("Durable copied embedding checksum differs from its source receipt.")
        temporary.replace(target)
    shutil.copyfile(source / "run-config.json", destination / "run-config.json")
    write_json(destination / "completed.json", completed)
    validate_completed(destination, task["rows"], task["start"])
    return completed


def remote_json(node: str, filename: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["ssh", "-n", "-o", "BatchMode=yes", node, "cat " + shlex.quote(str(filename))],
        check=True, text=True, capture_output=True, timeout=120,
    )
    return json.loads(result.stdout)


def fetch_shard(
    node: str, task: dict[str, Any], specification: dict[str, Any], destination: Path | None = None,
) -> None:
    remote_source = Path(task["output"])
    destination = remote_source if destination is None else destination
    completed = remote_json(node, remote_source / "completed.json")
    configuration = remote_json(node, remote_source / "run-config.json")
    validate_task_binding(completed, configuration, task, specification)
    if (destination / "completed.json").exists():
        existing = validate_completed(destination, task["rows"], task["start"])
        existing_config = json.loads((destination / "run-config.json").read_text(encoding="utf-8"))
        validate_task_binding(existing, existing_config, task, specification)
        if existing != completed or existing_config != configuration:
            raise ValueError("Existing local shard differs from the complete remote artifact.")
        return
    if (destination / "run-config.json").exists():
        if json.loads((destination / "run-config.json").read_text(encoding="utf-8")) != configuration:
            raise ValueError("Partial local transfer is bound to a different source/configuration.")
    destination.mkdir(parents=True, exist_ok=True)
    for filename, expected in representation_files(completed).items():
        target = destination / filename
        if target.exists() and sha256(target) == expected:
            continue
        temporary = target.with_suffix(target.suffix + ".transfer")
        subprocess.run(
            ["scp", "-q", node + ":" + shlex.quote(str(remote_source / filename)), str(temporary)],
            check=True, timeout=1200,
        )
        if sha256(temporary) != expected:
            raise ValueError("Cross-node embedding transfer checksum differs from its complete receipt.")
        temporary.replace(target)
    write_json(destination / "run-config.json", configuration)
    write_json(destination / "completed.json", completed)
    validate_completed(destination, task["rows"], task["start"])


def ready_on_node(node: str, tasks: list[dict[str, Any]]) -> set[str]:
    commands = [
        f"if test -f {shlex.quote(str(Path(task['output']) / 'completed.json'))}; "
        f"then printf '%s\\n' {shlex.quote(task['key'])}; fi"
        for task in tasks
    ]
    result = subprocess.run(
        ["ssh", "-n", "-o", "BatchMode=yes", node, "; ".join(commands) + "; true"],
        check=True, text=True, capture_output=True, timeout=120,
    )
    return set(result.stdout.splitlines())


def validate_bank_progress(plan: dict[str, Any], bank: Path, progress: dict[str, Any]) -> set[str]:
    if progress.get("plan_sha256") != configuration_digest(plan):
        raise ValueError("Existing bank progress is bound to a different frozen encoding plan.")
    if progress.get("bank_directory") != str(bank.resolve()):
        raise ValueError("Existing bank progress is bound to a different durable destination.")
    tasks = {task["key"]: task for task in plan["tasks"]}
    if not tasks or len(tasks) != len(plan["tasks"]) or set(progress["banked"]) - set(tasks):
        raise ValueError("Bank progress/task keys are duplicated or outside the expected finite task set.")
    verified = set()
    for key, recorded in progress["banked"].items():
        task = tasks[key]
        destination = bank / "embeddings" / task["corpus"] / task["model"] / key
        if sha256(destination / "completed.json") != recorded["receipt_sha256"]:
            raise ValueError("Previously banked completion receipt is missing or changed.")
        completed = validate_completed(destination, task["rows"], task["start"])
        configuration = json.loads((destination / "run-config.json").read_text(encoding="utf-8"))
        validate_task_binding(completed, configuration, task, plan["registry"]["models"][task["model"]])
        if recorded["rows"] != completed["rows"] or recorded["vectors_sha256"] != completed["vectors_sha256"]:
            raise ValueError("Saved banking progress does not match its checked artifact.")
        verified.add(key)
    return verified


def bank_loop(
    plan: dict[str, Any], bank: Path, progress: Path, nodes: int, deadline_seconds: int,
    artifact_requests: Path | None = None,
) -> None:
    if socket.gethostname() != "node-0":
        raise RuntimeError("Shared-storage publishing is restricted to the sole node-0 writer.")
    if nodes <= 0 or nodes * 8 != plan["workers"] or deadline_seconds <= 0:
        raise ValueError("Banker node count/deadline must match the finite eight-GPU-node plan.")
    plan_digest = configuration_digest(plan)
    prior = json.loads(progress.read_text(encoding="utf-8")) if progress.exists() else {
        "banked": {}, "plan_sha256": plan_digest, "bank_directory": str(bank.resolve()),
    }
    if prior["plan_sha256"] != plan_digest or prior["bank_directory"] != str(bank.resolve()):
        raise ValueError("Existing bank progress uses a different frozen plan or destination.")
    verified_requests = {}
    if artifact_requests is not None:
        process_artifact_requests(artifact_requests, bank, verified_requests)
    verified_keys = validate_bank_progress(plan, bank, prior)
    banked = prior["banked"]
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        if artifact_requests is not None:
            process_artifact_requests(artifact_requests, bank, verified_requests)
        ready = {}
        for node in range(1, nodes):
            pending = [
                task for task in plan["tasks"] if task["worker"] // 8 == node and task["key"] not in verified_keys
            ]
            ready[node] = ready_on_node(f"node-{node}", pending) if pending else set()
        for task in plan["tasks"]:
            if task["key"] in verified_keys:
                continue
            node = task["worker"] // 8
            source = Path(task["output"])
            specification = plan["registry"]["models"][task["model"]]
            if node == 0:
                if not (source / "completed.json").exists():
                    continue
            elif task["key"] in ready[node]:
                fetch_shard(f"node-{node}", task, specification)
            else:
                continue
            target = bank / "embeddings" / task["corpus"] / task["model"] / task["key"]
            completed = bank_shard(source, target, task, specification)
            banked[task["key"]] = {
                "rows": completed["rows"], "vectors_sha256": completed["vectors_sha256"],
                "receipt_sha256": sha256(target / "completed.json"),
            }
            verified_keys.add(task["key"])
            write_json(progress, {
                "complete": False, "banked": banked,
                "tasks": len(plan["tasks"]), "banked_tasks": len(banked),
                "banked_rows": sum(row["rows"] for row in banked.values()),
                "plan_sha256": plan_digest,
                "bank_directory": str(bank.resolve()),
            })
            print(f"SHARD_DURABLY_BANKED {task['key']} count={len(banked)}/{len(plan['tasks'])}", flush=True)
            if artifact_requests is not None:
                process_artifact_requests(artifact_requests, bank, verified_requests)
        if verified_keys == {task["key"] for task in plan["tasks"]}:
            if artifact_requests is not None:
                process_artifact_requests(artifact_requests, bank)
            validate_bank_progress(plan, bank, {
                "banked": banked, "plan_sha256": plan_digest, "bank_directory": str(bank.resolve()),
            })
            write_json(progress, {
                "complete": True, "banked": banked, "tasks": len(plan["tasks"]),
                "banked_tasks": len(verified_keys), "banked_rows": sum(row["rows"] for row in banked.values()),
                "plan_sha256": plan_digest, "bank_directory": str(bank.resolve()),
            })
            print("ALL_REAL_SEMANTIC_SHARDS_DURABLE", flush=True)
            return
        time.sleep(20)
    raise RuntimeError("Finite embedding bank deadline expired before every declared shard was durable.")


def main() -> None:
    import fcntl

    parser = argparse.ArgumentParser(description="Single-node serialized complete semantic-shard publisher.")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--progress", required=True)
    parser.add_argument("--nodes", type=int, default=3)
    parser.add_argument("--deadline-seconds", type=int, default=86400)
    parser.add_argument("--artifact-requests")
    args = parser.parse_args()
    progress = Path(args.progress)
    progress.parent.mkdir(parents=True, exist_ok=True)
    with Path("/scratch/rrbench-shared-publisher.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank_loop(
            json.loads(Path(args.plan).read_text(encoding="utf-8")), Path(args.bank), progress,
            args.nodes, args.deadline_seconds,
            Path(args.artifact_requests) if args.artifact_requests else None,
        )


if __name__ == "__main__":
    main()
