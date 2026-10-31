"""Fail-closed, CPU-only durable-ack timeout recovery around an immutable LC queue."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time
import uuid


VERSION = "lc-durable-ack-recovery-v1"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def require_hash(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Recovery checksum mismatch: {path}")


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_state(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".writing-" + str(os.getpid()))
    with staging.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    staging.replace(path)


def classify_failure(path, settings):
    path = Path(path)
    name = re.fullmatch(r"failed-rank-0-([1-9][0-9]*)\.json", path.name)
    if not name or path.is_symlink() or path.parent.resolve() != Path(settings["run"]).resolve():
        raise ValueError("Only the exact rank-zero durable-ack failure receipt is recoverable")
    failure = read(path)
    message = failure.get("message", "")
    match = re.fullmatch(r"No hash-bound durable acknowledgment for checkpoint-([1-9][0-9]*)", message)
    trace = failure.get("traceback", "")
    source = Path(settings["frozen_source"])
    if (failure.get("status") != "error" or failure.get("error_type") != "TimeoutError" or not match
            or str(source / "lc_run.py") not in trace or str(source / "lc_checkpoint.py") not in trace
            or "in on_save" not in trace or "in wait_for_ack" not in trace
            or 'wait_for_ack(sealed, args.output / "acks")' not in trace
            or not trace.rstrip().endswith("TimeoutError: " + message)
            or not isinstance(failure.get("unix"), (int, float))
            or not math.isfinite(failure["unix"])):
        raise ValueError("Unrecognized failure: recovery is limited to the frozen wait_for_ack TimeoutError")
    identity = {
        "run_config_sha256": settings["run_config_sha256"],
        "failure_name": path.name, "failure_sha256": sha(path),
    }
    return {
        **identity, "failure_id": fingerprint(identity), "failure_path": str(path),
        "step": int(match.group(1)), "rank_zero_pid": int(name.group(1)), "unix": failure["unix"],
    }


def verify_run_identity(settings):
    run, source = Path(settings["run"]), Path(settings["frozen_source"])
    require_hash(run / "run-config.json", settings["run_config_sha256"])
    require_hash(settings["admission"], settings["admission_sha256"])
    require_hash(settings["launcher"], settings["launcher_sha256"])
    config = read(run / "run-config.json")
    admission = read(settings["admission"])
    if (config.get("kind") != "full-fit" or config["inputs"].get("dataset") != "Instruments"
            or config.get("gate_report_sha256") != settings["admission_sha256"]
            or admission.get("status") != "pass"):
        raise ValueError("Recovery is restricted to the frozen admitted Instruments run")
    for name, digest in config["inputs"]["harness_sha256"].items():
        require_hash(source / name, digest)
    return config


def generation_identity(run):
    run = Path(run)
    names = ("supervisor.pid", "torchrun.pid", "runtime.json", "live.json", "latest-state.json")
    return fingerprint({name: sha(run / name) if (run / name).is_file() else None for name in names})


def verify_local_candidate(settings, candidate):
    import base64
    from lc_checkpoint import snapshot_identity, verify, verify_ack

    config = verify_run_identity(settings)
    run = Path(settings["run"])
    if (run / "completed.json").exists():
        raise ValueError("A completed run must never be relaunched")
    actual = classify_failure(Path(candidate["failure_path"]), settings)
    if actual != candidate:
        raise ValueError("Failure changed after recovery inspection")
    latest, live, runtime = read(run / "latest-state.json"), read(run / "live.json"), read(run / "runtime.json")
    step = candidate["step"]
    if (latest.get("global_step") != step or latest.get("checkpoint") != f"checkpoint-{step}"
            or live.get("global_step") != step or live.get("pid") != candidate["rank_zero_pid"]
            or runtime.get("pid") != candidate["rank_zero_pid"]
            or runtime.get("started_unix", float("inf")) > candidate["unix"]):
        raise ValueError("Stale failure or step/generation changed; never resume a different latest checkpoint")
    sealed = [int(path.parent.name.split("-")[-1]) for path in run.glob("checkpoint-*/complete.json")]
    if not sealed or max(sealed) != step:
        raise ValueError("Failure does not name the latest COMPLETE checkpoint")
    checkpoint, snapshot = run / f"checkpoint-{step}", run / "snapshots" / f"checkpoint-{step}"
    checkpoint_seal, snapshot_seal = verify(checkpoint), verify(snapshot)
    if checkpoint_seal != snapshot_seal or sha(checkpoint / "complete.json") != sha(snapshot / "complete.json"):
        raise ValueError("Checkpoint and independent frozen snapshot differ")
    if checkpoint_seal["world_size"] != 4 or checkpoint_seal["files"]["run-config.json"] != settings["run_config_sha256"]:
        raise ValueError("Checkpoint is not the complete admitted four-rank run")
    identity = snapshot_identity(snapshot)
    if any(latest.get(key) != value for key, value in identity.items()):
        raise ValueError("Latest-state receipt does not match the complete frozen snapshot")
    ack_path = run / "acks" / f"checkpoint-{step}.json"
    acknowledgment = read(ack_path)
    verify_ack(snapshot, acknowledgment)
    if (acknowledgment.get("writer_host") != settings["writer_host"]
            or acknowledgment.get("bank_helper_sha256") != settings["bank_helper_sha256"]
            or acknowledgment.get("bank_directory") != str(Path(settings["bank_directory"]) / f"checkpoint-{step}")
            or not isinstance(acknowledgment.get("verified_unix"), (int, float))
            or not math.isfinite(acknowledgment["verified_unix"])):
        raise ValueError("Acknowledgment has a different writer, helper, or durable destination")
    for path, digest in config["inputs"].get("verified_files_sha256", {}).items():
        require_hash(path, digest)
    return {
        "candidate": candidate, "checkpoint_identity": identity,
        "acknowledgment": acknowledgment, "ack_sha256": sha(ack_path),
        "generation_sha256": generation_identity(run),
        "run_config_sha256": settings["run_config_sha256"],
        "admission_sha256": settings["admission_sha256"],
        "launcher_sha256": settings["launcher_sha256"],
        "failure_bytes_b64": base64.b64encode(Path(candidate["failure_path"]).read_bytes()).decode(),
    }


def owns_process(record, settings):
    argv = record.get("argv", [])
    output = None
    for index, value in enumerate(argv):
        if value == "--output" and index + 1 < len(argv):
            output = argv[index + 1]
        elif value.startswith("--output="):
            output = value.split("=", 1)[1]
    if output == settings["run"] and any(Path(value).name in {"lc_run.py", "torchrun"} for value in argv):
        return True
    launcher = Path(settings["launcher"])
    cwd = Path(record.get("cwd", "/"))
    for value in argv:
        if Path(value).name == launcher.name and (cwd / value).resolve() == launcher.resolve():
            return True
        # A tmux pane shell may contain the owned launcher in its command string.
        if str(launcher) in value and any(Path(token).name in {"bash", "sh"} for token in argv[:1]):
            return True
    return False


def process_record(pid):
    directory = Path("/proc") / str(pid)
    stat = (directory / "stat").read_text().split(") ", 1)[1].split()
    return {
        "pid": int(pid), "ppid": int(stat[1]), "state": stat[0], "start_ticks": int(stat[19]),
        "argv": [part.decode(errors="replace") for part in (directory / "cmdline").read_bytes().split(b"\0") if part],
        "cwd": str((directory / "cwd").resolve()),
        "wchan": (directory / "wchan").read_text().strip(),
    }


def owned_processes(settings):
    result = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            record = process_record(directory.name)
            if owns_process(record, settings):
                result.append(record)
        except (FileNotFoundError, ProcessLookupError):
            continue
    return sorted(result, key=lambda record: record["pid"])


def gpus_free(devices, applications, allowed):
    if allowed != [0, 1, 2, 3]:
        raise ValueError("Recovery is limited to assigned GPUs0-3")
    uuids = {line.split(",")[1].strip() for line in devices.splitlines()
             if "," in line and int(line.split(",")[0]) in allowed}
    if len(uuids) != 4:
        raise ValueError("Assigned GPU inventory is incomplete")
    return not any(line.split(",")[-1].strip() in uuids
                   for line in applications.splitlines() if "," in line)


def assigned_gpus_free(settings):
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True)
    applications = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"], text=True)
    return gpus_free(devices, applications, settings["allowed_gpus"])


def ledger_directory(ledger, failure_id):
    if not re.fullmatch(r"[a-f0-9]{64}", failure_id):
        raise ValueError("Invalid recovery attempt identity")
    return Path(ledger) / failure_id


def consumed(ledger, failure_id):
    return ledger_directory(ledger, failure_id).exists()


def reserve_attempt(ledger, candidate, permit, failure_bytes=None):
    directory = ledger_directory(ledger, candidate["failure_id"])
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    write_once(directory / "reservation.json", {
        "version": VERSION, "candidate": candidate, "permit": permit, "reserved_unix": time.time(),
    })
    source = Path(candidate["failure_path"])
    if failure_bytes is None:
        require_hash(source, candidate["failure_sha256"])
        failure_bytes = source.read_bytes()
    if hashlib.sha256(failure_bytes).hexdigest() != candidate["failure_sha256"]:
        raise ValueError("Failure evidence changed before reservation")
    with (directory / "failure-original.json").open("xb") as target:
        target.write(failure_bytes)
        target.flush()
        os.fsync(target.fileno())
    return directory


def consume_external(ledger, candidate, details):
    directory = ledger_directory(ledger, candidate["failure_id"])
    if directory.exists():
        existing = read(directory / "external-handled.json")
        if existing["candidate"] != candidate or existing["details"] != details:
            raise ValueError("Previously consumed failure has different external handling")
        return directory
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()
    write_once(directory / "external-handled.json", {
        "version": VERSION, "candidate": candidate, "details": details,
        "recorded_unix": time.time(), "automatic_recovery_performed": False,
    })
    with (directory / "failure-original.json").open("xb") as stream:
        stream.write(Path(candidate["failure_path"]).read_bytes())
        stream.flush()
        os.fsync(stream.fileno())
    return directory


def known_failure_names(ledger):
    result = {}
    if not Path(ledger).exists():
        return result
    for directory in Path(ledger).iterdir():
        if not directory.is_dir():
            continue
        markers = [directory / name for name in ("reservation.json", "external-handled.json")]
        marker = next((path for path in markers if path.is_file()), None)
        if marker is None:
            raise ValueError("Incomplete recovery reservation requires explicit inspection")
        candidate = read(marker)["candidate"]
        result[candidate["failure_name"]] = candidate["failure_sha256"]
    return result


def probe(settings, check_gpus=False):
    verify_run_identity(settings)
    run = Path(settings["run"])
    owners = owned_processes(settings)
    known = known_failure_names(settings["worker_ledger"])
    unhandled, errors = [], []
    for path in sorted(run.glob("failed*.json")):
        if path.name in known:
            if sha(path) != known[path.name]:
                errors.append(f"Preserved failure receipt mutated: {path.name}")
            continue
        try:
            unhandled.append(classify_failure(path, settings))
        except Exception as error:
            errors.append(f"{path.name}: {error}")
    if len(unhandled) > 1:
        errors.append("Multiple unhandled failures require explicit inspection")
    return {
        "complete": (run / "completed.json").exists(), "owned_processes": owners,
        "unhandled": unhandled, "consumed_failure_names": sorted(known),
        "classification_error": "; ".join(errors) if errors else None,
        "generation_sha256": generation_identity(run),
        "gpus_free": assigned_gpus_free(settings) if check_gpus and not owners else False,
        "gpu_check_performed": bool(check_gpus and not owners), "observed_unix": time.time(),
        "candidate_ack_present": len(unhandled) == 1 and (
            run / "acks" / f'checkpoint-{unhandled[0]["step"]}.json').is_file(),
    }


def route_probe(value):
    if value["classification_error"]:
        return "blocked"
    if value["owned_processes"]:
        return "observe-current"
    if value["complete"]:
        return "frozen-queue"
    if len(value["unhandled"]) == 1:
        return "candidate"
    return "blocked"


def storage_processes_ready(records, broker_pid):
    return (any(record["pid"] == broker_pid for record in records)
            and all(record["state"] not in {"D", "Z", "X"}
                    and "request_wait_answer" not in record.get("wchan", "") for record in records))


def exporter_records(settings):
    broker = process_record(settings["broker_pid"])
    argv = broker["argv"]
    if ("broker" not in argv or settings["bank_directory"] not in argv
            or str(Path(settings["run"]) / "snapshots") not in argv
            or not any(Path(value).name == "lc_checkpoint.py" for value in argv)):
        raise ValueError("Exporter PID no longer identifies the original checkpoint broker")
    records = {broker["pid"]: broker}
    processes = []
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            try:
                if path.stat().st_uid == os.getuid():
                    processes.append(process_record(path.name))
            except (FileNotFoundError, ProcessLookupError):
                continue
    changed = True
    while changed:
        changed = False
        for record in processes:
            if record["ppid"] in records and record["pid"] not in records:
                records[record["pid"]] = record
                changed = True
    return list(records.values())


@contextmanager
def try_bank_lock(path):
    import fcntl
    with Path(path).open("r") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield None
            return
        try:
            yield lock
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def commit_reserved_launch(directory, expected, permit, inspect_current, spawn):
    directory = Path(directory)
    if any((directory / name).exists() for name in ("launch-intent.json", "launched.json", "aborted.json")):
        raise FileExistsError("A recovery attempt may never be launched twice")
    try:
        current = inspect_current()
        if (not math.isfinite(permit["expires_unix"]) or time.time() >= permit["expires_unix"]
                or current["owned_processes"] or not current["gpus_free"]
                or current["generation_sha256"] != expected["generation_sha256"]
                or current["candidate"] != expected["candidate"]):
            raise ValueError("Final recovery recheck failed: active process, GPU, generation, or permit race")
        write_once(directory / "launch-intent.json", {"proof": expected, "permit": permit, "unix": time.time()})
        launched = spawn()
        write_once(directory / "launched.json", {"result": launched, "unix": time.time()})
        return launched
    except Exception as error:
        write_once(directory / "aborted.json", {
            "error": str(error), "unix": time.time(), "automatic_retry_for_this_failure": False,
        })
        raise


def load_queue(settings):
    require_hash(settings["queue_module"], settings["queue_module_sha256"])
    require_hash(settings["queue_plan"], settings["queue_plan_sha256"])
    spec = importlib.util.spec_from_file_location("lc_immutable_queue_r2", settings["queue_module"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, read(settings["queue_plan"])


def worker_rpc(settings, action, payload=None):
    import base64
    command = [
        "env", "CUDA_VISIBLE_DEVICES=-1", "bash", str(Path(settings["frozen_source"]) / "env.sh"),
        "python", settings["worker_entry"], "worker", "--config", settings["config_path"],
        "--action", action,
    ]
    if payload is not None:
        command += ["--payload", base64.b64encode(json.dumps(payload).encode()).decode()]
    value = subprocess.check_output(
        ["ssh", "-n", "-o", "BatchMode=yes", settings["worker_host"], shlex.join(command)], text=True)
    return json.loads(value)


def worker_heartbeat(settings):
    heartbeat = read(Path(settings["run"]) / "exporter-heartbeat.json")
    if (heartbeat.get("pid") != settings["broker_pid"] or heartbeat.get("writer_host") != settings["writer_host"]
            or heartbeat.get("source") != str(Path(settings["run"]) / "snapshots")
            or heartbeat.get("destination") != settings["bank_directory"]
            or time.time() - heartbeat.get("unix", 0) > 180):
        raise ValueError("Original exporter heartbeat is absent, stale, or changed; no restart/bypass")
    return heartbeat


def worker_action(settings, action, payload):
    if socket.gethostname() != settings["worker_host"] or os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("Recovery worker guards must be CPU-only on the assigned worker")
    if action == "probe":
        return probe(settings)
    if action == "seed-external":
        for seed in settings["external_handled"]:
            path = Path(settings["run"]) / seed["failure_name"]
            require_hash(path, seed["failure_sha256"])
            candidate = classify_failure(path, settings)
            consume_external(settings["worker_ledger"], candidate, seed["details"])
        return {"status": "seeded-parent-handled-only", "automatic_launches": 0}
    if action == "verify":
        current = probe(settings, check_gpus=True)
        if route_probe(current) != "candidate" or not current["gpus_free"]:
            raise ValueError("Learner/ranks/supervisor remain or assigned GPUs are not free")
        if current["unhandled"][0] != payload["candidate"]:
            raise ValueError("Failure generation changed before verification")
        proof = verify_local_candidate(settings, payload["candidate"])
        proof["exporter_heartbeat"] = worker_heartbeat(settings)
        return proof
    if action != "launch":
        raise ValueError("Unknown CPU recovery action")
    import fcntl
    ledger = Path(settings["worker_ledger"])
    ledger.mkdir(parents=True, exist_ok=True)
    with (ledger / "recovery.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        proof, permit = payload["proof"], payload["permit"]
        if (permit["failure_id"] != proof["candidate"]["failure_id"]
                or permit["checkpoint_identity"] != proof["checkpoint_identity"]
                or permit["proof_sha256"] != fingerprint(proof)
                or permit["configuration_sha256"] != fingerprint(settings)):
            raise ValueError("Recovery permit identifies another failure")
        directory = reserve_attempt(ledger, proof["candidate"], permit)

        def recheck():
            current = probe(settings, check_gpus=True)
            if current["classification_error"] or current["complete"]:
                raise ValueError("Run/failure state changed before launch")
            if current["unhandled"]:
                # The just-created reservation intentionally consumes this exact failure.
                raise ValueError("A new unhandled failure appeared during reservation")
            verified = verify_local_candidate(settings, proof["candidate"])
            if verified["checkpoint_identity"] != proof["checkpoint_identity"] or verified["ack_sha256"] != proof["ack_sha256"]:
                raise ValueError("Checkpoint or acknowledgment changed after durable verification")
            worker_heartbeat(settings)
            command = [
                "env", "CUDA_VISIBLE_DEVICES=-1", "bash", str(Path(settings["frozen_source"]) / "env.sh"),
                "python", settings["writer_entry"], "lease-status", "--config", settings["config_path"],
                "--permit", permit["permit_id"],
            ]
            lease = json.loads(subprocess.check_output(
                ["ssh", "-n", "-o", "BatchMode=yes", settings["writer_host"], shlex.join(command)], text=True))
            if lease.get("status") != "held" or lease.get("permit") != permit:
                raise ValueError("Writer/storage lease was lost; fail closed")
            # Recheck after hashing and the cross-node lease check, immediately before spawn.
            verify_run_identity(settings)
            if classify_failure(Path(proof["candidate"]["failure_path"]), settings) != proof["candidate"]:
                raise ValueError("Failure receipt changed at the final launch boundary")
            run = Path(settings["run"])
            name = f'checkpoint-{proof["candidate"]["step"]}'
            require_hash(run / "acks" / (name + ".json"), proof["ack_sha256"])
            require_hash(run / name / "complete.json", proof["checkpoint_identity"]["complete_sha256"])
            require_hash(run / "snapshots" / name / "complete.json",
                         proof["checkpoint_identity"]["complete_sha256"])
            return {
                "candidate": proof["candidate"], "generation_sha256": generation_identity(settings["run"]),
                "owned_processes": owned_processes(settings), "gpus_free": assigned_gpus_free(settings),
            }

        def spawn():
            stream = (directory / "supervisor.log").open("x", encoding="utf-8")
            try:
                child = subprocess.Popen(
                    ["bash", settings["launcher"]], cwd=settings["project"],
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"), stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            finally:
                stream.close()
            return {"supervisor_pid": child.pid, "process": process_record(child.pid),
                    "launcher_sha256": settings["launcher_sha256"], "resume_checkpoint": proof["candidate"]["step"]}

        return commit_reserved_launch(directory, proof, permit, recheck, spawn)


def durable_ack_proof(settings, proof):
    from lc_checkpoint import snapshot_identity, verify_ack

    step = proof["candidate"]["step"]
    directory = Path(settings["bank_directory"]) / f"checkpoint-{step}"
    durable_ack = read(directory / "bank-receipt.json")
    local_ack = read(Path(settings["broker_inbox"]) / f"checkpoint-{step}.ack.json")
    if durable_ack != proof["acknowledgment"] or local_ack != durable_ack:
        raise ValueError("Worker, exporter, and durable bank acknowledgments differ")
    verify_ack(directory, durable_ack)
    if snapshot_identity(directory) != proof["checkpoint_identity"]:
        raise ValueError("Durable COMPLETE snapshot differs from worker state")
    return {"status": "verified", "bank_directory": str(directory), "identity": proof["checkpoint_identity"]}


def lease_status(settings, permit_id):
    if not re.fullmatch(r"[a-f0-9]{32}", permit_id):
        raise ValueError("Invalid lease identity")
    permit = read(Path(settings["writer_ledger"]) / "permits" / f"{permit_id}.json")
    if time.time() >= permit["expires_unix"]:
        return {"status": "expired"}
    owner = process_record(permit["writer_pid"])
    if owner["start_ticks"] != permit["writer_start_ticks"]:
        return {"status": "owner-changed"}
    info = Path(settings["bank_lock"]).stat()
    lock_key = f"{os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x}:{info.st_ino}"
    held = any(
        "FLOCK" in line and "WRITE" in line and str(permit["writer_pid"]) in line.split()
        and lock_key in line.split() for line in Path("/proc/locks").read_text().splitlines()
    )
    return {"status": "held" if held else "released", "permit": permit}


def attempt_recovery(settings, candidate):
    import base64

    writer = Path(settings["writer_ledger"])
    records = exporter_records(settings)
    if not storage_processes_ready(records, settings["broker_pid"]):
        return {"status": "waiting-storage", "reason": "original exporter or child is in uninterruptible I/O"}
    with try_bank_lock(settings["bank_lock"]) as lock:
        if lock is None:
            return {"status": "waiting-storage", "reason": "common bank lock is still held; no restart"}
        records = exporter_records(settings)
        if not storage_processes_ready(records, settings["broker_pid"]):
            return {"status": "waiting-storage", "reason": "exporter changed after lock acquisition"}
        write_state(writer / "activity.json", {
            "phase": "verifying-worker-and-durable-state", "candidate": candidate, "unix": time.time(),
        })
        proof = worker_rpc(settings, "verify", {"candidate": candidate})
        durable = durable_ack_proof(settings, proof)
        if not storage_processes_ready(exporter_records(settings), settings["broker_pid"]):
            return {"status": "waiting-storage", "reason": "exporter entered I/O wait after verification"}
        writer_process = process_record(os.getpid())
        permit = {
            "permit_id": uuid.uuid4().hex, "failure_id": candidate["failure_id"],
            "writer_pid": os.getpid(), "writer_start_ticks": writer_process["start_ticks"],
            "issued_unix": time.time(), "expires_unix": time.time() + settings.get("permit_seconds", 180),
            "checkpoint_identity": proof["checkpoint_identity"], "durable": durable,
            "proof_sha256": fingerprint(proof), "configuration_sha256": fingerprint(settings),
        }
        reservation = reserve_attempt(writer / "attempts", candidate, permit,
                                      base64.b64decode(proof["failure_bytes_b64"]))
        write_once(writer / "permits" / (permit["permit_id"] + ".json"), permit)
        write_once(reservation / "verified-state.json", proof)
        try:
            result = worker_rpc(settings, "launch", {"proof": proof, "permit": permit})
            write_once(reservation / "worker-launch-result.json", result)
            return {"status": "launched-once", "candidate": candidate, "result": result}
        except Exception as error:
            write_once(reservation / "launch-uncertain-or-blocked.json", {
                "error": str(error), "unix": time.time(), "automatic_retry_for_this_failure": False,
            })
            raise


def inspect(settings):
    queue, plan = load_queue(settings)
    value = worker_rpc(settings, "probe")
    return {
        "version": VERSION, "mode": "read-only", "automatic_launch_invoked": False,
        "route": route_probe(value), "worker": value,
        "queue_state": read(Path(plan["writer_state"]) / "state.json"),
        "queue_plan_sha256": sha(settings["queue_plan"]), "queue_source_sha256": sha(settings["queue_module"]),
        "recovery_source_sha256": sha(Path(__file__)),
        "writer_attempts": sorted(path.name for path in (Path(settings["writer_ledger"]) / "attempts").glob("*")
                                  if path.is_dir()),
    }


def tick_recovery(settings, queue, plan):
    observed = worker_rpc(settings, "probe")
    route = route_probe(observed)
    result = {"status": route, "automatic_recovery_of_parent_handled_failure": False}
    if route == "observe-current":
        phase = "waiting-instruments-learner"
        facts = queue.rpc(plan, Path(settings["queue_plan"]), "status")
    elif route == "frozen-queue":
        phase, facts = queue.tick(plan, Path(settings["queue_plan"]))
    elif route == "candidate":
        candidate = observed["unhandled"][0]
        result = (attempt_recovery(settings, candidate) if observed["candidate_ack_present"] else
                  {"status": "waiting-verified-bank-ack", "candidate": candidate})
        phase = "recovery-" + result["status"]
        facts = queue.rpc(plan, Path(settings["queue_plan"]), "status")
    else:
        raise ValueError(observed["classification_error"]
                         or "Incomplete/exited learner has no new admissible TimeoutError; fail closed")
    return phase, facts, result


def drive(settings, config_path):
    import fcntl
    import traceback

    if socket.gethostname() != settings["writer_host"] or os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("Recovery controller is CPU-only on the existing designated writer")
    queue, plan = load_queue(settings)
    state, recovery_state = Path(plan["writer_state"]), Path(settings["writer_ledger"])
    recovery_state.mkdir(parents=True, exist_ok=True)
    with (state / "controller.lock").open("a") as control_lock:
        fcntl.flock(control_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        worker_rpc(settings, "seed-external")
        write_state(recovery_state / "controller.pid.json", {
            "pid": os.getpid(), "source_sha256": sha(Path(__file__)), "config_sha256": sha(config_path),
            "wrapped_queue_source_sha256": settings["queue_module_sha256"],
            "wrapped_queue_plan_sha256": settings["queue_plan_sha256"],
            "cuda_visible_devices": "-1", "started_unix": time.time(),
        })
        write_state(state / "controller.pid.json", read(recovery_state / "controller.pid.json"))
        config_identity, last = sha(config_path), None
        while True:
            try:
                require_hash(config_path, config_identity)
                load_queue(settings)
                phase, facts, result = tick_recovery(settings, queue, plan)
                status = {
                    "status": "done" if phase == "all-complete-and-banked" else "active",
                    "phase": phase, "facts": facts, "recovery": result,
                    "pid": os.getpid(), "updated_unix": time.time(), "version": VERSION,
                }
                write_state(state / "state.json", status)
                write_state(recovery_state / "state.json", status)
                if phase != last:
                    print(json.dumps(status), flush=True)
                    last = phase
                if status["status"] == "done":
                    write_state(state / "completed.json", status)
                    return
            except Exception as error:
                status = {"status": "blocked", "error": str(error), "traceback": traceback.format_exc(),
                          "pid": os.getpid(), "updated_unix": time.time(), "version": VERSION}
                write_state(state / "state.json", status)
                write_state(recovery_state / "state.json", status)
                print(json.dumps(status), flush=True)
                return
            time.sleep(settings.get("poll_seconds", 30))


def main():
    import base64

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("drive", "inspect", "worker", "lease-status"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--action")
    parser.add_argument("--payload")
    parser.add_argument("--permit")
    args = parser.parse_args()
    settings = read(args.config)
    sys.path.insert(0, settings["frozen_source"])
    if settings.get("version") != VERSION:
        raise ValueError("Unexpected recovery configuration version")
    if args.mode == "drive":
        drive(settings, args.config)
    elif args.mode == "inspect":
        print(json.dumps(inspect(settings)), flush=True)
    elif args.mode == "lease-status":
        print(json.dumps(lease_status(settings, args.permit)), flush=True)
    else:
        payload = json.loads(base64.b64decode(args.payload)) if args.payload else None
        print(json.dumps(worker_action(settings, args.action, payload)), flush=True)


if __name__ == "__main__":
    main()
