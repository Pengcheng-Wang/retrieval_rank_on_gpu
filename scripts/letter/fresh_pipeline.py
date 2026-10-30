"""Task-bound node0 queue: wait on CPU, admit final RQ bundles, gate, train and bank."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

from fresh_ids import (
    FROZEN_BANK, KIND, TRACK, check_gate, file_sha256, load_execution, read_json,
    require_canonical, select_resume, stage_real, validate_audit, validate_scope,
    validate_training, verify_evaluation, verify_stage, write_json,
)
from protocol import require_hash


GPU_INDEXES = (2, 3)


class GpuOccupancyError(RuntimeError):
    """Assigned GPU ownership could not be proved without touching a foreign process."""


def queue_cells(spec: dict) -> list[dict]:
    cells = []
    for seed in spec["training"]["seeds"]:
        for dataset, values in spec["datasets"].items():
            for variant in values["variants"]:
                for temperature in spec["training"]["temperatures"][variant]:
                    cells.append({
                        "dataset": dataset, "variant": variant, "seed": seed,
                        "temperature": temperature,
                        "rq_run_id": f"openllama-v2-rq-{dataset}-{variant}-s42",
                        "stage_id": f"{dataset}-{variant}-ids-v1",
                        "run_id": f"openllama-v2-fresh-t5-{dataset}-{variant}-s{seed}-t{int(temperature * 10):02d}-v1",
                    })
    return cells


def find_bundle(roots: list[Path], run_id: str) -> Path | None:
    found = []
    for root in roots:
        for path in sorted(root.glob(run_id + "-completed-*")):
            if (path / "completed.json").is_file():
                if path.is_symlink():
                    raise ValueError("Final bundle may not redirect outside its incoming root")
                found.append(path)
    if len(found) > 1:
        raise ValueError("Ambiguous final RQ bundles; never select by a downstream result")
    return found[0] if found else None


def next_evaluation(run: Path) -> Path:
    for attempt in range(1, 1000):
        path = run / f"evaluation-attempt-{attempt:03d}"
        if not path.exists():
            return path
    raise ValueError("Evaluation attempt limit reached; existing partial predictions were preserved")


def torchrun_prefix(identity: str) -> list[str]:
    return [
        "torchrun", "--rdzv_backend=c10d", "--rdzv_endpoint=localhost:0",
        "--nproc_per_node=2", f"--rdzv_id={identity}-{uuid.uuid4().hex}",
    ]


def require_real_stage(stage: dict) -> None:
    lineage = stage["experiment_lineage"]
    if (lineage.get("synthetic_fixture") is not False or lineage.get("kind") != KIND
            or lineage.get("track_id") != TRACK or lineage.get("rq", {}).get("final_epoch") != 20000
            or not lineage.get("durable_proof")):
        raise ValueError("Synthetic/fixture or unfinished/undurable RQ stage cannot launch real T5")


def cell_is_banked(work: Path, cell: dict) -> bool:
    path = work / "acks" / f"{cell['run_id']}.json"
    if not path.exists():
        return False
    ack = read_json(path)
    result = work / "results" / cell["run_id"] / "result.json"
    if (ack.get("status") != "banked" or ack.get("run_id") != cell["run_id"]
            or not result.is_file() or ack.get("result_sha256") != file_sha256(result)):
        raise ValueError("Durable cell acknowledgment does not bind this immutable result")
    return True


def read_process_identity(pid: int, proc_root: Path = Path("/proc")) -> dict | None:
    try:
        text = (proc_root / str(pid) / "stat").read_text()
        close = text.rfind(")")
        fields = text[close + 2:].split()
        return {
            "pid": int(pid), "ppid": int(fields[1]), "pgid": int(fields[2]),
            "session_id": int(fields[3]), "start_time_ticks": int(fields[19]),
        }
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return None


def capture_process_identity(
    pid: int, label: str, gpu_uuids: list[str],
    process_reader: Callable[[int], dict | None] = read_process_identity,
) -> dict:
    identity = process_reader(pid)
    if identity is None:
        raise GpuOccupancyError(f"Could not record process identity for {label} PID {pid}")
    return {
        **identity, "label": label, "gpu_uuids": sorted(gpu_uuids),
        "members": {str(pid): identity["start_time_ticks"]},
    }


def _integer(value: str) -> int | None:
    try:
        return int(value.strip())
    except (AttributeError, ValueError):
        return None


def capture_gpu_snapshot(
    indexes: tuple[int, ...] = GPU_INDEXES,
    runner: Callable = subprocess.run, clock: Callable[[], float] = time.time,
) -> dict:
    gpu_result = runner(
        [
            "nvidia-smi", "--query-gpu=index,uuid,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True, capture_output=True, text=True, timeout=30,
    )
    client_result = runner(
        [
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True, capture_output=True, text=True, timeout=30,
    )
    gpus = {}
    for line in gpu_result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 3:
            continue
        index, memory = _integer(parts[0]), _integer(parts[2])
        if index in indexes:
            gpus[str(index)] = {"uuid": parts[1], "memory_used_mib": memory}
    clients = []
    for line in client_result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) != 4:
            continue
        clients.append({
            "gpu_uuid": parts[0], "pid": _integer(parts[1]),
            "process_name": parts[2], "used_memory_mib": _integer(parts[3]),
        })
    return {"timestamp": clock(), "gpus": gpus, "clients": clients}


def _descends_from(
    actual: dict, ancestor_pid: int, process_reader: Callable[[int], dict | None],
) -> bool:
    seen = set()
    current = actual
    while current is not None and current["pid"] not in seen:
        if current["pid"] == ancestor_pid:
            return True
        seen.add(current["pid"])
        if current["ppid"] <= 1:
            return False
        current = process_reader(current["ppid"])
    return False


def _matching_owner(
    pid: int, gpu_uuid: str, owners: list[dict],
    process_reader: Callable[[int], dict | None],
) -> dict | None:
    actual = process_reader(pid)
    if actual is None:
        return None
    for owner in owners:
        if gpu_uuid not in owner["gpu_uuids"]:
            continue
        member_start = owner.get("members", {}).get(str(pid))
        if (member_start == actual["start_time_ticks"]
                and actual["pgid"] == owner["pgid"]
                and actual["session_id"] == owner["session_id"]):
            return owner
        leader = process_reader(owner["pid"])
        if (leader is None or leader["start_time_ticks"] != owner["start_time_ticks"]
                or leader["pgid"] != owner["pgid"]
                or leader["session_id"] != owner["session_id"]):
            continue
        if (actual["session_id"] == owner["session_id"]
                and _descends_from(actual, owner["pid"], process_reader)):
            owner.setdefault("members", {})[str(pid)] = actual["start_time_ticks"]
            return owner
    return None


def classify_gpu_snapshot(
    snapshot: dict, baseline: dict, owners: list[dict],
    process_reader: Callable[[int], dict | None] = read_process_identity,
) -> dict:
    expected_uuids = baseline["gpu_uuids"]
    for index in map(str, baseline["gpu_indexes"]):
        gpu = snapshot.get("gpus", {}).get(index)
        if (gpu is None or gpu.get("uuid") != expected_uuids[index]
                or not isinstance(gpu.get("memory_used_mib"), int)):
            return {
                "status": "foreign", "owned_clients": [], "owned_owner_pids": [],
                "foreign_clients": [{"reason": "missing_or_changed_gpu_identity", "gpu_index": index}],
            }
    assigned = set(expected_uuids.values())
    owned_clients, foreign_clients, owned_pids = [], [], set()
    for value in snapshot.get("clients", []):
        gpu_uuid = value.get("gpu_uuid")
        if gpu_uuid not in assigned:
            if gpu_uuid in {None, "", "N/A"}:
                foreign_clients.append({**value, "reason": "unknown_gpu_uuid"})
            continue
        pid = value.get("pid")
        owner = _matching_owner(pid, gpu_uuid, owners, process_reader) if isinstance(pid, int) else None
        if owner is None:
            foreign_clients.append({**value, "reason": "unowned_or_reused_process"})
        else:
            owned_clients.append(value)
            owned_pids.add(owner["pid"])
    if foreign_clients:
        status = "foreign"
    elif owned_clients:
        status = "owned_live"
    else:
        memory = {
            index: snapshot["gpus"][index]["memory_used_mib"]
            for index in map(str, baseline["gpu_indexes"])
        }
        status = "clean" if memory == baseline["memory_used_mib"] else "residual_memory"
    return {
        "status": status, "owned_clients": owned_clients,
        "owned_owner_pids": sorted(owned_pids), "foreign_clients": foreign_clients,
    }


def _baseline_for(snapshot: dict, expected_indexes: tuple[int, ...]) -> dict:
    indexes = [int(index) for index in sorted(snapshot.get("gpus", {}), key=int)]
    if indexes != list(expected_indexes):
        raise GpuOccupancyError("Assigned GPU UUID/memory snapshot is incomplete")
    gpus = snapshot["gpus"]
    if any(
        not isinstance(gpus[str(index)].get("memory_used_mib"), int)
        for index in expected_indexes
    ):
        raise GpuOccupancyError("Assigned GPU memory accounting is unavailable")
    return {
        "gpu_indexes": list(expected_indexes),
        "gpu_uuids": {
            str(index): gpus[str(index)]["uuid"] for index in expected_indexes
        },
        "memory_used_mib": {
            str(index): gpus[str(index)]["memory_used_mib"] for index in expected_indexes
        },
    }


def wait_for_gpu_quiescence(
    snapshotter: Callable[[], dict], baseline_path: Path, evidence_path: Path, *,
    owners: tuple[dict, ...] | list[dict], drain_owned: Callable[[dict], object],
    quiet_samples: int = 3, quiet_seconds: float = 2.0, timeout_seconds: float = 120.0,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    process_reader: Callable[[int], dict | None] = read_process_identity,
    gpu_indexes: tuple[int, ...] = GPU_INDEXES,
) -> dict:
    if quiet_samples < 2 or quiet_seconds <= 0 or timeout_seconds < quiet_seconds:
        raise ValueError("GPU quiescence requires a bounded multi-sample quiet window")
    owners = list(owners)
    baseline = read_json(baseline_path) if baseline_path.is_file() else None
    baseline_candidate = None
    started, clean_started, clean_count = monotonic(), None, 0
    samples, drained = [], set()
    interval = quiet_seconds / (quiet_samples - 1)
    while True:
        observed = monotonic()
        try:
            current = snapshotter()
        except Exception as error:
            write_json(evidence_path, {
                "status": "snapshot_failed_prelaunch",
                "error": f"{type(error).__name__}: {error}", "samples": samples,
            })
            raise GpuOccupancyError(
                "GPU ownership snapshot failed before launch; no GPU process was started"
            ) from error
        provisional = baseline
        if provisional is None:
            current_candidate = _baseline_for(current, gpu_indexes)
            if baseline_candidate != current_candidate:
                baseline_candidate = current_candidate
                clean_started, clean_count = None, 0
            provisional = baseline_candidate
        classification = classify_gpu_snapshot(
            current, provisional, owners, process_reader=process_reader,
        )
        samples.append({
            "observed_monotonic": observed, "snapshot": current,
            "classification": classification,
        })
        status = classification["status"]
        if status == "foreign":
            write_json(evidence_path, {
                "status": "blocked_foreign", "samples": samples,
                "drained_owned_pids": sorted(drained),
            })
            raise GpuOccupancyError("Assigned GPUs contain a foreign or unidentifiable client")
        if status == "owned_live":
            for pid in classification["owned_owner_pids"]:
                if pid not in drained:
                    drain_owned(next(owner for owner in owners if owner["pid"] == pid))
                    drained.add(pid)
            clean_started, clean_count = None, 0
        elif status == "clean":
            if clean_started is None:
                clean_started, clean_count = observed, 1
            else:
                clean_count += 1
            quiet = observed - clean_started
            if clean_count >= quiet_samples and quiet >= quiet_seconds:
                if baseline is None:
                    baseline = baseline_candidate
                    write_json(baseline_path, baseline)
                write_json(evidence_path, {
                    "status": "launchable", "baseline": baseline,
                    "clean_samples": clean_count, "quiet_seconds": quiet,
                    "samples": samples, "drained_owned_pids": sorted(drained),
                })
                return current
        else:
            clean_started, clean_count = None, 0
        if monotonic() - started >= timeout_seconds:
            write_json(evidence_path, {
                "status": "blocked_ambiguous", "baseline": baseline,
                "samples": samples, "drained_owned_pids": sorted(drained),
            })
            raise GpuOccupancyError(
                "Assigned GPUs did not return to their recorded idle baseline within the bounded wait"
            )
        sleep(interval)


def monitor_gpu_process(
    process: subprocess.Popen, owner: dict, baseline: dict, evidence_path: Path, *,
    snapshotter: Callable[[], dict] = capture_gpu_snapshot,
    drain_owned: Callable[[dict], object],
    other_owners: tuple[dict, ...] | list[dict] = (),
    process_reader: Callable[[int], dict | None] = read_process_identity,
    exporter: subprocess.Popen | None = None,
    completed: Callable[[], bool] | None = None,
    sleep: Callable[[float], None] = time.sleep, poll_seconds: float = 1.0,
) -> int:
    if (exporter is None) != (completed is None):
        raise ValueError("Exporter supervision requires both the process and completion validator")
    samples = []
    while True:
        if exporter is not None:
            error = _exporter_error(exporter, completed)
            if error:
                receipt = {
                    "status": "exporter_failed", "exporter_pid": exporter.pid,
                    "exporter_returncode": exporter.poll(), "training_pid": process.pid,
                    "error": error, "drain": "requested", "unix": time.time(),
                }
                write_json(evidence_path.parent / "supervision.json", receipt)
                drain = drain_owned(owner)
                write_json(
                    evidence_path.parent / "supervision.json",
                    {**receipt, "drain": drain, "training_returncode": process.poll()},
                )
                raise ExporterFailure(error)
        try:
            current = snapshotter()
        except Exception as error:
            drain_owned(owner)
            write_json(evidence_path, {
                "status": "snapshot_failed_after_spawn", "owner": owner,
                "error": f"{type(error).__name__}: {error}", "samples": samples,
            })
            raise GpuOccupancyError(
                "GPU ownership snapshot failed after spawn; the owned process group was drained"
            ) from error
        owners = [owner, *other_owners]
        classification = classify_gpu_snapshot(current, baseline, owners, process_reader=process_reader)
        samples.append({"snapshot": current, "classification": classification})
        if classification["status"] == "foreign":
            drain_owned(owner)
            write_json(evidence_path, {
                "status": "foreign_after_spawn", "owner": owner, "samples": samples,
            })
            raise GpuOccupancyError(
                "A foreign GPU client appeared after spawn; the owned process group was drained"
            )
        unexpected = sorted(
            set(classification["owned_owner_pids"]) - {owner["pid"]}
        )
        if unexpected:
            for pid in unexpected:
                drain_owned(next(value for value in other_owners if value["pid"] == pid))
            drain_owned(owner)
            write_json(evidence_path, {
                "status": "owned_lifecycle_violation", "owner": owner,
                "unexpected_owned_pids": unexpected, "samples": samples,
            })
            raise GpuOccupancyError(
                "An unexpected owned GPU client appeared after spawn; lifecycle groups were drained"
            )
        code = process.poll()
        if code is not None:
            write_json(evidence_path, {
                "status": "process_exited", "returncode": code,
                "owner": owner, "samples": samples,
            })
            return code
        sleep(poll_seconds)


class ExporterFailure(RuntimeError):
    """The checkpoint exporter can no longer protect an owned native fit."""


def _exporter_error(exporter: subprocess.Popen, completed: Callable[[], bool]) -> str | None:
    code = exporter.poll()
    if code is None:
        return None
    try:
        valid_completion = completed()
    except Exception as error:
        return f"Checkpoint exporter exited {code}; invalid training completion: {error}"
    if code == 0 and valid_completion:
        return None
    return f"Checkpoint exporter unexpectedly exited {code}; validated training completion={valid_completion}"


def wait_for_exporter(exporter: subprocess.Popen, log: Path, completed: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 60
    while True:
        error = _exporter_error(exporter, completed)
        if error:
            raise ExporterFailure(error)
        if exporter.poll() is not None:
            return
        with log.open(encoding="utf-8") as stream:
            if "SINGLE_WRITER_EXPORTER_READY" in stream.read(4096):
                return
        if time.monotonic() >= deadline:
            raise ExporterFailure("Checkpoint exporter did not become ready within 60 seconds")
        time.sleep(.2)


def _drain_owned_training(training: subprocess.Popen, identity: dict | None = None) -> str:
    exited = training.poll() is not None
    current = read_process_identity(training.pid) if identity is not None else None
    live_members = []
    if identity is not None:
        for pid, start_time in identity.get("members", {}).items():
            member = read_process_identity(int(pid))
            if (member is not None and member["start_time_ticks"] == start_time
                    and member["pgid"] == identity["pgid"]
                    and member["session_id"] == identity["session_id"]):
                live_members.append(member)
    if exited and not live_members:
        return "already_exited"
    if identity is not None and (
        not exited and (
            current is None
            or current["start_time_ticks"] != identity["start_time_ticks"]
            or current["pgid"] != identity["pgid"]
            or current["session_id"] != identity["session_id"]
        )
    ):
        raise ExporterFailure("Refusing to signal a reused or changed training process identity")
    if os.name != "posix" or (not exited and os.getpgid(training.pid) != training.pid):
        raise ExporterFailure("Refusing to signal a training launcher outside its owned process session")
    # Native ranks handle SIGTERM by sealing state at a step boundary. The banker is not in this session.
    os.killpg(identity["pgid"] if identity is not None else training.pid, signal.SIGTERM)
    if not exited:
        try:
            training.wait(timeout=120)
        except subprocess.TimeoutExpired as error:
            raise ExporterFailure(
                f"Graceful drain requested for owned training PID {training.pid}, but exit is still pending"
            ) from error
    deadline = time.monotonic() + 120
    while True:
        remaining = []
        if identity is not None:
            for pid, start_time in identity.get("members", {}).items():
                member = read_process_identity(int(pid))
                if (member is not None and member["start_time_ticks"] == start_time
                        and member["pgid"] == identity["pgid"]
                        and member["session_id"] == identity["session_id"]):
                    remaining.append(pid)
        try:
            os.killpg(identity["pgid"] if identity is not None else training.pid, 0)
            group_exists = True
        except ProcessLookupError:
            group_exists = False
        if not remaining and not group_exists:
            break
        if time.monotonic() >= deadline:
            raise ExporterFailure(
                f"Owned GPU process group remains live after graceful drain: {remaining}"
            )
        time.sleep(.2)
    return "completed"


def wait_for_training(training: subprocess.Popen, exporter: subprocess.Popen,
                      completed: Callable[[], bool], evidence: Path) -> int:
    while True:
        error = _exporter_error(exporter, completed)
        if error:
            receipt = {
                "status": "exporter_failed", "exporter_pid": exporter.pid,
                "exporter_returncode": exporter.poll(), "training_pid": training.pid,
                "error": error, "drain": "requested", "unix": time.time(),
            }
            try:
                write_json(evidence / "supervision.json", receipt)
            finally:
                drain = _drain_owned_training(training)
            write_json(evidence / "supervision.json", {
                **receipt, "drain": drain, "training_returncode": training.poll(),
            })
            raise ExporterFailure(error)
        code = training.poll()
        if code is not None:
            return code
        try:
            training.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


class Runtime:
    def __init__(
        self, binding: Path, project: Path, control: Path, work: Path,
        gpu_indexes: tuple[int, int] = GPU_INDEXES,
    ):
        if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
            raise RuntimeError("The fresh-ID controller must be CPU-only; GPU children receive explicit leases")
        if (len(gpu_indexes) != 2 or len(set(gpu_indexes)) != 2
                or any(type(index) is not int or index < 0 for index in gpu_indexes)):
            raise ValueError("Select exactly two distinct physical GPU indexes")
        self.binding, self.project, self.control = binding, project, control
        self.gpu_indexes = tuple(gpu_indexes)
        self.rq_verifier = None
        self.root, self.destination = validate_scope(binding)
        if (not work.is_absolute() or work.is_symlink() or work.resolve().parent != self.root
                or not re.fullmatch(r"fresh-t5-[A-Za-z0-9][A-Za-z0-9_.-]*", work.name)
                or work.resolve().is_relative_to(control.resolve())
                or control.resolve().is_relative_to(work.resolve())):
            raise ValueError("Select a distinct, absolute, direct-child fresh-t5 work version inside the owned root")
        if project.resolve() != self.root / "project" or not control.resolve().is_relative_to(self.root):
            raise ValueError("Only this allocation's frozen environment and private control source are allowed")
        self.env = project / "scripts" / "letter" / "env.sh"
        require_canonical(self.env, "959317ce0565be80c55cd968ce7d62d5a29a50a7fbdb7abd1792c62ed9f4c64b")
        for name, digest in FROZEN_BANK.items():
            require_canonical(control / "scripts" / "letter" / name, digest)
        self.execution = control / "configs" / "letter-fresh-id-execution-v1.json"
        self.manifest = control / "configs" / "letter-reproduction-v1.json"
        self.method = control / "configs" / "letter-openllama-v2-method-v1.json"
        self.tokenizer_recipe = control / "configs" / "letter-tokenizer-reconstruction-v1.json"
        self.spec = load_execution(self.execution, self.manifest, self.method, self.tokenizer_recipe)
        self.work = work.resolve()
        self.work.mkdir(exist_ok=True)
        self.bank_root = self.destination / "openllama-v2" / self.work.name
        self.scripts = control / "scripts" / "letter"
        self.gpu_baseline = self.work / "gpu-idle-baseline.json"
        self._owned_gpu_processes = {}

    def command(self, argv: list, seed: int = 42, gpu: bool = False) -> tuple[list[str], dict]:
        assigned = ",".join(map(str, getattr(self, "gpu_indexes", GPU_INDEXES)))
        visible = assigned if gpu else ""
        command = [
            "bash", str(self.env), "env", f"CUDA_VISIBLE_DEVICES={visible}",
            "PYTHONDONTWRITEBYTECODE=1",
        ] + list(map(str, argv))
        # env.sh validates the physical allocation before the inner env command hides or exposes CUDA.
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": assigned, "LETTER_SEED": str(seed)}
        return command, env

    def invoke(self, argv: list, label: str, seed: int = 42,
               exporter: subprocess.Popen | None = None,
               completed: Callable[[], bool] | None = None, gpu: bool = False) -> None:
        if (exporter is None) != (completed is None):
            raise ValueError("Exporter supervision requires both the process and completion validator")
        directory = self.work / "commands" / f"{label}-{time.time_ns()}"
        directory.mkdir(parents=True, exist_ok=False)
        gpu_indexes = getattr(self, "gpu_indexes", GPU_INDEXES)
        prelaunch = None
        if gpu:
            prelaunch = wait_for_gpu_quiescence(
                lambda: capture_gpu_snapshot(gpu_indexes),
                self.gpu_baseline, directory / "gpu-prelaunch.json",
                owners=[entry[1] for entry in self._owned_gpu_processes.values()],
                drain_owned=self._drain_gpu_owner,
                gpu_indexes=gpu_indexes,
            )
        command, env = self.command(argv, seed, gpu=gpu)
        with (directory / "output.log").open("x", encoding="utf-8") as log:
            if exporter is not None:
                error = _exporter_error(exporter, completed)
                if error:
                    raise ExporterFailure(error)
            process = subprocess.Popen(command, cwd=self.root, env=env, stdout=log, stderr=log,
                                       start_new_session=gpu)
            identity = None
            if gpu:
                gpu_uuids = [
                    prelaunch["gpus"][str(index)]["uuid"] for index in gpu_indexes
                ]
                identity = self._record_owned_process(process, label, gpu_uuids)
            write_json(directory / "command.json", {
                "argv": command, "pid": process.pid,
                "cuda_visible_devices":
                    ",".join(map(str, gpu_indexes)) if gpu else "",
                "letter_seed": seed, "started_unix": time.time(),
                "supervised_exporter_pid": exporter.pid if exporter is not None else None,
                "owned_training_session": gpu,
                "process_identity": identity,
            })
            if gpu:
                baseline = read_json(self.gpu_baseline)
                try:
                    code = monitor_gpu_process(
                        process, identity, baseline, directory / "gpu-monitor.json",
                        drain_owned=self._drain_gpu_owner,
                        other_owners=[
                            entry[1] for pid, entry in self._owned_gpu_processes.items()
                            if pid != process.pid
                        ],
                        exporter=exporter, completed=completed,
                    )
                except GpuOccupancyError:
                    raise
                else:
                    wait_for_gpu_quiescence(
                        lambda: capture_gpu_snapshot(gpu_indexes),
                        self.gpu_baseline, directory / "gpu-release.json",
                        owners=[
                            entry[1] for entry in self._owned_gpu_processes.values()
                        ],
                        drain_owned=self._drain_gpu_owner,
                        gpu_indexes=gpu_indexes,
                    )
                    self._owned_gpu_processes.pop(process.pid, None)
            else:
                code = process.wait()
        if code:
            raise RuntimeError(f"{label} exited {code}; preserved {directory}")

    def _drain_gpu_owner(self, owner: dict) -> str:
        entry = self._owned_gpu_processes.get(owner["pid"])
        if entry is None or entry[1]["start_time_ticks"] != owner["start_time_ticks"]:
            raise GpuOccupancyError("Refusing to signal an unrecorded or reused GPU process identity")
        return _drain_owned_training(entry[0], owner)

    def _record_owned_process(
        self, process: subprocess.Popen, label: str, gpu_uuids: list[str],
    ) -> dict:
        try:
            identity = capture_process_identity(process.pid, label, gpu_uuids)
        except Exception:
            if process.poll() is None:
                _drain_owned_training(process)
            raise
        self._owned_gpu_processes[process.pid] = (process, identity)
        return identity

    def bank(self, source: Path, relative: str) -> None:
        self.invoke([
            "python", self.scripts / "bank.py", "--source", source,
            "--destination", self.bank_root / relative,
            "--lock-file", self.root / "bank.lock", "--writer-host", "node-0",
        ], "bank")

    def audit(self, stage: Path, output: Path) -> None:
        if not (output / "audit.json").exists():
            if output.exists() and any(output.iterdir()):
                raise ValueError("Interrupted audit directory preserved; use a new admitted stage attempt")
            self.invoke([
                "python", stage / "project" / "scripts" / "letter" / "prepare.py",
                "--upstream", stage / "upstream", "--manifest", stage / "manifest.json",
                "--dataset", verify_stage(stage)["dataset"], "--output", output,
            ], "full-audit")
        validate_audit(stage, output)

    def gate(self, stage: Path, audit: Path, gate: Path) -> dict:
        staged = verify_stage(stage)
        self.audit(stage, audit)
        if (gate / "gate.json").is_file():
            result = read_json(gate / "gate.json")
            if (result["status"] != "pass" or result["stage_sha256"] != file_sha256(stage / "stage.json")
                    or result["audit_sha256"] != file_sha256(audit / "audit.json")):
                raise ValueError("Stale or cross-index gate receipt")
            for part, digest in result["checkpoint_receipts"].items():
                from protocol import require_hash, verify_checkpoint
                require_hash(gate / part / "checkpoint-5" / "complete.json", digest)
                verify_checkpoint(gate / part / "checkpoint-5")
                validate_training(gate / part, stage, audit, 42, 1.0, smoke=True)
            return result
        if gate.exists():
            raise FileExistsError("Interrupted native gate preserved; choose a new gate attempt")
        gate.mkdir(parents=True)
        common = [
            "--upstream", stage / "upstream", "--manifest", stage / "manifest.json",
            "--audit", audit, "--dataset", staged["dataset"], "--seed", 42, "--temperature", 1.0,
            "--learning-rate", 0.0005, "--epochs", 200,
            "--batch-size", 16, "--effective-batch", 64, "--smoke",
        ]
        for name, output, extra in (
            ("continuous", "continuous", []),
            ("stopped", "resumed", ["--stop-after-steps", 3]),
            ("resumed", "resumed", ["--resume", "latest"]),
        ):
            if name == "resumed":
                validate_training(gate / output, stage, audit, 42, 1.0, smoke=True)
                extra = ["--resume", select_resume(gate / output)]
            self.invoke(
                torchrun_prefix(name) + [
                    stage / "project" / "scripts" / "letter" / "trace_train.py",
                    "--trace-dir", gate / "traces" / name, "--output", gate / output,
                ] + common + extra, f"gate-{name}",
                gpu=True,
            )
        self.invoke([
            "python", self.scripts / "fresh_ids.py", "check-gate",
            "--stage", stage, "--audit", audit, "--gate", gate,
        ], "gate-proof", gpu=True)
        return read_json(gate / "gate.json")

    def admit(self, cell: dict, bundle: Path) -> Path:
        stage = self.work / "stages" / cell["stage_id"]
        if not stage.exists():
            stage_real(SimpleNamespace(
                binding=self.binding, project=self.control, upstream=self.root / "upstream",
                manifest=self.manifest, execution=self.execution, method_spec=self.method,
                tokenizer_recipe=self.tokenizer_recipe, dataset=cell["dataset"], variant=cell["variant"],
                bundle=bundle, durable_bundle=self.destination / "openllama-v2" / "tokenizers" / bundle.name,
                semantic=self.root / "embeddings-openllama-v2-v1" / cell["dataset"],
                teacher=self.root / "incoming" / f"{cell['dataset']}-cf-bce-s42", output=stage,
            ), rq_verifier=self.rq_verifier)
        staged = verify_stage(stage)
        require_real_stage(staged)
        if (staged["experiment_lineage"]["execution_sha256"] != file_sha256(self.execution)
                or staged["experiment_lineage"]["rq"]["rq_run_id"] != cell["rq_run_id"]
                or staged["experiment_lineage"]["rq"]["bundle_completed_sha256"]
                != file_sha256(bundle / "completed.json")):
            raise ValueError("Existing stage is bound to another frozen RQ/execution")
        return stage

    def train(self, cell: dict, stage: Path, audit: Path, output: Path) -> None:
        import fcntl
        from bank import copy_artifact
        from bank_watch import checkpoint_artifacts

        require_real_stage(verify_stage(stage))
        gpu_indexes = getattr(self, "gpu_indexes", GPU_INDEXES)
        def completed() -> bool:
            from evaluate import verify_model_binding

            if not (output / "completed.json").is_file():
                return False
            run = validate_training(output, stage, audit, cell["seed"], cell["temperature"])
            verify_model_binding(output, read_json(output / "completed.json"), run,
                                 cell["dataset"], run["inputs"])
            return True

        if completed():
            return
        watcher = [
            "python", self.scripts / "bank_watch.py", "--runs", output,
            "--staging", self.work / "checkpoint-snapshots",
            "--destination", self.bank_root / "checkpoints", "--lock-file", self.root / "bank.lock",
            "--writer-host", "node-0", "--interval", 120,
        ]
        log_dir = self.work / "commands" / f"checkpoint-watcher-{cell['run_id']}-{time.time_ns()}"
        log_dir.mkdir(parents=True)
        idle = wait_for_gpu_quiescence(
            lambda: capture_gpu_snapshot(gpu_indexes),
            self.gpu_baseline, log_dir / "gpu-before-banker.json",
            owners=[entry[1] for entry in self._owned_gpu_processes.values()],
            drain_owned=self._drain_gpu_owner,
            gpu_indexes=gpu_indexes,
        )
        gpu_uuids = [idle["gpus"][str(index)]["uuid"] for index in gpu_indexes]
        command, env = self.command(watcher, gpu=False)
        with (log_dir / "output.log").open("x", encoding="utf-8") as log:
            banker = subprocess.Popen(
                command, cwd=self.root, env=env, stdout=log, stderr=log, start_new_session=True,
            )
            banker_identity = self._record_owned_process(
                banker, "checkpoint-watcher", gpu_uuids,
            )
            write_json(log_dir / "command.json", {
                "pid": banker.pid, "argv": command, "cuda_visible_devices": "",
                "process_identity": banker_identity,
            })
            try:
                wait_for_exporter(banker, log_dir / "output.log", completed)
                if completed():
                    return
                for attempt in range(self.spec["queue"]["max_invocation_failures"]):
                    extra = []
                    if output.exists() and any(output.iterdir()):
                        validate_training(output, stage, audit, cell["seed"], cell["temperature"])
                        extra = ["--resume", select_resume(output)]
                    try:
                        self.invoke(torchrun_prefix(cell["run_id"]) + [
                            stage / "project" / "scripts" / "letter" / "run.py",
                            "--upstream", stage / "upstream", "--manifest", stage / "manifest.json",
                            "--audit", audit, "--dataset", cell["dataset"], "--output", output,
                            "--seed", cell["seed"], "--temperature", cell["temperature"],
                            "--learning-rate", 0.0005, "--epochs", 200, "--effective-batch", 1024,
                            "--batch-size", self.spec["datasets"][cell["dataset"]]["micro_batch_size"],
                        ] + extra, "train-" + cell["run_id"], cell["seed"],
                            exporter=banker, completed=completed, gpu=True)
                    except ExporterFailure:
                        raise
                    except GpuOccupancyError:
                        raise
                    except RuntimeError:
                        if attempt + 1 == self.spec["queue"]["max_invocation_failures"]:
                            raise
                    if completed():
                        return
                    time.sleep(self.spec["queue"]["poll_seconds"])
                raise RuntimeError("Native T5 remains incomplete after bounded sealed-state retries")
            finally:
                # Holding the same lock guarantees we never stop our banker during durable IO.
                with (self.root / "bank.lock").open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    if banker.poll() is None:
                        current = read_process_identity(banker.pid)
                        if (current is None
                                or current["start_time_ticks"] != banker_identity["start_time_ticks"]):
                            raise ExporterFailure(
                                "Checkpoint banker PID identity changed before shutdown"
                            )
                        banker.terminate()
                        try:
                            banker.wait(timeout=30)
                        except subprocess.TimeoutExpired as error:
                            raise ExporterFailure(
                                "Checkpoint banker did not exit within 30 seconds"
                            ) from error
                    current = read_process_identity(banker.pid)
                    if (current is not None
                            and current["start_time_ticks"] == banker_identity["start_time_ticks"]):
                        raise ExporterFailure("Checkpoint banker identity remains live after shutdown")
                    self._owned_gpu_processes.pop(banker.pid, None)
                if (output / "run-config.json").exists():
                    for artifact in checkpoint_artifacts(output, self.work / "checkpoint-snapshots"):
                        with (self.root / "bank.lock").open("a") as lock:
                            fcntl.flock(lock, fcntl.LOCK_EX)
                            copy_artifact(Path(artifact["path"]),
                                          self.bank_root / "checkpoints" / artifact["identity"])

    def execute_cell(self, cell: dict, bundle: Path) -> dict:
        stage = self.admit(cell, bundle)
        audit = self.work / "audits" / cell["stage_id"]
        gates = self.work / "gates" / cell["stage_id"]
        gate = None
        for attempt in range(1, self.spec["queue"]["max_invocation_failures"] + 1):
            candidate = gates / f"attempt-{attempt:03d}"
            if (candidate / "gate.json").is_file() or not candidate.exists():
                gate = candidate
                break
        if gate is None:
            raise RuntimeError("Preserved failed compatibility gates exhausted the frozen attempt bound")
        self.gate(stage, audit, gate)
        self.bank(stage, f"stages/{cell['stage_id']}")
        self.bank(audit, f"audits/{cell['stage_id']}")
        self.bank(gate, f"gates/{cell['stage_id']}/{gate.name}")
        output = self.work / "runs" / cell["run_id"]
        self.train(cell, stage, audit, output)
        self.bank(output, f"runs/{cell['run_id']}")
        evaluations = self.work / "evaluations" / cell["run_id"]
        completed = [p for p in sorted(evaluations.glob("evaluation-attempt-*"))
                     if (p / "completed.json").is_file()]
        if len(completed) > 1:
            raise ValueError("Ambiguous completed evaluations; never select by test performance")
        if completed:
            evaluation = completed[0]
        else:
            if len(list(evaluations.glob("evaluation-attempt-*"))) >= 3:
                raise RuntimeError("Fixed-budget evaluation attempts exhausted; all partial outputs preserved")
            evaluation = next_evaluation(evaluations)
            self.invoke([
                "python", stage / "project" / "scripts" / "letter" / "evaluate.py",
                "--upstream", stage / "upstream", "--manifest", stage / "manifest.json",
                "--dataset", cell["dataset"], "--checkpoint", output, "--output", evaluation,
                "--seed", cell["seed"], "--split", "test", "--beams", 20, "--batch-size", 16,
            ], "evaluate-" + cell["run_id"], cell["seed"], gpu=True)
        result = verify_evaluation(stage, audit, output, evaluation)
        self.bank(evaluation, f"evaluations/{cell['run_id']}/{evaluation.name}")
        result["durability"] = "run and complete all-user predictions checksum-verified by node0 banker"
        report = self.work / "results" / cell["run_id"]
        if (report / "result.json").exists():
            if read_json(report / "result.json") != result:
                raise ValueError("Published compatible-method cell summary is immutable")
        else:
            write_json(report / "result.json", result)
        self.bank(report, f"results/{cell['run_id']}")
        write_json(self.work / "acks" / f"{cell['run_id']}.json", {
            "status": "banked", "run_id": cell["run_id"],
            "result_sha256": file_sha256(report / "result.json"),
        })
        return result


def queue_binding(runtime: Runtime) -> dict:
    binding = {
        "execution_sha256": file_sha256(runtime.execution), "job_binding_sha256": file_sha256(runtime.binding),
        "adapter_sha256": file_sha256(runtime.scripts / "fresh_ids.py"),
        "driver_sha256": file_sha256(Path(__file__)),
        "cuda_visible_devices": list(runtime.gpu_indexes),
        "root": str(runtime.root), "durable_root": str(runtime.bank_root),
        "control": str(runtime.control), "work": str(runtime.work),
        "queue": queue_cells(runtime.spec),
    }
    if getattr(runtime, "operational_admission", None) is not None:
        binding["operational_admission"] = runtime.operational_admission
    return binding


def bind_queue(work: Path, binding: dict, adoption_path: Path | None = None) -> Path:
    marker = work / "queue-binding.json"
    if adoption_path is not None:
        adoption = read_json(adoption_path)
        old_name, new_name = adoption["previous_binding_file"], adoption["new_binding_file"]
        allowed = r"queue-binding(?:-[a-z0-9-]+)?\.json"
        if (adoption.get("kind") != "fresh_queue_operational_adoption"
                or adoption["work"] != str(work) or binding["work"] != str(work)
                or not re.fullmatch(allowed, old_name) or not re.fullmatch(allowed, new_name)
                or old_name == new_name or adoption["new_binding"] != binding):
            raise ValueError("Invalid same-work, versioned operational adoption")
        previous = work / old_name
        require_hash(previous, adoption["previous_binding_sha256"])
        old = read_json(previous)
        operational = {"adapter_sha256", "driver_sha256", "control", "operational_admission"}
        if ({key: value for key, value in old.items() if key not in operational}
                != {key: value for key, value in binding.items() if key not in operational}):
            raise ValueError("Operational adoption must preserve recipe, queue, inputs and all work/run paths")
        protected = Path(adoption["protected_files"])
        require_hash(protected, adoption["protected_files_sha256"])
        protected_files = read_json(protected)
        if not isinstance(protected_files, dict) or not protected_files:
            raise ValueError("Protected artifact manifest must enumerate every existing file")
        actual_files = set()
        for path in work.rglob("*"):
            if path.is_symlink():
                raise ValueError("Protected artifact tree contains a redirected path")
            if path.is_file():
                actual_files.add(path.relative_to(work).as_posix())
        expected_files = set(protected_files)
        candidate = work / new_name
        if candidate.is_file():
            expected_files.add(new_name)
        if actual_files != expected_files:
            missing = sorted(expected_files - actual_files)
            extra = sorted(actual_files - expected_files)
            raise ValueError(
                f"Protected artifact file set differs: missing={missing}, extra={extra}"
            )
        for name, metadata in protected_files.items():
            path = work / name
            if not path.resolve().is_relative_to(work.resolve()) or path.is_symlink():
                raise ValueError("Protected artifact escapes the adopted work directory")
            require_hash(path, metadata["sha256"])
            if path.stat().st_size != metadata["bytes"]:
                raise ValueError("Protected artifact byte length differs")
        marker = work / new_name
    if marker.exists() and read_json(marker) != binding:
        raise ValueError("Readiness queue is frozen to different source/configuration bytes")
    if not marker.exists():
        write_json(marker, binding)
    return marker


def driver(runtime: Runtime, checks: int, *, adoption: Path | None = None) -> None:
    import fcntl

    binding = queue_binding(runtime)
    roots = [runtime.root / "rq-snapshots", runtime.root / "rq-bank-incoming" / "node-1",
             runtime.root / "rq-bank-incoming" / "node-2"]
    with (runtime.work / "driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = bind_queue(runtime.work, binding, adoption)
        controller_identity = capture_process_identity(
            os.getpid(), "fresh-controller", [],
        )
        write_json(runtime.work / "driver.json", {
            "pid": os.getpid(), "started_unix": time.time(), "argv": sys.argv,
            "process_identity": controller_identity,
            "queue_binding_sha256": file_sha256(marker), "max_idle_checks": checks,
            "queue_binding_file": marker.name,
            "adoption_sha256": file_sha256(adoption) if adoption is not None else None,
        })
        for check in range(checks):
            pending = [cell for cell in binding["queue"] if not cell_is_banked(runtime.work, cell)]
            if not pending:
                write_json(runtime.work / "status.json", {"status": "all_39_cells_durably_complete"})
                return
            primary = pending[0]["seed"]
            statuses, ready = [], []
            for cell in pending:
                bundle = find_bundle(roots, cell["rq_run_id"]) if cell["seed"] == primary else None
                state = "waiting_for_final_20000_epoch_RQ" if cell["seed"] == primary else "waiting_for_prior_seed"
                if bundle:
                    with (runtime.root / "bank.lock").open("a") as bank_lock:
                        fcntl.flock(bank_lock, fcntl.LOCK_EX)
                        persisted = (runtime.destination / "openllama-v2" / "tokenizers"
                                     / bundle.name / "bank-receipt.json").is_file()
                    state = "ready_for_strict_admission" if persisted else "waiting_for_durable_final_bundle"
                    if persisted:
                        ready.append((cell, bundle))
                statuses.append({"run_id": cell["run_id"], "status": state})
            status = {"status": "waiting", "pid": os.getpid(), "check": check + 1,
                      "unix": time.time(), "gpu_occupied_by_waiter": False, "cells": statuses}
            write_json(runtime.work / "status.json", status)
            print(json.dumps(status), flush=True)
            if ready:
                cell, bundle = ready[0]
                write_json(runtime.work / "status.json", {**status, "status": "executing", "cell": cell})
                try:
                    runtime.execute_cell(cell, bundle)
                except Exception as error:
                    write_json(runtime.work / "status.json", {
                        **status, "status": "blocked_preserving_artifacts", "cell": cell,
                        "error": f"{type(error).__name__}: {error}",
                    })
                    raise
            elif check + 1 < checks:
                time.sleep(runtime.spec["queue"]["poll_seconds"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--control", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--work", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    queue = commands.add_parser("driver")
    queue.add_argument("--checks", type=int, default=432)
    gate = commands.add_parser("gate")
    for name in ("stage", "audit", "gate"):
        gate.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    runtime = Runtime(args.binding, args.project, args.control, args.work)
    if args.command == "driver":
        if not 1 <= args.checks <= 1008:
            parser.error("Use 1..1008 checks (600 seconds apart when waiting)")
        driver(runtime, args.checks)
    else:
        print(json.dumps(runtime.gate(args.stage, args.audit, args.gate), indent=2), flush=True)


if __name__ == "__main__":
    main()
