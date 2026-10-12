from __future__ import annotations

import argparse
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import time
from typing import Callable, Literal

from rrbench.io import write_json


def guard_worker_parent(parent: int, start_ticks: int) -> None:
    if sys.platform != "linux":
        raise RuntimeError("Leased GPU workers require Linux parent-death signaling.")
    library = ctypes.CDLL(None, use_errno=True)
    if library.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot arm the worker parent-death signal.")
    current, state = read_process(parent, Path("/proc"))
    if os.getppid() != parent or current["start_ticks"] != start_ticks or state in ("Z", "X"):
        raise RuntimeError("Worker lease parent exited or changed before startup.")


def wait_for_start_gate(
    gate: Path, owner: str, worker: int, parent: int, parent_start: int,
    proc_root: Path = Path("/proc"), timeout: float = 60,
) -> None:
    if timeout < 0:
        raise ValueError("Startup gate timeout cannot be negative.")
    deadline = time.monotonic() + timeout
    while True:
        try:
            current, state = read_process(parent, proc_root)
        except (FileNotFoundError, ProcessLookupError) as error:
            raise RuntimeError("Worker lease parent exited before startup approval.") from error
        if current["start_ticks"] != parent_start or state in ("Z", "X"):
            raise RuntimeError("Worker lease parent identity changed before startup approval.")
        if gate.exists():
            approval = json.loads(gate.read_text(encoding="utf-8"))
            if approval != {"lease_owner": owner, "worker": worker}:
                raise ValueError("Startup gate owner/worker does not match this registered child.")
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Unregistered worker's startup gate never opened; no GPU work started.")
        time.sleep(0.05)


def fresh_start_gate(directory: Path, owner: str, launch: int) -> Path:
    if not isinstance(owner, str) or re.fullmatch(r"[A-Za-z0-9_-]+", owner) is None or type(launch) is not int or launch <= 0:
        raise ValueError("Startup launch identity must have a safe owner and positive integer sequence.")
    reserved = directory / f"{owner}-{launch}"
    reserved.mkdir(mode=0o700, exist_ok=False)
    return reserved / "approval.json"


def read_process(pid: int, proc_root: Path) -> tuple[dict[str, object], str]:
    directory = proc_root / str(pid)
    fields = (directory / "stat").read_text().rsplit(") ", 1)[1].split()
    state = fields[0]
    command = [
        value.decode("utf-8") for value in (directory / "cmdline").read_bytes().split(b"\0") if value
    ]
    return {
        "pid": pid, "parent": int(fields[1]), "start_ticks": int(fields[19]), "command": command,
    }, state


def register_worker(
    pid: int, parent: int, registry: Path, proc_root: Path = Path("/proc"),
    expected_module: str = "rrbench.semantic_encoding",
) -> None:
    identity, state = read_process(pid, proc_root)
    invocation = identity["command"]
    if identity["parent"] != parent or invocation[1:3] != ["-m", expected_module] or state in ("Z", "X"):
        raise RuntimeError("Cannot register a worker without its live parent/command identity.")
    entries = json.loads(registry.read_text(encoding="utf-8")) if registry.exists() else []
    if any(entry["pid"] == pid for entry in entries):
        raise ValueError("Worker is already registered to this lease.")
    write_json(registry, [*entries, identity])


def _owned_worker_state(
    entry: dict[str, object], parent: int, proc_root: Path,
    observation: dict[str, object] | None = None,
) -> Literal["live", "pending_exit", "exited", "changed"]:
    observation = {} if observation is None else observation
    try:
        current, state = read_process(entry["pid"], proc_root)
    except (FileNotFoundError, ProcessLookupError):
        observation["reason"] = "process_disappeared"
        return "exited"
    observation["first"] = {"identity": current, "task_state": state}
    if current["start_ticks"] != entry["start_ticks"] or current["parent"] != parent:
        observation["reason"] = "pid_start_or_parent_changed"
        return "changed"
    if state in ("Z", "X"):
        observation["reason"] = "verified_terminal_state"
        return "exited"
    if current == entry:
        observation["reason"] = "exact_live_identity"
        return "live"
    if current["command"]:
        observation["reason"] = "nonempty_command_changed"
        return "changed"
    try:
        latest, latest_state = read_process(entry["pid"], proc_root)
    except (FileNotFoundError, ProcessLookupError):
        observation["reason"] = "process_disappeared_after_empty_command"
        return "exited"
    observation["latest"] = {"identity": latest, "task_state": latest_state}
    if latest["start_ticks"] != entry["start_ticks"] or latest["parent"] != parent:
        observation["reason"] = "pid_start_or_parent_changed_after_empty_command"
        return "changed"
    if latest_state in ("Z", "X"):
        observation["reason"] = "verified_terminal_after_empty_command"
        return "exited"
    if latest == entry:
        observation["reason"] = "exact_identity_observable_again"
        return "live"
    observation["reason"] = "nonempty_command_changed" if latest["command"] else "empty_command_nonterminal"
    return "changed" if latest["command"] else "pending_exit"


def _lifecycle_event(
    expected: dict[str, object], state: str, observation: dict[str, object], pending_seconds: float,
) -> None:
    def summary(identity: dict[str, object]) -> dict[str, object]:
        command = identity["command"]
        if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
            raise ValueError("Process observation command has an invalid schema.")
        return {
            **{key: identity[key] for key in ("pid", "parent", "start_ticks")},
            "argv_count": len(command),
            "argv_sha256": hashlib.sha256(json.dumps(command, ensure_ascii=True).encode()).hexdigest(),
        }

    event = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "monotonic_seconds": time.monotonic(), "state": state, "pending_seconds": pending_seconds,
        "reason": observation["reason"], "expected": summary(expected),
    }
    for key in ("first", "latest"):
        if key in observation:
            value = observation[key]
            event[key] = {"identity": summary(value["identity"]), "task_state": value["task_state"]}
    print("OWNED_WORKER_LIFECYCLE " + json.dumps(event, sort_keys=True), flush=True)
    private = os.environ.get("RRBENCH_PRIVATE_LIFECYCLE_LOG")
    if private:
        descriptor = os.open(
            private, os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({**event, "raw_expected": expected, "raw_observation": observation}, stream)
            stream.write("\n")


def terminate_owned_workers(
    registry: Path, parent: int, proc_root: Path = Path("/proc"),
    send_signal: Callable[[int, int], None] = os.kill, timeout: float = 5,
) -> None:
    if timeout < 0:
        raise ValueError("Worker drain timeout cannot be negative.")
    entries = json.loads(registry.read_text(encoding="utf-8")) if registry.exists() else []
    problems = set()
    states, pending_start = {}, {}

    def state_of(entry: dict[str, object]) -> str:
        if entry["pid"] in problems:
            return "changed"
        observation = {}
        state = _owned_worker_state(entry, parent, proc_root, observation)
        if state != states.get(entry["pid"]):
            if state == "pending_exit":
                pending_start[entry["pid"]] = time.monotonic()
            elapsed = time.monotonic() - pending_start.get(entry["pid"], time.monotonic())
            _lifecycle_event(entry, state, observation, max(0, elapsed))
            states[entry["pid"]] = state
        if state == "changed":
            problems.add(entry["pid"])
        return state

    def unfinished(entry: dict[str, object]) -> bool:
        return state_of(entry) in ("live", "pending_exit")

    def deliver(entry: dict[str, object], requested_signal: int) -> None:
        if state_of(entry) != "live":
            return
        try:
            send_signal(entry["pid"], requested_signal)
        except ProcessLookupError:
            print(f"OWNED_WORKER_ALREADY_EXITED pid={entry['pid']}", flush=True)

    for entry in entries:
        deliver(entry, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and any(unfinished(entry) for entry in entries):
        time.sleep(0.1)
    for entry in entries:
        if unfinished(entry):
            deliver(entry, getattr(signal, "SIGKILL", 9))
    deadline = time.monotonic() + max(timeout, 0.1)
    while time.monotonic() < deadline and any(unfinished(entry) for entry in entries):
        time.sleep(0.1)
    remaining = [entry["pid"] for entry in entries if unfinished(entry)]
    if remaining:
        raise RuntimeError(f"Confirmed owned workers did not drain: {remaining}.")
    if problems:
        raise RuntimeError(f"Worker identity changed; these PIDs were not signalled: {sorted(problems)}.")


def wait_owned_worker_until(
    registry: Path, parent: int, pid: int, deadline_utc: str,
    proc_root: Path = Path("/proc"), send_signal: Callable[[int, int], None] = os.kill,
    now: Callable[[], float] = time.time, poll_seconds: float = 0.2,
    required_registry: Path | None = None,
) -> None:
    deadline = datetime.datetime.fromisoformat(deadline_utc.replace("Z", "+00:00"))
    if deadline.tzinfo is None or poll_seconds <= 0:
        raise ValueError("Owned-worker deadline must be timezone-aware and polling positive.")
    entries = json.loads(registry.read_text(encoding="utf-8"))
    matches = [entry for entry in entries if entry["pid"] == pid and entry["parent"] == parent]
    if len(matches) != 1:
        raise RuntimeError("Deadline monitor requires exactly one registered owned-worker identity.")
    expected = matches[0]
    required = json.loads(required_registry.read_text(encoding="utf-8")) if required_registry is not None else None
    if required is not None and (len(required) != 1 or required[0]["parent"] != parent):
        terminate_owned_workers(registry, parent, proc_root, send_signal, timeout=5)
        raise RuntimeError("Required reservation must contain exactly one child of this lease.")
    parent_identity, parent_state = read_process(parent, proc_root)
    if parent_state in ("Z", "X"):
        raise RuntimeError("Owned-worker parent exited before deadline monitoring.")
    previous, pending_start = None, None
    while True:
        current_parent, state = read_process(parent, proc_root)
        if current_parent != parent_identity or state in ("Z", "X"):
            raise RuntimeError("Owned-worker parent identity changed during deadline monitoring.")
        if required is not None and _owned_worker_state(required[0], parent, proc_root) != "live":
            terminate_owned_workers(registry, parent, proc_root, send_signal, timeout=5)
            raise RuntimeError("Required reservation exited or changed; actual owned work was drained.")
        observation = {}
        state = _owned_worker_state(expected, parent, proc_root, observation)
        if state != previous:
            if state == "pending_exit":
                pending_start = time.monotonic()
            elapsed = 0 if pending_start is None else time.monotonic() - pending_start
            _lifecycle_event(expected, state, observation, elapsed)
            previous = state
        if state == "exited":
            return
        if state == "changed":
            raise RuntimeError("Owned-worker identity changed; deadline monitor will not signal a reused PID.")
        remaining = deadline.timestamp() - now()
        if remaining <= 0:
            terminate_owned_workers(registry, parent, proc_root, send_signal, timeout=5)
            raise TimeoutError("Absolute compute deadline reached; registered owned workers were drained.")
        time.sleep(min(poll_seconds, remaining))


def main() -> None:
    parser = argparse.ArgumentParser(description="Identity-checked Linux worker registry and cleanup.")
    parser.add_argument("operation", choices=["register", "terminate", "wait"])
    parser.add_argument("--registry", required=True)
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--module", default="rrbench.semantic_encoding")
    parser.add_argument("--deadline")
    parser.add_argument("--required-registry")
    args = parser.parse_args()
    if args.operation == "register":
        if args.pid is None:
            parser.error("register requires --pid")
        register_worker(args.pid, args.parent, Path(args.registry), expected_module=args.module)
    elif args.operation == "terminate":
        terminate_owned_workers(Path(args.registry), args.parent)
    else:
        if args.pid is None or args.deadline is None:
            parser.error("wait requires --pid and an absolute --deadline")
        wait_owned_worker_until(
            Path(args.registry), args.parent, args.pid, args.deadline,
            required_registry=Path(args.required_registry) if args.required_registry else None,
        )


if __name__ == "__main__":
    main()
