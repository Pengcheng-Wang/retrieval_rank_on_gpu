from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Callable

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


def terminate_owned_workers(
    registry: Path, parent: int, proc_root: Path = Path("/proc"),
    send_signal: Callable[[int, int], None] = os.kill, timeout: float = 5,
) -> None:
    if timeout < 0:
        raise ValueError("Worker drain timeout cannot be negative.")
    entries = json.loads(registry.read_text(encoding="utf-8")) if registry.exists() else []
    problems = set()

    def live(entry: dict[str, object]) -> bool:
        try:
            current, state = read_process(entry["pid"], proc_root)
        except (FileNotFoundError, ProcessLookupError):
            return False
        if state in ("Z", "X"):
            return False
        if current != entry or current["parent"] != parent:
            problems.add(entry["pid"])
            return False
        return True

    def deliver(entry: dict[str, object], requested_signal: int) -> None:
        if not live(entry):
            return
        try:
            send_signal(entry["pid"], requested_signal)
        except ProcessLookupError:
            print(f"OWNED_WORKER_ALREADY_EXITED pid={entry['pid']}", flush=True)

    for entry in entries:
        deliver(entry, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and any(live(entry) for entry in entries):
        time.sleep(0.1)
    for entry in entries:
        if live(entry):
            deliver(entry, getattr(signal, "SIGKILL", 9))
    deadline = time.monotonic() + max(timeout, 0.1)
    while time.monotonic() < deadline and any(live(entry) for entry in entries):
        time.sleep(0.1)
    remaining = [entry["pid"] for entry in entries if live(entry)]
    if remaining:
        raise RuntimeError(f"Confirmed owned workers did not drain: {remaining}.")
    if problems:
        raise RuntimeError(f"Worker identity changed; these PIDs were not signalled: {sorted(problems)}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Identity-checked Linux worker registry and cleanup.")
    parser.add_argument("operation", choices=["register", "terminate"])
    parser.add_argument("--registry", required=True)
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--module", default="rrbench.semantic_encoding")
    args = parser.parse_args()
    if args.operation == "register":
        if args.pid is None:
            parser.error("register requires --pid")
        register_worker(args.pid, args.parent, Path(args.registry), expected_module=args.module)
    else:
        terminate_owned_workers(Path(args.registry), args.parent)


if __name__ == "__main__":
    main()
