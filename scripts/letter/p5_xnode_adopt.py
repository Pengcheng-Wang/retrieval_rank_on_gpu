"""Observe one explicitly bound fit without reparenting, restarting or expanding its queue."""

from __future__ import annotations

import argparse
import hashlib
import os
import socket
import time
from pathlib import Path

import p5_xnode_state as state


def process_identity(pid):
    root = Path("/proc") / str(pid)
    try:
        status = root.joinpath("status").read_text()
        if next(line for line in status.splitlines() if line.startswith("State:")).split()[1] == "Z":
            return None
        command = root.joinpath("cmdline").read_bytes()
        start = int(root.joinpath("stat").read_text().rsplit(")", 1)[1].split()[19])
    except FileNotFoundError:
        return None
    return {
        "pid": pid, "start_ticks": start,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "command_sha256": hashlib.sha256(command).hexdigest(),
    }


def adoption_state(expected, actual, completed):
    if actual is None:
        return "compute-complete" if completed else "blocked-exited-without-completion"
    if any(actual.get(key) != expected[key] for key in ("pid", "start_ticks", "boot_id", "command_sha256")):
        raise ValueError("Adopted PID lifetime identity changed")
    return "running"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--bindings-sha256")
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    state.configure_bindings(args.bindings, args.bindings_sha256)
    state.require_path(args.spec, writable=True)
    bindings = state.configuration()
    spec = state.read(args.spec)
    cell = bindings["adopted_cell"]
    if (cell is None or socket.gethostname() != state.WRITER_HOST
            or spec.get("kind") != "p5-xnode-source-adoption-v1"
            or spec.get("cell") != cell or spec.get("gpu") != bindings["cells"][cell]["gpu"]
            or spec.get("cells") != [cell]):
        raise ValueError("Only the explicitly bound one-cell writer-host continuation may be adopted")
    run = state.P5 / "runs" / spec["cell"]
    expected = spec["identity"]
    pidfd = None
    with state.locked(args.spec.with_suffix(".lock"), nonblocking=True):
        if adoption_state(expected, process_identity(expected["pid"]), False) != "running":
            raise RuntimeError("The live fit vanished before adoption")
        if hasattr(os, "pidfd_open"):
            pidfd = os.pidfd_open(expected["pid"])
            adoption_state(expected, process_identity(expected["pid"]), False)
        try:
            while True:
                status = adoption_state(
                    expected, process_identity(expected["pid"]), (run / "completed.json").exists(),
                )
                state.write(args.spec.with_suffix(".status.json"), {
                    "status": status, "controller_pid": os.getpid(), "fit_identity": expected,
                    "cell": spec["cell"], "gpu": bindings["cells"][cell]["gpu"],
                    "automatic_restart": False, "time": time.time(),
                })
                if status == "blocked-exited-without-completion":
                    raise RuntimeError("Adopted fit exited ambiguously; no automatic replacement is permitted")
                if status == "compute-complete" and (run / "results-banked.json").exists():
                    return
                time.sleep(10)
        finally:
            if pidfd is not None:
                os.close(pidfd)


if __name__ == "__main__":
    main()
