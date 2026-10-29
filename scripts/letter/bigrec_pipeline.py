"""Supervise only one admitted BIGRec cell, with bank-before-test handshakes."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

from bigrec_bank import require_ack, seal_payload, stage_payload
from bigrec_checkpoint import latest, verify
from bigrec_data import verify_admission
from bigrec_gate import require_gates
from protocol import file_sha256, write_json


def retryable_exit(returncode: int, stopping: bool) -> bool:
    return not stopping and returncode in (0, -15, -9, 137, 143)


def quarantine_unsealed_checkpoints(run: Path) -> None:
    for path in run.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            verify(path)
        except (ValueError, KeyError, FileNotFoundError, json.JSONDecodeError):
            path.rename(run / f".incomplete-{path.name}-{uuid.uuid4().hex}")


def pending_run_exports(outbox: Path, run_name: str) -> list[str]:
    pending = []
    for ready in outbox.glob("*/ready.json"):
        request = json.loads(ready.read_text(encoding="utf-8"))
        if not request["name"].startswith(run_name + "-"):
            continue
        marker = ready.parent / "banked.json"
        if not marker.is_file():
            pending.append(request["identity"])
            continue
        ack = json.loads(marker.read_text(encoding="utf-8"))
        if (ack.get("status") != "banked" or ack["identity"] != request["identity"]
                or ack["complete_sha256"] != request["complete_sha256"]):
            raise ValueError("A run artifact has an invalid durable acknowledgment")
    return sorted(pending)


def supervise(args) -> None:
    report = verify_admission(args.admission)
    require_gates(args.gates, args.admission)
    args.run.mkdir(parents=True, exist_ok=True)
    args.evaluation.mkdir(parents=True, exist_ok=True)
    status_path = args.run / "pipeline-status.json"
    state = {
        "pid": os.getpid(), "host": socket.gethostname(), "dataset": report["dataset"],
        "run": str(args.run), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "stopping": False,
        "admission_sha256": file_sha256(args.admission / "admission.json"),
        "approved_gates_sha256": file_sha256(args.gates), "child_pid": None,
    }
    active = {"child": None}

    def drain(signum, frame):
        state["stopping"] = True
        child = active["child"]
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, drain)

    def publish(phase):
        state.update(phase=phase, updated_unix=time.time())
        write_json(status_path, state)

    def await_bank(name, source, phase):
        while not state["stopping"]:
            try:
                acknowledgment = require_ack(args.outbox, name, source)
                state["last_bank_ack"] = acknowledgment
                publish(phase + "_acknowledged")
                return True
            except FileNotFoundError:
                publish(phase + "_waiting")
                time.sleep(20)
        return False

    handshake = args.run / "exporter-handshake"
    if not handshake.exists():
        handshake.mkdir()
        write_json(handshake / "request.json", {
            "run": args.run.name, "host": socket.gethostname(),
            "admission_sha256": state["admission_sha256"],
            "approved_gates_sha256": state["approved_gates_sha256"],
        })
        seal_payload(handshake, "bigrec_exporter_preflight_handshake", {
            "track": report["track"], "dataset": report["dataset"],
        })
    handshake_name = f"{args.run.name}-exporter-handshake"
    stage_payload(handshake, args.outbox, handshake_name)
    if not await_bank(handshake_name, handshake, "exporter_preflight"):
        return
    directory = Path(__file__).parent
    shared = [
        "--admission", str(args.admission), "--model", str(args.model), "--outbox", str(args.outbox),
    ]
    train_command = [
        sys.executable, str(directory / "bigrec_train.py"), *shared,
        "--output", str(args.run), "--gates", str(args.gates),
    ]
    evaluation_shared = [
        *shared, "--run", str(args.run), "--output", str(args.evaluation), "--catalog", str(args.catalog),
    ]
    stages = (
        ("training", args.run / "completed.json", train_command),
        ("generation", args.evaluation / "generation-completed.json", [
            sys.executable, str(directory / "bigrec_evaluate.py"), "generate", *evaluation_shared,
        ]),
        ("grounding", args.evaluation / "completed.json", [
            sys.executable, str(directory / "bigrec_evaluate.py"), "ground", *evaluation_shared,
        ]),
    )
    for phase, completion, command in stages:
        if phase == "generation":
            if not await_bank(f"{args.run.name}-selected", args.run / "selected", "selected_before_test"):
                return
        attempt = 0
        while not completion.exists() and not state["stopping"]:
            attempt += 1
            invocation = list(command)
            if phase == "training":
                quarantine_unsealed_checkpoints(args.run)
                if latest(args.run) is not None:
                    invocation.extend(["--resume", "latest"])
            log = args.run / f"{phase}-invocation-{time.time_ns()}.log"
            with log.open("w", encoding="utf-8") as stream:
                child = subprocess.Popen(invocation, stdout=stream, stderr=subprocess.STDOUT)
                active["child"] = child
                state.update(child_pid=child.pid, child_log=str(log), attempt=attempt)
                publish(phase)
                print(f"BIGREC_PIPELINE {report['dataset']} {phase} pid={child.pid}", flush=True)
                while child.poll() is None:
                    publish(phase)
                    time.sleep(15)
                returncode = child.returncode
            active["child"] = None
            state.update(child_pid=None, returncode=returncode)
            if completion.exists():
                break
            if not retryable_exit(returncode, state["stopping"]):
                publish("stopped" if state["stopping"] else phase + "_blocked")
                if not state["stopping"]:
                    write_json(args.run / "pipeline-blocked.json", state)
                    raise RuntimeError(f"BIGRec {phase} failed; retain its fixed recipe and inspect {log}")
                return
            publish(phase + "_resume_pending")
            time.sleep(30)
    if state["stopping"]:
        publish("drained")
        return
    if not await_bank(f"{args.run.name}-evaluation", args.evaluation / "summary", "evaluation_final"):
        return
    if not await_bank(f"bigrec-{report['dataset']}-catalog", args.catalog / "catalog", "catalog_final"):
        return
    while not state["stopping"]:
        pending = pending_run_exports(args.outbox, args.run.name)
        if not pending:
            break
        state["pending_final_artifacts"] = len(pending)
        publish("all_expensive_artifacts_waiting")
        time.sleep(20)
    if state["stopping"]:
        publish("drained")
        return
    state["pending_final_artifacts"] = 0
    publish("complete")
    write_json(args.run / "pipeline-completed.json", state)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("admission", "model", "gates", "run", "evaluation", "outbox", "catalog"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    for name in ("admission", "model", "gates", "run", "evaluation", "outbox", "catalog"):
        setattr(args, name, getattr(args, name).resolve())
    supervise(args)


if __name__ == "__main__":
    main()
