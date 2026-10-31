"""Resource-guarded P5-only queues; restart only complete-state continuations."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from p5_source import read_json, sha256, write_json


def ensure_free(gpu):
    if gpu not in (4, 5):
        raise ValueError("Only allocated node0 GPU4/GPU5 are admitted")
    gpu_uuid = subprocess.check_output([
        "nvidia-smi", f"--id={gpu}", "--query-gpu=uuid", "--format=csv,noheader",
    ], text=True).strip()
    applications = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader",
    ], text=True)
    if any(line.split(",")[0].strip() == gpu_uuid for line in applications.splitlines()):
        raise RuntimeError(f"Assigned GPU{gpu} is not free; no process will be displaced")
    return gpu_uuid


def command(common, cell, script, output, microbatch, gate=None):
    result = [sys.executable, str(Path(__file__).with_name(script))]
    for name in ("native", "upstream", "manifest", "recipe", "model"):
        result.extend(["--" + name, common[name]])
    result.extend(["--corpus", cell["corpus"], "--output", str(output),
                   "--microbatch-size", str(microbatch)])
    if gate is not None:
        result.extend(["--gate", str(gate)])
    return result


def launch(arguments, logfile, environment, status_path, status):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(arguments, stdout=log, stderr=subprocess.STDOUT, env=environment)
        write_json(status_path, {**status, "pid": process.pid, "log": str(logfile), "command": arguments})
        return process.wait()


def queue(args):
    import socket

    if socket.gethostname() != "node-0":
        raise RuntimeError("P5 is restricted to the authorized node0 allocation")
    spec = read_json(args.spec)
    gpu = spec["gpu"]
    environment = dict(os.environ)
    environment.update(CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=":4096:8",
                       OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       RAYON_NUM_THREADS="2", TOKENIZERS_PARALLELISM="true", PYTHONHASHSEED="0")
    status_path = args.spec.with_suffix(".status.json")
    write_json(args.spec.with_suffix(".pid.json"), {"pid": os.getpid(), "gpu": gpu})
    for cell in spec["cells"]:
        if Path(cell["run"], "completed.json").exists():
            continue
        if not Path(cell["corpus"], "admission.json").exists():
            write_json(status_path, {"status": "blocked", "cell": cell, "reason": "No complete corpus admission"})
            raise RuntimeError("Do not start a P5 fit before full corpus admission")
        gate_path = None
        for microbatch in (64, 32, 16, 8, 4, 2, 1):
            candidate = Path(cell["gate"]) / f"micro-{microbatch}"
            if (candidate / "gate.json").exists():
                gate = read_json(candidate / "gate.json")
                if gate["status"] == "pass":
                    gate_path = candidate
                    break
                if gate["status"] == "memory-gate-failed":
                    continue
                raise RuntimeError("Failed native gate is not a memory-admission opportunity")
            ensure_free(gpu)
            result = launch(
                command(spec["common"], cell, "p5_gate.py", candidate, microbatch),
                candidate / "gate.log", environment, status_path,
                {"status": "gating", "cell": cell["name"], "gpu": gpu, "microbatch": microbatch},
            )
            if result == 0:
                gate_path = candidate
                break
            if result != 13:
                raise RuntimeError(f"Native gate failed for {cell['name']}; inspect {candidate / 'gate.log'}")
        if gate_path is None:
            raise RuntimeError("No complete-target native training microbatch fits the assigned GPU")
        gate = read_json(gate_path / "gate.json")
        write_json(Path(cell["run"]) / "admitted-operating-recipe.json", {
            "source_recipe_sha256": sha256(spec["common"]["recipe"]),
            "gate_sha256": sha256(gate_path / "gate.json"),
            "microbatch_size": gate["operating_microbatch_size"],
            "gradient_accumulation": gate["operating_accumulation"],
            "effective_batch_size": 128, "selection": "hardware admission only, no score-based tuning",
        })
        if args.gates_only:
            continue
        for attempt in range(3):
            ensure_free(gpu)
            arguments = command(spec["common"], cell, "p5_run.py", cell["run"],
                                gate["operating_microbatch_size"], gate_path)
            if args.prime_steps:
                arguments.extend(["--stop-steps", str(args.prime_steps)])
                latest = Path(cell["run"]) / "latest.json"
                if latest.exists() and read_json(latest)["global_step"] >= args.prime_steps:
                    break
            result = launch(
                arguments,
                Path(cell["run"]) / "train.log", environment, status_path,
                {"status": "priming" if args.prime_steps else "training",
                 "cell": cell["name"], "gpu": gpu, "attempt": attempt + 1},
            )
            if result == 0 and Path(cell["run"], "completed.json").exists():
                break
            if (args.prime_steps and result == 0 and Path(cell["run"], "latest.json").exists()
                    and read_json(Path(cell["run"]) / "latest.json")["global_step"] >= args.prime_steps):
                break
            if result != 0:
                raise RuntimeError(f"P5 fit failed visibly; no silent recipe fallback: {cell['name']}")
            time.sleep(30)
        else:
            raise RuntimeError("P5 repeatedly drained before completion; retain its sealed continuation")
    status = ("gates-complete" if args.gates_only else "priming-complete" if args.prime_steps
              else "queue-verified-complete" if all(
        Path(cell["run"], "results-banked.json").exists() for cell in spec["cells"]
    ) else "queue-compute-complete-bank-pending")
    write_json(status_path, {"status": status, "gpu": gpu})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--gates-only", action="store_true")
    parser.add_argument("--prime-steps", type=int, default=0)
    queue(parser.parse_args())
