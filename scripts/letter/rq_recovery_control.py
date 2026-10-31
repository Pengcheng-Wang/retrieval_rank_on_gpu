"""Isolated, full-state continuation of the two explicitly leased RQ runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import zipfile

import numpy as np

from rq_recovery_compat import POLICY
from rq_recovery_runtime import array_sha, file_manifest, rng_sha, sha


VARIANTS = {
    "semantic": {
        "gpu": "1", "alpha": "0",
        "config": "41fd4d11003fe4fe43a424d4755354d2d60d43ba0e327e2d373d141a0995d38d",
        "base_state": "23433094b3df639028bd89e13ab423e977d0063a28feb44ee94f121d36a9a467",
    },
    "cf": {
        "gpu": "6", "alpha": "0.02",
        "config": "6d6206feab94f6d695439bf1522e682a209354ea42bd1be4a3d03ac2a6fa6f28",
        "base_state": "8d2ce28f81eafe84d521cdfc05271020b89ffc9d99056725a1a71af62b9b7e2d",
    },
}
HARNESS = "17ec8afe40408ec7c316ef257a38b88960995924aa2eeeb2a39881924f42191e"
ORIGINAL_SEAL = "d8aba7ad8a66bf29b73118b15e49f3ff2d9ae96fca5fa0b3c052cbb235757a42"


def read_json(path):
    return json.loads(path.read_text())


def require_sha(path, expected):
    if path.is_symlink() or sha(path) != expected:
        raise ValueError(f"Recovery checksum mismatch: {path}")


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    next_path = path.with_name(path.name + ".next")
    next_path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    next_path.replace(path)


def validate_scope(root, variant, output, gpu, hostname):
    if variant not in VARIANTS or gpu != VARIANTS[variant]["gpu"] or hostname != "node-1":
        raise ValueError("Recovery is restricted to node1 GPU1 semantic / GPU6 CF")
    if not output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Recovery output is redirected outside its node-local root")
    original = root / "runs" / f"openllama-v2-rq-Instruments-{variant}-s42"
    if output.resolve() == original.resolve() and not output.is_symlink():
        return "production"
    probes = root / "rq_recovery_v1" / "probes"
    if output.resolve().is_relative_to(probes.resolve()) and output.resolve() != probes.resolve():
        return "probe"
    raise ValueError("Output is outside the two leased runs and isolated recovery probes")


def require_equal(left, right, where="state"):
    import torch
    if torch.is_tensor(left):
        equal = (torch.is_tensor(right) and left.dtype == right.dtype
                 and torch.equal(left.detach().cpu(), right.detach().cpu()))
    elif isinstance(left, np.ndarray):
        equal = isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)
    elif isinstance(left, dict):
        equal = isinstance(right, dict) and left.keys() == right.keys()
        if equal:
            for key in left:
                require_equal(left[key], right[key], f"{where}.{key}")
    elif isinstance(left, (list, tuple)):
        equal = type(left) is type(right) and len(left) == len(right)
        if equal:
            for index, value in enumerate(left):
                require_equal(value, right[index], f"{where}[{index}]")
    else:
        equal = type(left) is type(right) and left == right
    if not equal:
        raise ValueError(f"Exact native state mismatch: {where}")


def audit_archive(events, destination):
    summary = {"tasks": 0, "failed_tasks": 0, "corrections": 0, "phases": {}, "worker_pids": []}
    pids = set()
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(events.rglob("*")):
            if path.is_symlink():
                raise ValueError("Redirected runtime evidence")
            if not path.is_file() or path.name == "task-context.json":
                continue
            archive.write(path, path.relative_to(events).as_posix())
            if path.name.startswith("tasks-") and path.suffix == ".jsonl":
                for line in path.read_text().splitlines():
                    event = json.loads(line)
                    summary["tasks"] += 1
                    summary["failed_tasks"] += event["status"] != "success"
                    summary["corrections"] += len(event["corrections"])
                    phase = event.get("context", {}).get("phase", "unlabeled")
                    summary["phases"][phase] = summary["phases"].get(phase, 0) + len(event["corrections"])
                    pids.add(event["pid"])
    summary["worker_pids"] = sorted(pids)
    return summary


def seal_compatibility(directory, envelope, events, admission):
    target = directory / "compatibility"
    target.mkdir(exist_ok=False)
    summary = audit_archive(events, target / "events.zip")
    if admission is not None:
        for name in ("admission.json", "source.zip", "gates.zip"):
            shutil.copyfile(admission.parent / name, target / name)
    result = {
        **envelope, "policy": POLICY, "runtime_summary": summary,
        "files": file_manifest(target),
    }
    atomic_json(target / "attestation.json", result)
    return result


def run(args):
    import fcntl
    import torch

    root, output = args.root.resolve(), args.output.resolve()
    private = root / "rq_recovery_v1"
    source = root / "project" / "scripts" / "letter"
    scope = validate_scope(root, args.variant, output, os.environ.get("CUDA_VISIBLE_DEVICES"),
                           socket.gethostname())
    fixed = VARIANTS[args.variant]
    base = read_json(private / "originals" / args.variant / "base.json")
    require_sha(private / "originals" / "seal.json", ORIGINAL_SEAL)
    require_sha(source / "tokenizer_run.py", HARNESS)
    require_sha(private / "originals" / args.variant / "run-config.json", fixed["config"])
    if base["state_sha256"] != fixed["base_state"]:
        raise ValueError("Wrong original recovery base")
    if scope == "production":
        if args.admission is None:
            raise ValueError("Production requires the sealed and banked cold-continuation admission")
        admission = read_json(args.admission)
        if (admission["status"] != "admitted" or admission["policy"] != POLICY
                or admission["original_seal_sha256"] != ORIGINAL_SEAL
                or admission["variants"] != VARIANTS):
            raise ValueError("Wrong production admission")
        for name, digest in admission["control_sha256"].items():
            require_sha(Path(__file__).parent / name, digest)
        for name, digest in admission["artifacts"].items():
            require_sha(args.admission.parent / name, digest)
        require_sha(output / "run-config.json", fixed["config"])
    runtime = private / ("production" if scope == "production" else "probe-runtime") / args.variant
    if scope == "probe":
        runtime = output.parent / "runtime-control"
    runtime.mkdir(parents=True, exist_ok=True)
    lease = (runtime / "lease.lock").open("a")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    occupied = subprocess.check_output([
        "nvidia-smi", "-i", fixed["gpu"], "--query-compute-apps=pid", "--format=csv,noheader",
    ], text=True).strip()
    if occupied:
        raise RuntimeError(f"Owned GPU is already occupied; no signal sent: {occupied}")
    if (output / "completed.json").exists():
        raise FileExistsError("Run already completed")
    sys.path.insert(0, str(source))
    import tokenizer_run as native
    from bank_tokenizers import read_state
    import k_means_constrained.sklearn_import.cluster.k_means_ as module
    overlay = private / "compat-overlay-001"
    overlay_receipt = read_json(overlay / "overlay.json")
    if (Path(module.__file__).resolve() !=
            overlay / "k_means_constrained" / "sklearn_import" / "cluster" / "k_means_.py"
            or not hasattr(module, "_rq_compat_k_init")
            or file_manifest(overlay / "k_means_constrained") != overlay_receipt["overlay_files"]):
        raise ValueError("Actual imported package differs from the sealed private overlay")
    if scope == "production":
        require_sha(overlay / "overlay.json", admission["overlay_receipt_sha256"])

    config = read_json(private / "originals" / args.variant / "run-config.json")
    state = read_state(args.resume, config)
    epoch = state["epoch"] + 1
    if not 16400 <= epoch <= 20000:
        raise ValueError("Recovery checkpoint is outside the authorized continuation")
    if epoch == 16400:
        require_sha(args.resume / "state.pt", fixed["base_state"])
    elif scope == "production":
        from rq_recovery_lineage import verify_checkpoint_attestation
        verify_checkpoint_attestation(args.resume, output.name, fixed["config"], sha(args.admission))
    if args.stop_after_epoch is not None and not epoch < args.stop_after_epoch <= 20000:
        raise ValueError("Invalid bounded continuation endpoint")
    if scope == "probe" and args.stop_after_epoch is None:
        raise ValueError("Diagnostic probes must have an explicit epoch bound")
    del state
    events = runtime / "events"
    (events / "contexts").mkdir(parents=True, exist_ok=True)
    os.environ["RQ_RECOVERY_EVENTS"] = str(events)
    os.environ["RQ_RECOVERY_REQUIRE_CONTEXT"] = "1"
    invocation = f"{os.getpid()}-{time.time_ns()}"
    common = {
        "kind": "rq_compatibility_attestation", "purpose": scope,
        "run_id": Path(base["origin"]).name, "base_epoch": 16400,
        "base_state_sha256": fixed["base_state"], "run_config_sha256": fixed["config"],
        "original_seal_sha256": ORIGINAL_SEAL, "native_harness_sha256": HARNESS,
        "admission_sha256": sha(args.admission) if args.admission else None,
        "invocation": invocation, "resumed_from_epoch": epoch,
        "resumed_from_state_sha256": sha(args.resume / "state.pt"),
    }
    atomic_json(runtime / f"invocation-{invocation}.json", {
        **common, "pid": os.getpid(), "gpu": fixed["gpu"], "started_unix": time.time(),
        "output": str(output), "stop_after_epoch": args.stop_after_epoch,
    })
    context = {"epoch": epoch + 1, "calls": 0, "phase": "training", "loaded": epoch}
    original_cluster, original_load = native.cluster_codebook, native.load_state
    original_save, original_append = native.save_state, native.append_jsonl
    original_export, original_write = native.export_indices, native.write_json

    def load(path, trainer, spec):
        loaded = original_load(path, trainer, spec)
        import random
        require_equal(loaded["model"], trainer.model.state_dict(), "restored.model")
        require_equal(loaded["optimizer"], trainer.optimizer.state_dict(), "restored.optimizer")
        require_equal(loaded["python_rng"], random.getstate(), "restored.python_rng")
        require_equal(loaded["numpy_rng"], np.random.get_state(), "restored.numpy_rng")
        require_equal(loaded["torch_rng"], torch.get_rng_state(), "restored.torch_rng")
        require_equal(loaded["cuda_rng"], torch.cuda.get_rng_state_all(), "restored.cuda_rng")
        context["loaded"] = loaded["epoch"] + 1
        atomic_json(runtime / f"restored-{invocation}-{loaded['epoch'] + 1}.json", {
            "status": "exact", "epoch": loaded["epoch"] + 1, "state_sha256": sha(path / "state.pt"),
            "model_optimizer_python_numpy_torch_cuda_rng": "all_exact",
        })
        return loaded

    def cluster(data, clusters):
        before = np.random.get_state()
        predictor = np.random.RandomState(0)
        predictor.set_state(before)
        seeds = predictor.randint(np.iinfo(np.int32).max, size=10)
        centered = np.array(data, copy=True)
        centered -= centered.mean(axis=0)
        call_id = f"{invocation}-{context['epoch']}-{context['calls']}-{time.time_ns()}"
        call = {
            "call_id": call_id, "invocation": invocation, "epoch": context["epoch"],
            "phase": ("export" if context["phase"] == "export" else
                      "training" if context["calls"] < 4 else "validation"),
            "level": context["calls"] % 4, "expected_input_sha256": array_sha(centered),
            "initialization_seeds": seeds.tolist(), "rng_before_sha256": rng_sha(before),
        }
        atomic_json(events / "contexts" / f"{call_id}.json", call)
        atomic_json(events / "task-context.json", call)
        result = original_cluster(data, clusters)
        require_equal(predictor.get_state(), np.random.get_state(), "native_clustering_rng_budget")
        context["calls"] += 1
        return result

    def append(path, record):
        original_append(path, record)
        context.update(epoch=record["epoch"] + 1, calls=0)

    def save(directory, trainer, completed_epoch, spec, selected, recovery_root):
        original_save(directory, trainer, completed_epoch, spec, selected, recovery_root)
        receipt = read_json(directory / "completed.json")
        seal_compatibility(directory, {
            **common, "artifact": "checkpoint", "epoch": completed_epoch + 1,
            "state_sha256": receipt["state_sha256"], "selected_checkpoint": selected,
        }, events, args.admission)

    def export(model, dataset, clusters, directory, seed, recovery_root):
        context.update(phase="export", epoch=context["loaded"], calls=0)
        if (directory / "compatibility").exists():
            raise FileExistsError("A preserved export attestation requires explicit readmission")
        result = original_export(model, dataset, clusters, directory, seed, recovery_root)
        latest = read_json(output / "latest-state.json")
        latest_directory = Path(latest["checkpoint"])
        seal_compatibility(directory, {
            **common, "artifact": "export", "selected_epoch": context["loaded"],
            "index_sha256": result["index_sha256"], "export_sha256": sha(directory / "export.json"),
            "final_checkpoint_attestation_sha256":
                sha(latest_directory / "compatibility" / "attestation.json"),
        }, events, args.admission)
        return result

    def write(path, value):
        if path.name == "source-receipt.json" and path.exists():
            if read_json(path) != value:
                raise ValueError("Preserved author source receipt differs")
            return
        if path == output / "completed.json":
            value = {**value, "rq_compatibility": {
                "policy": POLICY, "admission_sha256": common["admission_sha256"],
                "export_attestation_sha256":
                    sha(output / "export" / "compatibility" / "attestation.json"),
                "base_state_sha256": fixed["base_state"],
            }}
        return original_write(path, value)

    native.load_state, native.cluster_codebook, native.save_state = load, cluster, save
    native.append_jsonl, native.export_indices, native.write_json = append, export, write
    sys.argv = [
        str(source / "tokenizer_run.py"), "--upstream", str(root / "upstream"),
        "--semantic", str(root / "embeddings-openllama-v2-v1/Instruments/embeddings.npy"),
        "--cf", str(root / "runs/Instruments-cf-bce-s42/item-embeddings.pt"),
        "--embedding-receipt", str(root / "embeddings-openllama-v2-v1/Instruments/embedding-receipt.json"),
        "--method-spec", str(root / "project/configs/letter-openllama-v2-method-v1.json"),
        "--manifest", str(root / "project/configs/letter-reproduction-v1.json"),
        "--dataset", "Instruments", "--output", str(output), "--alpha", fixed["alpha"], "--beta", "0",
        "--seed", "42", "--epochs", "20000", "--batch-size", "1024", "--eval-step", "2000",
        "--checkpoint-every", "100", "--resume", str(args.resume),
    ]
    if args.stop_after_epoch is not None:
        sys.argv.extend(["--stop-after-epoch", str(args.stop_after_epoch)])
    native.main()
    atomic_json(runtime / f"finished-{invocation}.json", {
        "status": "completed" if (output / "completed.json").is_file() else "drained",
        "latest": read_json(output / "latest-state.json"), "pid": os.getpid(),
        "finished_unix": time.time(),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--stop-after-epoch", type=int)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
