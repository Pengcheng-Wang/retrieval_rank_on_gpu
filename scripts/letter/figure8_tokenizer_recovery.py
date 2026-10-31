"""Sidecar recovery contract for two deterministic Figure 8 tokenizer faults."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import socket
import sys
import time

import numpy as np

from rq_recovery_compat import POLICY
from rq_recovery_runtime import file_manifest


RECOVERY_ARMS = {
    "beta-0p001": {
        "host": "node-1",
        "gpu": 0,
        "base_epoch": 16000,
        "first_failed_epoch": 16401,
    },
    "clusters-20": {
        "host": "node-1",
        "gpu": 1,
        "base_epoch": 11000,
        "first_failed_epoch": 11431,
    },
}
ENDPOINT_ERROR = "IndexError: index 256 is out of bounds for axis 0 with size 256"
CONTROL_FILES = (
    "figure8_tokenizer_recovery.py",
    "figure8_tokenizer_recovery_runtime.py",
    "rq_recovery_compat.py",
    "rq_recovery_runtime.py",
    "rq_recovery_lineage.py",
    "rq_recovery_control.py",
)


def validate_recovery_identity(spec: dict, arm: str, hostname: str, gpu: int) -> dict:
    fixed = RECOVERY_ARMS.get(arm)
    if fixed is None:
        raise ValueError("Figure 8 tokenizer recovery is restricted to two observed faults")
    output = PurePosixPath(spec.get("output", ""))
    if (
        spec.get("host") != fixed["host"]
        or spec.get("gpu") != fixed["gpu"]
        or hostname != fixed["host"]
        or gpu != fixed["gpu"]
        or output.name != arm
        or output.parent.name != "runs"
        or output.parent.parent.name != "figure8-sweeps-v1"
        or not output.is_absolute()
    ):
        raise ValueError("Figure 8 tokenizer recovery host, GPU or output identity differs")
    return {"arm": arm, **fixed}


def parse_native_failures(content: str, arm: str) -> dict:
    fixed = RECOVERY_ARMS.get(arm)
    if fixed is None:
        raise ValueError("Unknown Figure 8 tokenizer recovery arm")
    traceback_positions = [
        match.start()
        for match in re.finditer(
            r"joblib\.externals\.loky\.process_executor\._RemoteTraceback:", content,
        )
    ]
    if len(traceback_positions) != 5 or content.count(ENDPOINT_ERROR) != 10:
        raise ValueError("Native failure evidence must contain five exact repeated endpoint faults")
    epochs = []
    for position in traceback_positions:
        matches = list(re.finditer(r'\{"epoch":\s*(\d+),[^\n]*\}', content[:position]))
        if not matches:
            raise ValueError("Native failure evidence lacks the completed epoch before a fault")
        epochs.append(int(matches[-1].group(1)))
    expected = fixed["first_failed_epoch"] - 1
    if epochs != [expected] * 5:
        raise ValueError("Native endpoint fault did not repeat at the exact frozen state")
    return {
        "attempts": 5,
        "completed_epoch_before_fault": expected,
        "first_failed_epoch": fixed["first_failed_epoch"],
        "candidate_index": 256,
        "sample_count": 256,
    }


def native_arguments(
    spec: dict,
    output: Path,
    resume: Path,
    stop_after_epoch: int | None = None,
) -> list:
    control = Path(spec["control"])
    argv = [
        control / "scripts/letter/tokenizer_run.py",
        "--upstream", str(Path(spec["root"]) / "upstream"),
        "--semantic", spec["semantic"],
        "--cf", spec["cf"],
        "--output", str(output),
        "--dataset", spec["dataset"],
        "--embedding-receipt", spec["embedding_receipt"],
        "--manifest", spec["manifest"],
        "--method-spec", spec["method_spec"],
        "--codes", *map(str, spec["codes"]),
        "--epsilons", *map(str, spec["epsilons"]),
        "--dimension", str(spec["dimension"]),
        "--layers", *map(str, spec["layers"]),
        "--clusters", str(spec["clusters"]),
        "--alpha", str(spec["alpha"]),
        "--beta", str(spec["beta"]),
        "--seed", str(spec["seed"]),
        "--epochs", str(spec["epochs"]),
        "--batch-size", str(spec["batch_size"]),
        "--eval-step", str(spec["eval_step"]),
        "--checkpoint-every", str(spec["checkpoint_every"]),
        "--resume", str(resume),
    ]
    if stop_after_epoch is not None:
        argv.extend(["--stop-after-epoch", str(stop_after_epoch)])
    return argv


def _digest(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_probe_admission(value: dict) -> dict:
    required = {
        "status", "kind", "policy", "identity", "spec_sha256", "run_config_sha256",
        "resume", "overlay_receipt_sha256", "native_module_sha256", "failure",
        "failure_log_sha256", "control_sha256",
    }
    if (
        set(value) != required
        or value.get("status") != "admitted"
        or value.get("kind") != "figure8_tokenizer_kmeans_probe_admission_v1"
        or value.get("policy") != POLICY
    ):
        raise ValueError("Figure 8 tokenizer probe admission schema differs")
    identity = value["identity"]
    fixed = RECOVERY_ARMS.get(identity.get("arm")) if isinstance(identity, dict) else None
    if fixed is None or identity != {"arm": identity["arm"], **fixed}:
        raise ValueError("Figure 8 tokenizer probe admission identity differs")
    resume = value["resume"]
    if (
        set(resume) != {"epoch", "completed_sha256", "state_sha256"}
        or resume["epoch"] != fixed["base_epoch"]
    ):
        raise ValueError("Figure 8 tokenizer probe resume identity differs")
    digests = [
        value["spec_sha256"],
        value["run_config_sha256"],
        resume["completed_sha256"],
        resume["state_sha256"],
        value["overlay_receipt_sha256"],
        value["native_module_sha256"],
        value["failure_log_sha256"],
        *value["control_sha256"].values(),
    ]
    if not value["control_sha256"] or not all(_digest(digest) for digest in digests):
        raise ValueError("Figure 8 tokenizer probe admission contains an invalid digest")
    failure = value["failure"]
    expected_failure = {
        "attempts": 5,
        "completed_epoch_before_fault": fixed["first_failed_epoch"] - 1,
        "first_failed_epoch": fixed["first_failed_epoch"],
        "candidate_index": 256,
        "sample_count": 256,
    }
    if failure != expected_failure:
        raise ValueError("Figure 8 tokenizer native failure evidence differs")
    return value


def probe_admission(
    *,
    identity: dict,
    spec_sha256: str,
    run_config_sha256: str,
    resume_completed_sha256: str,
    resume_state_sha256: str,
    overlay_receipt_sha256: str,
    native_module_sha256: str,
    failure_log_sha256: str,
    failure: dict,
    control_sha256: dict,
) -> dict:
    value = {
        "status": "admitted",
        "kind": "figure8_tokenizer_kmeans_probe_admission_v1",
        "policy": POLICY,
        "identity": identity,
        "spec_sha256": spec_sha256,
        "run_config_sha256": run_config_sha256,
        "resume": {
            "epoch": identity["base_epoch"],
            "completed_sha256": resume_completed_sha256,
            "state_sha256": resume_state_sha256,
        },
        "overlay_receipt_sha256": overlay_receipt_sha256,
        "native_module_sha256": native_module_sha256,
        "failure_log_sha256": failure_log_sha256,
        "failure": failure,
        "control_sha256": control_sha256,
    }
    return validate_probe_admission(value)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".next")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def require_hash(path: Path, expected: str) -> None:
    if path.is_symlink() or sha(path) != expected:
        raise ValueError(f"Figure 8 tokenizer recovery checksum mismatch: {path}")


def control_hashes(directory: Path) -> dict:
    return {name: sha(directory / name) for name in CONTROL_FILES}


def validate_overlay(overlay: Path, native_package: Path) -> dict:
    from figure8_tokenizer_recovery_runtime import MODULE, NATIVE_SHA256

    receipt = read_json(overlay / "overlay.json")
    expected = {
        "policy": POLICY,
        "native_module_sha256": NATIVE_SHA256,
        "changed": [MODULE],
        "original_files": file_manifest(native_package),
        "overlay_files": file_manifest(overlay / "k_means_constrained"),
    }
    if receipt != expected:
        raise ValueError("Figure 8 tokenizer recovery overlay receipt differs")
    return receipt


def prepare_probe(args) -> dict:
    from bank_tokenizers import read_state
    from figure8_tokenizer import (
        ARM_ORDER,
        FIGURE8_TOKENIZER_HARNESS_SHA256,
        arm_tokenizer_spec,
        validate_arm_launch_spec,
    )

    spec = read_json(args.spec)
    identity = validate_recovery_identity(spec, args.arm, socket.gethostname(), args.gpu)
    expected = arm_tokenizer_spec({"arms": list(ARM_ORDER)}, args.arm)
    validate_arm_launch_spec(spec, expected, args.arm)
    production = Path(spec["output"]).resolve()
    resume = args.resume.resolve()
    if (
        production.is_symlink()
        or resume.parent != production / "checkpoints"
        or resume.name != f"epoch-{identity['base_epoch']:05d}"
    ):
        raise ValueError("Probe resume checkpoint differs from the exact failed production run")
    config_path = production / "run-config.json"
    config = read_json(config_path)
    if (
        config.get("harness_sha256") != FIGURE8_TOKENIZER_HARNESS_SHA256
        or {key: config.get(key) for key in expected} != expected
    ):
        raise ValueError("Frozen Figure 8 tokenizer run config differs")
    state = read_state(resume, config)
    if state["epoch"] + 1 != identity["base_epoch"]:
        raise ValueError("Probe base checkpoint epoch differs")
    latest = read_json(production / "latest-state.json")
    if (
        latest.get("epoch") != identity["base_epoch"]
        or Path(latest.get("checkpoint", "")).resolve() != resume
    ):
        raise ValueError("Production latest-state does not name the admitted base checkpoint")
    native_package = args.native_package.resolve()
    overlay = args.overlay.resolve()
    validate_overlay(overlay, native_package)
    failure_content = args.failure_log.read_text(encoding="utf-8")
    failure = parse_native_failures(failure_content, args.arm)
    admission = probe_admission(
        identity=identity,
        spec_sha256=sha(args.spec),
        run_config_sha256=sha(config_path),
        resume_completed_sha256=sha(resume / "completed.json"),
        resume_state_sha256=sha(resume / "state.pt"),
        overlay_receipt_sha256=sha(overlay / "overlay.json"),
        native_module_sha256=sha(
            native_package / "sklearn_import" / "cluster" / "k_means_.py",
        ),
        failure_log_sha256=sha(args.failure_log),
        failure=failure,
        control_sha256=control_hashes(args.recovery_control),
    )
    if args.output.exists():
        raise FileExistsError("Figure 8 tokenizer probe admissions are immutable")
    args.output.mkdir(parents=True)
    atomic_json(args.output / "admission.json", admission)
    shutil.copyfile(args.spec, args.output / "arm-spec.json")
    shutil.copyfile(args.failure_log, args.output / "native-failures.log")
    shutil.copyfile(overlay / "overlay.json", args.output / "overlay.json")
    atomic_json(args.output / "failure-summary.json", failure)
    return admission


def equal_state(left, right, where: str = "state") -> None:
    import torch

    if torch.is_tensor(left):
        equal = (
            torch.is_tensor(right)
            and left.dtype == right.dtype
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    elif isinstance(left, np.ndarray):
        equal = (
            isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and np.array_equal(left, right)
        )
    elif isinstance(left, dict):
        equal = isinstance(right, dict) and left.keys() == right.keys()
        if equal:
            for key in left:
                equal_state(left[key], right[key], f"{where}.{key}")
    elif isinstance(left, (list, tuple)):
        equal = type(left) is type(right) and len(left) == len(right)
        if equal:
            for index, value in enumerate(left):
                equal_state(value, right[index], f"{where}[{index}]")
    else:
        equal = type(left) is type(right) and left == right
    if not equal:
        raise ValueError(f"Exact tokenizer recovery state mismatch: {where}")


def archive_events(events: Path, destination: Path, invocation: str) -> dict:
    from rq_recovery_control import audit_archive
    from figure8_tokenizer_recovery_runtime import audit_event_archive

    audit_archive(events, destination)
    return audit_event_archive(destination, invocation)


def seal_attestation(
    directory: Path,
    *,
    artifact: str,
    common: dict,
    events: Path,
    admission_path: Path,
    artifact_fields: dict,
) -> dict:
    target = directory / "compatibility"
    if target.exists():
        return verify_attestation(target, common["admission_sha256"], artifact)
    target.mkdir()
    shutil.copyfile(admission_path, target / "admission.json")
    audited = archive_events(events, target / "events.zip", common["invocation"])
    result = {
        **common,
        **artifact_fields,
        "artifact": artifact,
        "runtime_summary": audited["summary"],
        "corrections_by_epoch_level_phase": audited["corrections_by_epoch_level_phase"],
        "files": file_manifest(target),
    }
    atomic_json(target / "attestation.json", result)
    return result


def verify_attestation(path: Path, admission_sha256: str, artifact: str) -> dict:
    from figure8_tokenizer_recovery_runtime import audit_event_archive

    result = read_json(path / "attestation.json")
    actual = file_manifest(path)
    actual.pop("attestation.json", None)
    if (
        result.get("kind") != "figure8_tokenizer_kmeans_compatibility_attestation_v1"
        or result.get("policy") != POLICY
        or result.get("artifact") != artifact
        or result.get("admission_sha256") != admission_sha256
        or actual != result.get("files")
    ):
        raise ValueError("Figure 8 tokenizer compatibility attestation differs")
    require_hash(path / "admission.json", admission_sha256)
    audited = audit_event_archive(path / "events.zip", result["invocation"])
    if (
        audited["summary"] != result.get("runtime_summary")
        or audited["corrections_by_epoch_level_phase"]
        != result.get("corrections_by_epoch_level_phase")
    ):
        raise ValueError("Figure 8 tokenizer compatibility event audit differs")
    return result


def validate_probe_files(args, admission: dict, spec: dict) -> None:
    identity = validate_recovery_identity(
        spec, args.arm, socket.gethostname(), args.gpu,
    )
    if identity != admission["identity"]:
        raise ValueError("Figure 8 tokenizer probe identity differs from admission")
    require_hash(args.spec, admission["spec_sha256"])
    require_hash(Path(spec["output"]) / "run-config.json", admission["run_config_sha256"])
    require_hash(args.resume / "completed.json", admission["resume"]["completed_sha256"])
    require_hash(args.resume / "state.pt", admission["resume"]["state_sha256"])
    require_hash(args.overlay / "overlay.json", admission["overlay_receipt_sha256"])
    validate_overlay(args.overlay.resolve(), args.native_package.resolve())
    require_hash(
        args.native_package / "sklearn_import" / "cluster" / "k_means_.py",
        admission["native_module_sha256"],
    )
    for name, digest in admission["control_sha256"].items():
        require_hash(args.recovery_control / name, digest)
    if args.admission.is_symlink():
        raise ValueError("Figure 8 tokenizer probe admission is redirected")
    evidence_root = args.admission.parent
    require_hash(evidence_root / "native-failures.log", admission["failure_log_sha256"])
    if read_json(evidence_root / "failure-summary.json") != admission["failure"]:
        raise ValueError("Figure 8 tokenizer failure summary differs from admission")


def run_probe(args) -> dict:
    import torch

    admission = validate_probe_admission(read_json(args.admission))
    spec = read_json(args.spec)
    validate_probe_files(args, admission, spec)
    identity = admission["identity"]
    if (
        args.output.resolve() == Path(spec["output"]).resolve()
        or not args.output.resolve().is_relative_to(
            Path(spec["root"]).resolve() / "figure8-tokenizer-recovery-v1" / "probes",
        )
        or args.stop_after_epoch != identity["first_failed_epoch"]
    ):
        raise ValueError("Probe output or endpoint escapes the isolated recovery namespace")
    if args.output.exists():
        raise FileExistsError("Figure 8 tokenizer recovery probes are immutable")
    scripts = Path(spec["control"]) / "scripts" / "letter"
    sys.path.insert(0, str(scripts))
    import bank_tokenizers
    import tokenizer_run as native
    import k_means_constrained.sklearn_import.cluster.k_means_ as overlay_module
    from figure8_tokenizer_recovery_runtime import MODULE

    if (
        sha(Path(native.__file__)) != read_json(Path(spec["output"]) / "run-config.json")[
            "harness_sha256"
        ]
        or Path(overlay_module.__file__).resolve()
        != args.overlay.resolve() / "k_means_constrained" / MODULE
    ):
        raise ValueError("Probe did not import the frozen tokenizer harness and overlay")
    config = read_json(Path(spec["output"]) / "run-config.json")
    expected_state = bank_tokenizers.read_state(args.resume, config)
    first_epoch = expected_state["epoch"] + 1
    invocation = f"{args.arm}-{os.getpid()}-{time.time_ns()}"
    events = args.output.parent / "runtime" / invocation
    (events / "contexts").mkdir(parents=True)
    os.environ["FIGURE8_TOKENIZER_RECOVERY_EVENTS"] = str(events)
    os.environ["FIGURE8_TOKENIZER_RECOVERY_OVERLAY"] = str(args.overlay.resolve())
    os.environ["FIGURE8_TOKENIZER_NATIVE_KMEANS_MODULE"] = str(
        args.native_package / "sklearn_import" / "cluster" / "k_means_.py",
    )
    levels = len(spec["codes"])
    context = {"epoch": first_epoch + 1, "calls": 0, "phase": "training"}
    original_load = native.load_state
    original_cluster = native.cluster_codebook
    original_save = native.save_state
    original_append = native.append_jsonl
    original_export = native.export_indices
    original_write = native.write_json
    common = {
        "kind": "figure8_tokenizer_kmeans_compatibility_attestation_v1",
        "policy": POLICY,
        "purpose": "probe",
        "arm": args.arm,
        "run_id": Path(spec["output"]).name,
        "run_config_sha256": admission["run_config_sha256"],
        "native_harness_sha256": config["harness_sha256"],
        "admission_sha256": sha(args.admission),
        "base_epoch": identity["base_epoch"],
        "base_state_sha256": admission["resume"]["state_sha256"],
        "invocation": invocation,
        "resumed_from_epoch": first_epoch,
        "resumed_from_state_sha256": admission["resume"]["state_sha256"],
    }

    def load(path, trainer, run_spec):
        loaded = original_load(path, trainer, run_spec)
        equal_state(expected_state, loaded, "loaded")
        equal_state(loaded["model"], trainer.model.state_dict(), "restored.model")
        equal_state(loaded["optimizer"], trainer.optimizer.state_dict(), "restored.optimizer")
        equal_state(loaded["python_rng"], random.getstate(), "restored.python_rng")
        equal_state(loaded["numpy_rng"], np.random.get_state(), "restored.numpy_rng")
        equal_state(loaded["torch_rng"], torch.get_rng_state(), "restored.torch_rng")
        equal_state(loaded["cuda_rng"], torch.cuda.get_rng_state_all(), "restored.cuda_rng")
        atomic_json(events / "restored.json", {
            "status": "exact",
            "epoch": loaded["epoch"] + 1,
            "state_sha256": sha(path / "state.pt"),
            "model_optimizer_python_numpy_torch_cuda_rng": "all_exact",
        })
        return loaded

    def cluster(data, clusters):
        before = np.random.get_state()
        predictor = np.random.RandomState()
        predictor.set_state(before)
        seeds = predictor.randint(np.iinfo(np.int32).max, size=10)
        centered = np.array(data, copy=True)
        centered -= centered.mean(axis=0)
        call_id = f"{invocation}-{context['epoch']}-{context['calls']}-{time.time_ns()}"
        call = {
            "call_id": call_id,
            "invocation": invocation,
            "epoch": context["epoch"],
            "phase": context["phase"] if context["phase"] == "export"
            else ("training" if context["calls"] < levels else "validation"),
            "level": context["calls"] % levels,
            "expected_input_sha256":
                __import__("rq_recovery_runtime").array_sha(centered),
            "initialization_seeds": seeds.tolist(),
            "rng_before_sha256": __import__("rq_recovery_runtime").rng_sha(before),
        }
        atomic_json(events / "contexts" / f"{call_id}.json", call)
        atomic_json(events / "task-context.json", call)
        result = original_cluster(data, clusters)
        equal_state(predictor.get_state(), np.random.get_state(), "clustering_rng_budget")
        context["calls"] += 1
        return result

    def save(directory, trainer, epoch, run_spec, selected, recovery_root):
        original_save(directory, trainer, epoch, run_spec, selected, recovery_root)
        receipt = read_json(directory / "completed.json")
        seal_attestation(
            directory,
            artifact="checkpoint",
            common=common,
            events=events,
            admission_path=args.admission,
            artifact_fields={
                "epoch": epoch + 1,
                "state_sha256": receipt["state_sha256"],
                "selected_checkpoint": selected,
            },
        )

    def append(path, value):
        original_append(path, value)
        context.update(epoch=value["epoch"] + 1, calls=0)

    def export(model, dataset, clusters, directory, seed, recovery_root):
        context.update(phase="export", calls=0)
        result = original_export(model, dataset, clusters, directory, seed, recovery_root)
        seal_attestation(
            directory,
            artifact="export",
            common=common,
            events=events,
            admission_path=args.admission,
            artifact_fields={
                "index_sha256": result["index_sha256"],
                "selected_checkpoint":
                    read_json(args.output / "latest-state.json")["checkpoint"],
            },
        )
        return result

    def write(path, value):
        return original_write(path, value)

    native.load_state = load
    native.cluster_codebook = cluster
    native.save_state = save
    native.append_jsonl = append
    native.export_indices = export
    native.write_json = write
    sys.argv = [str(value) for value in native_arguments(
        spec, args.output, args.resume, args.stop_after_epoch,
    )]
    native.main()
    latest = read_json(args.output / "latest-state.json")
    checkpoint = Path(latest["checkpoint"])
    attestation = verify_attestation(
        checkpoint / "compatibility", sha(args.admission), "checkpoint",
    )
    if (
        latest["epoch"] != identity["first_failed_epoch"]
        or attestation["runtime_summary"]["native_faults"] < 1
    ):
        raise ValueError("Probe did not cross and attest the exact native endpoint fault")
    result = {
        "status": "pass",
        "kind": "figure8_tokenizer_kmeans_probe_result_v1",
        "arm": args.arm,
        "admission_sha256": sha(args.admission),
        "checkpoint": str(checkpoint),
        "checkpoint_state_sha256": sha(checkpoint / "state.pt"),
        "attestation_sha256": sha(checkpoint / "compatibility" / "attestation.json"),
        "runtime_summary": attestation["runtime_summary"],
        "corrections_by_epoch_level_phase":
            attestation["corrections_by_epoch_level_phase"],
    }
    atomic_json(args.output.parent / "probe-result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    overlay = commands.add_parser("make-overlay")
    overlay.add_argument("--native-package", type=Path, required=True)
    overlay.add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser("prepare-probe")
    run = commands.add_parser("run-probe")
    for command in (prepare, run):
        command.add_argument("--arm", choices=tuple(RECOVERY_ARMS), required=True)
        command.add_argument("--spec", type=Path, required=True)
        command.add_argument("--resume", type=Path, required=True)
        command.add_argument("--overlay", type=Path, required=True)
        command.add_argument("--native-package", type=Path, required=True)
        command.add_argument("--recovery-control", type=Path, required=True)
        command.add_argument("--gpu", type=int, required=True)
    prepare.add_argument("--failure-log", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    run.add_argument("--admission", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--stop-after-epoch", type=int, required=True)
    args = parser.parse_args()
    if args.command == "make-overlay":
        from figure8_tokenizer_recovery_runtime import make_overlay
        result = make_overlay(args.native_package.resolve(), args.output.resolve())
    elif args.command == "prepare-probe":
        result = prepare_probe(args)
    else:
        result = run_probe(args)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
