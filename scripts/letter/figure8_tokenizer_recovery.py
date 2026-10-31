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
import signal
import socket
import subprocess
import sys
import time
import zipfile

import numpy as np

from rq_recovery_compat import POLICY
from rq_recovery_runtime import file_manifest


RECOVERY_ARMS = {
    "beta-0p001": {
        "host": "node-1",
        "gpu": 2,
        "base_epoch": 16000,
        "first_failed_epoch": 16401,
    },
    "clusters-20": {
        "host": "node-1",
        "gpu": 7,
        "base_epoch": 11000,
        "first_failed_epoch": 11431,
    },
}
ENDPOINT_ERROR = "IndexError: index 256 is out of bounds for axis 0 with size 256"
CONTROL_FILES = (
    "bank.py",
    "bank_tokenizers.py",
    "figure8_tokenizer.py",
    "figure8_tokenizer_recovery.py",
    "figure8_tokenizer_recovery_runtime.py",
    "protocol.py",
    "rq_recovery_compat.py",
    "rq_recovery_runtime.py",
    "rq_recovery_lineage.py",
    "rq_recovery_control.py",
    "tokenizer_run.py",
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


def validate_production_admission(value: dict) -> dict:
    required = {
        "status", "kind", "policy", "probe_admission", "probe",
        "probe_bank_receipt_sha256", "probe_result_sha256",
        "production_control_sha256",
    }
    if (
        set(value) != required
        or value.get("status") != "admitted"
        or value.get("kind") != "figure8_tokenizer_kmeans_production_admission_v1"
        or value.get("policy") != POLICY
    ):
        raise ValueError("Figure 8 tokenizer production admission schema differs")
    base = validate_probe_admission(value["probe_admission"])
    probe = value["probe"]
    required_probe = {
        "admission_sha256", "checkpoint_state_sha256", "attestation_sha256",
        "native_defined", "native_faults", "corrections",
        "corrections_by_epoch_level_phase",
    }
    if set(probe) != required_probe:
        raise ValueError("Figure 8 tokenizer production probe summary differs")
    if not value["production_control_sha256"] or not all(_digest(value[name]) for name in (
        "probe_bank_receipt_sha256", "probe_result_sha256",
    )) or not all(_digest(digest) for digest in value["production_control_sha256"].values()
    ) or not all(_digest(probe[name]) for name in (
        "admission_sha256", "checkpoint_state_sha256", "attestation_sha256",
    )):
        raise ValueError("Figure 8 tokenizer production admission contains an invalid digest")
    corrections = probe["corrections_by_epoch_level_phase"]
    failure_epoch = str(base["identity"]["first_failed_epoch"])
    if (
        probe["native_faults"] < 1
        or probe["native_defined"] < 1
        or probe["corrections"] < probe["native_faults"]
        or failure_epoch not in corrections
    ):
        raise ValueError("Figure 8 tokenizer production admission lacks the exact native fault gate")
    return value


def production_admission(
    probe_contract: dict,
    probe_result: dict,
    *,
    probe_bank_receipt_sha256: str,
    probe_result_sha256: str,
    production_control_sha256: dict,
) -> dict:
    base = validate_probe_admission(probe_contract)
    runtime = probe_result.get("runtime_summary", {})
    if (
        probe_result.get("status") != "pass"
        or probe_result.get("kind") != "figure8_tokenizer_kmeans_probe_result_v1"
        or probe_result.get("arm") != base["identity"]["arm"]
    ):
        raise ValueError("Figure 8 tokenizer probe result identity differs")
    value = {
        "status": "admitted",
        "kind": "figure8_tokenizer_kmeans_production_admission_v1",
        "policy": POLICY,
        "probe_admission": base,
        "probe": {
            "admission_sha256": probe_result["admission_sha256"],
            "checkpoint_state_sha256": probe_result["checkpoint_state_sha256"],
            "attestation_sha256": probe_result["attestation_sha256"],
            "native_defined": runtime.get("native_defined"),
            "native_faults": runtime.get("native_faults"),
            "corrections": runtime.get("corrections"),
            "corrections_by_epoch_level_phase":
                probe_result["corrections_by_epoch_level_phase"],
        },
        "probe_bank_receipt_sha256": probe_bank_receipt_sha256,
        "probe_result_sha256": probe_result_sha256,
        "production_control_sha256": production_control_sha256,
    }
    return validate_production_admission(value)


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


def reset_event_segment(events: Path) -> None:
    for path in list(events.iterdir()):
        if path.is_symlink():
            raise ValueError("Compatibility runtime evidence is redirected")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    (events / "contexts").mkdir()


def seal_attestation(
    directory: Path,
    *,
    artifact: str,
    common: dict,
    events: Path,
    admission_path: Path,
    artifact_fields: dict,
    parent: Path | None = None,
) -> dict:
    target = directory / "compatibility"
    if target.exists():
        return verify_attestation(
            target, common["admission_sha256"], artifact, common,
        )
    target.mkdir()
    shutil.copyfile(admission_path, target / "admission.json")
    parent_result = None
    if parent is not None:
        parent_result = verify_attestation(
            parent, common["admission_sha256"], "checkpoint", common,
        )
        shutil.copytree(parent, target / "parent")
    validate_parent_binding(common, parent_result)
    audited = archive_events(events, target / "events.zip", common["invocation"])
    restoration = events / "restored.json"
    result = {
        **common,
        **artifact_fields,
        "artifact": artifact,
        "runtime_summary": audited["summary"],
        "corrections_by_epoch_level_phase": audited["corrections_by_epoch_level_phase"],
        "parent_attestation_sha256":
            sha(parent / "attestation.json") if parent is not None else None,
        "restoration_sha256": sha(restoration) if restoration.is_file() else None,
        "files": file_manifest(target),
    }
    atomic_json(target / "attestation.json", result)
    return result


def validate_parent_binding(common: dict, parent: dict | None) -> str:
    if parent is None:
        if (
            common.get("parent_epoch") != common.get("base_epoch")
            or common.get("parent_state_sha256") != common.get("base_state_sha256")
        ):
            raise ValueError("Compatibility continuation lacks its exact native base")
        return "native-base"
    if (
        parent.get("artifact") != "checkpoint"
        or parent.get("epoch") != common.get("parent_epoch")
        or parent.get("state_sha256") != common.get("parent_state_sha256")
    ):
        raise ValueError("Compatibility parent differs from the exact resume state")
    return "compatible-parent"


def validate_attestation_identity(result: dict, expected: dict) -> None:
    keys = (
        "kind", "policy", "purpose", "arm", "run_id", "run_config_sha256",
        "native_harness_sha256", "admission_sha256", "base_epoch", "base_state_sha256",
    )
    if any(result.get(key) != expected.get(key) for key in keys):
        raise ValueError("Figure 8 tokenizer compatibility identity differs")


def verify_attestation(
    path: Path,
    admission_sha256: str,
    artifact: str,
    expected: dict | None = None,
) -> dict:
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
    if expected is not None:
        validate_attestation_identity(result, expected)
    require_hash(path / "admission.json", admission_sha256)
    audited = audit_event_archive(path / "events.zip", result["invocation"])
    if (
        audited["summary"] != result.get("runtime_summary")
        or audited["corrections_by_epoch_level_phase"]
        != result.get("corrections_by_epoch_level_phase")
    ):
        raise ValueError("Figure 8 tokenizer compatibility event audit differs")
    restoration_sha256 = result.get("restoration_sha256")
    with zipfile.ZipFile(path / "events.zip") as archive:
        if (
            "restoration_sha256" not in result
            and result.get("purpose") == "probe"
            and "restored.json" in archive.namelist()
        ):
            restoration_sha256 = hashlib.sha256(
                archive.read("restored.json"),
            ).hexdigest()
        if restoration_sha256 is None:
            if "restored.json" in archive.namelist():
                raise ValueError("Unclaimed tokenizer restoration receipt")
        else:
            try:
                restored_bytes = archive.read("restored.json")
            except KeyError as error:
                raise ValueError("Tokenizer restoration receipt is missing") from error
            if hashlib.sha256(restored_bytes).hexdigest() != restoration_sha256:
                raise ValueError("Tokenizer restoration receipt checksum differs")
            restored = json.loads(restored_bytes)
            if restored != {
                "status": "exact",
                "epoch": result["resumed_from_epoch"],
                "state_sha256": result["resumed_from_state_sha256"],
                "model_optimizer_python_numpy_torch_cuda_rng": "all_exact",
            }:
                raise ValueError("Tokenizer restoration was not exact full-state recovery")
    parent_path = path / "parent"
    if parent_path.is_dir():
        parent = verify_attestation(
            parent_path, admission_sha256, "checkpoint", expected,
        )
        if sha(parent_path / "attestation.json") != result.get("parent_attestation_sha256"):
            raise ValueError("Figure 8 tokenizer compatibility parent checksum differs")
    else:
        parent = None
        if result.get("parent_attestation_sha256") is not None:
            raise ValueError("Figure 8 tokenizer compatibility parent is missing")
    validate_parent_binding(result, parent)
    return result


def attestation_chain(
    path: Path, admission_sha256: str, artifact: str, expected: dict,
) -> list[dict]:
    result = []
    current = path
    current_artifact = artifact
    while True:
        result.append(
            verify_attestation(current, admission_sha256, current_artifact, expected),
        )
        parent = current / "parent"
        if not parent.is_dir():
            return result
        current = parent
        current_artifact = "checkpoint"


def validate_invocation_restorations(chain: list[dict]) -> None:
    invocations = {}
    for attestation in chain:
        identity = attestation["invocation"]
        group = invocations.setdefault(identity, {
            "resume": (
                attestation["resumed_from_epoch"],
                attestation["resumed_from_state_sha256"],
            ),
            "restorations": 0,
        })
        if group["resume"] != (
            attestation["resumed_from_epoch"],
            attestation["resumed_from_state_sha256"],
        ):
            raise ValueError("Tokenizer invocation changed its exact resume identity")
        group["restorations"] += attestation.get("restoration_sha256") is not None
    if not invocations or any(group["restorations"] != 1 for group in invocations.values()):
        raise ValueError("Every tokenizer invocation requires one exact restoration receipt")


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


def prepare_production(args) -> dict:
    from figure8_tokenizer import verify_durable_tree

    probe_contract = validate_probe_admission(read_json(args.probe_admission))
    if sha(args.probe_admission) != read_json(
        args.probe_bank / "probe-result.json",
    ).get("admission_sha256"):
        raise ValueError("Durable probe result names another probe admission")
    durable = verify_durable_tree(args.probe_bank)
    probe_result_path = args.probe_bank / "probe-result.json"
    probe_result = read_json(probe_result_path)
    identity = probe_contract["identity"]
    checkpoint = (
        args.probe_bank / "run" / "checkpoints"
        / f"epoch-{identity['first_failed_epoch']:05d}"
    )
    require_hash(checkpoint / "state.pt", probe_result["checkpoint_state_sha256"])
    require_hash(
        checkpoint / "compatibility" / "attestation.json",
        probe_result["attestation_sha256"],
    )
    attestation = verify_attestation(
        checkpoint / "compatibility",
        probe_result["admission_sha256"],
        "checkpoint",
    )
    if (
        attestation["epoch"] != identity["first_failed_epoch"]
        or attestation["runtime_summary"] != probe_result["runtime_summary"]
        or attestation["corrections_by_epoch_level_phase"]
        != probe_result["corrections_by_epoch_level_phase"]
    ):
        raise ValueError("Durable probe attestation differs from its result")
    admission = production_admission(
        probe_contract,
        probe_result,
        probe_bank_receipt_sha256=durable["bank_receipt_sha256"],
        probe_result_sha256=sha(probe_result_path),
        production_control_sha256=control_hashes(args.recovery_control),
    )
    if args.output.exists():
        raise FileExistsError("Figure 8 tokenizer production admissions are immutable")
    args.output.mkdir(parents=True)
    atomic_json(args.output / "admission.json", admission)
    for name in (
        "arm-spec.json", "native-failures.log", "overlay.json", "failure-summary.json",
    ):
        shutil.copyfile(args.probe_admission.parent / name, args.output / name)
    shutil.copyfile(args.probe_admission, args.output / "probe-admission.json")
    shutil.copyfile(probe_result_path, args.output / "probe-result.json")
    shutil.copyfile(args.probe_bank / "bank-receipt.json", args.output / "probe-bank-receipt.json")
    return admission


def validate_production_files(args, admission: dict, spec: dict) -> None:
    from figure8_tokenizer import verify_durable_tree

    base = admission["probe_admission"]
    identity = validate_recovery_identity(
        spec, args.arm, socket.gethostname(), args.gpu,
    )
    if identity != base["identity"]:
        raise ValueError("Figure 8 tokenizer production identity differs from admission")
    require_hash(args.spec, base["spec_sha256"])
    require_hash(Path(spec["output"]) / "run-config.json", base["run_config_sha256"])
    require_hash(args.overlay / "overlay.json", base["overlay_receipt_sha256"])
    validate_overlay(args.overlay.resolve(), args.native_package.resolve())
    require_hash(
        args.native_package / "sklearn_import" / "cluster" / "k_means_.py",
        base["native_module_sha256"],
    )
    for name, digest in admission["production_control_sha256"].items():
        require_hash(args.recovery_control / name, digest)
    if args.admission.is_symlink():
        raise ValueError("Figure 8 tokenizer production admission is redirected")
    durable = verify_durable_tree(args.admission.parent)
    if durable["bank_receipt_sha256"] != sha(args.admission.parent / "bank-receipt.json"):
        raise ValueError("Figure 8 tokenizer production admission bank differs")
    evidence = args.admission.parent
    require_hash(
        evidence / "probe-admission.json",
        admission["probe"]["admission_sha256"],
    )
    require_hash(
        evidence / "probe-result.json",
        admission["probe_result_sha256"],
    )
    require_hash(
        evidence / "probe-bank-receipt.json",
        admission["probe_bank_receipt_sha256"],
    )
    require_hash(evidence / "native-failures.log", base["failure_log_sha256"])
    if read_json(evidence / "failure-summary.json") != base["failure"]:
        raise ValueError("Figure 8 tokenizer production failure evidence differs")


def run_compatibility(args, purpose: str) -> dict:
    import torch

    spec = read_json(args.spec)
    if purpose == "probe":
        admission = validate_probe_admission(read_json(args.admission))
        base = admission
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
    elif purpose == "production":
        admission = validate_production_admission(read_json(args.admission))
        base = admission["probe_admission"]
        validate_production_files(args, admission, spec)
        identity = base["identity"]
        if (
            args.output.resolve() != Path(spec["output"]).resolve()
            or args.stop_after_epoch is not None
            or (args.output / "completed.json").exists()
        ):
            raise ValueError("Production recovery must continue the exact incomplete arm")
    else:
        raise ValueError("Unknown Figure 8 tokenizer recovery purpose")
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
    parent = None
    if first_epoch == identity["base_epoch"]:
        require_hash(args.resume / "completed.json", base["resume"]["completed_sha256"])
        require_hash(args.resume / "state.pt", base["resume"]["state_sha256"])
    elif purpose == "production" and first_epoch > identity["base_epoch"]:
        parent = args.resume / "compatibility"
    else:
        raise ValueError("Compatibility continuation starts from an unadmitted epoch")
    latest = read_json(args.output / "latest-state.json") if args.output.exists() else None
    if purpose == "production" and (
        latest is None
        or latest.get("epoch") != first_epoch
        or Path(latest.get("checkpoint", "")).resolve() != args.resume.resolve()
    ):
        raise ValueError("Production recovery resume differs from latest-state")
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
        "purpose": purpose,
        "arm": args.arm,
        "run_id": Path(spec["output"]).name,
        "run_config_sha256": base["run_config_sha256"],
        "native_harness_sha256": config["harness_sha256"],
        "admission_sha256": sha(args.admission),
        "base_epoch": identity["base_epoch"],
        "base_state_sha256": base["resume"]["state_sha256"],
        "invocation": invocation,
        "resumed_from_epoch": first_epoch,
        "resumed_from_state_sha256": sha(args.resume / "state.pt"),
    }
    segment_parent = [parent]
    segment_epoch = [first_epoch]
    segment_state = [sha(args.resume / "state.pt")]

    def current_common():
        return {
            **common,
            "parent_epoch": segment_epoch[0],
            "parent_state_sha256": segment_state[0],
        }

    if parent is not None:
        verify_attestation(
            parent, common["admission_sha256"], "checkpoint", current_common(),
        )

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
            common=current_common(),
            events=events,
            admission_path=args.admission,
            artifact_fields={
                "epoch": epoch + 1,
                "state_sha256": receipt["state_sha256"],
                "selected_checkpoint": selected,
            },
            parent=segment_parent[0],
        )
        segment_parent[0] = directory / "compatibility"
        segment_epoch[0] = epoch + 1
        segment_state[0] = receipt["state_sha256"]
        reset_event_segment(events)

    def append(path, value):
        original_append(path, value)
        context.update(epoch=value["epoch"] + 1, calls=0)

    def export(model, dataset, clusters, directory, seed, recovery_root):
        context.update(phase="export", calls=0)
        result = original_export(model, dataset, clusters, directory, seed, recovery_root)
        seal_attestation(
            directory,
            artifact="export",
            common=current_common(),
            events=events,
            admission_path=args.admission,
            artifact_fields={
                "index_sha256": result["index_sha256"],
                "selected_checkpoint":
                    read_json(args.output / "latest-state.json")["checkpoint"],
            },
            parent=segment_parent[0],
        )
        return result

    def write(path, value):
        if purpose == "production" and path == args.output / "completed.json":
            latest = read_json(args.output / "latest-state.json")
            final_checkpoint = Path(latest["checkpoint"])
            final_path = final_checkpoint / "compatibility"
            export_path = args.output / "export" / "compatibility"
            final_chain = attestation_chain(
                final_path, common["admission_sha256"], "checkpoint", common,
            )
            validate_invocation_restorations(final_chain)
            exported = verify_attestation(
                export_path, common["admission_sha256"], "export", common,
            )
            correction_epochs = []
            for attestation in final_chain:
                for epoch, phases in attestation["corrections_by_epoch_level_phase"].items():
                    if any(
                        count
                        for phase, levels_by_id in phases.items()
                        if phase != "export"
                        for count in levels_by_id.values()
                    ):
                        correction_epochs.append(int(epoch))
            selected = Path(value["selected_checkpoint"])
            selected_epoch = int(selected.name.split("-")[1])
            if selected_epoch > identity["base_epoch"]:
                selected_path = args.output / selected / "compatibility"
                verify_attestation(
                    selected_path, common["admission_sha256"], "checkpoint", common,
                )
            first_correction = min(correction_epochs) if correction_epochs else None
            selected_dependent = (
                first_correction is not None and selected_epoch >= first_correction
            )
            export_corrections = exported["runtime_summary"]["phases"].get("export", 0)
            value = {
                **value,
                "kmeans_compatibility": {
                    "policy": POLICY,
                    "classification":
                        "compatibility correction at undefined native CDF endpoint indexing",
                    "not_claimed": "bitwise native execution past the deterministic crash",
                    "admission_sha256": common["admission_sha256"],
                    "base_epoch": identity["base_epoch"],
                    "base_state_sha256": base["resume"]["state_sha256"],
                    "final_attestation_sha256": sha(final_path / "attestation.json"),
                    "export_attestation_sha256": sha(export_path / "attestation.json"),
                    "completion_used_compatibility": True,
                    "native_artifact_identity_claimed": False,
                    "selected_model_repair_dependent": selected_dependent,
                    "first_training_correction_epoch": first_correction,
                    "export_corrections": export_corrections,
                    "downstream_ids_repair_dependent":
                        selected_dependent or export_corrections > 0,
                    "original_scientific_recipe_and_selection_unchanged": True,
                },
            }
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
    if purpose == "production":
        completed = read_json(args.output / "completed.json")
        compatibility = completed.get("kmeans_compatibility")
        if (
            completed.get("status") != "complete"
            or compatibility is None
            or compatibility.get("admission_sha256") != common["admission_sha256"]
        ):
            raise ValueError("Production tokenizer completion lost compatibility provenance")
        return {
            "status": "complete",
            "arm": args.arm,
            "completed_sha256": sha(args.output / "completed.json"),
            "kmeans_compatibility": compatibility,
        }
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


def run_probe(args) -> dict:
    return run_compatibility(args, "probe")


def run_production(args) -> dict:
    return run_compatibility(args, "production")


def production_resume(args, admission: dict, spec: dict) -> Path:
    from bank_tokenizers import read_state

    output = Path(spec["output"])
    latest = read_json(output / "latest-state.json")
    resume = Path(latest["checkpoint"]).resolve()
    if (
        resume.parent != output.resolve() / "checkpoints"
        or resume.name != f"epoch-{latest['epoch']:05d}"
    ):
        raise ValueError("Production compatibility resume escapes its tokenizer run")
    config = read_json(output / "run-config.json")
    state = read_state(resume, config)
    if state["epoch"] + 1 != latest["epoch"]:
        raise ValueError("Production compatibility latest checkpoint epoch differs")
    base = admission["probe_admission"]
    if latest["epoch"] == base["identity"]["base_epoch"]:
        require_hash(resume / "completed.json", base["resume"]["completed_sha256"])
        require_hash(resume / "state.pt", base["resume"]["state_sha256"])
    elif latest["epoch"] > base["identity"]["base_epoch"]:
        expected = {
            "kind": "figure8_tokenizer_kmeans_compatibility_attestation_v1",
            "policy": POLICY,
            "purpose": "production",
            "arm": args.arm,
            "run_id": output.name,
            "run_config_sha256": base["run_config_sha256"],
            "native_harness_sha256": config["harness_sha256"],
            "admission_sha256": sha(args.admission),
            "base_epoch": base["identity"]["base_epoch"],
            "base_state_sha256": base["resume"]["state_sha256"],
        }
        verify_attestation(
            resume / "compatibility", sha(args.admission), "checkpoint", expected,
        )
    else:
        raise ValueError("Production compatibility latest checkpoint predates its base")
    return resume


def gpu_is_free(index: int) -> bool:
    result = subprocess.run(
        [
            "nvidia-smi", f"--id={index}", "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return int(result.stdout.strip()) == 0


def supervise_production(args) -> dict:
    import fcntl

    admission = validate_production_admission(read_json(args.admission))
    spec = read_json(args.spec)
    validate_production_files(args, admission, spec)
    output = Path(spec["output"])
    lock_path = output.parent / f"{output.name}.runner.lock"
    status_path = output.parent / f"{output.name}.compat-runner-status.json"
    child = [None]
    stopping = [False]

    def stop(*_):
        stopping[0] = True
        if child[0] is not None and child[0].poll() is None:
            os.killpg(child[0].pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        failures = 0
        while not (output / "completed.json").is_file():
            resume = production_resume(args, admission, spec)
            if not gpu_is_free(spec["gpu"]):
                raise RuntimeError("Assigned Figure 8 tokenizer GPU is occupied; no process was signaled")
            command = [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "run-production",
                "--arm", args.arm,
                "--spec", str(args.spec),
                "--resume", str(resume),
                "--overlay", str(args.overlay),
                "--native-package", str(args.native_package),
                "--recovery-control", str(args.recovery_control),
                "--gpu", str(args.gpu),
                "--admission", str(args.admission),
                "--output", str(output),
            ]
            log = output.parent / f"{output.name}.compat.log"
            with log.open("a", encoding="utf-8") as stream:
                child[0] = subprocess.Popen(
                    command,
                    cwd=spec["root"],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                atomic_json(status_path, {
                    "status": "training",
                    "runner_pid": os.getpid(),
                    "child_pid": child[0].pid,
                    "host": spec["host"],
                    "gpu": spec["gpu"],
                    "resume": str(resume),
                    "started_unix": time.time(),
                })
                returncode = child[0].wait()
            child[0] = None
            if stopping[0]:
                atomic_json(status_path, {
                    "status": "drained",
                    "runner_pid": os.getpid(),
                    "unix": time.time(),
                })
                return {"status": "drained", "arm": args.arm}
            if returncode == 0:
                failures = 0
            else:
                failures += 1
                atomic_json(status_path, {
                    "status": "retry_pending",
                    "returncode": returncode,
                    "consecutive_failures": failures,
                    "unix": time.time(),
                })
                if failures >= 5:
                    raise RuntimeError(
                        "Figure 8 compatibility continuation failed five consecutive invocations",
                    )
                time.sleep(30)
        atomic_json(status_path, {
            "status": "complete",
            "runner_pid": os.getpid(),
            "completed_sha256": sha(output / "completed.json"),
            "unix": time.time(),
        })
    return {
        "status": "complete",
        "arm": args.arm,
        "completed_sha256": sha(output / "completed.json"),
    }


def verify_completed_compatibility(
    payload: Path,
    arm: str,
    expected_admission_sha256: str | None,
) -> dict | None:
    if arm not in RECOVERY_ARMS:
        return None
    if not _digest(expected_admission_sha256):
        raise ValueError("A trusted Figure 8 tokenizer compatibility admission is required")
    completed = read_json(payload / "completed.json")
    compatibility = completed.get("kmeans_compatibility")
    if not isinstance(compatibility, dict):
        raise ValueError("Repaired Figure 8 tokenizer lacks compatibility provenance")
    identity = RECOVERY_ARMS[arm]
    admission_path = payload / "checkpoints" / "epoch-20000" / "compatibility" / "admission.json"
    require_hash(admission_path, expected_admission_sha256)
    admission = validate_production_admission(read_json(admission_path))
    base = admission["probe_admission"]
    if base["identity"] != {"arm": arm, **identity}:
        raise ValueError("Figure 8 tokenizer compatibility admission identity differs")
    config = read_json(payload / "run-config.json")
    expected = {
        "kind": "figure8_tokenizer_kmeans_compatibility_attestation_v1",
        "policy": POLICY,
        "purpose": "production",
        "arm": arm,
        "run_id": f"{arm}",
        "run_config_sha256": sha(payload / "run-config.json"),
        "native_harness_sha256": config["harness_sha256"],
        "admission_sha256": expected_admission_sha256,
        "base_epoch": identity["base_epoch"],
        "base_state_sha256": base["resume"]["state_sha256"],
    }
    final_path = payload / "checkpoints" / "epoch-20000" / "compatibility"
    raw_final = read_json(final_path / "attestation.json")
    if raw_final.get("arm") != arm or raw_final.get("run_id") != arm:
        raise ValueError("Figure 8 tokenizer compatibility arm differs")
    final_chain = attestation_chain(
        final_path, expected_admission_sha256, "checkpoint", expected,
    )
    validate_invocation_restorations(final_chain)
    export_path = payload / "export" / "compatibility"
    exported = verify_attestation(
        export_path, expected_admission_sha256, "export", expected,
    )
    selected = Path(completed["selected_checkpoint"])
    selected_epoch = int(selected.name.split("-")[1])
    if selected_epoch > identity["base_epoch"]:
        verify_attestation(
            payload / selected / "compatibility",
            expected_admission_sha256,
            "checkpoint",
            expected,
        )
    correction_epochs = []
    for attestation in final_chain:
        for epoch, phases in attestation["corrections_by_epoch_level_phase"].items():
            if any(
                count
                for phase, levels_by_id in phases.items()
                if phase != "export"
                for count in levels_by_id.values()
            ):
                correction_epochs.append(int(epoch))
    first_correction = min(correction_epochs) if correction_epochs else None
    selected_dependent = (
        first_correction is not None and selected_epoch >= first_correction
    )
    export_corrections = exported["runtime_summary"]["phases"].get("export", 0)
    required = {
        "policy": POLICY,
        "classification":
            "compatibility correction at undefined native CDF endpoint indexing",
        "not_claimed": "bitwise native execution past the deterministic crash",
        "admission_sha256": expected_admission_sha256,
        "base_epoch": identity["base_epoch"],
        "base_state_sha256": final_chain[-1]["base_state_sha256"],
        "final_attestation_sha256": sha(final_path / "attestation.json"),
        "export_attestation_sha256": sha(export_path / "attestation.json"),
        "completion_used_compatibility": True,
        "native_artifact_identity_claimed": False,
        "selected_model_repair_dependent": selected_dependent,
        "first_training_correction_epoch": first_correction,
        "export_corrections": export_corrections,
        "downstream_ids_repair_dependent":
            selected_dependent or export_corrections > 0,
        "original_scientific_recipe_and_selection_unchanged": True,
    }
    if compatibility != required:
        raise ValueError("Figure 8 tokenizer final compatibility summary differs")
    return {
        **required,
        "final_runtime_summary": final_chain[0]["runtime_summary"],
        "export_runtime_summary": exported["runtime_summary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    overlay = commands.add_parser("make-overlay")
    overlay.add_argument("--native-package", type=Path, required=True)
    overlay.add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser("prepare-probe")
    production = commands.add_parser("prepare-production")
    run = commands.add_parser("run-probe")
    run_production_parser = commands.add_parser("run-production")
    supervisor = commands.add_parser("supervise-production")
    for command in (prepare, run, run_production_parser):
        command.add_argument("--arm", choices=tuple(RECOVERY_ARMS), required=True)
        command.add_argument("--spec", type=Path, required=True)
        command.add_argument("--resume", type=Path, required=True)
        command.add_argument("--overlay", type=Path, required=True)
        command.add_argument("--native-package", type=Path, required=True)
        command.add_argument("--recovery-control", type=Path, required=True)
        command.add_argument("--gpu", type=int, required=True)
    prepare.add_argument("--failure-log", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    production.add_argument("--probe-admission", type=Path, required=True)
    production.add_argument("--probe-bank", type=Path, required=True)
    production.add_argument("--recovery-control", type=Path, required=True)
    production.add_argument("--output", type=Path, required=True)
    run.add_argument("--admission", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--stop-after-epoch", type=int, required=True)
    run_production_parser.add_argument("--admission", type=Path, required=True)
    run_production_parser.add_argument("--output", type=Path, required=True)
    run_production_parser.set_defaults(stop_after_epoch=None)
    for name in (
        "arm", "spec", "overlay", "native-package", "recovery-control",
        "gpu", "admission", "output",
    ):
        option = f"--{name}"
        if name == "arm":
            supervisor.add_argument(option, choices=tuple(RECOVERY_ARMS), required=True)
        elif name == "gpu":
            supervisor.add_argument(option, type=int, required=True)
        else:
            supervisor.add_argument(option, type=Path, required=True)
    supervisor.set_defaults(stop_after_epoch=None)
    args = parser.parse_args()
    if args.command == "make-overlay":
        from figure8_tokenizer_recovery_runtime import make_overlay
        result = make_overlay(args.native_package.resolve(), args.output.resolve())
    elif args.command == "prepare-probe":
        result = prepare_probe(args)
    elif args.command == "prepare-production":
        result = prepare_production(args)
    elif args.command == "run-production":
        result = run_production(args)
    elif args.command == "supervise-production":
        result = supervise_production(args)
    else:
        result = run_probe(args)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
