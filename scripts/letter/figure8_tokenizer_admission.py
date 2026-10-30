"""Create one immutable durable admission for a Figure 8 tokenizer arm."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from bank import copy_artifact
from figure8_tokenizer import (
    ARM_ORDER,
    arm_tokenizer_spec,
    load_plan,
    verify_arm_admission,
)
from fresh_ids import file_sha256, load_execution, validate_scope, write_json
from fresh_rq_admission import require_unredirected


CONTROL_FILES = (
    "figure8_tokenizer.py",
    "figure8_tokenizer_admission.py",
    "figure8_tokenizer_audit.py",
    "figure8_tokenizer_queue.py",
    "fresh_ids.py",
    "fresh_pipeline.py",
    "fresh_rq_admission.py",
    "run.py",
    "evaluate.py",
    "prepare.py",
    "author.py",
    "protocol.py",
    "runtime_patch.py",
    "bank.py",
    "bank_watch.py",
)


def source_hashes(scripts: Path) -> dict:
    return {name: file_sha256(scripts / name) for name in CONTROL_FILES}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("binding", "control", "work", "plan", "audit", "spec", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--arm", choices=ARM_ORDER, required=True)
    args = parser.parse_args()
    import fcntl

    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1"):
        raise RuntimeError("Figure8 tokenizer admission is CPU-only")
    root, destination = validate_scope(args.binding)
    control = require_unredirected(args.control, root, "Figure8 control")
    work = require_unredirected(args.work, root, "Figure8 tokenizer work")
    plan_path = require_unredirected(args.plan, control, "Figure8 tokenizer plan")
    audit_path = require_unredirected(args.audit, root, f"{args.arm} final audit")
    spec_path = require_unredirected(args.spec, root, f"{args.arm} specification")
    output_dir = require_unredirected(
        args.output_dir, work, f"{args.arm} admission output",
    )
    if (work != root / "fresh-t5-figure8-tokenizer-v1"
            or output_dir != work / "admissions" / args.arm
            or plan_path.name != "letter-figure8-tokenizer-v1.json"):
        raise ValueError("Figure8 tokenizer admission paths escape the isolated namespace")
    plan = load_plan(plan_path)
    expected_spec = arm_tokenizer_spec(plan, args.arm)
    execution = load_execution(
        control / "configs" / "letter-fresh-id-execution-v1.json",
        control / "configs" / "letter-reproduction-v1.json",
        control / "configs" / "letter-openllama-v2-method-v1.json",
        control / "configs" / "letter-tokenizer-reconstruction-v1.json",
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    with (root / "bank.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        arm = verify_arm_admission(
            args.arm,
            audit,
            spec_path,
            destination / "openllama-v2" / "figure8-tokenizers",
            expected_spec,
            execution,
        )
    if output_dir.exists():
        raise FileExistsError("Figure8 tokenizer arm admission is immutable")
    output_dir.mkdir(parents=True)
    admission = {
        "status": "pass",
        "kind": "figure8_tokenizer_arm_admission_v1",
        "arm": args.arm,
        "plan_sha256": file_sha256(plan_path),
        "audit_sha256": file_sha256(audit_path),
        "arm_identity": arm,
        "tokenizer_spec": expected_spec,
        "source_sha256": source_hashes(control / "scripts" / "letter"),
        "work": str(work),
        "durable_root": str(
            destination / "openllama-v2" / "fresh-t5-figure8-tokenizer-v1"
        ),
        "claim":
            "Admission uses tokenizer specification and durable hashes only; "
            "no downstream validation or test metric was read.",
    }
    write_json(output_dir / "admission.json", admission)
    shutil.copyfile(spec_path, output_dir / "arm-spec.json")
    target = (
        destination / "openllama-v2" / "fresh-t5-figure8-tokenizer-v1"
        / "operational" / "admissions" / args.arm
    )
    with (root / "bank.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        copy_artifact(output_dir, target)
    print(json.dumps({
        "status": "pass",
        "arm": args.arm,
        "admission_sha256": file_sha256(output_dir / "admission.json"),
        "durable": str(target),
    }), flush=True)


if __name__ == "__main__":
    main()
