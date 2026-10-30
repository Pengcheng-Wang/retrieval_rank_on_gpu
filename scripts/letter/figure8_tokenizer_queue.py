"""Execute the isolated 15-arm Figure 8 tokenizer downstream queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path, PurePosixPath

import numpy as np

from author import SOURCE_HASHES
from bank import source_manifest
from figure8_tokenizer import (
    ARM_ORDER,
    arm_tokenizer_spec,
    bind_tokenizer_queue,
    completed_evaluation,
    derive_arm_manifest,
    execute_cell,
    expand_items,
    load_plan,
    next_admitted_cell,
    queue_binding,
    require_durable_admission_path,
    tokenizer_cells,
    validate_evaluation_rows,
    validate_identifier_bank,
    validate_stage_lineage,
    validate_variable_audit,
    verify_arm_admission,
    verify_durable_tree,
)
from figure8_tokenizer_admission import source_hashes
from fresh_ids import (
    FROZEN_HARNESS,
    T5_ASSETS,
    AFTER,
    BEFORE,
    canonical_bytes,
    file_sha256,
    patch_run,
    read_json,
    require_canonical,
    select_resume,
    verify_inputs,
    verify_original,
    write_json,
)
from fresh_pipeline import (
    ExporterFailure,
    GpuOccupancyError,
    Runtime,
    capture_gpu_snapshot,
    next_evaluation,
    read_process_identity,
    torchrun_prefix,
    wait_for_exporter,
    wait_for_gpu_quiescence,
)
from protocol import code_to_items, rank_metrics, require_hash, verify_checkpoint


def build_stage(
    output: Path, runtime: Runtime, cell: dict, admission: dict, plan: dict,
) -> dict:
    if output.exists():
        raise FileExistsError("Figure8 stages are immutable, including partial stages")
    arm = cell["arm"]
    arm_identity = admission["arm_identity"]
    bundle = (
        runtime.destination / "openllama-v2" / "figure8-tokenizers"
        / arm_identity["bundle"]
    )
    payload = bundle / "payload"
    parent = read_json(runtime.manifest)
    verify_original(runtime.root / "upstream", parent, "Instruments")
    verify_inputs(
        runtime.root / "embeddings-openllama-v2-v1" / "Instruments",
        runtime.root / "incoming" / "Instruments-cf-bce-s42",
        runtime.spec,
        parent,
        "Instruments",
    )
    for name, digest in FROZEN_HARNESS.items():
        require_canonical(runtime.control / name, digest)
    native_files = {
        **SOURCE_HASHES["tiger"],
        **{f"ckpt/TIGER/{name}": digest for name, digest in T5_ASSETS.items()},
    }
    for name, digest in native_files.items():
        require_hash(runtime.root / "upstream" / "LETTER-TIGER" / name, digest)
    require_hash(payload / "export" / "index.json", arm_identity["index_sha256"])
    require_hash(payload / "export" / "codes.npy", arm_identity["codes_sha256"])
    indices = read_json(payload / "export" / "index.json")
    codes = np.load(payload / "export" / "codes.npy", allow_pickle=False)
    lineage = {
        "kind": "figure8-tokenizer-downstream-v1",
        "track_id": runtime.spec["track_id"],
        "claim_type": runtime.spec["claim_type"],
        "synthetic_fixture": False,
        "exact_original_llama1_reproduction": False,
        "dataset": "Instruments",
        "arm": arm,
        "tokenizer_spec": admission["tokenizer_spec"],
        "bundle": arm_identity["bundle"],
        "bundle_completed_sha256": arm_identity["bundle_completed_sha256"],
        "bank_receipt_sha256": arm_identity["bank_receipt_sha256"],
        "index_sha256": arm_identity["index_sha256"],
        "codes_sha256": arm_identity["codes_sha256"],
        "plan_sha256": admission["plan_sha256"],
        "admission_sha256": file_sha256(admission["_path"]),
        "downstream": {
            key: plan[key] for key in (
                "seed", "temperature", "learning_rate", "epochs",
                "world_size", "effective_batch_size", "beams",
            )
        },
    }
    derived = derive_arm_manifest(
        parent,
        file_sha256(runtime.manifest),
        lineage,
        indices,
        codes,
        admission["tokenizer_spec"]["codes"],
    )
    output.mkdir(parents=True, exist_ok=False)
    source_receipt = {}
    for name, digest in FROZEN_HARNESS.items():
        original = runtime.control / name
        target = output / "project" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        content = patch_run(original.read_bytes()) if name.endswith("/run.py") else canonical_bytes(original)
        target.write_bytes(content)
        source_receipt[name] = {
            "original_canonical_sha256": digest,
            "executed_sha256": file_sha256(target),
            "change": "receipt lineage only" if name.endswith("/run.py") else "canonical LF only",
        }
    for name in native_files:
        target = output / "upstream" / "LETTER-TIGER" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(runtime.root / "upstream" / "LETTER-TIGER" / name, target)
    for kind in ("item", "inter", "index"):
        target = output / "upstream" / "data" / "Instruments" / f"Instruments.{kind}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        source = (
            payload / "export" / "index.json"
            if kind == "index"
            else runtime.root / "upstream" / "data" / "Instruments" / target.name
        )
        shutil.copyfile(source, target)
    provenance = output / "provenance"
    provenance.mkdir()
    shutil.copyfile(runtime.manifest, provenance / "parent-manifest.json")
    shutil.copyfile(payload / "export" / "codes.npy", provenance / "codes.npy")
    shutil.copyfile(admission["_path"], provenance / "admission.json")
    write_json(provenance / "arm-spec.json", admission["tokenizer_spec"])
    write_json(provenance / "execution.json", runtime.spec)
    write_json(output / "manifest.json", derived)
    receipt = {
        "status": "ready",
        "kind": "isolated_figure8_tokenizer_inputs_and_source",
        "dataset": "Instruments",
        "arm": arm,
        "experiment_lineage": lineage,
        "inputs": derived["datasets"]["Instruments"]["sha256"],
        "manifest_sha256": file_sha256(output / "manifest.json"),
        "source_patch": source_receipt,
        "files": source_manifest(output)["files"],
    }
    write_json(output / "stage.json", receipt)
    return verify_stage(output, cell, admission)


def verify_stage(stage: Path, cell: dict, admission: dict) -> dict:
    receipt = read_json(stage / "stage.json")
    if (receipt.get("status") != "ready"
            or receipt.get("kind") != "isolated_figure8_tokenizer_inputs_and_source"
            or receipt.get("arm") != cell["arm"]):
        raise ValueError("Figure8 stage identity differs")
    for name, metadata in receipt["files"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Figure8 stage file escapes its immutable directory")
        path = stage / name
        if path.is_symlink():
            raise ValueError("Figure8 stage contains a redirected file")
        require_hash(path, metadata["sha256"])
    actual = {
        path.relative_to(stage).as_posix()
        for path in stage.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path != stage / "stage.json"
    }
    if actual != set(receipt["files"]):
        raise ValueError("Figure8 immutable stage file set changed")
    manifest = read_json(stage / "manifest.json")
    validate_stage_lineage(manifest, receipt["experiment_lineage"])
    lineage = receipt["experiment_lineage"]
    if (lineage["arm"] != cell["arm"]
            or lineage["bundle"] != admission["arm_identity"]["bundle"]
            or lineage["bundle_completed_sha256"]
            != admission["arm_identity"]["bundle_completed_sha256"]
            or lineage["admission_sha256"] != file_sha256(admission["_path"])):
        raise ValueError("cross-arm or stale Figure8 stage")
    indices = read_json(
        stage / "upstream" / "data" / "Instruments" / "Instruments.index.json"
    )
    codes = np.load(stage / "provenance" / "codes.npy", allow_pickle=False)
    validate_identifier_bank(
        indices, codes, len(indices), admission["tokenizer_spec"]["codes"],
    )
    for name, digest in FROZEN_HARNESS.items():
        source = canonical_bytes(stage / "project" / name)
        if name.endswith("/run.py"):
            if source.count(AFTER) != 1:
                raise ValueError("Figure8 staged run source lost its lineage patch")
            source = source.replace(AFTER, BEFORE)
        if hashlib.sha256(source).hexdigest() != digest:
            raise ValueError("Figure8 staged source changed beyond the guarded patch")
    return receipt


def validate_audit(stage: Path, audit: Path, cell: dict, admission: dict) -> dict:
    staged = verify_stage(stage, cell, admission)
    report = read_json(audit / "audit.json")
    manifest = read_json(stage / "manifest.json")
    reference = manifest["datasets"]["Instruments"]
    if (report.get("dataset") != "Instruments"
            or report.get("inputs") != staged["inputs"]
            or report.get("manifest_sha256") != staged["manifest_sha256"]
            or report.get("source") != SOURCE_HASHES["tiger"]
            or report.get("identifier_bank") != reference["figure8_identifier_bank"]):
        raise ValueError("Figure8 audit does not bind this stage")
    return validate_variable_audit(report, reference["figure8_identifier_bank"], {
        "train": reference["training_prefix_examples"],
        "valid": reference["validation_users"],
        "test": reference["test_users"],
    })


def validate_training(
    output: Path, stage: Path, audit: Path, cell: dict, admission: dict,
    runtime: Runtime,
) -> dict:
    from run import training_options
    from runtime_patch import require_resume_patch

    staged = verify_stage(stage, cell, admission)
    audited = validate_audit(stage, audit, cell, admission)
    spec = read_json(stage / "provenance" / "execution.json")
    run = read_json(output / "run-config.json")
    if (run.get("kind") != "fresh-id-compatible-rebuild"
            or run.get("experiment_lineage") != staged["experiment_lineage"]
            or run.get("dataset") != "Instruments"
            or run.get("inputs") != staged["inputs"]
            or run.get("seed") != 42 or run.get("temperature") != 0.7
            or run.get("initialization") != "author random T5 configuration"
            or run.get("tokenizer_vocab_sha256") != audited["tokenizer_vocab_sha256"]):
        raise ValueError("cross-arm, warm-start or stale Figure8 training identity")
    options = training_options(
        seed=42,
        batch_size=spec["datasets"]["Instruments"]["micro_batch_size"],
        effective_batch=1024,
        world_size=2,
        learning_rate=0.0005,
        epochs=200,
    )
    expected_harness = {
        name: file_sha256(stage / "project" / "scripts" / "letter" / name)
        for name in ("run.py", "author.py", "protocol.py", "prepare.py", "runtime_patch.py")
    }
    if (run.get("options") != options or run.get("world_size") != 2
            or run.get("author_source") != SOURCE_HASHES["tiger"]
            or run.get("audit_sha256") != file_sha256(audit / "audit.json")
            or run.get("harness_sha256") != expected_harness
            or run.get("trainer_source_sha256") != require_resume_patch()):
        raise ValueError("Figure8 native T5 recipe/audit/harness differs")
    for key in ("d_model", "d_ff", "d_kv", "num_heads", "num_layers", "dropout_rate"):
        if run.get("model_config", {}).get(key) != spec["generator"][key]:
            raise ValueError("Figure8 native random T5 architecture changed")
    return run


def verify_evaluation(
    stage: Path, audit: Path, run_path: Path, evaluation: Path,
    cell: dict, admission: dict, runtime: Runtime,
) -> dict:
    import torch
    from fresh_ids import equal_state

    staged = verify_stage(stage, cell, admission)
    run = validate_training(run_path, stage, audit, cell, admission, runtime)
    done = read_json(evaluation / "completed.json")
    trained = read_json(run_path / "completed.json")
    config = read_json(evaluation / "evaluation-config.json")
    manifest = read_json(stage / "manifest.json")
    interactions, _, indices = verify_original(stage / "upstream", manifest, "Instruments")
    bank = manifest["datasets"]["Instruments"]["figure8_identifier_bank"]
    if (done.get("status") != "complete" or trained.get("status") != "complete"
            or config.get("source_run") != run or config.get("dataset") != "Instruments"
            or config.get("checkpoint_sha256") != trained.get("model_sha256")
            or config.get("split") != "test" or config.get("beams") != 20
            or config.get("max_new_tokens") != 10
            or config.get("seen_item_exclusion") is not False
            or config.get("expected_users") != len(interactions)):
        raise ValueError("Figure8 final evaluation binding differs")
    require_hash(evaluation / "predictions.jsonl", done["predictions_sha256"])
    groups = code_to_items(indices)
    rows, raw_rows = [], {}
    with (evaluation / "predictions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            rows.append({"user": row["user"], "codes": row["codes"]})
            raw_rows[row["user"]] = row
    validate_evaluation_rows(
        rows, list(interactions), groups, 20, 10, bank,
    )
    totals = {"code": {}, "item": {}}
    for user, target_items in interactions.items():
        row = raw_rows[user]
        target = target_items[-1]
        target_code = "".join(indices[str(target)])
        expanded = expand_items(row["codes"], groups)
        metrics = {
            "code": rank_metrics(row["codes"], target_code),
            "item": rank_metrics(expanded, target),
        }
        if (row["target_item"] != target or row["target_code"] != target_code
                or row["item_predictions"] != expanded or row["metrics"] != metrics):
            raise ValueError("Figure8 saved predictions do not independently recompute")
        for kind, values in metrics.items():
            for key, value in values.items():
                totals[kind][key] = totals[kind].get(key, 0.0) + value
    metrics = {
        kind: {key: value / len(interactions) for key, value in values.items()}
        for kind, values in totals.items()
    }
    if done.get("users") != len(interactions) or done.get("metrics") != metrics:
        raise ValueError("Figure8 evaluation is not the complete all-user result")
    selected_name = Path(trained["best_checkpoint"]).name
    selected = run_path / selected_name
    verify_checkpoint(selected)
    state = read_json(run_path / "trainer_state.json")
    selected_state = read_json(selected / "trainer_state.json")
    losses = [row["eval_loss"] for row in state["log_history"] if "eval_loss" in row]
    if (not losses or min(losses) != trained["best_metric"]
            or selected_state["best_metric"] != trained["best_metric"]
            or Path(selected_state["best_model_checkpoint"]).name != selected_name
            or not equal_state(
                torch.load(selected / "pytorch_model.bin", map_location="cpu"),
                torch.load(run_path / "pytorch_model.bin", map_location="cpu"),
            )):
        raise ValueError("Figure8 final T5 is not its within-arm validation minimum")
    return {
        "status": "verified_figure8_tokenizer_cell",
        "arm": cell["arm"],
        "experiment_lineage": staged["experiment_lineage"],
        "seed": 42,
        "temperature": 0.7,
        "all_users": len(interactions),
        "beams": 20,
        "code_metrics": metrics["code"],
        "original_item_diagnostic": metrics["item"],
        "model_sha256": trained["model_sha256"],
        "predictions_sha256": done["predictions_sha256"],
        "selected_checkpoint": selected_name,
        "selected_epoch": selected_state["epoch"],
        "selection": "within-arm validation only; test descriptive",
    }


class ArmRuntime(Runtime):
    def __init__(self, *args, plan: dict, admissions: dict[str, dict], **kwargs):
        super().__init__(*args, **kwargs)
        self.plan = plan
        self.admissions = admissions

    def admit(self, cell: dict, admission: dict) -> Path:
        stage = self.work / "stages" / cell["stage_id"]
        if not stage.exists():
            build_stage(stage, self, cell, admission, self.plan)
        return Path(verify_stage(stage, cell, admission) and stage)

    def gate(self, cell: dict, stage: Path) -> None:
        admission = self.admissions[cell["arm"]]
        audit = self.work / "audits" / cell["stage_id"]
        if not (audit / "audit.json").is_file():
            if audit.exists():
                raise FileExistsError("Interrupted Figure8 audit is preserved")
            self.invoke([
                "python", self.scripts / "figure8_tokenizer_audit.py",
                "--upstream", stage / "upstream",
                "--manifest", stage / "manifest.json",
                "--dataset", "Instruments",
                "--codes", stage / "provenance" / "codes.npy",
                "--codebook-sizes", *admission["tokenizer_spec"]["codes"],
                "--output", audit,
            ], f"audit-{cell['arm']}")
        validate_audit(stage, audit, cell, admission)
        self.bank(stage, f"stages/{cell['stage_id']}")
        self.bank(audit, f"audits/{cell['stage_id']}")

    def train(self, cell: dict, stage: Path) -> Path:
        import fcntl

        from bank import copy_artifact
        from bank_watch import checkpoint_artifacts
        from evaluate import verify_model_binding

        admission = self.admissions[cell["arm"]]
        audit = self.work / "audits" / cell["stage_id"]
        output = self.work / "runs" / cell["run_id"]

        def completed() -> bool:
            if not (output / "completed.json").is_file():
                return False
            run = validate_training(output, stage, audit, cell, admission, self)
            verify_model_binding(
                output, read_json(output / "completed.json"), run,
                "Instruments", run["inputs"],
            )
            return True

        if completed():
            return output
        watcher = [
            "python", self.scripts / "bank_watch.py",
            "--runs", output,
            "--staging", self.work / "checkpoint-snapshots",
            "--destination", self.bank_root / "checkpoints",
            "--lock-file", self.root / "bank.lock",
            "--writer-host", "node-0",
            "--interval", 120,
        ]
        log_dir = (
            self.work / "commands"
            / f"checkpoint-watcher-{cell['run_id']}-{time.time_ns()}"
        )
        log_dir.mkdir(parents=True)
        idle = wait_for_gpu_quiescence(
            lambda: capture_gpu_snapshot(self.gpu_indexes),
            self.gpu_baseline,
            log_dir / "gpu-before-banker.json",
            owners=[entry[1] for entry in self._owned_gpu_processes.values()],
            drain_owned=self._drain_gpu_owner,
            gpu_indexes=self.gpu_indexes,
        )
        gpu_uuids = [
            idle["gpus"][str(index)]["uuid"] for index in self.gpu_indexes
        ]
        command, env = self.command(watcher, gpu=False)
        with (log_dir / "output.log").open("x", encoding="utf-8") as log:
            banker = subprocess.Popen(
                command,
                cwd=self.root,
                env=env,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            banker_identity = self._record_owned_process(
                banker, "checkpoint-watcher", gpu_uuids,
            )
            write_json(log_dir / "command.json", {
                "pid": banker.pid,
                "argv": command,
                "cuda_visible_devices": "",
                "process_identity": banker_identity,
            })
            try:
                wait_for_exporter(banker, log_dir / "output.log", completed)
                if completed():
                    return output
                for attempt in range(self.spec["queue"]["max_invocation_failures"]):
                    extra = []
                    if output.exists() and any(output.iterdir()):
                        validate_training(
                            output, stage, audit, cell, admission, self,
                        )
                        extra = ["--resume", select_resume(output)]
                    try:
                        self.invoke(torchrun_prefix(cell["run_id"]) + [
                            stage / "project" / "scripts" / "letter" / "run.py",
                            "--upstream", stage / "upstream",
                            "--manifest", stage / "manifest.json",
                            "--audit", audit,
                            "--dataset", "Instruments",
                            "--output", output,
                            "--seed", 42,
                            "--temperature", 0.7,
                            "--learning-rate", 0.0005,
                            "--epochs", 200,
                            "--effective-batch", 1024,
                            "--batch-size",
                            self.spec["datasets"]["Instruments"]["micro_batch_size"],
                        ] + extra, f"train-{cell['run_id']}", 42,
                            exporter=banker, completed=completed, gpu=True)
                    except (ExporterFailure, GpuOccupancyError):
                        raise
                    except RuntimeError:
                        if attempt + 1 == self.spec["queue"]["max_invocation_failures"]:
                            raise
                    if completed():
                        return output
                    time.sleep(self.spec["queue"]["poll_seconds"])
                raise RuntimeError(
                    "Figure8 native T5 remains incomplete after bounded retries"
                )
            finally:
                with (self.root / "bank.lock").open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    if banker.poll() is None:
                        current = read_process_identity(banker.pid)
                        if (current is None
                                or current["start_time_ticks"]
                                != banker_identity["start_time_ticks"]):
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
                            and current["start_time_ticks"]
                            == banker_identity["start_time_ticks"]):
                        raise ExporterFailure(
                            "Checkpoint banker identity remains live after shutdown"
                        )
                    self._owned_gpu_processes.pop(banker.pid, None)
                if (output / "run-config.json").exists():
                    for artifact in checkpoint_artifacts(
                        output, self.work / "checkpoint-snapshots",
                    ):
                        with (self.root / "bank.lock").open("a") as lock:
                            fcntl.flock(lock, fcntl.LOCK_EX)
                            copy_artifact(
                                Path(artifact["path"]),
                                self.bank_root / "checkpoints" / artifact["identity"],
                            )

    def bank_run(self, cell: dict, run: Path) -> None:
        self.bank(run, f"runs/{cell['run_id']}")

    def evaluate(self, cell: dict, stage: Path, run: Path) -> Path:
        admission = self.admissions[cell["arm"]]
        root = self.work / "evaluations" / cell["run_id"]
        evaluation = completed_evaluation(root)
        if evaluation is None:
            if len(list(root.glob("evaluation-attempt-*"))) >= 3:
                raise RuntimeError("Figure8 evaluation attempts exhausted")
            evaluation = next_evaluation(root)
            self.invoke([
                "python", stage / "project" / "scripts" / "letter" / "evaluate.py",
                "--upstream", stage / "upstream",
                "--manifest", stage / "manifest.json",
                "--dataset", "Instruments",
                "--checkpoint", run,
                "--output", evaluation,
                "--seed", 42,
                "--split", "test",
                "--beams", 20,
                "--batch-size", 16,
            ], f"evaluate-{cell['run_id']}", 42, gpu=True)
        verify_evaluation(
            stage, self.work / "audits" / cell["stage_id"], run,
            evaluation, cell, admission, self,
        )
        return evaluation

    def bank_evaluation(self, cell: dict, evaluation: Path) -> None:
        self.bank(
            evaluation,
            f"evaluations/{cell['run_id']}/{evaluation.name}",
        )

    def publish(
        self, cell: dict, stage: Path, run: Path, evaluation: Path,
    ) -> dict:
        admission = self.admissions[cell["arm"]]
        result = verify_evaluation(
            stage, self.work / "audits" / cell["stage_id"], run,
            evaluation, cell, admission, self,
        )
        report = self.work / "results" / cell["run_id"]
        if (report / "result.json").is_file():
            if read_json(report / "result.json") != result:
                raise ValueError("Figure8 published result is immutable")
        else:
            report.mkdir(parents=True)
            write_json(report / "result.json", result)
        self.bank(report, f"results/{cell['run_id']}")
        ack = self.work / "acks" / f"{cell['run_id']}.json"
        ack.parent.mkdir(parents=True, exist_ok=True)
        write_json(ack, {
            "status": "banked",
            "arm": cell["arm"],
            "run_id": cell["run_id"],
            "result_sha256": file_sha256(report / "result.json"),
        })
        return result


def load_admissions(values: list[str], plan: dict, destination: Path) -> dict[str, dict]:
    admissions = {}
    for value in values:
        arm, separator, raw_path = value.partition("=")
        if not separator or arm not in ARM_ORDER or arm in admissions:
            raise ValueError("Use each --arm-admission ARM=PATH once")
        path = require_durable_admission_path(Path(raw_path), destination, arm)
        verify_durable_tree(path.parent)
        receipt = read_json(path)
        if (receipt.get("status") != "pass"
                or receipt.get("kind") != "figure8_tokenizer_arm_admission_v1"
                or receipt.get("arm") != arm
                or receipt.get("tokenizer_spec") != arm_tokenizer_spec(plan, arm)
                or receipt.get("plan_sha256") != plan["_sha256"]
                or receipt.get("source_sha256") != source_hashes(Path(__file__).parent)):
            raise ValueError(f"Figure8 arm admission differs: {arm}")
        audit = {
            "status": "pass",
            "arm": arm,
            **{
                key: receipt["arm_identity"][key]
                for key in (
                    "bundle", "bundle_completed_sha256", "run_completed_sha256",
                    "index_sha256", "codes_sha256", "bank_receipt_sha256",
                )
            },
            "selected_epoch": receipt["arm_identity"]["selected_epoch"],
            "final_epoch": receipt["arm_identity"]["final_epoch"],
            "durable_bank_receipt_sha256":
                receipt["arm_identity"]["bank_receipt_sha256"],
        }
        # Re-check the live durable bytes instead of trusting the admission alone.
        spec_path = path.parent / "arm-spec.json"
        if not spec_path.is_file():
            raise ValueError(f"Figure8 arm admission has no immutable spec: {arm}")
        verify_arm_admission(
            arm, audit, spec_path,
            destination / "openllama-v2" / "figure8-tokenizers",
            receipt["tokenizer_spec"],
        )
        receipt["_path"] = path
        admissions[arm] = receipt
    return admissions


def cell_is_banked(work: Path, cell: dict) -> bool:
    ack = work / "acks" / f"{cell['run_id']}.json"
    if not ack.is_file():
        return False
    receipt = read_json(ack)
    result = work / "results" / cell["run_id"] / "result.json"
    if (receipt.get("status") != "banked" or receipt.get("arm") != cell["arm"]
            or receipt.get("run_id") != cell["run_id"]
            or not result.is_file()
            or receipt.get("result_sha256") != file_sha256(result)):
        raise ValueError("Figure8 cell acknowledgment differs")
    return True


def parse_gpu_pair(value: str) -> tuple[int, int]:
    if not re.fullmatch(r"\d+,\d+", value):
        raise ValueError("GPU pair must contain two comma-separated indexes")
    indexes = tuple(map(int, value.split(",")))
    if len(set(indexes)) != 2:
        raise ValueError("GPU pair must contain two distinct indexes")
    return indexes


def main() -> None:
    import fcntl

    parser = argparse.ArgumentParser()
    for name in ("binding", "project", "control", "work", "plan"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--arm-admission", action="append", default=[])
    parser.add_argument("--gpu-pair", required=True)
    parser.add_argument("--checks", type=int, default=15)
    args = parser.parse_args()
    plan = load_plan(args.plan.resolve())
    plan["_sha256"] = file_sha256(args.plan)
    runtime = ArmRuntime(
        args.binding.resolve(), args.project.resolve(), args.control.resolve(),
        args.work.resolve(), gpu_indexes=parse_gpu_pair(args.gpu_pair),
        plan=plan, admissions={},
    )
    if runtime.work.name != plan["work_basename"]:
        raise ValueError("Figure8 tokenizer work namespace differs")
    with (runtime.root / "bank.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        admissions = load_admissions(
            args.arm_admission, plan, runtime.destination,
        )
    runtime.admissions = admissions
    cells = tokenizer_cells()
    binding = queue_binding(
        plan_sha256=plan["_sha256"],
        work=runtime.work,
        durable_root=runtime.bank_root,
        gpu_indexes=runtime.gpu_indexes,
        source_sha256=source_hashes(runtime.scripts),
    )
    with (runtime.work / "driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = bind_tokenizer_queue(runtime.work, binding)
        write_json(runtime.work / "driver.json", {
            "pid": os.getpid(),
            "started_unix": time.time(),
            "binding_sha256": file_sha256(marker),
            "checks": args.checks,
        })
        for check in range(args.checks):
            banked = {cell["run_id"] for cell in cells if cell_is_banked(runtime.work, cell)}
            cell = next_admitted_cell(cells, admissions, banked)
            if cell is None:
                status = (
                    "all_15_arms_durably_complete"
                    if len(banked) == len(cells)
                    else "waiting_for_individual_arm_admissions"
                )
                write_json(runtime.work / "status.json", {
                    "status": status,
                    "banked": len(banked),
                    "admitted": list(admissions),
                })
                return
            execute_cell(runtime, cell, admissions[cell["arm"]])
            write_json(runtime.work / "status.json", {
                "status": "completed_one_arm",
                "check": check + 1,
                "arm": cell["arm"],
            })


if __name__ == "__main__":
    main()
