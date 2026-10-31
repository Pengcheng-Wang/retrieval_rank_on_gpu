"""Finalize a sealed terminal BIGRec decision without entering a training loop."""

from __future__ import annotations

import json
import math
import re
import shutil
import time
import uuid
from pathlib import Path

from bigrec_bank import seal_payload, stage_payload, verify_payload
from bigrec_checkpoint import latest, validate_position, verify
from protocol import file_sha256, require_hash, write_json


SELECTED_FILES = (
    "adapter_model.safetensors", "adapter_config.json", "tokenizer.model",
    "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
    "tokenizer.json", "README.md",
)


def frozen_policy(recipe: dict) -> dict:
    training = recipe["training"]
    expected = {
        "sample": 1024, "micro_batch": 4, "accumulation": 32, "effective_batch": 128,
        "epochs": 50, "early_stopping_patience": 5, "validation_sample": 5000,
    }
    if any(training.get(key) != value for key, value in expected.items()):
        raise ValueError("Terminal maintenance requires the unchanged frozen BIGRec policy")
    return {"max_steps": 400, "patience": 5, "steps_per_epoch": 8, "epochs": 50}


def terminal_decision(state: dict, continuation: dict, recipe: dict) -> dict:
    policy = frozen_policy(recipe)
    if state.get("max_steps") != policy["max_steps"] or state.get("num_train_epochs") != policy["epochs"]:
        raise ValueError("Sealed Trainer state differs from the frozen 400-step/50-epoch schedule")
    step = state["global_step"]
    counter = continuation["early_stopping_patience_counter"]
    if type(step) is not int or not 0 <= step <= policy["max_steps"] or type(counter) is not int or counter < 0:
        raise ValueError("Invalid native step or patience counter")
    if continuation["evaluations"] != step // 8 or continuation["microbatches"] != step * 32:
        raise ValueError("Terminal state requires a complete native validation/position boundary")
    patience_exhausted = counter >= policy["patience"]
    if patience_exhausted and step % policy["steps_per_epoch"]:
        raise ValueError("Patience-terminal state has already advanced past its epoch boundary")
    reason = "maximum_steps" if step == policy["max_steps"] else "early_stopping" if patience_exhausted else None
    return {
        "terminal": reason is not None, "reason": reason, "global_step": step,
        "patience_counter": counter, "max_steps": policy["max_steps"], "patience": policy["patience"],
        "evaluations": continuation["evaluations"], "microbatches": continuation["microbatches"],
        "best_checkpoint": Path(state["best_model_checkpoint"]).name if state.get("best_model_checkpoint") else None,
        "best_metric": state.get("best_metric"),
    }


def checkpoint_decision(checkpoint: Path, recipe: dict) -> tuple[dict, dict, dict]:
    verify(checkpoint)
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    continuation = json.loads((checkpoint / "continuation.json").read_text(encoding="utf-8"))
    decision = terminal_decision(state, continuation, recipe)
    marker = checkpoint / "terminal-status.json"
    if marker.is_file() and json.loads(marker.read_text(encoding="utf-8"))["decision"] != decision:
        raise ValueError("Persisted terminal decision disagrees with the sealed native state")
    return decision, state, continuation


def persist_decision(checkpoint: Path, recipe: dict, maintenance: dict) -> None:
    if (checkpoint / "complete.json").exists():
        raise ValueError("Never add terminal metadata to an already sealed checkpoint")
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    continuation = json.loads((checkpoint / "continuation.json").read_text(encoding="utf-8"))
    validate_position(checkpoint, state)
    write_json(checkpoint / "terminal-status.json", {
        "schema_version": 1, "maintenance": maintenance,
        "decision": terminal_decision(state, continuation, recipe),
    })


def _recorded_best(run: Path, state: dict) -> Path:
    name = state.get("best_model_checkpoint")
    if not name or not isinstance(state.get("best_metric"), (float, int)) or not math.isfinite(state["best_metric"]):
        raise ValueError("Terminal finalization needs a finite, recorded best checkpoint")
    best = Path(name).resolve()
    if (best.parent != run.resolve() or not re.fullmatch(r"checkpoint-\d+", best.name)
            or int(best.name.split("-")[-1]) > state["global_step"]):
        raise ValueError("The recorded best checkpoint escapes or postdates the original run")
    verify(best)
    best_state = json.loads((best / "trainer_state.json").read_text(encoding="utf-8"))
    if best_state["best_metric"] != state["best_metric"]:
        raise ValueError("The recorded best checkpoint and terminal metric disagree")
    return best


def finalize_terminal(
    run: Path, checkpoint: Path, recipe: dict, report: dict, maintenance: dict, outbox: Path,
) -> dict:
    started = time.monotonic()
    decision, state, continuation = checkpoint_decision(checkpoint, recipe)
    if not decision["terminal"]:
        raise ValueError("An active checkpoint cannot be finalized as terminal")
    run_config_path = run / "run-config.json"
    require_hash(run_config_path, continuation["run_config_sha256"])
    config = json.loads(run_config_path.read_text(encoding="utf-8"))
    if (config["track"] != recipe["track"] or config["dataset"] != report["dataset"]
            or config["admission_sha256"] != report["admission_sha256"]
            or config["recipe_sha256"] != report["recipe_sha256"] or config["model"] != report["model"]):
        raise ValueError("Terminal checkpoint identity differs from the frozen run/admission/recipe")
    best = _recorded_best(run, state)
    checkpoint_sha = file_sha256(checkpoint / "complete.json")
    selection = {
        "track": recipe["track"], "dataset": report["dataset"], "best_checkpoint": best.name,
        "best_checkpoint_sha256": file_sha256(best / "complete.json"),
        "best_validation_loss": state["best_metric"], "final_global_step": state["global_step"],
        "admission_sha256": report["admission_sha256"], "run_config_sha256": file_sha256(run_config_path),
        "recipe_sha256": report["recipe_sha256"], "source": config["source"],
    }
    selected = run / "selected"
    if selected.exists():
        if (selected / "complete.json").is_file():
            saved = verify_payload(selected)
            if (saved["kind"] != "bigrec_selected_adapter"
                    or json.loads((selected / "selection.json").read_text(encoding="utf-8")) != selection):
                raise ValueError("Existing selected artifact disagrees with the recorded terminal best")
            require_hash(selected / "adapter_model.safetensors", file_sha256(best / "adapter_model.safetensors"))
        else:
            selected.rename(run / f".terminal-incomplete-selected-{uuid.uuid4().hex}")
    if not selected.exists():
        pending = run / f".terminal-selected-partial-{uuid.uuid4().hex}"
        pending.mkdir()
        for name in SELECTED_FILES:
            source = best / name
            if source.is_file():
                shutil.copyfile(source, pending / name)
                require_hash(pending / name, file_sha256(source))
        write_json(pending / "selection.json", selection)
        write_json(pending / "terminal-maintenance.json", {
            "maintenance": maintenance, "decision": decision,
            "terminal_checkpoint": checkpoint.name, "terminal_checkpoint_sha256": checkpoint_sha,
            "selection": "unchanged_recorded_best_not_reselected",
            "extra_optimizer_updates": 0, "extra_microbatches": 0, "extra_validations": 0,
        })
        seal_payload(pending, "bigrec_selected_adapter", {
            "admission_sha256": report["admission_sha256"], "recipe_sha256": report["recipe_sha256"],
            "model": report["model"], "maintenance": maintenance,
        })
        pending.rename(selected)
    marker = run / "completed.json"
    if marker.is_file():
        completed = json.loads(marker.read_text(encoding="utf-8"))
        if (completed.get("status") != "complete" or completed["global_step"] != state["global_step"]
                or completed["selected_complete_sha256"] != file_sha256(selected / "complete.json")
                or completed["best_validation_loss"] != state["best_metric"]):
            raise ValueError("Existing completion marker differs from the recorded terminal outcome")
        return completed
    request = stage_payload(selected, outbox, f"{run.name}-selected")
    completed = {
        "status": "complete", "track": recipe["track"], "global_step": state["global_step"],
        "optimizer_updates": continuation["optimizer_updates"], "best_validation_loss": state["best_metric"],
        "selected_complete_sha256": file_sha256(selected / "complete.json"), "staged": request["identity"],
        "test_evaluated": False, "wall_seconds_this_invocation": time.monotonic() - started,
        "terminal_checkpoint_sha256": checkpoint_sha, "terminal_decision": decision,
        "maintenance": maintenance, "extra_optimizer_updates": 0, "extra_microbatches": 0,
    }
    write_json(marker, completed)
    return completed


def finish_if_terminal(args, maintenance: dict, phase: str = "before_training") -> bool:
    resume = getattr(args, "resume", None)
    if phase == "before_training" and resume and resume != "latest":
        checkpoint = Path(resume).resolve()
    else:
        checkpoint = latest(args.output)
    if checkpoint is None:
        return False
    recipe = json.loads((args.admission / "recipe.json").read_text(encoding="utf-8"))
    decision, _, _ = checkpoint_decision(checkpoint, recipe)
    if not decision["terminal"]:
        return False
    current = latest(args.output)
    if current is None or current.resolve() != checkpoint.resolve():
        raise ValueError("Refuse to reselect an older terminal outcome after subsequent progress")
    from bigrec_data import verify_admission
    from bigrec_gate import require_gates
    report = verify_admission(args.admission)
    if not getattr(args, "gate_run", False):
        require_gates(args.gates, args.admission)
    report = {**report, "admission_sha256": file_sha256(args.admission / "admission.json")}
    finalize_terminal(args.output, checkpoint, recipe, report, maintenance, args.outbox)
    return True
