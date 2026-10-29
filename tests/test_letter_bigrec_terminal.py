"""Terminal continuation is a completed decision, not another training epoch."""

import hashlib
import importlib
import json
import os
import shutil
import stat
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "letter"))


def implementation(name):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        pytest.fail(f"Missing terminal-maintenance implementation: {name}")


@pytest.fixture
def work():
    path = ROOT / ".local" / "letter" / "bigrec-terminal-tests" / uuid.uuid4().hex
    path.mkdir(parents=True)
    yield path
    if os.name == "nt":
        for artifact in path.rglob("*"):
            if artifact.is_file():
                artifact.chmod(stat.S_IREAD | stat.S_IWRITE)
    shutil.rmtree(path)


def recipe():
    return json.loads((ROOT / "configs" / "letter-bigrec-compatible-v1.json").read_text())


@pytest.mark.parametrize("autocrlf", ["false", "true"])
def test_git_checkout_preserves_hash_bound_maintenance_bytes(work, autocrlf):
    guard = implementation("bigrec_maintenance")
    repository, checkout = work / "repository", work / "checkout"
    scripts = repository / "scripts" / "letter"
    scripts.mkdir(parents=True)
    checkout.mkdir()
    shutil.copyfile(ROOT / ".gitattributes", repository / ".gitattributes")
    expected = {
        "bigrec_train.py": guard.BASE_TRAIN_SHA256,
        "bigrec_pipeline.py": guard.BASE_PIPELINE_SHA256,
    }
    for name in ("bigrec_maintenance.py", "bigrec_terminal.py"):
        expected[name] = hashlib.sha256((ROOT / "scripts" / "letter" / name).read_bytes()).hexdigest()
    for name in expected:
        shutil.copyfile(ROOT / "scripts" / "letter" / name, scripts / name)
    commands = (
        ["init", "--quiet"],
        ["add", "."],
        ["checkout-index", "--all", "--prefix=" + str(checkout.resolve()) + os.sep],
    )
    for command in commands:
        subprocess.run(
            ["git", "-c", "core.autocrlf=" + autocrlf, *command],
            cwd=repository, check=True, capture_output=True,
        )
    for name, digest in expected.items():
        restored = checkout / "scripts" / "letter" / name
        assert hashlib.sha256(restored.read_bytes()).hexdigest() == digest, name


def terminal_records(step=48, counter=5):
    return (
        {"global_step": step, "epoch": step / 8, "max_steps": 400, "num_train_epochs": 50,
         "best_metric": 1.0, "best_model_checkpoint": "checkpoint-8"},
        {"early_stopping_patience_counter": counter, "evaluations": step // 8,
         "microbatches": step * 32, "microbatches_per_epoch": 256, "accumulation": 32},
    )


def make_run(work, terminal_step=48, counter=5):
    import bigrec_checkpoint
    from bigrec_train import source_identity, training_options
    from protocol import file_sha256, write_json

    run = work / "run"
    run.mkdir()
    spec = recipe()
    binding = {
        "track": spec["track"], "dataset": "Instruments", "model": {"fixture": True},
        "source": source_identity(), "options": training_options(spec),
        "admission_sha256": "a" * 64, "recipe_sha256": "b" * 64,
    }
    write_json(run / "run-config.json", binding)
    best = run / "checkpoint-8"
    terminal = run / f"checkpoint-{terminal_step}"
    for path, step, patience, weights in (
        (best, 8, 0, b"best-native-adapter-fixture"),
        (terminal, terminal_step, counter, b"later-native-adapter-fixture"),
    ):
        path.mkdir()
        for name in bigrec_checkpoint.REQUIRED:
            (path / name).write_bytes(b"fixture")
        (path / "adapter_model.safetensors").write_bytes(weights)
        (path / "adapter_config.json").write_text("{}")
        for name in ("tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"):
            (path / name).write_bytes(b"fixture-tokenizer")
        state, continuation = terminal_records(step, patience)
        state["best_model_checkpoint"] = str(best.resolve())
        continuation.update(
            sampler={"fixture": True}, optimizer_updates=step,
            run_config_sha256=file_sha256(run / "run-config.json"),
        )
        write_json(path / "trainer_state.json", state)
        write_json(path / "continuation.json", continuation)
        bigrec_checkpoint.seal(path)
    report = {
        "dataset": "Instruments", "model": binding["model"], "recipe_sha256": "b" * 64,
        "admission_sha256": "a" * 64,
    }
    maintenance = {"id": "bigrec-terminal-v1", "deployment_sha256": "c" * 64}
    return run, best, terminal, spec, report, maintenance


def test_restored_patience_is_terminal_without_another_improvement_opportunity():
    terminal = implementation("bigrec_terminal")
    state, continuation = terminal_records()
    decision = terminal.terminal_decision(state, continuation, recipe())
    assert decision["terminal"] and decision["reason"] == "early_stopping"
    assert decision["global_step"] == 48 and decision["patience_counter"] == 5
    assert decision["max_steps"] == 400
    state, continuation = terminal_records(44, 4)
    assert not terminal.terminal_decision(state, continuation, recipe())["terminal"]


def test_maximum_schedule_is_terminal_without_extending_or_shortening_it():
    terminal = implementation("bigrec_terminal")
    state, continuation = terminal_records(400, 0)
    assert terminal.terminal_decision(state, continuation, recipe())["reason"] == "maximum_steps"
    changed = recipe()
    changed["training"]["epochs"] = 49
    with pytest.raises(ValueError, match="frozen"):
        terminal.terminal_decision(state, continuation, changed)


def test_terminal_patience_cannot_legitimize_an_already_extra_partial_epoch():
    terminal = implementation("bigrec_terminal")
    state, continuation = terminal_records(49, 5)
    with pytest.raises(ValueError, match="boundary"):
        terminal.terminal_decision(state, continuation, recipe())


def test_cold_terminal_finalization_copies_only_the_recorded_best_without_mutating_seals(work):
    terminal = implementation("bigrec_terminal")
    run, best, checkpoint, spec, report, maintenance = make_run(work)
    before = {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in run.rglob("*") if p.is_file()}
    result = terminal.finalize_terminal(run, checkpoint, spec, report, maintenance, work / "outbox")
    assert result["status"] == "complete" and result["global_step"] == 48
    assert result["optimizer_updates"] == 48
    assert (run / "selected" / "adapter_model.safetensors").read_bytes() == (
        best / "adapter_model.safetensors").read_bytes()
    selection = json.loads((run / "selected" / "selection.json").read_text())
    assert selection["best_checkpoint"] == "checkpoint-8"
    assert selection["best_validation_loss"] == 1.0
    assert selection["final_global_step"] == 48
    assert not (run / "drained.json").exists()
    assert all(hashlib.sha256((run / name).read_bytes()).hexdigest() == digest for name, digest in before.items())
    terminal.finalize_terminal(run, checkpoint, spec, report, maintenance, work / "outbox")
    assert json.loads((run / "completed.json").read_text()) == result


def test_crash_before_selected_publication_preserves_partial_files_and_finalizes_recorded_best(work):
    terminal = implementation("bigrec_terminal")
    run, best, checkpoint, spec, report, maintenance = make_run(work)
    partial = run / "selected"
    partial.mkdir()
    (partial / "adapter_model.safetensors").write_bytes(b"interrupted-write")
    terminal.finalize_terminal(run, checkpoint, spec, report, maintenance, work / "outbox")
    displaced = list(run.glob(".terminal-incomplete-selected-*"))
    assert len(displaced) == 1
    assert (displaced[0] / "adapter_model.safetensors").read_bytes() == b"interrupted-write"
    assert (run / "selected" / "adapter_model.safetensors").read_bytes() == (
        best / "adapter_model.safetensors").read_bytes()


def test_crash_after_selected_seal_retries_publication_without_reselection(work, monkeypatch):
    terminal = implementation("bigrec_terminal")
    run, best, checkpoint, spec, report, maintenance = make_run(work)
    original = terminal.stage_payload
    monkeypatch.setattr(terminal, "stage_payload", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("crash")))
    with pytest.raises(RuntimeError, match="crash"):
        terminal.finalize_terminal(run, checkpoint, spec, report, maintenance, work / "outbox")
    sealed = (run / "selected" / "complete.json").read_bytes()
    assert not (run / "completed.json").exists()
    monkeypatch.setattr(terminal, "stage_payload", original)
    terminal.finalize_terminal(run, checkpoint, spec, report, maintenance, work / "outbox")
    assert (run / "selected" / "complete.json").read_bytes() == sealed


def test_recorded_best_must_not_escape_its_run(work):
    terminal = implementation("bigrec_terminal")
    import bigrec_checkpoint
    run, best, checkpoint, spec, report, maintenance = make_run(work)
    path = checkpoint / "trainer_state.json"
    state = json.loads(path.read_text())
    state["best_model_checkpoint"] = str(work / "foreign" / "checkpoint-8")
    path.write_text(json.dumps(state))
    bigrec_checkpoint.seal(checkpoint)
    with pytest.raises(ValueError, match="recorded best"):
        terminal.finalize_terminal(run, checkpoint, spec, report, maintenance, work / "outbox")


def test_frozen_source_patch_has_only_three_guarded_control_flow_insertions():
    maintenance = implementation("bigrec_maintenance")
    source = (ROOT / "scripts" / "letter" / "bigrec_train.py").read_bytes()
    patched = maintenance.patched_training_source(source)
    assert patched.count(b"_terminal.finish_if_terminal(") == 2
    assert patched.count(b"_terminal.persist_decision(") == 1
    assert patched.index(b"_terminal.finish_if_terminal(") < patched.index(b"model = load_training_model")
    end = patched.index(b"    trainer.train(")
    assert patched.index(b"_terminal.finish_if_terminal(", end) < patched.index(b"    stopped =", end)
    with pytest.raises(ValueError, match="frozen"):
        maintenance.patched_training_source(source + b"\n# concurrent change\n")


def test_terminal_status_is_persisted_before_sealing_and_never_added_to_existing_seals(work):
    terminal = implementation("bigrec_terminal")
    run, best, checkpoint, spec, report, maintenance = make_run(work)
    with pytest.raises(ValueError, match="sealed"):
        terminal.persist_decision(checkpoint, spec, maintenance)
    (checkpoint / "complete.json").unlink()
    terminal.persist_decision(checkpoint, spec, maintenance)
    marker = json.loads((checkpoint / "terminal-status.json").read_text())
    assert marker["decision"]["terminal"] and marker["decision"]["patience_counter"] == 5


def test_legacy_native_callback_does_not_restore_the_terminal_decision():
    from transformers import EarlyStoppingCallback, TrainerControl, TrainerState

    args = SimpleNamespace(
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        evaluation_strategy="epoch", eval_strategy="epoch", greater_is_better=False,
    )
    callback = EarlyStoppingCallback(early_stopping_patience=5)
    callback.early_stopping_patience_counter = 5
    state = TrainerState(global_step=48, best_metric=1.0)
    control = TrainerControl()
    callback.on_train_begin(args, state, control)
    assert not control.should_training_stop
    callback.on_evaluate(args, state, control, metrics={"eval_loss": 0.5})
    assert callback.early_stopping_patience_counter == 0
