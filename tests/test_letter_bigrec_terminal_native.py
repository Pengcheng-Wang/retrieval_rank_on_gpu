"""Pinned native Trainer fault injection; synthetic weights, never benchmark fits."""

import argparse
import gc
import hashlib
import json
import os
import signal
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "letter"))


def run_native_faults(project: Path, model_path: Path, real_admission: Path, output: Path) -> dict:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    sys.path.insert(0, str(project / "scripts" / "letter"))
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import EarlyStoppingCallback, LlamaConfig, LlamaForCausalLM, Trainer
    import bigrec_data
    import bigrec_gate
    from bigrec_checkpoint import verify
    from bigrec_source import load_tokenizer, verify_model
    from bigrec_maintenance import BASE_TRAIN_SHA256, OVERLAY_FILES, maintained_module, patched_training_source
    import bigrec_maintenance
    from protocol import file_sha256, write_json

    assert transformers.__version__ == "4.31.0"
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    bigrec_data.verify_admission(real_admission)
    verify_model(model_path)
    if output.exists():
        raise FileExistsError("Native fault proof directories are immutable")
    output.mkdir(parents=True)
    frozen_train = project / "scripts/letter/bigrec_train.py"
    source = frozen_train.read_bytes()
    assert hashlib.sha256(source).hexdigest() == BASE_TRAIN_SHA256
    spec = json.loads((real_admission / "recipe.json").read_text())
    maintenance = {
        "id": "bigrec-terminal-v1-native-fault-fixture",
        "effective_training_sha256": hashlib.sha256(patched_training_source(source)).hexdigest(),
    }
    original_eval_loop = Trainer.evaluation_loop
    original_early = EarlyStoppingCallback.on_evaluate
    original_train = Trainer.train
    results = {}

    class InjectedCrash(RuntimeError):
        pass

    for case in ("ordinary", "cold_terminal", "simultaneous_drain", "crash_after_seal"):
        folder = output / case
        folder.mkdir()
        admission = folder / "fixture-admission"
        admission.mkdir()
        (admission / "recipe.json").write_bytes((real_admission / "recipe.json").read_bytes())
        report = {
            "status": "synthetic_native_fault_fixture_not_benchmark_admission",
            "dataset": "Instruments", "model": {"synthetic_native_llama": True},
            "recipe_sha256": file_sha256(admission / "recipe.json"),
        }
        write_json(admission / "admission.json", report)
        run = folder / "run"
        args = SimpleNamespace(
            admission=admission, model=model_path, output=run, outbox=folder / "fixture-outbox",
            gates=folder / "fixture-gates.json", gate_run=False, stop_after_steps=None, resume=None,
        )
        write_json(args.gates, {"kind": "synthetic_native_fault_fixture_only"})
        if case in ("ordinary", "cold_terminal"):
            module = ModuleType("_legacy_bigrec_native_fault_fixture")
            module.__file__ = str(frozen_train)
            exec(compile(source, str(frozen_train), "exec"), module.__dict__)
        else:
            module = maintained_module(frozen_train, maintenance)

        def tiny_model(*unused):
            base = LlamaForCausalLM(LlamaConfig(
                vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                num_attention_heads=2, max_position_embeddings=2048, pad_token_id=0,
                bos_token_id=1, eos_token_id=2,
            ))
            result = get_peft_model(base, LoraConfig(
                r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "v_proj"],
                bias="none", task_type="CAUSAL_LM",
            ))
            result.config.use_cache = False
            return result

        def fixture_rows(path):
            count = 1024 if path.name == "train.jsonl" else 5000
            return [{
                "input_ids": [1, 3 + i % 8, 11 + i % 8, 2],
                "attention_mask": [1, 1, 1, 1], "labels": [1, 3 + i % 8, 11 + i % 8, 2],
            } for i in range(count)]

        module.verify_admission = lambda path: report
        module.verify_model = lambda path: report["model"]
        module.load_training_model = tiny_model
        module.load_tokenizer = lambda path: load_tokenizer(model_path)
        module.read_rows = fixture_rows
        evaluations, train_entries = [], []
        signal_observed = []

        def controlled_evaluation(trainer, *values, **options):
            result = original_eval_loop(trainer, *values, **options)
            assert result.num_samples == 5000
            evaluations.append(result.num_samples)
            # A predeclared synthetic sequence forces patience exhaustion.
            result.metrics["eval_loss"] = float(len(evaluations)) if len(evaluations) <= 6 else 0.25
            return result

        def observe_train(trainer, *values, **options):
            train_entries.append(True)
            return original_train(trainer, *values, **options)

        def terminal_signal(callback, training_args, state, control, **kwargs):
            original_early(callback, training_args, state, control, **kwargs)
            if callback.early_stopping_patience_counter >= 5 and case in ("cold_terminal", "simultaneous_drain"):
                signal_observed.append(state.global_step)
                signal.raise_signal(signal.SIGTERM)

        original_seal = module.seal

        def crash_after_seal(path):
            result = original_seal(path)
            continuation = json.loads((path / "continuation.json").read_text())
            if case == "crash_after_seal" and continuation["early_stopping_patience_counter"] >= 5:
                raise InjectedCrash("after terminal full-state seal; before selected publication")
            return result

        module.seal = crash_after_seal
        prior_signal = signal.getsignal(signal.SIGTERM)
        try:
            with patch.object(bigrec_data, "verify_admission", lambda path: report), \
                 patch.object(bigrec_gate, "require_gates", lambda *a, **k: {"synthetic_fixture": True}), \
                 patch.object(Trainer, "evaluation_loop", controlled_evaluation), \
                 patch.object(Trainer, "train", observe_train), \
                 patch.object(EarlyStoppingCallback, "on_evaluate", terminal_signal):
                if case == "crash_after_seal":
                    with pytest.raises(InjectedCrash):
                        module.fit(args)
                else:
                    module.fit(args)
                terminal = run / "checkpoint-48"
                verify(terminal)
                state = json.loads((terminal / "trainer_state.json").read_text())
                continuation = json.loads((terminal / "continuation.json").read_text())
                assert state["global_step"] == 48 and state["max_steps"] == 400
                assert state["num_train_epochs"] == 50 and continuation["evaluations"] == 6
                assert continuation["microbatches"] == 1536
                assert continuation["early_stopping_patience_counter"] == 5
                assert Path(state["best_model_checkpoint"]).name == "checkpoint-8"
                assert len(evaluations) == 6 and all(n == 5000 for n in evaluations)
                before_updates = (run / "updates.jsonl").read_bytes()
                before_terminal = (terminal / "complete.json").read_bytes()
                before_entries = len(train_entries)
                if case == "cold_terminal":
                    assert not (run / "completed.json").exists()
                    assert json.loads((run / "drained.json").read_text())["global_step"] == 48
                if case in ("cold_terminal", "crash_after_seal"):
                    resumed = maintained_module(frozen_train, maintenance)
                    resumed.load_training_model = lambda *a, **k: (_ for _ in ()).throw(
                        AssertionError("terminal recovery must not construct a training model"))
                    resumed.read_rows = lambda *a, **k: (_ for _ in ()).throw(
                        AssertionError("terminal recovery must not consume another training or validation row"))
                    args.resume = "latest"
                    resumed.fit(args)
                assert (run / "updates.jsonl").read_bytes() == before_updates
                assert (terminal / "complete.json").read_bytes() == before_terminal
                assert len(train_entries) == before_entries
                assert len(evaluations) == 6
                completed = json.loads((run / "completed.json").read_text())
                selection = json.loads((run / "selected/selection.json").read_text())
                assert completed["global_step"] == 48
                assert selection["best_checkpoint"] == "checkpoint-8"
                assert selection["best_validation_loss"] == 1.0
                best_sha = file_sha256(run / "checkpoint-8/adapter_model.safetensors")
                assert file_sha256(run / "selected/adapter_model.safetensors") == best_sha
                if case == "simultaneous_drain":
                    assert signal_observed == [48]
                    assert not (run / "drained.json").exists()
                results[case] = {
                    "status": "pass", "final_global_step": 48, "max_steps": 400, "epochs": 50,
                    "patience": 5, "training_examples": 1024, "validation_examples": 5000,
                    "microbatches": 1536, "validation_calls": 6,
                    "optimizer_updates": continuation["optimizer_updates"],
                    "best_checkpoint": "checkpoint-8", "selected_adapter_sha256": best_sha,
                    "extra_optimizer_updates": 0, "extra_microbatches": 0, "extra_validations": 0,
                    "native_train_entries": len(train_entries), "signal_observed": signal_observed,
                }
        finally:
            signal.signal(signal.SIGTERM, prior_signal)
            gc.collect()
            torch.cuda.empty_cache()
    hashes = {value["selected_adapter_sha256"] for value in results.values()}
    assert len(hashes) == 1
    result = {
        "status": "pass", "kind": "bigrec_native_terminal_faults",
        "scope": "synthetic native Llama/PEFT weights; real Transformers 4.31 Trainer, AMP, sampler and callbacks",
        "base_train_sha256": BASE_TRAIN_SHA256,
        "effective_training_sha256": maintenance["effective_training_sha256"],
        "overlay_files": {name: file_sha256(Path(bigrec_maintenance.__file__).parent / name) for name in OVERLAY_FILES},
        "cases": results, "all_recorded_best_adapters_identical": True,
        "no_production_fit_or_test_metric_used": True,
    }
    write_json(output / "native-terminal-faults.json", result)
    return result


def test_native_terminal_faults():
    location = os.environ.get("BIGREC_TERMINAL_NATIVE_ROOT")
    if not location:
        pytest.skip("Explicit pinned-runtime native terminal fault gate required")
    result = run_native_faults(
        Path(os.environ["BIGREC_FROZEN_PROJECT"]), Path(os.environ["BIGREC_MODEL"]),
        Path(os.environ["BIGREC_ADMISSION"]), Path(location),
    )
    assert result["status"] == "pass"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run_native_faults(arguments.project, arguments.model, arguments.admission, arguments.output)
    print(json.dumps({"status": result["status"], "cases": list(result["cases"])}))
