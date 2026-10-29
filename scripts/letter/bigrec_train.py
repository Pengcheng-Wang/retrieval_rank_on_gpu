"""Native BIGRec training geometry with complete, auditable cold continuation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import signal
import time
from pathlib import Path

from bigrec_checkpoint import latest, seal, verify
from bigrec_data import read_rows, verify_admission
from bigrec_source import load_tokenizer, verify_model
from protocol import bind_run_config, file_sha256, require_hash, write_json


CORE_FILES = (
    "bigrec_train.py", "bigrec_checkpoint.py", "bigrec_source.py", "bigrec_data.py", "bigrec_evaluate.py",
)


def source_identity() -> dict:
    return {name: file_sha256(Path(__file__).parent / name) for name in CORE_FILES}


def deterministic_seed(seed: int = 42) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from transformers import set_seed

    set_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def training_options(recipe: dict) -> dict:
    t = recipe["training"]
    return {
        "seed": t["trainer_seed"], "data_seed": t["trainer_seed"],
        "per_device_train_batch_size": t["micro_batch"],
        "per_device_eval_batch_size": t["micro_batch"],
        "gradient_accumulation_steps": t["accumulation"],
        "num_train_epochs": t["epochs"], "learning_rate": t["learning_rate"],
        "warmup_steps": t["warmup_steps"], "weight_decay": t["weight_decay"],
        "lr_scheduler_type": t["lr_scheduler_type"], "optim": t["optimizer"],
        "fp16": t["fp16"], "bf16": False, "max_grad_norm": t["max_grad_norm"],
        "logging_steps": t["logging_steps"], "group_by_length": t["group_by_length"],
        "evaluation_strategy": "epoch", "save_strategy": "epoch",
        "load_best_model_at_end": True, "metric_for_best_model": "eval_loss",
        "greater_is_better": False, "save_total_limit": None,
        "report_to": [], "disable_tqdm": True, "dataloader_num_workers": 0,
        "dataloader_pin_memory": True, "remove_unused_columns": True,
    }


def load_training_model(model_path: Path, recipe: dict):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
    from transformers import LlamaForCausalLM

    base = LlamaForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, use_safetensors=True,
        load_in_8bit=True, torch_dtype=torch.float16, device_map={"": 0},
    )
    base = prepare_model_for_int8_training(base)
    lora = recipe["training"]["lora"]
    model = get_peft_model(base, LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"], bias="none", task_type="CAUSAL_LM",
    ))
    model.config.use_cache = False
    model.config.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if (not trainable or any(".lora_" not in n or not any(f".{q}." in n for q in ("q_proj", "v_proj"))
                             for n in trainable)):
        raise ValueError("Only native q_proj/v_proj LoRA parameters may train")
    if sum(p.numel() for p in trainable.values()) != 4194304:
        raise ValueError("Unexpected compatible 7B rank-eight q/v LoRA parameter count")
    return model


def load_adapter_weights(model, path: Path) -> None:
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file

    weights = load_file(str(path / "adapter_model.safetensors"), device="cpu")
    expected = get_peft_model_state_dict(model)
    if set(weights) != set(expected) or any(weights[k].shape != expected[k].shape for k in weights):
        raise ValueError("Adapter keys or shapes do not match native q/v LoRA")
    set_peft_model_state_dict(model, weights)


def optimizer_updates(optimizer) -> int:
    state = optimizer.state_dict()["state"]
    steps = {int(v["step"].item() if hasattr(v["step"], "item") else v["step"])
             for v in state.values() if "step" in v}
    if len(steps) > 1:
        raise ValueError("Native q/v AdamW parameters have inconsistent update counts")
    return next(iter(steps), 0)


def save_on_step_end(step: int, draining: bool, interval: int, steps_per_epoch: int) -> bool:
    return step % steps_per_epoch != 0 and (step % interval == 0 or draining)


def batch_digest(inputs: dict) -> str:
    import torch

    digest = hashlib.sha256()
    for name, value in sorted(inputs.items()):
        if torch.is_tensor(value):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(name.encode())
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()

def recording_sampler(batch_size: int, lengths: list[int], state: dict | None = None):
    from transformers.trainer_pt_utils import LengthGroupedSampler

    class RecordingSampler(LengthGroupedSampler):
        def __init__(self):
            super().__init__(batch_size, lengths=lengths)
            self.epoch_indices = {} if state is None else dict(state["epoch_indices"])
            self.iteration = 0

        def __iter__(self):
            key = str(self.iteration)
            self.iteration += 1
            if key not in self.epoch_indices:
                self.epoch_indices[key] = list(super().__iter__())
            indices = self.epoch_indices[key]
            if sorted(indices) != list(range(len(lengths))):
                raise ValueError("Cached native length-grouped permutation is invalid")
            return iter(indices)

        def export_state(self):
            return {"epoch_indices": dict(self.epoch_indices), "iterations": self.iteration,
                    "batch_size": batch_size, "lengths": lengths}

    if state is not None and (state["batch_size"] != batch_size or state["lengths"] != lengths):
        raise ValueError("Native sampler geometry or token lengths changed")
    return RecordingSampler()


def fit(args) -> None:
    report = verify_admission(args.admission)
    recipe = json.loads((args.admission / "recipe.json").read_text(encoding="utf-8"))
    model_identity = verify_model(args.model)
    if model_identity != report["model"]:
        raise ValueError("The admitted and training backbone identities differ")
    if args.gate_run:
        if args.stop_after_steps is None or not 0 < args.stop_after_steps <= recipe["gates"]["production_shaped_compare_step"]:
            raise ValueError("Gate runs require an explicit bounded step drain, not a shorter epoch budget")
    else:
        if args.gates is None:
            raise ValueError("Long fitting requires approved model and complete-state continuation gates")
        from bigrec_gate import require_gates
        require_gates(args.gates, args.admission)
    if (args.output / "completed.json").exists():
        raise FileExistsError("This immutable BIGRec fit is already complete")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Only the frozen single-GPU BIGRec geometry is admitted")

    deterministic_seed(recipe["training"]["initialization_seed"])
    import torch
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback, Trainer, TrainerCallback, TrainingArguments
    from bigrec_bank import seal_payload, stage_payload
    from runtime_patch import require_resume_patch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one explicitly allocated GPU; no CPU fallback")
    trainer_sha = require_resume_patch()
    rows = read_rows(args.admission / "train.jsonl")
    validation_rows = read_rows(args.admission / "validation.jsonl")
    if len(rows) != 1024 or len(validation_rows) != 5000:
        raise ValueError("The native few-shot/validation geometry changed")
    features = lambda values: [
        {k: r[k] for k in ("input_ids", "attention_mask", "labels")} for r in values
    ]
    train_data = Dataset.from_list(features(rows))
    validation_data = Dataset.from_list(features(validation_rows))
    tokenizer = load_tokenizer(args.model)
    model = load_training_model(args.model, recipe)
    options = training_options(recipe)
    binding = {
        "kind": "bigrec-gate" if args.gate_run else "bigrec-compatible-reference",
        "track": recipe["track"], "dataset": report["dataset"],
        "recipe_sha256": report["recipe_sha256"],
        "admission_sha256": file_sha256(args.admission / "admission.json"),
        "model": model_identity, "source": source_identity(),
        "trainer_source_sha256": trainer_sha, "options": options,
        "versions": {n: importlib.metadata.version(n) for n in
                     ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "safetensors")},
    }
    bind_run_config(args.output, binding)
    write_json(args.output / "runtime.json", {
        "pid": os.getpid(), "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
        "started_unix": time.time(), "visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
    })
    resume = latest(args.output) if args.resume == "latest" else Path(args.resume) if args.resume else None
    continuation = {}
    if resume:
        verify(resume)
        continuation = json.loads((resume / "continuation.json").read_text(encoding="utf-8"))
        if continuation["run_config_sha256"] != file_sha256(args.output / "run-config.json"):
            raise ValueError("Resume state is bound to a different training configuration")
    sampler = recording_sampler(128, [len(r["input_ids"]) for r in rows], continuation.get("sampler"))
    early = EarlyStoppingCallback(early_stopping_patience=recipe["training"]["early_stopping_patience"])
    early.early_stopping_patience_counter = continuation.get("early_stopping_patience_counter", 0)
    callback_state = {
        "evaluations": continuation.get("evaluations", 0),
        "microbatches": continuation.get("microbatches", 0),
    }
    start_step = int(json.loads((resume / "trainer_state.json").read_text())["global_step"]) if resume else 0
    trace_file = args.output / f"trace-from-{start_step}.jsonl"
    if args.gate_run and trace_file.exists():
        raise FileExistsError("Do not overwrite a gate invocation trace")

    class NativeTrainer(Trainer):
        def _get_train_sampler(self):
            return sampler

        def _save(self, output_dir=None, state_dict=None):
            path = Path(output_dir or self.args.output_dir)
            path.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(str(path), safe_serialization=True)
            self.tokenizer.save_pretrained(str(path))

        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            verify(Path(resume_from_checkpoint))
            load_adapter_weights(model or self.model, Path(resume_from_checkpoint))

        def _load_best_model(self):
            if self.state.best_model_checkpoint:
                verify(Path(self.state.best_model_checkpoint))
                load_adapter_weights(self.model, Path(self.state.best_model_checkpoint))

        def _load_optimizer_and_scheduler(self, checkpoint):
            super()._load_optimizer_and_scheduler(checkpoint)
            if checkpoint:
                verify(Path(checkpoint))
                scaler = self.accelerator.scaler
                if scaler is None:
                    raise ValueError("FP16 continuation requires the active Accelerator GradScaler")
                scaler.load_state_dict(torch.load(Path(checkpoint) / "scaler.pt", map_location="cpu"))

        def training_step(self, model, inputs):
            digest = batch_digest(inputs) if args.gate_run else None
            loss = super().training_step(model, inputs)
            callback_state["microbatches"] += 1
            if not torch.isfinite(loss):
                raise ValueError("Native BIGRec training loss is not finite")
            if args.gate_run:
                with trace_file.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({
                        "batch": digest, "loss": float(loss.detach().cpu()),
                        "global_step_before": self.state.global_step,
                        "microbatch": callback_state["microbatches"],
                    }) + "\n")
            return loss

        def _save_checkpoint(self, model, trial, metrics=None):
            super()._save_checkpoint(model, trial, metrics)
            checkpoint = args.output / f"checkpoint-{self.state.global_step}"
            scaler = self.accelerator.scaler
            if scaler is None:
                raise ValueError("A complete FP16 checkpoint requires the Accelerator scaler")
            torch.save(scaler.state_dict(), checkpoint / "scaler.pt")
            write_json(checkpoint / "continuation.json", {
                **callback_state, "sampler": sampler.export_state(),
                "early_stopping_patience_counter": early.early_stopping_patience_counter,
                "optimizer_updates": optimizer_updates(self.optimizer),
                "run_config_sha256": file_sha256(args.output / "run-config.json"),
                "microbatches_per_epoch": 256, "accumulation": 32,
            })
            seal(checkpoint)
            request = stage_payload(checkpoint, args.outbox, f"{args.output.name}-{checkpoint.name}")
            write_json(args.output / "latest-state.json", {
                "global_step": self.state.global_step, "epoch": self.state.epoch,
                "optimizer_updates": optimizer_updates(self.optimizer),
                "checkpoint": checkpoint.name, "checkpoint_sha256": file_sha256(checkpoint / "complete.json"),
                "best_model_checkpoint": self.state.best_model_checkpoint, "best_metric": self.state.best_metric,
                "evaluations": callback_state["evaluations"], "staged": request["identity"], "unix": time.time(),
            })

    class Control(TrainerCallback):
        draining = False

        def on_train_begin(self, training_args, state, control, train_dataloader=None, **kwargs):
            if len(train_dataloader) != 256 or len(train_dataloader) % training_args.gradient_accumulation_steps:
                raise ValueError("Prepared native loader must have 256 batches and integral accumulation")
            write_json(args.output / "loader.json", {
                "prepared_microbatches": len(train_dataloader), "accumulation": 32,
                "steps_per_epoch": 8, "max_steps": state.max_steps, "num_train_epochs": state.num_train_epochs,
                "sampler": "native_LengthGroupedSampler_with_persisted_permutations",
            })

        def on_step_end(self, training_args, state, control, **kwargs):
            updates = optimizer_updates(trainer.optimizer)
            with (args.output / "updates.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "global_step": state.global_step, "epoch": state.epoch,
                    "optimizer_updates": updates, "scale": trainer.accelerator.scaler.get_scale(),
                    "learning_rate": trainer.lr_scheduler.get_last_lr()[0], "unix": time.time(),
                }) + "\n")
            print(f"BIGREC_UPDATE {report['dataset']} step={state.global_step} optimizer_updates={updates}", flush=True)
            requested = self.draining or (args.stop_after_steps is not None and state.global_step >= args.stop_after_steps)
            if save_on_step_end(state.global_step, requested,
                                recipe["training"]["operational_checkpoint_steps"], 8):
                control.should_save = True
            elif state.global_step % 8 == 0:
                control.should_save = False
            if requested:
                control.should_training_stop = True
            return control

        def on_epoch_end(self, training_args, state, control, **kwargs):
            if state.epoch is not None and not float(state.epoch).is_integer():
                if self.draining or args.stop_after_steps is not None:
                    control.should_evaluate = False
                    control.should_save = False
            return control

        def on_evaluate(self, training_args, state, control, **kwargs):
            callback_state["evaluations"] += 1

    controller = Control()
    signal.signal(signal.SIGTERM, lambda sig, frame: setattr(controller, "draining", True))
    trainer = NativeTrainer(
        model=model, args=TrainingArguments(output_dir=str(args.output), **options),
        train_dataset=train_data, eval_dataset=validation_data, tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True, label_pad_token_id=-100,
        ),
        callbacks=[controller, early],
    )
    start = time.monotonic()
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    stopped = controller.draining or (
        args.stop_after_steps is not None and trainer.state.global_step >= args.stop_after_steps
    )
    if stopped:
        final = latest(args.output)
        if final is None or verify(final)["global_step"] != trainer.state.global_step:
            raise ValueError("Drain did not persist a complete checkpoint at its final step")
        write_json(args.output / "drained.json", {
            "status": "drained", "global_step": trainer.state.global_step,
            "optimizer_updates": optimizer_updates(trainer.optimizer),
            "checkpoint": final.name, "wall_seconds": time.monotonic() - start,
        })
        return
    if not trainer.state.best_model_checkpoint:
        raise ValueError("Training cannot select a model without epoch validation")
    selected = args.output / "selected"
    if selected.exists():
        raise FileExistsError("Selected adapters are immutable")
    trainer.save_model(str(selected))
    write_json(selected / "selection.json", {
        "track": recipe["track"], "dataset": report["dataset"],
        "best_checkpoint": Path(trainer.state.best_model_checkpoint).name,
        "best_checkpoint_sha256": file_sha256(Path(trainer.state.best_model_checkpoint) / "complete.json"),
        "best_validation_loss": trainer.state.best_metric,
        "final_global_step": trainer.state.global_step,
        "admission_sha256": file_sha256(args.admission / "admission.json"),
        "run_config_sha256": file_sha256(args.output / "run-config.json"),
        "recipe_sha256": report["recipe_sha256"], "source": source_identity(),
    })
    seal_payload(selected, "bigrec_selected_adapter", {
        "admission_sha256": file_sha256(args.admission / "admission.json"),
        "recipe_sha256": report["recipe_sha256"], "model": model_identity,
    })
    request = stage_payload(selected, args.outbox, f"{args.output.name}-selected")
    write_json(args.output / "completed.json", {
        "status": "complete", "track": recipe["track"], "global_step": trainer.state.global_step,
        "optimizer_updates": optimizer_updates(trainer.optimizer),
        "best_validation_loss": trainer.state.best_metric,
        "selected_complete_sha256": file_sha256(selected / "complete.json"),
        "staged": request["identity"], "test_evaluated": False,
        "wall_seconds_this_invocation": time.monotonic() - start,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("admission", "model", "output", "outbox"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--gates", type=Path)
    parser.add_argument("--gate-run", action="store_true")
    parser.add_argument("--stop-after-steps", type=int)
    parser.add_argument("--resume")
    args = parser.parse_args()
    for name in ("admission", "model", "output", "outbox", "gates"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    try:
        fit(args)
    except Exception as error:
        write_json(args.output / "failed.json", {
            "status": "fail", "type": type(error).__name__, "error": str(error), "unix": time.time(),
        })
        raise


if __name__ == "__main__":
    main()
