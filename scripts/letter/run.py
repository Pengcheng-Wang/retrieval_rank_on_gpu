"""Reproduce the author's T5 training recipe using unchanged model/data classes."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import signal
import time
from pathlib import Path

from author import SOURCE_HASHES, default_args, load_backend
from prepare import make_tiger_tokenizer, read_release, tokenizer_digest
from runtime_patch import require_resume_patch
from protocol import (
    accumulation_steps, bind_run_config, file_sha256, seal_checkpoint,
    verify_checkpoint, write_json,
)


def require_integral_epoch(microbatches: int, accumulation: int) -> None:
    if microbatches <= 0 or accumulation <= 0 or microbatches % accumulation:
        raise ValueError("Legacy exact resume requires integral accumulation epochs")


def training_options(
    *, seed: int, batch_size: int, effective_batch: int, world_size: int,
    learning_rate: float, epochs: int,
) -> dict:
    return {
        "seed": seed, "data_seed": seed,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": accumulation_steps(effective_batch, batch_size, world_size),
        "warmup_ratio": 0.01, "num_train_epochs": epochs,
        "learning_rate": learning_rate, "weight_decay": 0.01,
        "lr_scheduler_type": "cosine", "logging_steps": 10, "optim": "adamw_torch",
        "evaluation_strategy": "epoch", "save_strategy": "epoch",
        "save_total_limit": 2, "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss", "greater_is_better": False,
        "ddp_find_unused_parameters": False if world_size > 1 else None,
        "eval_delay": 1, "fp16": False, "bf16": False, "report_to": [],
        "disable_tqdm": True,
    }


def fit(args) -> None:
    import torch
    import transformers
    from torch.utils.data import Subset
    from transformers import EarlyStoppingCallback, T5Config, Trainer, TrainerCallback, TrainingArguments

    if not torch.cuda.is_available():
        raise RuntimeError("The training job requires CUDA; no CPU fallback")
    trainer_sha256 = require_resume_patch()
    if not 0 < args.temperature < float("inf"):
        raise ValueError("Temperature must be finite and positive")
    if (args.output / "completed.json").exists():
        raise FileExistsError("This run is already complete; never overwrite it")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        torch.distributed.init_process_group("nccl")
    author = load_backend(args.upstream, "tiger")
    author.utils.set_seed(args.seed)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    _, _, indices = read_release(args.upstream, manifest, args.dataset)
    audit = json.loads((args.audit / "audit.json").read_text(encoding="utf-8"))
    tokenizer = make_tiger_tokenizer(author, indices)
    if (audit["status"] != "pass" or audit["dataset"] != args.dataset
            or audit["inputs"] != manifest["datasets"][args.dataset]["sha256"]
            or audit["manifest_sha256"] != file_sha256(args.manifest)
            or audit["source"] != SOURCE_HASHES["tiger"]
            or audit["tokenizer_vocab_sha256"] != tokenizer_digest(tokenizer)):
        raise ValueError("Full-data audit is missing, stale or bound to different inputs")
    upstream_args = default_args(author, args.upstream, args.dataset)
    train_data, valid_data = author.utils.load_datasets(upstream_args)
    if args.smoke:
        train_data = Subset(train_data, list(range(100)))
        valid_data = Subset(valid_data, list(range(100)))
    config = T5Config.from_pretrained(author.directory / "ckpt" / "TIGER", local_files_only=True)
    config.vocab_size = len(tokenizer)
    model = author.model.LETTER(config)
    model.set_hyper(args.temperature)
    model.resize_token_embeddings(len(tokenizer))
    model.to(torch.device("cuda", local_rank))
    model.config.use_cache = False

    options = training_options(
        seed=args.seed, batch_size=args.batch_size, effective_batch=args.effective_batch,
        world_size=world_size, learning_rate=args.learning_rate, epochs=args.epochs,
    )
    if args.smoke:
        options.update(
            max_steps=5, logging_steps=1, save_strategy="steps", save_steps=1,
            evaluation_strategy="no", load_best_model_at_end=False,
        )
    receipt = {
        "dataset": args.dataset, "seed": args.seed, "temperature": args.temperature,
        "kind": "smoke" if args.smoke else "released-id-replay",
        "inputs": audit["inputs"], "audit_sha256": file_sha256(args.audit / "audit.json"),
        "author_source": SOURCE_HASHES["tiger"], "options": options,
        "world_size": world_size, "initialization": "author random T5 configuration",
        "tokenizer_vocab_sha256": tokenizer_digest(tokenizer),
        "model_config": config.to_dict(),
        "harness_sha256": {
            name: file_sha256(Path(__file__).parent / name)
            for name in ("run.py", "author.py", "protocol.py", "prepare.py", "runtime_patch.py")
        },
        "trainer_source_sha256": trainer_sha256,
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "accelerate")},
    }
    receipt = json.loads(json.dumps(receipt))
    if local_rank == 0:
        bind_run_config(args.output, receipt)
        tokenizer.save_pretrained(args.output)
        write_json(args.output / "runtime.json", {
            "gpu": torch.cuda.get_device_name(local_rank),
            "cuda": torch.version.cuda, "torch": torch.__version__,
            "transformers": transformers.__version__, "pid": os.getpid(),
        })
    if world_size > 1:
        torch.distributed.barrier()
    resume = None
    if args.resume:
        if args.resume == "latest":
            checkpoints = sorted(args.output.glob("checkpoint-*"),
                                 key=lambda p: int(p.name.split("-")[-1]), reverse=True)
            complete = [p for p in checkpoints if (p / "complete.json").is_file()]
            if not complete:
                raise ValueError("No complete checkpoint is available for exact resume")
            resume = complete[0]
        else:
            resume = Path(args.resume)
        verify_checkpoint(resume)

    class Receipts(TrainerCallback):
        draining = False

        def on_train_begin(self, training_args, state, control, train_dataloader=None, **kwargs):
            require_integral_epoch(len(train_dataloader), training_args.gradient_accumulation_steps)
            if state.is_world_process_zero:
                bind_run_config(args.output / "loader", {
                    "microbatches_per_epoch": len(train_dataloader),
                    "accumulation": training_args.gradient_accumulation_steps,
                    "world_size": world_size, "training_examples": len(train_data),
                })

        def on_step_end(self, training_args, state, control, **kwargs):
            requested_stop = self.draining or (
                args.stop_after_steps is not None and state.global_step >= args.stop_after_steps
            )
            if requested_stop:
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_epoch_end(self, training_args, state, control, **kwargs):
            partial = state.epoch is not None and not float(state.epoch).is_integer()
            if partial and (self.draining or args.stop_after_steps is not None):
                control.should_evaluate = False
            return control

        def on_save(self, training_args, state, control, **kwargs):
            if world_size > 1:
                torch.distributed.barrier()
            if state.is_world_process_zero:
                checkpoint = args.output / f"checkpoint-{state.global_step}"
                write_json(checkpoint / "callback-state.json", {
                    "early_stopping_patience_counter":
                        early_stopping.early_stopping_patience_counter if early_stopping else None,
                })
                seal_checkpoint(checkpoint)
                write_json(args.output / "latest-state.json", {
                    "global_step": state.global_step, "epoch": state.epoch,
                    "checkpoint": checkpoint.name, "best_model_checkpoint": state.best_model_checkpoint,
                    "best_metric": state.best_metric, "updated_unix": time.time(),
                })
            if world_size > 1:
                torch.distributed.barrier()

    early_stopping = None if args.smoke else EarlyStoppingCallback(early_stopping_patience=20)
    callbacks = [Receipts()] + ([early_stopping] if early_stopping else [])
    if resume:
        callback_state = json.loads((resume / "callback-state.json").read_text(encoding="utf-8"))
        if early_stopping:
            early_stopping.early_stopping_patience_counter = callback_state["early_stopping_patience_counter"]
    # A drain requests a complete save at a step boundary rather than a partial tensor write.
    signal.signal(signal.SIGTERM, lambda sig, frame: setattr(callbacks[0], "draining", True))
    trainer = Trainer(
        model=model, args=TrainingArguments(output_dir=str(args.output), **options),
        train_dataset=train_data, eval_dataset=valid_data, tokenizer=tokenizer,
        data_collator=author.collator.Collator(upstream_args, tokenizer), callbacks=callbacks,
    )
    started = time.monotonic()
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    trainer.save_state()
    intentional_stop = callbacks[0].draining or (
        args.stop_after_steps is not None and trainer.state.global_step >= args.stop_after_steps
    )
    if not intentional_stop:
        trainer.save_model(str(args.output))
        if local_rank == 0:
            write_json(args.output / "completed.json", {
                "status": "complete", "global_step": trainer.state.global_step,
                "epoch": trainer.state.epoch, "best_metric": trainer.state.best_metric,
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "wall_seconds_this_invocation": time.monotonic() - started,
                "model_sha256": file_sha256(args.output / "pytorch_model.bin"),
                "run_config_sha256": file_sha256(args.output / "run-config.json"),
            })
    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=("Instruments", "Beauty", "Yelp"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--effective-batch", type=int, default=1024)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--stop-after-steps", type=int)
    args = parser.parse_args()
    for name in ("upstream", "manifest", "audit", "output"):
        setattr(args, name, getattr(args, name).resolve())
    try:
        fit(args)
    except Exception as error:
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            write_json(args.output / "failed.json", {
                "error_type": type(error).__name__, "message": str(error), "unix": time.time(),
            })
        raise


if __name__ == "__main__":
    main()
