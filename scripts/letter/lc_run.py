"""Gated native LC-Rec execution on a released-ID compatible-backbone hybrid."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import signal
import time
import types
from pathlib import Path

from author import SOURCE_HASHES
from lc_checkpoint import (
    assert_identical, compare, seal, snapshot, snapshot_identity, verify, wait_for_ack,
)
from protocol import bind_run_config, file_sha256, require_hash, write_json
from runtime_patch import require_resume_patch


CLAIM = "released-ID compatible-backbone hybrid"
HARNESS = ("lc_run.py", "lc_checkpoint.py", "lc_evaluate.py", "lc_audit.py",
           "author.py", "prepare.py", "protocol.py", "runtime_patch.py", "env.sh")


def epoch_geometry(examples, micro_batch, world_size, accumulation):
    if min(examples, micro_batch, world_size, accumulation) < 1:
        raise ValueError("Invalid positive training geometry")
    batches = math.ceil(math.ceil(examples / micro_batch) / world_size)
    if batches % accumulation:
        raise ValueError("Legacy exact resume requires integral accumulation epochs")
    return {
        "examples": examples, "microbatches_per_epoch": batches,
        "updates_per_epoch": batches // accumulation,
        "effective_batch": micro_batch * world_size * accumulation,
        "distributed_padding_examples": batches * micro_batch * world_size - examples,
    }


def recipe_for(manifest, dataset):
    if dataset == "Beauty":
        raise ValueError("Beauty full corpus FAIL: no filtering, correction, or training is admitted")
    if dataset == "Yelp":
        raise ValueError("Yelp continuation geometry requires a separate production-shaped gate")
    if dataset != "Instruments":
        raise ValueError("Unknown LC dataset")
    shared = manifest["author_lc_rec_recipes"]["shared"]
    native = manifest["author_lc_rec_recipes"][dataset]
    return {
        "seed": 42, "epochs": native["epochs"], "learning_rate": native["learning_rate"],
        "temperature": native["temperature"], "lora_r": native["lora_r"],
        "lora_alpha": native["lora_alpha"], "lora_dropout": shared["lora_dropout"],
        "target_modules": shared["lora_target_modules"],
        "modules_to_save": shared["lora_modules_to_save"], "tasks": native["tasks"],
        "micro_batch": native["per_device_batch_size"],
        "world_size": shared["world_size_convention"],
        "accumulation": shared["gradient_accumulation_steps"],
        "effective_batch": native["effective_batch_size_convention"],
        "weight_decay": shared["weight_decay"], "warmup_ratio": shared["warmup_ratio"],
        "lr_scheduler_type": shared["lr_scheduler_type"], "optimizer": shared["optimizer"],
        "bf16": shared["bf16"], "fp16": shared["fp16"],
        "context": shared["model_max_length"], "gradient_checkpointing": True,
    }


def read_admitted_audit(directory, dataset):
    marker = json.loads((directory / "completed.json").read_text())
    if marker.get("status") != "pass" or marker.get("corpus_admission") != "PASS":
        raise ValueError("LC training requires a completed full corpus PASS audit")
    for name, digest in marker["files_sha256"].items():
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Unsafe audit artifact name")
        require_hash(path, digest)
    audit = json.loads((directory / "audit.json").read_text())
    config = json.loads((directory / "run-config.json").read_text())
    require_hash(directory / "run-config.json", audit["run_config_sha256"])
    if (audit.get("status") != "pass" or audit.get("corpus_admission") != "PASS"
            or audit.get("probe") is not False or audit.get("coverage_status") != "pass"
            or audit.get("all_fingerprints_reverified_after_audit") is not True
            or audit.get("dataset") != dataset or config.get("dataset") != dataset
            or config.get("probe_base_records") is not None):
        raise ValueError("LC training requires an unmodified full corpus PASS audit")
    return audit, config


def setup_runtime(require_world=True):
    import torch

    versions = {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in (
            "torch", "transformers", "peft", "accelerate", "bitsandbytes",
            "sentencepiece", "tokenizers", "safetensors",
        )},
    }
    expected = {"torch": "1.13.1+cu117", "transformers": "4.31.0", "peft": "0.4.0",
                "accelerate": "0.21.0", "bitsandbytes": "0.41.1"}
    if (not versions["python"].startswith("3.10.")
            or any(versions[key] != value for key, value in expected.items())):
        raise RuntimeError("LC execution requires the existing pinned private LETTER runtime")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("LC execution requires the explicitly assigned BF16 CUDA devices")
    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("LOCAL_RANK", "0"))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in {"0,1,2,3", "0"} or (require_world and world != 4):
        raise RuntimeError("Training is assigned node2 GPUs0–3 with world size four")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if world > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
    versions["trainer_source_sha256"] = require_resume_patch()
    versions["torch_deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    return rank, world, versions


def load_inputs(args):
    from lc_audit import (
        _native_args, _tokenizer, _verify_inputs, fingerprint, verify_sequence_rows,
    )
    from author import load_backend
    from prepare import tokenizer_digest
    from protocol import sequence_records

    for name in ("upstream", "manifest", "method_spec", "model", "model_receipt", "audit",
                 "execution_spec"):
        setattr(args, name, getattr(args, name).resolve())
    audit, audited = read_admitted_audit(args.audit, args.dataset)
    spec, manifest, model_receipt, verified, corpus = _verify_inputs(args)
    if verified != audited["verified_files_sha256"]:
        raise ValueError("Executed model/tokenizer/corpus/source bytes differ from the full audit")
    execution = json.loads(args.execution_spec.read_text())
    if (execution.get("claim") != CLAIM or execution.get("optimizer_sharding") != "none"
            or execution.get("attention") != "eager" or execution.get("schema_version") != 1
            or execution.get("author_position_sets_primary") is not True):
        raise ValueError("Unexpected LC execution amendment")
    recipe = recipe_for(manifest, args.dataset)
    author = load_backend(args.upstream, "lc-rec")
    author.utils.set_seed(42)
    native = _native_args(author, args, manifest)
    train, valid = author.utils.load_datasets(native)
    sequential = train.datasets[0]
    verify_sequence_rows(sequential, sequence_records(corpus[0], "train"), corpus[2])
    verify_sequence_rows(valid, sequence_records(corpus[0], "valid"), corpus[2])
    tokenizer = _tokenizer(args.model, corpus[2], 2048)
    test_tokenizer = _tokenizer(args.model, corpus[2], 2048)
    author.collator.TestCollator(native, test_tokenizer)
    if (tokenizer_digest(tokenizer) != audited["tokenizer_vocab_sha256"]
            or tokenizer.padding_side != "right" or test_tokenizer.padding_side != "left"):
        raise ValueError("LC tokenizer vocabulary or isolated padding policy changed")
    identity = {
        "claim": CLAIM, "dataset": args.dataset, "track_id": spec["track_id"],
        "index_origin": "pinned author-released indices; not fresh compatible-backbone tokenizer",
        "model_id": model_receipt["model_id"], "model_revision": model_receipt["revision"],
        "audit_sha256": file_sha256(args.audit / "audit.json"),
        "audit_completed_sha256": file_sha256(args.audit / "completed.json"),
        "verified_files_sha256": verified, "tokenizer_vocab_sha256": tokenizer_digest(tokenizer),
        "model_receipt_sha256": file_sha256(args.model_receipt),
        "execution_spec_sha256": file_sha256(args.execution_spec),
        "validation_render_sha256": fingerprint(valid.valid_text_data),
        "validation_examples": len(valid), "train_examples": len(train),
        "harness_sha256": {name: file_sha256(Path(__file__).parent / name) for name in HARNESS},
        "source_sha256": SOURCE_HASHES["lc-rec"],
        "format": {"tasks": "seqrec", "train_prompt_variants": 17, "validation_prompts": 2,
                   "validation_seed": 42, "test_prompt_id": 0, "only_train_response": True,
                   "history": 20, "his_sep": ", ", "add_prefix": False, "context": 2048,
                   "automatic_eos": False, "explicit_training_eos": True, "chat_template": False,
                   "finish_reason": "N/A (static native targets)", "dropped_examples": 0},
    }
    return {
        "author": author, "native": native, "tokenizer": tokenizer, "test_tokenizer": test_tokenizer,
        "train": train, "valid": valid, "indices": corpus[2], "interactions": corpus[0],
        "identity": identity, "recipe": recipe, "model": args.model, "audit": audit,
    }


def freeze_original_modules(model):
    active = {}
    for name, parameter in model.named_parameters():
        if "original_module" in name and any(x in name for x in ("embed_tokens", "lm_head")):
            parameter.requires_grad = False
        if "modules_to_save.default" in name:
            active[name] = parameter
    if len(active) != 2 or not all(parameter.requires_grad for parameter in active.values()):
        raise ValueError("Both active saved embedding/head modules must remain trainable")
    return active


def load_adapter_exact(model, checkpoint):
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file

    receipt = verify(checkpoint)
    path = checkpoint / receipt["adapter_file"]
    saved = load_file(str(path)) if path.suffix == ".safetensors" else torch.load(
        path, map_location="cpu",
    )
    current = get_peft_model_state_dict(model)
    if current.keys() != saved.keys() or any(
        current[key].shape != saved[key].shape or current[key].dtype != saved[key].dtype
        for key in current
    ):
        raise ValueError("Saved LC adapter has different keys, shapes, or dtypes")
    set_peft_model_state_dict(model, saved)
    assert_identical(get_peft_model_state_dict(model), saved, "adapter_reload")
    freeze_original_modules(model)


def load_model(inputs, rank, adapter=None):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_int8_training
    from transformers import LlamaConfig

    recipe = inputs["recipe"]
    inputs["author"].utils.set_seed(recipe["seed"])
    config = LlamaConfig.from_pretrained(inputs["model"], local_files_only=True)
    config.pretraining_tp = 1
    config._attn_implementation = "eager"
    model = inputs["author"].model.LETTER.from_pretrained(
        inputs["model"], config=config, load_in_8bit=True, device_map={"": rank},
        local_files_only=True,
    )
    model.set_hyper(recipe["temperature"])
    model.resize_token_embeddings(len(inputs["tokenizer"]))
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, LoraConfig(
        r=recipe["lora_r"], lora_alpha=recipe["lora_alpha"],
        target_modules=recipe["target_modules"], modules_to_save=recipe["modules_to_save"],
        lora_dropout=recipe["lora_dropout"], bias="none", inference_mode=False,
        task_type=TaskType.CAUSAL_LM,
    ))
    freeze_original_modules(model)
    model.config.use_cache = False
    attentions = [module for name, module in model.named_modules() if name.endswith("self_attn")]
    if len(attentions) != 32 or any(type(module).__name__ != "LlamaAttention" for module in attentions):
        raise ValueError("Real 7B gate requires all 32 unpatched native eager attention layers")
    if not model.is_loaded_in_8bit or model.config.pretraining_tp != 1:
        raise ValueError("LC model is not using the declared int8/pretraining_tp path")
    if adapter is not None:
        load_adapter_exact(model, adapter)
    return model


def rng_state():
    import numpy as np
    import torch

    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state_all(state["cuda"])


def preflight(model, inputs, rank):
    import torch

    saved_rng = rng_state()
    collator = inputs["author"].collator.Collator(inputs["native"], inputs["tokenizer"])
    batch = collator([inputs["train"][index] for index in range(8)])
    batch = {key: value.to(rank) for key, value in batch.items()}
    model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(**batch)
        oracle = torch.nn.functional.cross_entropy(
            (output.logits[:, :-1] / inputs["recipe"]["temperature"]).reshape(
                -1, len(inputs["tokenizer"])),
            batch["labels"][:, 1:].reshape(-1), ignore_index=-100,
        )
    if not torch.isfinite(output.loss):
        raise ValueError("Non-finite real LC preflight loss")
    assert_identical(output.loss.detach(), oracle.detach(), "native_loss_logits_div_temperature")
    output.loss.backward()
    gradients = {}
    for name, parameter in freeze_original_modules(model).items():
        gradient = parameter.grad
        if gradient is None or not torch.isfinite(gradient).all() or gradient[32000:].abs().sum() <= 0:
            raise ValueError(f"No finite active semantic-ID embedding/head gradient: {name}")
        gradients[name] = float(gradient[32000:].abs().sum())
    if any(p.grad is not None for name, p in model.named_parameters() if "original_module" in name):
        raise ValueError("Frozen original embedding/head received a gradient")
    result = {
        "status": "pass", "native_loss": float(output.loss), "loss_formula": "CE(logits / temperature)",
        "temperature": inputs["recipe"]["temperature"], "active_sid_gradient_l1": gradients,
        "base_originals_frozen": True, "real_model_layers": 32, "int8": True,
        "batch_shape": list(batch["input_ids"].shape), "rank": rank,
    }
    model.zero_grad(set_to_none=True)
    restore_rng(saved_rng)
    return result


def sampler_states(loader):
    result, seen = {}, set()

    def visit(obj, path):
        if obj is None or id(obj) in seen:
            return
        seen.add(id(obj))
        generator = getattr(obj, "generator", None)
        if generator is not None:
            result[path + ".generator"] = generator.get_state().clone()
        for name in ("batch_sampler", "sampler"):
            visit(getattr(obj, name, None), path + "." + name)

    visit(loader, "loader")
    return result


def trainer_class():
    """Reuse guarded Trainer bytecode; scope the stochastic-loader fix to this trainer only."""
    import numpy as np
    import torch
    from accelerate.data_loader import DataLoaderShard
    from accelerate.utils import send_to_device, synchronize_rng_states
    from transformers import Trainer
    import transformers.trainer as trainer_module

    require_resume_patch()

    class NoPrefetchShard(DataLoaderShard):
        def __iter__(self):
            if self.rng_types is not None:
                synchronize_rng_states(self.rng_types, self.synchronized_generator)
            self.begin()
            iterator = torch.utils.data.DataLoader.__iter__(self)
            checkpoint = getattr(self, "_lc_before_fetch", None)
            if checkpoint is not None:
                saved = torch.load(checkpoint / f"rng_state_{self._lc_owner.args.process_index}.pth",
                                   map_location="cpu")
                random.setstate(saved["python"])
                np.random.set_state(saved["numpy"])
                self._lc_owner._lc_preserve_collated_rng = True
            for index in range(len(self)):
                batch = next(iterator)
                self.end_of_dataloader = index == len(self) - 1
                if self.device is not None:
                    batch = send_to_device(batch, self.device)
                yield batch
            try:
                next(iterator)
            except StopIteration:
                pass
            else:
                raise ValueError("Prepared LC loader length did not match its iterator")
            self.end()

    def adopt(loader, owner, checkpoint=None):
        if not isinstance(loader, DataLoaderShard) or loader.num_workers != 0:
            raise RuntimeError("LC resume fix requires the pinned map-style zero-worker DataLoaderShard")
        loader.__class__ = NoPrefetchShard
        loader._lc_owner = owner
        loader._lc_before_fetch = checkpoint
        return loader

    class LCTrainer(Trainer):
        def __init__(self, *args, lc_state=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.lc_state = lc_state or {"microbatches": 0, "ordered_inputs_sha256": "0" * 64,
                                         "losses": []}
            self._lc_preserve_collated_rng = False
            self._lc_prime_reducer = False

        def get_train_dataloader(self):
            return adopt(super().get_train_dataloader(), self)

        def _inner_training_loop(self, *args, **kwargs):
            checkpoint = kwargs.get("resume_from_checkpoint")
            original = Trainer._inner_training_loop
            namespace = dict(original.__globals__)

            def skip(loader, count):
                return adopt(trainer_module.skip_first_batches(loader, count), self, Path(checkpoint))

            namespace["skip_first_batches"] = skip
            scoped = types.FunctionType(
                original.__code__, namespace, original.__name__, original.__defaults__,
                original.__closure__,
            )
            return scoped(self, *args, **kwargs)

        def _load_rng_state(self, checkpoint):
            if self._lc_preserve_collated_rng:
                python_state, numpy_state = random.getstate(), np.random.get_state()
                super()._load_rng_state(checkpoint)
                random.setstate(python_state)
                np.random.set_state(numpy_state)
                self._lc_preserve_collated_rng = False
            else:
                super()._load_rng_state(checkpoint)

        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            load_adapter_exact(model or self.model, Path(resume_from_checkpoint))
            self._lc_prime_reducer = True

        def _prime_resume_reducer(self, model, inputs):
            if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
                raise RuntimeError("LC reducer restoration requires native DDP")
            state = rng_state()
            parameters = {
                name: parameter.detach().clone()
                for name, parameter in self.model.named_parameters() if parameter.requires_grad
            }
            backward_sync = model.require_backward_grad_sync
            forward_sync = model.require_forward_param_sync
            model.require_backward_grad_sync = True
            # Native DDP initially has one large bucket, then changes reduction order on rebuild.
            # Recreate that initialization state without an optimizer update or an RNG advance.
            super().training_step(model, inputs)
            model.zero_grad(set_to_none=True)
            rebuilt = model.reducer._rebuild_buckets()
            if not rebuilt:
                raise RuntimeError("Cold-resume DDP reducer did not rebuild its initial buckets")
            model._has_rebuilt_buckets = True
            model.require_backward_grad_sync = backward_sync
            model.require_forward_param_sync = forward_sync
            restore_rng(state)
            assert_identical(
                {name: parameter.detach() for name, parameter in self.model.named_parameters()
                 if parameter.requires_grad}, parameters, "resume_reducer_priming_parameters",
            )
            self._lc_prime_reducer = False
            print(f"LC_DDP_RESUME_PRIMED rank={self.args.process_index}", flush=True)

        def training_step(self, model, inputs):
            if self._lc_prime_reducer:
                self._prime_resume_reducer(model, inputs)
            digest = hashlib.sha256(bytes.fromhex(self.lc_state["ordered_inputs_sha256"]))
            for key in sorted(inputs):
                value = inputs[key].detach().cpu().contiguous()
                digest.update(key.encode())
                digest.update(str(list(value.shape)).encode())
                digest.update(value.numpy().tobytes())
            self.lc_state["ordered_inputs_sha256"] = digest.hexdigest()
            self.lc_state["microbatches"] += 1
            loss = super().training_step(model, inputs)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite native LC training loss")
            self.lc_state["losses"].append(float(loss))
            if self.lc_state["microbatches"] % self.args.gradient_accumulation_steps == 0:
                for name, parameter in self.model.named_parameters():
                    if parameter.requires_grad and (
                        parameter.grad is None or not torch.isfinite(parameter.grad).all()
                    ):
                        raise ValueError(f"Missing or non-finite intended LC gradient: {name}")
            return loss

    return LCTrainer


def options_for(recipe, gate):
    return {
        "seed": recipe["seed"], "data_seed": recipe["seed"],
        "per_device_train_batch_size": recipe["micro_batch"],
        "per_device_eval_batch_size": recipe["micro_batch"],
        "gradient_accumulation_steps": recipe["accumulation"],
        "num_train_epochs": recipe["epochs"], "learning_rate": recipe["learning_rate"],
        "weight_decay": recipe["weight_decay"], "warmup_ratio": recipe["warmup_ratio"],
        "lr_scheduler_type": recipe["lr_scheduler_type"], "optim": recipe["optimizer"],
        "bf16": True, "fp16": False, "gradient_checkpointing": True,
        "evaluation_strategy": "epoch", "save_strategy": "steps",
        "save_steps": 1 if gate else 50, "save_total_limit": None,
        "load_best_model_at_end": False, "metric_for_best_model": "eval_loss",
        "greater_is_better": False, "eval_delay": 1, "report_to": [],
        "logging_steps": 1 if gate else 10, "logging_nan_inf_filter": False,
        "ddp_find_unused_parameters": False, "dataloader_num_workers": 0,
        "dataloader_drop_last": False, "remove_unused_columns": False,
        "disable_tqdm": True, "save_safetensors": True, "max_grad_norm": 1.0,
        "label_names": ["labels"],
    }


def fit(args):
    import torch
    from torch.utils.data import Subset
    from transformers import TrainerCallback, TrainingArguments

    rank, world, runtime = setup_runtime()
    inputs = load_inputs(args)
    recipe = inputs["recipe"]
    if world != recipe["world_size"]:
        raise ValueError("LC effective batch requires the author world size")
    production = epoch_geometry(len(inputs["train"]), recipe["micro_batch"], world, recipe["accumulation"])
    train, valid = inputs["train"], inputs["valid"]
    if args.gate:
        train, valid = Subset(train, range(100)), Subset(valid, range(100))
    geometry = epoch_geometry(len(train), recipe["micro_batch"], world, recipe["accumulation"])
    options = options_for(recipe, args.gate)
    config = {
        "schema_version": 1, "claim": CLAIM, "exact_original_llama1_reproduction": False,
        "kind": "real7b-gate" if args.gate else "full-fit", "inputs": inputs["identity"],
        "recipe": recipe, "options": options, "runtime": runtime, "geometry": geometry,
        "production_geometry": production, "optimizer_sharding": "none",
        "parallelism_deviation": "Native int8 LoRA with DDP and unsharded AdamW; no DeepSpeed ZeRO2.",
        "attention": "native eager; no FlashAttention monkey patch",
        "continuation": "complete rank RNG/sampler/callback state; no adapter-only restart",
        "loader_correction": "No lookahead; restore Python/NumPy before first resumed collation; "
                             "existing source-guarded full-epoch sampler replay unchanged.",
        "ddp_resume_correction": "Resume-only RNG-neutral, no-update native DDP backward and "
                                 "bucket rebuild before the next actual microbatch.",
        "selection": "minimum validation loss at complete epochs, earliest tie; full budget; test untouched",
    }
    args.output = args.output.resolve()
    if (args.output / "completed.json").exists():
        raise FileExistsError("Completed LC runs are immutable")
    if not args.gate:
        if args.gate_report is None:
            raise ValueError("Full LC fitting requires a real 7B continuation admission receipt")
        gate = json.loads(args.gate_report.read_text())
        if (gate.get("status") != "pass" or gate.get("inputs") != inputs["identity"]
                or gate.get("recipe") != recipe or gate.get("runtime") != runtime):
            raise ValueError("Real 7B gate is missing or bound to a different execution")
        config["gate_report_sha256"] = file_sha256(args.gate_report)
        heartbeat = json.loads((args.output / "exporter-heartbeat.json").read_text())
        if time.time() - heartbeat["unix"] > 180:
            raise ValueError("Node-zero LC exporter is not responsive; do not start an unbanked fit")
    if rank == 0:
        bind_run_config(args.output, config)
        (args.output / "snapshots").mkdir(exist_ok=True)
        write_json(args.output / "runtime.json", {
            "pid": os.getpid(), "gpu": torch.cuda.get_device_name(rank), "cuda": torch.version.cuda,
            "started_unix": time.time(), "versions": runtime,
        })
    torch.distributed.barrier()
    resume = None
    if args.resume:
        if args.resume == "latest":
            checkpoints = sorted(args.output.glob("checkpoint-*/complete.json"),
                                 key=lambda path: int(path.parent.name.split("-")[-1]), reverse=True)
            if not checkpoints:
                raise ValueError("No sealed LC checkpoint is available for exact continuation")
            resume = checkpoints[0].parent
        else:
            resume = Path(args.resume).resolve()
        receipt = verify(resume)
        if receipt["files"]["run-config.json"] != file_sha256(args.output / "run-config.json"):
            raise ValueError("Cannot resume a differently bound LC checkpoint")
    model = load_model(inputs, rank)
    proof = preflight(model, inputs, rank)
    write_json(args.output / f"preflight-rank-{rank}-{os.getpid()}.json", proof)
    lc_state, callbacks_state = None, {"validation_calls": []}
    if resume is not None:
        rank_state = torch.load(resume / f"rank-state-{rank}.pt", map_location="cpu")
        lc_state = rank_state["ordered_training"]
        callbacks_state = json.loads((resume / "callback-state.json").read_text())
    trainer = None

    class Receipts(TrainerCallback):
        draining = False

        def on_train_begin(self, training_args, state, control, train_dataloader=None, **kwargs):
            actual = epoch_geometry(len(train), recipe["micro_batch"], world, recipe["accumulation"])
            if len(train_dataloader) != actual["microbatches_per_epoch"]:
                raise ValueError("Prepared LC dataloader differs from the verified integral geometry")
            if rank == 0:
                bind_run_config(args.output / "loader", {
                    **actual, "accumulation": recipe["accumulation"], "world_size": world,
                    "prepared_loader": type(train_dataloader).__name__,
                })

        def on_step_end(self, training_args, state, control, **kwargs):
            if state.global_step == 1:
                control.should_save = True
            requested = self.draining or (
                args.stop_after_steps is not None and state.global_step >= args.stop_after_steps)
            if requested:
                control.should_save = True
                control.should_training_stop = True
            if state.epoch is not None and float(state.epoch).is_integer():
                control.should_save = False
            if rank == 0:
                write_json(args.output / "live.json", {
                    "global_step": state.global_step, "epoch": state.epoch,
                    "last_microbatch_loss": trainer.lc_state["losses"][-1],
                    "updated_unix": time.time(), "pid": os.getpid(),
                })
            return control

        def on_epoch_end(self, training_args, state, control, **kwargs):
            if state.epoch is not None and float(state.epoch).is_integer():
                control.should_save = True
            else:
                control.should_evaluate = False
            return control

        def on_evaluate(self, training_args, state, control, metrics=None, **kwargs):
            value = float(metrics["eval_loss"])
            if not math.isfinite(value):
                raise ValueError("Non-finite native LC validation loss")
            callbacks_state["validation_calls"].append({
                "global_step": state.global_step, "epoch": state.epoch, "eval_loss": value,
            })
            return control

        def on_save(self, training_args, state, control, **kwargs):
            checkpoint = args.output / f"checkpoint-{state.global_step}"
            torch.save({
                "ordered_training": trainer.lc_state,
                "sampler_generators": sampler_states(trainer.callback_handler.train_dataloader),
            }, checkpoint / f"rank-state-{rank}.pt")
            torch.distributed.barrier()
            if rank == 0:
                write_json(checkpoint / "callback-state.json", callbacks_state)
                write_json(checkpoint / "run-config.json", config)
                seal(checkpoint, world)
                sealed = args.output / "snapshots" / checkpoint.name
                snapshot(checkpoint, sealed)
                write_json(args.output / "latest-state.json", {
                    "global_step": state.global_step, "epoch": state.epoch,
                    "checkpoint": checkpoint.name, "best_checkpoint": state.best_model_checkpoint,
                    "best_metric": state.best_metric, **snapshot_identity(sealed),
                })
                print(f"LC_SEALED step={state.global_step} checkpoint={sealed}", flush=True)
                if not args.gate:
                    wait_for_ack(sealed, args.output / "acks")
                    print(f"LC_BANK_ACK step={state.global_step}", flush=True)
            torch.distributed.barrier()
            return control

    receipts = Receipts()
    signal.signal(signal.SIGTERM, lambda sig, frame: setattr(receipts, "draining", True))
    trainer = trainer_class()(
        model=model, args=TrainingArguments(output_dir=str(args.output), **options),
        train_dataset=train, eval_dataset=valid, tokenizer=inputs["tokenizer"],
        data_collator=inputs["author"].collator.Collator(inputs["native"], inputs["tokenizer"]),
        callbacks=[receipts], lc_state=lc_state,
    )
    started = time.monotonic()
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    interrupted = receipts.draining or (
        args.stop_after_steps is not None and trainer.state.global_step >= args.stop_after_steps)
    if not interrupted:
        expected = geometry["updates_per_epoch"] * recipe["epochs"]
        if (trainer.state.global_step != expected or trainer.state.epoch != recipe["epochs"]
                or len(callbacks_state["validation_calls"]) != recipe["epochs"]):
            raise ValueError("LC fit did not complete its full declared budget and validation schedule")
        if args.gate:
            probe_batch = inputs["author"].collator.Collator(inputs["native"], inputs["tokenizer"])(
                [inputs["valid"][index] for index in range(8)])
            probe_batch = {key: value.to(rank) for key, value in probe_batch.items()}
            model.eval()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**probe_batch).logits.cpu()
            torch.save({"inputs": {key: value.cpu() for key, value in probe_batch.items()},
                        "logits": logits}, args.output / f"reload-reference-rank-{rank}.pt")
        if rank == 0:
            chosen = Path(trainer.state.best_model_checkpoint).name
            write_json(args.output / "completed.json", {
                "status": "complete", "kind": config["kind"], "global_step": trainer.state.global_step,
                "epoch": trainer.state.epoch, "validation_calls": callbacks_state["validation_calls"],
                "selected_checkpoint": chosen,
                "selected_complete_sha256": file_sha256(
                    args.output / "snapshots" / chosen / "complete.json"),
                "run_config_sha256": file_sha256(args.output / "run-config.json"),
                "wall_seconds_this_invocation": time.monotonic() - started,
            })
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def reload_probe(args):
    import torch

    rank, world, _ = setup_runtime()
    inputs = load_inputs(args)
    reference = torch.load(args.run / f"reload-reference-rank-{rank}.pt", map_location="cpu")
    checkpoint = args.run / "checkpoint-8"
    model = load_model(inputs, rank)
    preflight(model, inputs, rank)
    load_adapter_exact(model, checkpoint)
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        actual = model(**{key: value.to(rank) for key, value in reference["inputs"].items()}).logits
    assert_identical(reference["logits"], actual.cpu(), f"cold_adapter_logits_rank_{rank}")
    write_json(args.run / f"cold-reload-rank-{rank}.json", {
        "status": "pass", "bitwise_logits": True, "rank": rank,
        "checkpoint": snapshot_identity(checkpoint),
    })
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def admit(args):
    baseline_config = json.loads((args.baseline / "run-config.json").read_text())
    results = []
    for candidate in (args.mid_epoch, args.epoch_boundary):
        if json.loads((candidate / "run-config.json").read_text()) != baseline_config:
            raise ValueError("Gate candidate recipes differ")
        results.append(compare(args.baseline / "checkpoint-8", candidate / "checkpoint-8"))
    for run in (args.baseline, args.mid_epoch, args.epoch_boundary):
        done = json.loads((run / "completed.json").read_text())
        if done.get("kind") != "real7b-gate" or done.get("global_step") != 8:
            raise ValueError("Gate is not a completed native real-7B run")
        for rank in range(4):
            probes = list(run.glob(f"preflight-rank-{rank}-*.json"))
            if not probes or any(json.loads(path.read_text())["status"] != "pass" for path in probes):
                raise ValueError("Missing real-7B loss/gradient preflight")
    for rank in range(4):
        if json.loads((args.baseline / f"cold-reload-rank-{rank}.json").read_text())["status"] != "pass":
            raise ValueError("Missing cold real-7B adapter reload evidence")
    production = json.loads(args.production_loader_report.read_text())
    if (production.get("status") != "pass" or production.get("training_prefixes") != 131837
            or production.get("microbatches_per_epoch") != 4120
            or production.get("runner_sha256") != file_sha256(Path(__file__))
            or len(production.get("cases", [])) != 12):
        raise ValueError("Missing production-shaped native prompt/sampler continuation gate")
    generation_path = args.baseline / "generation-probe.json"
    generation = json.loads(generation_path.read_text())
    if (generation.get("status") != "pass" or generation.get("inputs") != baseline_config["inputs"]
            or [p["policy"] for p in generation["policies"]] != [
                "author-position-sets", "full-prefix-trie-correction"]):
        raise ValueError("Missing fixed-beam native-precision validation generation gate")
    receipt = {
        "status": "pass", "claim": CLAIM, "scope": "real7B int8/LoRA execution; not model quality",
        "inputs": baseline_config["inputs"], "recipe": baseline_config["recipe"],
        "runtime": baseline_config["runtime"], "production_geometry": baseline_config["production_geometry"],
        "gates": results, "cold_reload_all_ranks_bitwise": True,
        "production_loader_report_sha256": file_sha256(args.production_loader_report),
        "generation_probe_sha256": file_sha256(generation_path),
        "generation_precision": generation["inference_precision"],
        "cuts": {"mid_epoch": 1, "epoch_boundary": 2}, "gate_training_rows": 100,
        "checks": ["native loss CE(logits/temperature)", "finite SID embedding/head gradients",
                   "8 actual optimizer updates", "exact cold adapter logits",
                   "every-rank RNG and sampler generator", "ordered tokenized inputs",
                   "optimizer/scheduler/adapter/callback state", "all four validation calls"],
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "pass", "receipt": str(args.output)}), flush=True)


def add_input_arguments(parser):
    for name in ("upstream", "manifest", "method-spec", "model", "model-receipt", "audit",
                 "execution-spec"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--dataset", choices=("Instruments", "Beauty", "Yelp"), required=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("fit")
    add_input_arguments(train)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--gate", action="store_true")
    train.add_argument("--gate-report", type=Path)
    train.add_argument("--resume")
    train.add_argument("--stop-after-steps", type=int)
    reload = commands.add_parser("reload-probe")
    add_input_arguments(reload)
    reload.add_argument("--run", type=Path, required=True)
    admission = commands.add_parser("admit")
    for name in ("baseline", "mid-epoch", "epoch-boundary", "output"):
        admission.add_argument(f"--{name}", type=Path, required=True)
    admission.add_argument("--production-loader-report", type=Path, required=True)
    args = parser.parse_args()
    try:
        {"fit": fit, "reload-probe": reload_probe, "admit": admit}[args.command](args)
    except Exception as error:
        import traceback
        output = getattr(args, "output", None) or getattr(args, "run", None)
        if output is not None and output.is_dir():
            write_json(output / f"failed-rank-{os.environ.get('LOCAL_RANK', '0')}-{os.getpid()}.json", {
                "status": "error", "error_type": type(error).__name__, "message": str(error),
                "traceback": traceback.format_exc(), "unix": time.time(),
            })
        raise


if __name__ == "__main__":
    main()
