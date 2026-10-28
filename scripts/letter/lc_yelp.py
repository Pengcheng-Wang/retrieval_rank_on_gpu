"""Isolated Yelp micro64/accum1 control; imports, but never rewrites, frozen LC v4."""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import secrets
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

from lc_lifecycle import read, require_hash, sha, write


CONTROL_VERSION = "lc-yelp-m64-v1"
GATE_ROWS = 512
GATE_STEPS = 12


def geometry(examples, micro, world, accumulation):
    batches = math.ceil(math.ceil(examples / micro) / world)
    integral = batches % accumulation == 0
    return {
        "examples": examples, "micro_batch": micro, "world_size": world,
        "accumulation": accumulation, "effective_batch": micro * world * accumulation,
        "microbatches_per_epoch": batches, "integral_accumulation_epoch": integral,
        "updates_per_epoch": batches // accumulation if integral else None,
        "distributed_padding_examples": batches * micro * world - examples,
    }


def yelp_recipe(manifest, dataset):
    if dataset == "Beauty":
        raise ValueError("Beauty remains unadmitted; no filtering or task changes are authorized")
    if dataset != "Yelp":
        raise ValueError("This private control is exclusively for Yelp; Instruments v4 is immutable")
    common = manifest["author_lc_rec_recipes"]["shared"]
    native = manifest["author_lc_rec_recipes"]["Yelp"]
    return {
        "seed": 42, "epochs": native["epochs"], "learning_rate": native["learning_rate"],
        "temperature": native["temperature"], "lora_r": native["lora_r"],
        "lora_alpha": native["lora_alpha"], "lora_dropout": common["lora_dropout"],
        "target_modules": common["lora_target_modules"], "modules_to_save": common["lora_modules_to_save"],
        "tasks": native["tasks"], "micro_batch": 64, "world_size": 4, "accumulation": 1,
        "effective_batch": 256, "weight_decay": common["weight_decay"],
        "warmup_ratio": common["warmup_ratio"], "lr_scheduler_type": common["lr_scheduler_type"],
        "optimizer": common["optimizer"], "bf16": common["bf16"], "fp16": common["fp16"],
        "context": common["model_max_length"], "gradient_checkpointing": True,
    }


def require_gpu_admission(report, identity, recipe):
    if (report.get("kind") != "real7b-full-microbatch-admission" or report.get("status") != "pass"
            or report.get("inputs") != identity or report.get("recipe") != recipe
            or report.get("micro_batch") != 64):
        raise ValueError("Yelp fit requires a source-bound real GPU full-microbatch admission, not CPU preparation")


def adapt_fit_source(source):
    old = ('torch.save({"inputs": {key: value.cpu() for key, value in probe_batch.items()},\n'
           '                        "logits": logits}, args.output / f"reload-reference-rank-{rank}.pt")')
    if source.count("range(100)") != 2 or source.count("range(8)") != 1 or source.count(old) != 1:
        raise ValueError("Frozen LC fit source no longer matches the narrow private gate adaptation")
    return source.replace("range(100)", "range(512)").replace("range(8)", "range(64)").replace(
        old, "write_reload_reference(args.output, rank, probe_batch, logits)")


def tensor_identity(tensor):
    value = tensor.detach().cpu().contiguous()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest()}


def write_reload_reference(output, rank, batch, logits):
    import torch

    torch.save({"inputs": {key: value.cpu() for key, value in batch.items()},
                "logits_identity": tensor_identity(logits)},
               Path(output) / f"reload-reference-rank-{rank}.pt")


def maximum_training_witness(inputs, audit):
    witnesses = [
        read(path)["train/seqrec/longest"] for path in Path(audit).glob("worker-*/examples.json")
    ]
    if len(witnesses) != 4:
        raise ValueError("Full audit maximum-length witnesses are missing")
    witness = max(witnesses, key=lambda row: row["audit"]["lengths"]["full"])
    raw = witness["rendered"]
    sequential = inputs["train"].datasets[0]
    prompt, full = sequential._get_text_data(
        sequential.inter_data[raw["record_number"]], sequential.prompts[raw["prompt_id"]])
    if prompt != raw["prompt"] or full != raw["full"] or raw["native"] != {
        "input_ids": prompt, "labels": full,
    }:
        raise ValueError("Maximum-length witness differs from unchanged native dataset rendering")
    tokens = inputs["tokenizer"](full + inputs["tokenizer"].eos_token, truncation=False)["input_ids"]
    if len(tokens) != witness["audit"]["lengths"]["full"]:
        raise ValueError("Maximum-length witness tokenization changed")
    return {"native": raw["native"], "record_number": raw["record_number"], "prompt_id": raw["prompt_id"],
            "full_tokens": len(tokens), "rendered_sha256": hashlib.sha256(full.encode()).hexdigest()}


def bootstrap(args):
    spec = read(args.execution_spec)
    if (spec.get("control_version") != CONTROL_VERSION or spec.get("dataset") != "Yelp"
            or spec.get("training_geometry") != {"world_size": 4, "micro_batch": 64, "accumulation": 1,
                                                  "effective_batch": 256}
            or args.dataset != "Yelp"):
        raise ValueError("Unexpected private Yelp execution specification")
    if args.command != "prepare" and socket.gethostname() != spec["worker_host"]:
        raise RuntimeError("Yelp GPU work is restricted to the explicitly assigned worker")
    source = Path(args.frozen_source).resolve()
    for name, digest in spec["frozen_sources_sha256"].items():
        require_hash(source / name, digest)
    sys.path.insert(0, str(source))
    import lc_run as frozen

    if Path(frozen.__file__).resolve().parent != source:
        raise ValueError("Yelp control imported a different frozen source deployment")
    original_load = frozen.load_inputs
    original_options = frozen.options_for
    original_trainer = frozen.trainer_class
    original_preflight = frozen.preflight
    original_fit = frozen.fit
    frozen.recipe_for = yelp_recipe

    def load_inputs(values):
        inputs = original_load(values)
        if len(inputs["train"]) != 225061 or len(inputs["valid"]) != 60862:
            raise ValueError("Yelp native full-data coverage changed")
        inputs["identity"]["private_execution"] = {
            "control_version": CONTROL_VERSION, "control_source_sha256": sha(Path(__file__)),
            "control_path": str(Path(__file__).resolve()), "frozen_source": str(source),
            "lifecycle_source_sha256": sha(Path(__file__).with_name("lc_lifecycle.py")),
            "audit_path_policy": "No relocation or rebinding: original audited source/model/corpus paths retained.",
            "native_geometry": geometry(225061, 32, 4, 2),
            "amended_geometry": geometry(225061, 64, 4, 1),
        }
        inputs["worst_witness"] = maximum_training_witness(inputs, values.audit)
        if getattr(values, "cpu_receipt", None) is not None and values.command != "prepare":
            prepared = read(values.cpu_receipt)
            if (prepared.get("kind") != "cpu-preparation" or prepared.get("status") != "pass"
                    or prepared.get("inputs") != inputs["identity"]):
                raise ValueError("CPU preparation/source/corpus fingerprint differs from the queued run")
        return inputs

    def options(recipe, gate):
        values = original_options(recipe, gate)
        values["per_device_eval_batch_size"] = 32
        return values

    def trainer_class():
        base = original_trainer()

        class FullMicrobatchTrainer(base):
            def training_step(self, model, inputs):
                result = super().training_step(model, inputs)
                import torch
                rank = self.args.process_index
                write(Path(self.args.output_dir) / f"memory-rank-{rank}-{os.getpid()}.json", {
                    "status": "observed", "micro_batch": int(inputs["input_ids"].shape[0]),
                    "tokens_this_batch": int(inputs["input_ids"].shape[1]),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(rank),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(rank),
                    "device_total_bytes": torch.cuda.get_device_properties(rank).total_memory,
                    "global_step_before_update": self.state.global_step,
                })
                return result

        return FullMicrobatchTrainer

    preflight_source = inspect.getsource(original_preflight)
    before = 'batch = collator([inputs["train"][index] for index in range(8)])'
    if preflight_source.count(before) != 1:
        raise ValueError("Frozen native preflight source changed")
    preflight_source = preflight_source.replace(
        before, 'batch = collator([inputs["worst_witness"]["native"] for _ in range(64)])')
    preflight_namespace = dict(frozen.__dict__)
    exec(compile(preflight_source, str(Path(__file__)) + ":full-microbatch-preflight", "exec"),
         preflight_namespace)
    native_preflight = preflight_namespace["preflight"]

    def preflight(model, inputs, rank):
        import torch
        torch.cuda.reset_peak_memory_stats(rank)
        result = native_preflight(model, inputs, rank)
        if result["batch_shape"] != [64, inputs["worst_witness"]["full_tokens"]]:
            raise ValueError("Real Yelp memory gate must use 64 maximum-length native rows per rank")
        result.update(
            full_microbatch=True, maximum_native_tokens=inputs["worst_witness"]["full_tokens"],
            peak_allocated_bytes=torch.cuda.max_memory_allocated(rank),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(rank),
            device_total_bytes=torch.cuda.get_device_properties(rank).total_memory,
        )
        return result

    frozen.load_inputs, frozen.options_for = load_inputs, options
    frozen.trainer_class, frozen.preflight = trainer_class, preflight
    namespace = dict(frozen.__dict__, write_reload_reference=write_reload_reference)
    source_text = adapt_fit_source(inspect.getsource(original_fit))
    exec(compile(source_text, str(Path(__file__)) + ":private-fit", "exec"), namespace)
    return frozen, namespace["fit"], spec


def production_loader_contract(frozen, inputs, output):
    import numpy as np
    import torch
    from accelerate.data_loader import BatchSamplerShard, skip_first_batches
    from torch.utils.data import BatchSampler, RandomSampler
    from transformers import TrainingArguments
    from lc_checkpoint import assert_identical

    if torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("Production-shaped preparation is CPU-only")
    native = inputs["train"].datasets[0]

    class IndexedNative:
        def __len__(self):
            return len(native)

        def __getitem__(self, index):
            row = native[index]
            return index, row["input_ids"], row["labels"]

    dataset = IndexedNative()
    trainer = frozen.trainer_class()(
        model=torch.nn.Linear(1, 1),
        args=TrainingArguments(output_dir=str(Path(output) / "cpu-loader"), no_cuda=True,
                               report_to=[], per_device_train_batch_size=64,
                               remove_unused_columns=False, disable_tqdm=True),
        train_dataset=dataset, data_collator=lambda rows: rows,
    )
    loader_class = type(trainer.get_train_dataloader())

    def reseed():
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

    def rng():
        return {"python": random.getstate(), "numpy": np.random.get_state(),
                "cpu": torch.random.get_rng_state()}

    def restore(state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.random.set_rng_state(state["cpu"])

    def loader(rank, batch):
        generator = torch.Generator().manual_seed(42)
        sampler = BatchSamplerShard(
            BatchSampler(RandomSampler(dataset, generator=generator), batch_size=batch, drop_last=False),
            num_processes=4, process_index=rank, split_batches=False, even_batches=True)
        result = loader_class(dataset, batch_sampler=sampler, collate_fn=lambda rows: rows, num_workers=0)
        result._lc_before_fetch = None
        result._lc_owner = SimpleNamespace(
            args=SimpleNamespace(process_index=rank), _lc_preserve_collated_rng=False)
        return result, generator

    measured = {}
    for batch in (32, 64):
        coverage = np.zeros(len(dataset), dtype=np.int64)
        lengths = []
        for rank in range(4):
            reseed()
            iterator, _ = loader(rank, batch)
            lengths.append(len(iterator))
            observed_batches = 0
            for rows in iterator:
                observed_batches += 1
                if len(rows) != batch:
                    raise ValueError("Prepared rank batch did not retain the declared full microbatch")
                np.add.at(coverage, [row[0] for row in rows], 1)
            if observed_batches != len(iterator):
                raise ValueError("Actual prepared iterator length mismatch")
        expected = geometry(len(dataset), batch, 4, 2 if batch == 32 else 1)
        if (lengths != [expected["microbatches_per_epoch"]] * 4 or (coverage == 0).any()
                or int(coverage.sum() - len(dataset)) != expected["distributed_padding_examples"]):
            raise ValueError("Measured full-data padding/coverage differs from the execution contract")
        measured[str(batch)] = {
            **expected, "prepared_lengths": lengths, "observed_rows": int(coverage.sum()),
            "unique_prefixes": int((coverage > 0).sum()), "padding_rows": int(coverage.sum() - len(dataset)),
            "coverage_sha256": hashlib.sha256(coverage.tobytes()).hexdigest(),
        }

    cases = []

    def scalar_optimizer():
        from transformers import get_cosine_schedule_with_warmup
        parameter = torch.nn.Parameter(torch.tensor(0.5))
        optimizer = torch.optim.AdamW([parameter], lr=3e-4, weight_decay=0.01)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 53, 5280)
        return parameter, optimizer, scheduler

    def scalar_update(parameter, optimizer, scheduler, rows):
        target = int(hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:8], 16) / (2 ** 32)
        loss = (parameter - target).square()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    for rank in range(4):
        reseed()
        base, generator = loader(rank, 64)
        batches = len(base)
        digest, count, cuts = "0" * 64, 0, {}
        parameter, optimizer, scheduler = scalar_optimizer()
        validation_calls = []

        def advance(value, rows):
            return hashlib.sha256(bytes.fromhex(value) + json.dumps(rows).encode()).hexdigest()

        for epoch in range(2):
            for rows in base:
                count += 1
                digest = advance(digest, rows)
                scalar_update(parameter, optimizer, scheduler, rows)
                if count in (1, batches + 1):
                    cuts[count] = copy.deepcopy({
                        "rng": rng(), "digest": digest, "model": parameter.detach(),
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "validation_calls": validation_calls,
                    })
            validation_calls.append({"epoch": epoch + 1, "step": count,
                                     "loss": float((parameter.detach() - 0.25).square())})
            if count == batches:
                cuts[count] = copy.deepcopy({
                    "rng": rng(), "digest": digest, "model": parameter.detach(),
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "validation_calls": validation_calls,
                })
        expected = {"rng": rng(), "ordered_inputs_sha256": digest,
                    "sampler": generator.get_state(), "microbatches": count,
                    "model": parameter.detach().clone(), "optimizer": copy.deepcopy(optimizer.state_dict()),
                    "scheduler": scheduler.state_dict(), "validation_calls": validation_calls}
        for cut, saved in cuts.items():
            directory = Path(output) / f"rank-{rank}-cut-{cut}"
            directory.mkdir()
            torch.save(saved["rng"], directory / f"rng_state_{rank}.pth")
            reseed()
            resumed, resumed_generator = loader(rank, 64)
            parameter, optimizer, scheduler = scalar_optimizer()
            with torch.no_grad():
                parameter.copy_(saved["model"])
            optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
            scheduler.load_state_dict(copy.deepcopy(saved["scheduler"]))
            validation_calls = copy.deepcopy(saved["validation_calls"])
            epochs, skip = divmod(cut, batches)
            for epoch in range(epochs):
                for _ in resumed:
                    pass
            if skip:
                first = skip_first_batches(resumed, skip)
                first.__class__ = loader_class
                first._lc_before_fetch, first._lc_owner = directory, resumed._lc_owner
            else:
                restore(saved["rng"])
                first = resumed
            value, count, sync_cpu = saved["digest"], cut, bool(skip)
            for epoch in range(epochs, 2):
                for rows in (first if epoch == epochs else resumed):
                    if sync_cpu:
                        torch.random.set_rng_state(saved["rng"]["cpu"])
                        sync_cpu = False
                    value = advance(value, rows)
                    count += 1
                    scalar_update(parameter, optimizer, scheduler, rows)
                validation_calls.append({"epoch": epoch + 1, "step": count,
                                         "loss": float((parameter.detach() - 0.25).square())})
            actual = {"rng": rng(), "ordered_inputs_sha256": value,
                      "sampler": resumed_generator.get_state(), "microbatches": count,
                      "model": parameter.detach().clone(), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "validation_calls": validation_calls}
            assert_identical(expected, actual, f"Yelp.production.rank{rank}.cut{cut}")
            cases.append({"rank": rank, "cut_microbatch": cut, "microbatches": count,
                          "ordered_inputs_sha256": value, "status": "pass"})
    return {"status": "pass", "measured": measured, "continuation_cases": cases,
            "scope": "Full native rendered inputs, exact sample/padding coverage, sampler/Python/NumPy/CPU RNG",
            "cpu_surrogate_state": "Bitwise scalar-model/AdamW/cosine scheduler and epoch-validation continuation; not a real-model gate",
            "gpu_model_continuation": "NOT_RUN; requires separately queued real GPU gate"}


def prepare(args):
    import torch
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or torch.cuda.is_available():
        raise RuntimeError("Yelp preparation must not use a GPU")
    args.output.mkdir(parents=True, exist_ok=False)
    frozen, _, spec = bootstrap(args)
    inputs = frozen.load_inputs(args)
    contract = production_loader_contract(frozen, inputs, args.output)
    write(args.output / "production-loader.json", contract)
    result = {
        "kind": "cpu-preparation", "status": "pass", "inputs": inputs["identity"],
        "recipe": inputs["recipe"], "production_loader": contract,
        "maximum_training_witness": {key: value for key, value in inputs["worst_witness"].items()
                                     if key != "native"},
        "gate_rows": GATE_ROWS, "gate_updates": GATE_STEPS, "gpu_gate_status": "queued-not-run",
        "control_source_sha256": sha(Path(__file__)),
        "execution_spec_sha256": sha(args.execution_spec),
        "template_label_temperature_epoch_changes": False,
        "audit_artifact_relocation": False,
    }
    write(args.output / "prepared.json", result)
    print(json.dumps({"status": "pass", "kind": result["kind"], "prepared": str(args.output),
                      "native": contract["measured"]["32"], "amended": contract["measured"]["64"]}), flush=True)


def fit(args):
    if args.gate_mode and args.cpu_receipt is None:
        raise ValueError("A GPU gate requires the completed full-data CPU preparation receipt")
    frozen, private_fit, _ = bootstrap(args)
    args.gate = args.gate_mode
    if not args.gate:
        report = read(args.gate_report)
        require_gpu_admission(report, report.get("inputs"), report.get("recipe"))
    private_fit(args)


def reload_probe(args):
    import torch
    from lc_checkpoint import snapshot_identity
    frozen, _, _ = bootstrap(args)
    rank, world, _ = frozen.setup_runtime()
    inputs = frozen.load_inputs(args)
    checkpoint = args.run / f"checkpoint-{GATE_STEPS}"
    reference = torch.load(args.run / f"reload-reference-rank-{rank}.pt", map_location="cpu")
    if reference["inputs"]["input_ids"].shape[0] != 64:
        raise ValueError("Reload gate is not a full microbatch")
    model = frozen.load_model(inputs, rank)
    frozen.preflight(model, inputs, rank)
    frozen.load_adapter_exact(model, checkpoint)
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        actual = model(**{key: value.to(rank) for key, value in reference["inputs"].items()}).logits
    if tensor_identity(actual) != reference["logits_identity"]:
        raise ValueError("Real Yelp full-microbatch cold adapter reload is not bitwise identical")
    write(args.run / f"cold-reload-rank-{rank}.json", {
        "status": "pass", "micro_batch": 64, "rank": rank, "bitwise_logits": True,
        "checkpoint": snapshot_identity(checkpoint),
    })
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def generation_probe(args):
    import torch
    frozen, _, _ = bootstrap(args)
    rank, world, runtime = frozen.setup_runtime(require_world=False)
    if world != 1 or rank != 0:
        raise ValueError("Grammar gate uses only assigned GPU0 after the four-rank gate exits")
    inputs = frozen.load_inputs(args)
    import lc_evaluate as evaluator
    model = evaluator.load_evaluation_model(inputs, 0, args.run / f"checkpoint-{GATE_STEPS}")
    tokenizer = inputs["test_tokenizer"]
    batch, _ = inputs["author"].collator.TestCollator(inputs["native"], tokenizer)([inputs["valid"][0]])
    batch = {key: value.to(0) for key, value in batch.items()}
    codes = [[tokenizer.convert_tokens_to_ids(token) for token in code] for code in inputs["indices"].values()]
    catalog = {"".join(code) for code in inputs["indices"].values()}
    results = []
    with torch.inference_mode():
        for policy in evaluator.POLICIES:
            width = batch["input_ids"].shape[1]
            generated = model.generate(
                **batch, num_beams=20, num_return_sequences=20, max_new_tokens=5,
                do_sample=False, return_dict_in_generate=True, output_scores=True,
                prefix_allowed_tokens_fn=evaluator.make_constraint(codes, 2, width, policy),
                eos_token_id=2, pad_token_id=0, length_penalty=1.0, early_stopping=True)
            tokens = generated.sequences[:, width:].cpu().tolist()
            if (len(tokens) != 20 or any(len(row) != 5 or row[-1] != 2 or 2 in row[:-1] for row in tokens)
                    or not torch.isfinite(generated.sequences_scores).all()):
                raise ValueError("Yelp fixed-beam validation grammar gate failed")
            texts = [text.strip().replace(" ", "") for text in tokenizer.batch_decode(
                tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)]
            invalid = sum(text not in catalog for text in texts)
            if policy == "full-prefix-trie-correction" and invalid:
                raise ValueError("Catalog trie emitted an invalid code")
            results.append({"policy": policy, "beams": 20, "tokens": 5,
                            "finite_scores": True, "invalid_codes": invalid, "status": "pass"})
    write(args.run / "generation-probe.json", {
        "status": "pass", "scope": "one validation prompt; no test quality measurement",
        "inputs": inputs["identity"], "policies": results, "runtime": runtime,
        "inference_precision": evaluator.INFERENCE_PRECISION,
    })


def admit(args):
    from lc_checkpoint import compare
    prepared = read(args.cpu_receipt)
    config = read(args.baseline / "run-config.json")
    if prepared["status"] != "pass" or prepared["inputs"] != config["inputs"]:
        raise ValueError("GPU gate differs from full-data CPU preparation")
    results = []
    for candidate in (args.mid_epoch, args.epoch_boundary):
        if read(candidate / "run-config.json") != config:
            raise ValueError("Yelp gate recipes are not identical")
        results.append(compare(args.baseline / f"checkpoint-{GATE_STEPS}",
                               candidate / f"checkpoint-{GATE_STEPS}"))
    for run in (args.baseline, args.mid_epoch, args.epoch_boundary):
        completed = read(run / "completed.json")
        if completed["global_step"] != GATE_STEPS or completed["epoch"] != 6:
            raise ValueError("Yelp real gate must finish six bounded epochs and twelve updates")
        for rank in range(4):
            proofs = list(run.glob(f"preflight-rank-{rank}-*.json"))
            if not proofs or any(
                read(path).get("status") != "pass" or read(path).get("batch_shape") != [
                    64, prepared["maximum_training_witness"]["full_tokens"]] for path in proofs
            ):
                raise ValueError("Missing real full-microbatch maximum-length loss/gradient/memory proof")
            memory = list(run.glob(f"memory-rank-{rank}-*.json"))
            if not memory or any(read(path)["micro_batch"] != 64 for path in memory):
                raise ValueError("Actual training updates did not use full per-rank microbatch64")
    for rank in range(4):
        if read(args.baseline / f"cold-reload-rank-{rank}.json").get("status") != "pass":
            raise ValueError("Missing full-microbatch real cold-reload gate")
    generation = read(args.baseline / "generation-probe.json")
    if generation["status"] != "pass" or generation["inputs"] != config["inputs"]:
        raise ValueError("Generation gate is missing or belongs to another execution")
    result = {
        "kind": "real7b-full-microbatch-admission", "status": "pass",
        "inputs": config["inputs"], "recipe": config["recipe"], "runtime": config["runtime"],
        "micro_batch": 64, "world_size": 4, "accumulation": 1,
        "native_geometry": geometry(225061, 32, 4, 2),
        "amended_geometry": geometry(225061, 64, 4, 1),
        "cpu_receipt_sha256": sha(args.cpu_receipt), "gates": results,
        "cold_reload": "bitwise full microbatch, every rank", "generation": generation["policies"],
        "no_template_label_temperature_epoch_changes": True,
    }
    require_gpu_admission(result, config["inputs"], config["recipe"])
    if args.output.exists():
        raise FileExistsError("GPU admissions are immutable")
    write(args.output, result)
    print(json.dumps({"status": "pass", "admission": str(args.output)}), flush=True)


def common_cli(args):
    result = ["--frozen-source", str(args.frozen_source)]
    for key in ("upstream", "manifest", "method_spec", "execution_spec", "model", "model_receipt", "audit"):
        result += ["--" + key.replace("_", "-"), str(getattr(args, key))]
    return result + ["--dataset", "Yelp"]


def invoke(args, label, arguments, gpu="0,1,2,3", distributed=True):
    target = Path(args.gate if hasattr(args, "gate") else args.output)
    log = target / f"{label}.log"
    if log.exists():
        raise FileExistsError("Queued GPU attempts are single-use; preserve the failed attempt")
    launcher = ["torchrun", "--nproc_per_node=4", "--master_port", str(20000 + secrets.randbelow(20000)),
                "--log_dir", str(target / ("elastic-" + label))] if distributed else ["python"]
    command = ["bash", str(args.env_script)] + launcher + [str(Path(__file__).resolve())] + arguments
    with log.open("x", encoding="utf-8") as stream:
        child = subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu),
                               stdout=stream, stderr=subprocess.STDOUT)
    if child.returncode:
        raise RuntimeError(f"Yelp {label} failed with exit {child.returncode}; inspect {log}")


def gate_sequence(args):
    bootstrap(args)
    prepared = read(args.cpu_receipt)
    if prepared.get("kind") != "cpu-preparation" or prepared.get("status") != "pass":
        raise ValueError("No full-data CPU preparation; GPU gate is not authorized")
    require_hash(Path(__file__), prepared["control_source_sha256"])
    require_hash(args.execution_spec, prepared["execution_spec_sha256"])
    require_hash(Path(__file__).with_name("lc_lifecycle.py"),
                 prepared["inputs"]["private_execution"]["lifecycle_source_sha256"])
    args.gate.mkdir(parents=True, exist_ok=False)
    common = common_cli(args) + ["--cpu-receipt", str(args.cpu_receipt)]
    phases = [
        ("baseline", "baseline", None, False), ("mid-stop", "mid-epoch", 1, False),
        ("mid-resume", "mid-epoch", None, True), ("epoch-stop", "epoch-boundary", 2, False),
        ("epoch-resume", "epoch-boundary", None, True),
    ]
    for label, directory, stop, resume in phases:
        command = ["fit"] + common + ["--output", str(args.gate / directory), "--gate-mode"]
        if stop is not None:
            command += ["--stop-after-steps", str(stop)]
        if resume:
            command += ["--resume", "latest"]
        invoke(args, label, command)
    invoke(args, "cold-reload", ["reload-probe"] + common + ["--run", str(args.gate / "baseline")])
    invoke(args, "generation", ["generation-probe"] + common + ["--run", str(args.gate / "baseline")],
           gpu="0", distributed=False)
    invoke(args, "admit", ["admit", "--frozen-source", str(args.frozen_source),
                          "--baseline", str(args.gate / "baseline"),
                          "--mid-epoch", str(args.gate / "mid-epoch"),
                          "--epoch-boundary", str(args.gate / "epoch-boundary"),
                          "--cpu-receipt", str(args.cpu_receipt),
                          "--output", str(args.gate / "admission.json")], gpu="-1", distributed=False)


def supervise_fit(args):
    bootstrap(args)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "snapshots").mkdir(exist_ok=True)
    (args.output / "acks").mkdir(exist_ok=True)
    (args.output / "supervisor.pid").write_text(str(os.getpid()) + "\n")
    while not (args.output / "completed.json").exists():
        complete = list(args.output.glob("checkpoint-*/complete.json"))
        for path in args.output.glob("checkpoint-*"):
            if path.is_dir() and not (path / "complete.json").is_file():
                quarantine = args.output / "quarantine"
                quarantine.mkdir(exist_ok=True)
                path.rename(quarantine / (path.name + "-partial-" + str(time.time_ns())))
        stamp = str(time.time_ns())
        command = ["bash", str(args.env_script), "torchrun", "--nproc_per_node=4",
                   "--master_port", str(20000 + secrets.randbelow(20000)),
                   "--log_dir", str(args.output / ("elastic-" + stamp)), str(Path(__file__).resolve()),
                   "fit"] + common_cli(args) + ["--output", str(args.output),
                                                "--gate-report", str(args.gate_report)]
        if complete:
            command += ["--resume", "latest"]
        with (args.output / ("fit-" + stamp + ".log")).open("x", encoding="utf-8") as stream:
            child = subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3"),
                                     stdout=stream, stderr=subprocess.STDOUT)
            (args.output / "torchrun.pid").write_text(str(child.pid) + "\n")
            code = child.wait()
        if code not in (0, 137, 143):
            raise RuntimeError(f"Yelp fit failed with {code}; inspect preserved failure/checkpoint evidence")
        if not (args.output / "completed.json").exists():
            time.sleep(10)


def add_inputs(parser):
    for name in ("upstream", "manifest", "method-spec", "model", "model-receipt", "audit", "execution-spec"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--dataset", choices=("Yelp",), required=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "fit", "reload-probe", "generation-probe", "gate-sequence", "supervise-fit",
                 "evaluate", "admit"):
        sub = commands.add_parser(name)
        sub.add_argument("--frozen-source", type=Path, required=True)
        if name != "admit":
            add_inputs(sub)
        if name in ("prepare", "fit", "supervise-fit", "evaluate", "admit"):
            sub.add_argument("--output", type=Path, required=True)
        if name in ("fit", "reload-probe", "generation-probe", "gate-sequence"):
            sub.add_argument("--cpu-receipt", type=Path, required=name != "fit")
        if name == "fit":
            sub.add_argument("--gate-mode", action="store_true")
            sub.add_argument("--gate-report", type=Path)
            sub.add_argument("--resume")
            sub.add_argument("--stop-after-steps", type=int)
        if name in ("reload-probe", "generation-probe", "evaluate"):
            sub.add_argument("--run", type=Path, required=True)
        if name == "evaluate":
            sub.add_argument("--decoder-policy", required=True)
            sub.add_argument("--batch-size", type=int, default=1)
        if name == "gate-sequence":
            sub.add_argument("--gate", type=Path, required=True)
        if name in ("gate-sequence", "supervise-fit"):
            sub.add_argument("--env-script", type=Path, required=True)
        if name == "supervise-fit":
            sub.add_argument("--gate-report", type=Path, required=True)
        if name == "admit":
            for key in ("baseline", "mid-epoch", "epoch-boundary", "cpu-receipt"):
                sub.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.frozen_source))
    try:
        if args.command == "evaluate":
            bootstrap(args)
            import lc_evaluate
            lc_evaluate.evaluate(args)
        else:
            {"prepare": prepare, "fit": fit, "reload-probe": reload_probe,
             "generation-probe": generation_probe, "gate-sequence": gate_sequence,
             "supervise-fit": supervise_fit, "admit": admit}[args.command](args)
    except Exception as error:
        import traceback
        target = getattr(args, "output", None) or getattr(args, "gate", None) or getattr(args, "run", None)
        if target is not None and target.is_dir():
            write(target / f"failed-control-{os.getpid()}.json", {
                "status": "failed", "error": str(error), "traceback": traceback.format_exc(),
                "control_version": CONTROL_VERSION, "unix": time.time(),
            })
        raise


if __name__ == "__main__":
    main()
