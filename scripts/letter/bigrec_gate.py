"""Real-weight, admitted-data BIGRec gates and immutable approval receipts."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

from bigrec_checkpoint import verify
from bigrec_data import read_rows, verify_admission, write_rows
from bigrec_source import load_tokenizer, verify_model
from bigrec_train import (
    deterministic_seed, load_adapter_weights, load_training_model, source_identity,
)
from protocol import file_sha256, require_hash, write_json


def parameter_digest(model, trainable: bool) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad == trainable:
            array = parameter.detach().cpu().contiguous().numpy()
            digest.update(name.encode())
            digest.update(str(array.dtype).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def model_gate(admission: Path, model_path: Path, output: Path) -> dict:
    report = verify_admission(admission)
    model_identity = verify_model(model_path)
    if report["model"] != model_identity:
        raise ValueError("Model gate backbone is not the admitted backbone")
    if (output / "model-gate.json").exists():
        raise FileExistsError("Native model gates are immutable")
    output.mkdir(parents=True, exist_ok=True)
    recipe = json.loads((admission / "recipe.json").read_text(encoding="utf-8"))
    rows = read_rows(admission / "train.jsonl")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    deterministic_seed()
    import torch
    from peft import PeftModel
    from safetensors.torch import save_file
    from transformers import DataCollatorForSeq2Seq
    from bigrec_evaluate import encode_title_batch, generate_batch, last_token_pool, load_frozen_base, rank_l2

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expose one explicitly allocated GPU for the admitted native gate")
    start = time.monotonic()
    tokenizer = load_tokenizer(model_path)
    model = load_training_model(model_path, recipe)
    collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    probes = sorted(rows, key=lambda r: len(r["input_ids"]), reverse=True)[:4]
    features = lambda values: [{k: r[k] for k in ("input_ids", "attention_mask", "labels")} for r in values]
    batch = {k: v.to("cuda") for k, v in collator(features(probes)).items()}
    model.train()
    with torch.autocast("cuda", dtype=torch.float16):
        first = model(**batch)
    oracle = torch.nn.functional.cross_entropy(
        first.logits[:, :-1].contiguous().view(-1, first.logits.shape[-1]),
        batch["labels"][:, 1:].contiguous().view(-1), ignore_index=-100,
    )
    torch.testing.assert_close(first.loss, oracle, rtol=0, atol=0)
    if not torch.isfinite(first.loss):
        raise ValueError("Native vanilla causal CE is not finite")
    before_base = parameter_digest(model, False)
    before_adapter = parameter_digest(model, True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0)
    # This low-scale API probe is not the native-GradScaler training/resume gate below.
    scaler = torch.cuda.amp.GradScaler(init_scale=1.0)
    scaler.scale(first.loss).backward()
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainable.values()):
        raise ValueError("Every intended q/v LoRA parameter needs a finite gradient")
    for target in ("q_proj", "v_proj"):
        if sum(p.grad.abs().sum().item() for n, p in trainable.items()
               if f".{target}." in n and ".lora_B." in n) <= 0:
            raise ValueError("Both native q and v LoRA factors must receive nonzero gradients")
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise ValueError("A frozen base parameter received a gradient")
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    if parameter_digest(model, False) != before_base or parameter_digest(model, True) == before_adapter:
        raise ValueError("The adapter did not update independently of the frozen base")
    del first, oracle, optimizer, scaler
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        expected_logits = model(**batch).logits.detach().cpu()
    adapter = output / "api-adapter"
    model.save_pretrained(str(adapter), safe_serialization=True)
    del model, trainable
    gc.collect()
    torch.cuda.empty_cache()
    restored = load_training_model(model_path, recipe).eval()
    load_adapter_weights(restored, adapter)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        actual_logits = restored(**batch).logits.detach().cpu()
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
    finite_examples = 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for offset in range(0, 100, 4):
            sample = {k: v.to("cuda") for k, v in collator(features(rows[offset:offset + 4])).items()}
            loss = restored(**sample).loss
            if not torch.isfinite(loss):
                raise ValueError("100-real-row native smoke has non-finite loss")
            finite_examples += len(sample["input_ids"])
    del restored, batch, actual_logits, expected_logits, sample, loss
    gc.collect()
    torch.cuda.empty_cache()

    base = load_frozen_base(model_path)
    if any("lora_" in n for n, _ in base.named_parameters()):
        raise ValueError("The title grounder was accidentally adapted")
    catalog = read_rows(admission / "catalog.jsonl")
    fixture = [catalog[0]["title"], catalog[0]["title"], catalog[1]["title"], ""]
    embeddings = encode_title_batch(base, tokenizer, fixture)
    encoded = tokenizer(fixture, return_tensors="pt", padding=True, truncation=False)
    encoded = {k: v.to(base.device) for k, v in encoded.items()}
    with torch.inference_mode():
        native = base(**encoded, output_hidden_states=True, use_cache=False).hidden_states[-1][:, -1, :].cpu()
    torch.testing.assert_close(embeddings, native, rtol=0, atol=0)
    torch.testing.assert_close(embeddings[0], embeddings[1], rtol=0, atol=0)
    rank, distances = rank_l2(embeddings[:1].to(base.device), embeddings.to(base.device))
    direct = torch.linalg.vector_norm(embeddings[0].float() - embeddings.float(), dim=1)
    torch.testing.assert_close(distances[0].cpu(), direct, rtol=1e-5, atol=1e-5)
    if rank[0, :2].tolist() != [0, 1]:
        raise ValueError("Duplicate-title grounding did not preserve original-ID tie order")
    try:
        half_distances = torch.cdist(embeddings[:1].to(base.device), embeddings.to(base.device), p=2)
        half_status = {"supported": True, "finite": bool(torch.isfinite(half_distances).all())}
    except RuntimeError as error:
        half_status = {"supported": False, "error_type": type(error).__name__,
                       "diagnostic": str(error)[:240]}
    save_file({"embeddings": embeddings.contiguous()}, str(output / "grounding-fixture.safetensors"))
    generated_model = PeftModel.from_pretrained(
        base, str(adapter), is_trainable=False, torch_dtype=torch.float16, device_map={"": 0},
    ).half().eval()
    parity_prompts = [r["prompt_ids"] for r in rows[:8]]
    native_generation = generate_batch(generated_model, tokenizer, parity_prompts, recipe,
                                       manage_output_memory=False)
    managed_generation = generate_batch(generated_model, tokenizer, parity_prompts, recipe)
    if native_generation != managed_generation:
        raise ValueError("Output-lifetime correction changed native beam tokens or sequence scores")
    generation_rows = sorted(rows, key=lambda r: len(r["prompt_ids"]), reverse=True)[:8]
    generated = generate_batch(generated_model, tokenizer, [r["prompt_ids"] for r in generation_rows], recipe)
    if len(generated) != 8 or any(len(beams) != 4 for beams in generated):
        raise ValueError("Native full-batch generation gate is incomplete")
    write_rows(output / "probe-generations.jsonl", [
        {"user": r["user"], "position": r["position"], "beams": beams}
        for r, beams in zip(generation_rows, generated)
    ])
    result = {
        "status": "pass", "dataset": report["dataset"], "source": source_identity(),
        "admission_sha256": file_sha256(admission / "admission.json"),
        "recipe_sha256": report["recipe_sha256"], "model": model_identity,
        "vanilla_ce": True, "qv_only_gradients": True, "frozen_base": True, "adapter_reload": True,
        "last_token_pooling": True, "four_beam_generation": True, "real_smoke_rows": finite_examples,
        "generation_prompt_source": "training_only_no_heldout_scores",
        "generated_users": 8, "beams_per_user": 4, "generation_budget": 128,
        "native_generation_memory_parity": True,
        "generation_memory": "last-logit clone; release obsolete output KV mapping; exact in-place beam permutation; max_split_size_mb:128",
        "fp16_distance_probe": half_status, "rank_distance_dtype": "float32_direct_cdist",
        "adapter_trainable_parameters": 4194304, "frozen_base_sha256": before_base,
        "api_probe_scaler": "init_scale=1; not a substitute for native-65536 exact-resume gates",
        "elapsed_seconds": time.monotonic() - start,
        "maximum_real_training_tokens": max(len(r["input_ids"]) for r in probes),
    }
    write_json(output / "model-gate.json", result)
    return result


def same_training_source(previous: dict, current: dict) -> bool:
    names = ("bigrec_train.py", "bigrec_checkpoint.py", "bigrec_source.py", "bigrec_data.py")
    return all(name in previous and previous[name] == current.get(name) for name in names)


def exact_tree(left, right) -> bool:
    import numpy as np
    import torch

    if type(left) is not type(right):
        return False
    if torch.is_tensor(left):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(exact_tree(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(exact_tree(a, b) for a, b in zip(left, right))
    return left == right


def critical_trainer_state(state: dict) -> dict:
    return {
        "global_step": state["global_step"], "epoch": state["epoch"],
        "max_steps": state["max_steps"], "num_train_epochs": state["num_train_epochs"],
        "best_metric": state["best_metric"],
        "best_checkpoint": Path(state["best_model_checkpoint"]).name if state["best_model_checkpoint"] else None,
        "evaluations": [{k: v for k, v in row.items() if k in ("eval_loss", "step", "epoch")}
                        for row in state["log_history"] if "eval_loss" in row],
    }


def continuation_cases(root: Path) -> list:
    if (root / "chained").is_dir():
        return [(root / "chained", (0, 5, 8, 9), (5, 8, 9))]
    return [(root / f"interrupted-{step}", (0, step), (step,)) for step in (5, 8, 9)]


def compare_continuations(root: Path) -> dict:
    import torch
    from safetensors.torch import load_file

    reference = root / "continuous"
    expected_config = json.loads((reference / "run-config.json").read_text(encoding="utf-8"))
    if not same_training_source(expected_config["source"], source_identity()):
        raise ValueError("Training/continuation implementation changed after its runs")
    reference_checkpoint = reference / "checkpoint-12"
    verify(reference_checkpoint)
    reference_model = load_file(str(reference_checkpoint / "adapter_model.safetensors"), device="cpu")
    trusted_names = ("optimizer.pt", "scheduler.pt", "scaler.pt", "rng_state.pth")
    reference_states = {n: torch.load(reference_checkpoint / n, map_location="cpu") for n in trusted_names}
    reference_continuation = json.loads((reference_checkpoint / "continuation.json").read_text())
    reference_trainer = critical_trainer_state(json.loads((reference_checkpoint / "trainer_state.json").read_text()))
    reference_trace = read_rows(reference / "trace-from-0.jsonl")
    updates = read_rows(reference / "updates.jsonl")
    previous_count, previous_lr, positive_lr_updates = 0, 0.0, 0
    for update in updates:
        if update["optimizer_updates"] > previous_count and previous_lr > 0:
            positive_lr_updates += 1
        previous_count, previous_lr = update["optimizer_updates"], update["learning_rate"]
    if (reference_trainer["max_steps"] != 400 or reference_trainer["num_train_epochs"] != 50
            or len(reference_trace) != 384 or reference_continuation["optimizer_updates"] < 5
            or reference_continuation["evaluations"] != 1 or positive_lr_updates < 5):
        raise ValueError("The gate changed the native schedule or lacks five actual updates and epoch validation")
    checks = []
    for run, starts, drains in continuation_cases(root):
        label = run.name
        if json.loads((run / "run-config.json").read_text()) != expected_config:
            raise ValueError("Continuous and resumed native configurations differ")
        for drain in drains:
            if verify(run / f"checkpoint-{drain}")["global_step"] != drain:
                raise ValueError(f"Missing sealed native drain point {drain}")
        checkpoint = run / "checkpoint-12"
        verify(checkpoint)
        if not exact_tree(reference_model, load_file(str(checkpoint / "adapter_model.safetensors"), device="cpu")):
            raise ValueError(f"Adapter continuation is not bitwise identical in {label}")
        for name, expected in reference_states.items():
            actual = torch.load(checkpoint / name, map_location="cpu")
            if not exact_tree(expected, actual):
                raise ValueError(f"{name} continuation differs in {label}")
        continuation = json.loads((checkpoint / "continuation.json").read_text())
        if not exact_tree(continuation, reference_continuation):
            raise ValueError(f"Sampler, early-stop, validation or position state differs in {label}")
        trainer = critical_trainer_state(json.loads((checkpoint / "trainer_state.json").read_text()))
        if trainer != reference_trainer:
            raise ValueError(f"Native validation/selection progression differs in {label}")
        trace = [row for start in starts for row in read_rows(run / f"trace-from-{start}.jsonl")]
        if trace != reference_trace:
            raise ValueError(f"Ordered native training batches or losses differ in {label}")
        checks.extend({
            "drain_step": drain, "checkpoint_sha256": file_sha256(run / f"checkpoint-{drain}/complete.json"),
            "final_checkpoint_sha256": file_sha256(checkpoint / "complete.json"),
            "exact": True, "chain": label,
        } for drain in drains)
    result = {
        "status": "pass", "source": expected_config["source"],
        "admission_sha256": expected_config["admission_sha256"],
        "actual_optimizer_updates": reference_continuation["optimizer_updates"],
        "nonzero_learning_rate_optimizer_updates": positive_lr_updates,
        "adapter_bitwise": True, "optimizer_bitwise": True, "scheduler_exact": True,
        "scaler_exact": True, "rng_exact": True, "sampler_exact": True, "ordered_batches_exact": True,
        "validation_state_exact": True, "prepared_batches": 256, "accumulation": 32,
        "compared_microbatches": 384, "validation_examples": 5000, "evaluations": 1,
        "unchanged_epoch_budget": 50, "unchanged_scheduler_steps": 400, "checks": checks,
        "reference_checkpoint_sha256": file_sha256(reference_checkpoint / "complete.json"),
    }
    write_json(root / "resume-gate.json", result)
    return result


def approve(model_gates: list[Path], resume_gate: Path, output: Path) -> dict:
    source = source_identity()
    resume = json.loads(resume_gate.read_text(encoding="utf-8"))
    if (resume["status"] != "pass" or not same_training_source(resume["source"], source)
            or resume["actual_optimizer_updates"] < 5
            or resume.get("nonzero_learning_rate_optimizer_updates", 0) < 5):
        raise ValueError("The native continuation gate is missing or stale")
    models = {}
    for path in model_gates:
        result = json.loads(path.read_text(encoding="utf-8"))
        if (result["status"] != "pass" or result["source"] != source or result["real_smoke_rows"] != 100
                or result.get("native_generation_memory_parity") is not True):
            raise ValueError("An admitted real-backbone model gate is missing or stale")
        models[result["dataset"]] = {
            "admission_sha256": result["admission_sha256"], "model_gate_sha256": file_sha256(path),
        }
    if set(models) != {"Instruments", "Beauty", "Yelp"}:
        raise ValueError("All three dataset-specific real model/length gates are required")
    result = {
        "status": "pass", "source": source, "models": models,
        "resume_gate_sha256": file_sha256(resume_gate), "actual_optimizer_updates": resume["actual_optimizer_updates"],
        "nonzero_learning_rate_optimizer_updates": resume["nonzero_learning_rate_optimizer_updates"],
        "training_evidence_source": resume["source"],
        "training_evidence_components_unchanged": True,
        "evaluation_source_independently_gated": True,
        "recipe_frozen_before_test": True, "native_tests_only_training_prompts": True,
    }
    if output.exists():
        raise FileExistsError("BIGRec approval receipts are immutable")
    write_json(output, result)
    return result


def require_gates(path: Path, admission: Path) -> dict:
    report = verify_admission(admission)
    gates = json.loads(path.read_text(encoding="utf-8"))
    if (gates["status"] != "pass" or gates["source"] != source_identity()
            or gates["models"][report["dataset"]]["admission_sha256"] != file_sha256(admission / "admission.json")
            or gates["actual_optimizer_updates"] < 5):
        raise ValueError("BIGRec execution gates are missing, failed, stale or for different data")
    return gates


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    model = commands.add_parser("model")
    for name in ("admission", "model", "output"):
        model.add_argument(f"--{name}", type=Path, required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--root", type=Path, required=True)
    approval = commands.add_parser("approve")
    approval.add_argument("--model-gates", type=Path, nargs=3, required=True)
    approval.add_argument("--resume-gate", type=Path, required=True)
    approval.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "model":
        result = model_gate(args.admission, args.model, args.output)
    elif args.command == "compare":
        result = compare_continuations(args.root)
    else:
        result = approve(args.model_gates, args.resume_gate, args.output)
    print(json.dumps({"status": result["status"], "gate": args.command}))


if __name__ == "__main__":
    main()
