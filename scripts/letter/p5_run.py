"""Source-native P5 training/evaluation with full-state continuation and bank gating."""

from __future__ import annotations

import argparse
import math
import os
import random
import signal
import shutil
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, T5Config

from p5_assets import verify_assets
from p5_checkpoint import IndexSampler, pointer_path, restore_checkpoint, save_checkpoint
from p5_checkpoint import verify_checkpoint
from p5_data import CompleteCollator, datasets, verify_admission
from p5_source import digest_json, load_native, load_recipe, native_args, read_json
from p5_source import require_hash, runtime_versions, sha256, write_json
from prepare import read_release


STOP_REQUESTED = False


def request_stop(*_):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def reconcile_trace(path, checkpoint_step):
    import json

    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    committed, changed = [], False
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise ValueError("Corrupt committed P5 update trace")
            changed = True
            continue
        if row["step"] <= checkpoint_step:
            committed.append(line)
        else:
            changed = True
    if changed:
        shutil.copyfile(path, path.with_name("updates-uncommitted-" + uuid.uuid4().hex + ".jsonl"))
        partial = path.with_name(path.name + ".partial")
        partial.write_text("".join(line + "\n" for line in committed), encoding="utf-8")
        partial.replace(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def make_model(native, model_path, tokenizer, dropout=.1):
    verify_assets(model_path)
    config = T5Config.from_pretrained(str(model_path), local_files_only=True)
    expected = {
        "d_model": 512, "d_ff": 2048, "d_kv": 64, "num_heads": 8,
        "num_layers": 6, "num_decoder_layers": 6, "vocab_size": 32128,
        "decoder_start_token_id": 0, "pad_token_id": 0, "eos_token_id": 1,
        "tie_word_embeddings": True, "feed_forward_proj": "relu",
    }
    if any(getattr(config, key) != value for key, value in expected.items()):
        raise ValueError("Pretrained architecture is not the pinned T5-small")
    config.dropout_rate = dropout
    model = native.modeling_p5.P5(config)
    tensors = load_file(str(Path(model_path) / "model.safetensors"))
    result = model.load_state_dict(tensors, strict=False)
    allowed_missing = {
        "encoder.whole_word_embeddings.weight", "encoder.position_embeddings.weight",
        "encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight",
    }
    native_unused = {"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"}
    if not set(result.unexpected_keys) <= native_unused or not set(result.missing_keys) <= allowed_missing:
        raise ValueError(f"Unexplained pretrained P5 weights: {result}")
    model.pretrained_load_diagnostics = {
        "initialized_or_tied_missing_keys": result.missing_keys,
        "native_explicitly_ignored_unused_keys": result.unexpected_keys,
    }
    if "shared.weight" not in tensors:
        raise ValueError("Safe pretrained snapshot lacks the actual shared embedding")
    if not torch.equal(model.shared.weight.detach(), tensors["shared.weight"]):
        raise ValueError("Pretrained shared embedding was not loaded exactly")
    model.tie_weights()
    model.resize_token_embeddings(len(tokenizer))
    if (model.shared.weight.data_ptr() != model.lm_head.weight.data_ptr()
            or model.shared.weight.data_ptr() != model.encoder.embed_tokens.weight.data_ptr()
            or not model.encoder.whole_word_embeddings.weight.requires_grad):
        raise ValueError("Native tied embeddings/whole-word trainability changed")
    if not torch.equal(
        model.shared.weight.detach()[:min(len(tokenizer), len(tensors["shared.weight"]))],
        tensors["shared.weight"][:min(len(tokenizer), len(tensors["shared.weight"]))],
    ):
        raise ValueError("Resizing the native vocabulary changed pretrained rows")
    old_prepare = native.modeling_p5.P5.prepare_inputs_for_generation

    def prepare(self, input_ids, past_key_values=None, past=None, **kwargs):
        if past is not None and past_key_values is not None:
            raise ValueError("Ambiguous native generation cache")
        return old_prepare(self, input_ids, past=past if past is not None else past_key_values, **kwargs)

    model.prepare_inputs_for_generation = MethodType(prepare, model)
    return model


def optimizer_and_scheduler(native, model, training_rows, recipe):
    training = recipe["training"]
    updates = math.ceil(training_rows / training["effective_batch_size"])
    args = SimpleNamespace(
        gpu="cuda:0", epochs=training["epochs"], gradient_accumulation_steps=1,
        warmup_prop=training["warmup_ratio"], weight_decay=training["weight_decay"],
        lr=training["learning_rate"], adam_eps=training["adam_epsilon"],
    )
    logger = SimpleNamespace(log=lambda *a, **kw: None)
    return native.utils.create_optimizer_and_scheduler(args, logger, model, updates)


def native_loss(model, batch):
    inputs, attention, whole, targets, target_attention = batch
    result = model(
        input_ids=inputs, attention_mask=attention, whole_word_ids=whole,
        whole_word_embedding_type="shijie", labels=targets, return_dict=True,
    )
    token_loss = result.loss.view_as(targets)
    if not torch.isfinite(token_loss).all():
        raise ValueError("Nonfinite native P5 token loss")
    mask = target_attention.float()
    per_example = (token_loss * mask).sum(1) / mask.sum(1)
    return per_example.mean()


def update(model, optimizer, scheduler, train, indices, collator, microbatch, clip):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    weighted_loss, batch_digest, word_gradient = 0., [], 0.
    for offset in range(0, len(indices), microbatch):
        rows = [train[i] for i in indices[offset:offset + microbatch]]
        batch_digest.extend(rows)
        batch = tuple(t.to("cuda") for t in collator(rows))
        loss = native_loss(model, batch)
        weight = len(rows) / len(indices)
        (loss * weight).backward()
        weighted_loss += loss.item() * weight
        del batch, loss
    gradient = model.encoder.whole_word_embeddings.weight.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise ValueError("Native whole-word embedding is inactive or has invalid gradients")
    word_gradient = gradient.norm().item()
    if word_gradient <= 0:
        raise ValueError("Native whole-word embedding has no training gradient")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    if not torch.isfinite(norm):
        raise ValueError("Nonfinite P5 gradient norm")
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "loss": weighted_loss, "gradient_norm": norm.item(), "whole_word_gradient": word_gradient,
        "examples": len(indices), "sampled_rows_sha256": digest_json(batch_digest),
        "lr_after_update": scheduler.get_last_lr()[0],
    }


def generation(model, batch, tokenizer, catalog, native, recipe, cache=True, logits_processor=None):
    inputs, attention, whole, _, _ = batch
    encoded = model.encoder(
        input_ids=inputs, attention_mask=attention, whole_word_ids=whole,
        whole_word_embedding_type="shijie", return_dict=True,
    )
    if "_trie" not in catalog:
        catalog["_trie"] = native.generation_trie.Trie([[0] + ids for ids in catalog["token_ids"]])
        catalog["_valid_sequences"] = {tuple(ids) for ids in catalog["token_ids"]}
        catalog["_generation_max_length"] = max(map(len, catalog["token_ids"])) + 1
    trie = catalog["_trie"]

    def allowed(_batch, prefix):
        result = trie.get(prefix.tolist())
        if not result:
            # Native HF beam search also asks about padding rows of finished examples.
            # Keep native [] masking; validate every returned hypothesis below.
            catalog["_empty_beam_prefix_count"] = catalog.get("_empty_beam_prefix_count", 0) + 1
            if "_first_empty_beam_prefix" not in catalog:
                catalog["_first_empty_beam_prefix"] = prefix.tolist()
        return result

    evaluation = recipe["evaluation"]
    generated = model.generate(
        encoder_outputs=encoded, attention_mask=attention,
        decoder_start_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        num_beams=evaluation["beams"], num_return_sequences=evaluation["returns"],
        max_length=catalog["_generation_max_length"],
        length_penalty=evaluation["length_penalty"], early_stopping=evaluation["early_stopping"],
        min_length=0, do_sample=False, use_cache=cache, repetition_penalty=1.,
        no_repeat_ngram_size=0, prefix_allowed_tokens_fn=allowed,
        output_scores=True, return_dict_in_generate=True, logits_processor=logits_processor,
    )
    expected = inputs.shape[0] * evaluation["returns"]
    if generated.sequences.shape[0] != expected or not torch.isfinite(generated.sequences_scores).all():
        raise ValueError("Generation lost beams or produced nonfinite sequence scores")
    valid = catalog["_valid_sequences"]
    for output in generated.sequences.tolist():
        if output[0] != 0 or 1 not in output[1:]:
            raise ValueError("Generation lost decoder prefix or complete EOS")
        stop = output.index(1, 1)
        if tuple(output[1:stop + 1]) not in valid or any(i != 0 for i in output[stop + 1:]):
            raise ValueError("Generation returned a truncated or non-catalog sequence")
    return generated


def per_user_metrics(canonical_predictions, target, target_item, groups):
    names = ("recall@5", "recall@10", "ndcg@5", "ndcg@10")

    def score(rank):
        return {key: (float(rank is not None and rank <= int(key.split("@")[1]))
                      if key.startswith("recall") else
                      1 / math.log2(rank + 1) if rank is not None and rank <= int(key.split("@")[1]) else 0.)
                for key in names}

    if any(p not in groups for p in canonical_predictions):
        raise ValueError("Generated identity is outside the catalog")
    rank = canonical_predictions.index(target) + 1 if target in canonical_predictions else None
    distinct, seen = [], set()
    for code in dict.fromkeys(canonical_predictions):
        for item in groups[code]:
            if item not in seen:
                distinct.append(item)
                seen.add(item)
    item_rank = distinct.index(target_item) + 1 if target_item in distinct else None
    return {
        "primary": score(rank), "item_diagnostic": score(item_rank),
        "primary_rank": rank, "item_rank": item_rank,
        "top_original_items": distinct[:20],
        "duplicate_beams": len(canonical_predictions) - len(set(canonical_predictions)),
    }


def evaluate(model, data, users, interactions, split, collator, tokenizer, catalog, native,
             recipe, output, limit=None):
    import json

    if split not in ("valid", "test"):
        raise ValueError("Evaluation split must be valid or test")
    count = len(users) if limit is None else min(limit, len(users))
    batch_size = recipe["evaluation"]["batch_size"]
    totals = {"primary": Counter(), "item_diagnostic": Counter()}
    seen, duplicates, resumed_chunks = set(), 0, 0
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    frozen = (read_json(output.parent / "test-started.json")["selected_seal_sha256"]
              if split == "test" else None)
    parts = output.parent / (output.stem + "-parts")
    model.eval()
    with partial.open("w", encoding="utf-8") as stream, torch.inference_mode():
        for start in range(0, count, batch_size):
            indices = list(range(start, min(start + batch_size, count)))
            chunk = parts / f"{start:08d}.json"
            if frozen and chunk.exists():
                saved = read_json(chunk)
                if (saved["frozen_model_sha256"] != frozen or saved["first_user_index"] != start
                        or saved["status"] != "complete" or len(saved["records"]) != len(indices)):
                    raise ValueError("Cannot resume test chunks from a different frozen model/user range")
                records = saved["records"]
                resumed_chunks += 1
            else:
                batch = tuple(t.to("cuda") for t in collator([data[i] for i in indices]))
                result = generation(model, batch, tokenizer, catalog, native, recipe)
                decoded = tokenizer.batch_decode(
                    result.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=True,
                )
                scores = result.sequences_scores.cpu().tolist()
                records = []
                for local, index in enumerate(indices):
                    begin, end = local * 20, (local + 1) * 20
                    ordered = sorted(zip(decoded[begin:end], scores[begin:end]), key=lambda p: -p[1])
                    user = users[index]
                    original = interactions[user][-2 if split == "valid" else -1]
                    records.append({
                        "original_user": user, "original_target_item": original,
                        "target": catalog["canonical_targets"][original],
                        "predictions": [p for p, _ in ordered], "scores": [s for _, s in ordered],
                    })
                if frozen:
                    write_json(chunk, {
                        "status": "complete", "frozen_model_sha256": frozen,
                        "first_user_index": start, "records": records,
                    })
            for record, index in zip(records, indices):
                user = users[index]
                if user in seen:
                    raise ValueError("Repeated evaluation user")
                seen.add(user)
                original = interactions[user][-2 if split == "valid" else -1]
                target = catalog["canonical_targets"][original]
                predictions = record["predictions"]
                if (record["original_user"] != user or record["original_target_item"] != original
                        or record["target"] != target or len(predictions) != 20
                        or len(record["scores"]) != 20 or not all(map(math.isfinite, record["scores"]))):
                    raise ValueError("Test continuation changed a user/target/beam or lost finite scores")
                metrics = per_user_metrics(predictions, target, original, catalog["canonical_to_original_ids"])
                for identity in totals:
                    totals[identity].update(metrics[identity])
                duplicates += metrics["duplicate_beams"]
                stream.write(json.dumps({**record, **metrics}, ensure_ascii=False, allow_nan=False) + "\n")
            if start % (batch_size * 100) == 0:
                print(f"EVALUATE split={split} users={len(seen)}/{count}", flush=True)
            if STOP_REQUESTED:
                raise InterruptedError("Validation/test interrupted; incomplete file is not a result")
    if seen != set(users[:count]):
        raise ValueError("Evaluation omitted users")
    partial.replace(output)
    return {
        "split": split, "users": count, "expected_users": count, "beams_per_user": 20,
        "seen_item_exclusion": False, "duplicate_canonical_beams": duplicates,
        "resumed_frozen_test_chunks": resumed_chunks,
        **{identity: {key: value / count for key, value in values.items()}
           for identity, values in totals.items()},
        "predictions_sha256": sha256(output),
    }


def setup(args):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("P5 uses exactly one explicitly assigned CUDA-visible GPU")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"4", "5"}:
        raise RuntimeError("This P5 allocation is restricted to node0 GPU4/GPU5")
    recipe = load_recipe(args.recipe)
    if args.microbatch_size not in (1, 2, 4, 8, 16, 32, 64):
        raise ValueError("Memory admission may change only a divisor microbatch, preserving effective128")
    runtime = runtime_versions()
    audit = verify_admission(args.corpus, args.recipe)
    native = load_native(args.native)
    interactions, _, _ = read_release(args.upstream, read_json(args.manifest), audit["dataset"])
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.corpus / "tokenizer"), local_files_only=True, use_fast=True,
    )
    catalog = read_json(args.corpus / "catalog.json")
    if digest_json(tokenizer.get_vocab()) != audit["tokenizer_vocab_sha256"]:
        raise ValueError("Saved P5 tokenizer changed")
    model_assets = verify_assets(args.model)
    if sha256(args.model / "asset-receipt.json") != audit["model_receipt_sha256"]:
        raise ValueError("Corpus admitted for a different base model")
    args_ = native_args(audit["dataset"], audit["representation"], len(catalog["token_ids"]))
    users, views = datasets(native, interactions, catalog["original_item_codes"], args_)
    collator = CompleteCollator(
        native, args_, tokenizer, audit["input_bound"], audit["target_bound_including_eos"],
    )
    contract = {
        "recipe_sha256": sha256(args.recipe), "admission_sha256": sha256(args.corpus / "admission.json"),
        "dataset": audit["dataset"], "representation": audit["representation"], "runtime": runtime,
        "model_weights_sha256": model_assets["files_sha256"]["model.safetensors"],
        "driver_sha256": {
            p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("p5_*.py"))
        },
        "gate_subset": args.gate_subset,
        "operating_microbatch_size": args.microbatch_size,
        "operating_accumulation": 128 // args.microbatch_size,
        "workers": 0, "test_once": True, "mode": "gate" if args.gate_subset else "reference",
    }
    if args.gate_subset:
        # Deterministic coverage-spread indices, not the first (usually single-user) rows.
        positions = np.linspace(0, len(views["train"]) - 1, args.gate_subset, dtype=int).tolist()
        from torch.utils.data import Subset
        views["train"] = Subset(views["train"], positions)
    seed_everything(recipe["training"]["seed"])
    model = make_model(native, args.model, tokenizer, recipe["training"]["dropout"]).to("cuda")
    optimizer, scheduler = optimizer_and_scheduler(native, model, len(views["train"]), recipe)
    sampler = IndexSampler(len(views["train"]), recipe["training"]["seed"] + 1000)
    return SimpleNamespace(
        recipe=recipe, native=native, audit=audit, interactions=interactions, tokenizer=tokenizer,
        catalog=catalog, users=users, views=views, collator=collator, contract=contract,
        model=model, optimizer=optimizer, scheduler=scheduler, sampler=sampler,
    )


def train(args):
    context = setup(args)
    recipe, model, optimizer, scheduler, sampler = (
        context.recipe, context.model, context.optimizer, context.scheduler, context.sampler,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    binding = args.output / "p5-run-config.json"
    if binding.exists() and read_json(binding) != context.contract:
        raise ValueError("Run directory already belongs to a different P5 contract")
    write_json(binding, context.contract)
    if not args.gate_subset:
        gate = read_json(args.gate / "gate.json")
        if (gate["status"] != "pass" or gate["recipe_sha256"] != sha256(args.recipe)
                or gate["admission_sha256"] != sha256(args.corpus / "admission.json")
                or gate["driver_sha256"] != context.contract["driver_sha256"]
                or gate["operating_microbatch_size"] != args.microbatch_size):
            raise ValueError("A matching native GPU gate is mandatory before this long fit")
        if not (args.output / "bank-watcher-ready.json").exists():
            raise ValueError("Start and verify the checkpoint bank watcher before training")
        watcher = read_json(args.output / "bank-watcher-ready.json")
        os.kill(watcher["pid"], 0)
        if watcher["status"] != "ready" or watcher["host"] != "node-0":
            raise ValueError("The P5 node0 bank watcher is not ready")
    state = {
        "epoch": 0, "global_step": 0, "phase": "train", "best_metric": None,
        "selected_artifact": None, "validation_history": [], "examples_seen": 0,
        "test_started": False,
    }
    if args.resume is not None:
        state = restore_checkpoint(
            args.resume, model, optimizer, scheduler, sampler, context.contract,
        )
    elif (args.output / "latest.json").exists():
        state = restore_checkpoint(
            pointer_path(args.output), model, optimizer, scheduler, sampler, context.contract,
        )
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    last_save = time.monotonic()
    trace_path = args.output / "updates.jsonl"
    reconcile_trace(trace_path, state["global_step"])
    while state["epoch"] < recipe["training"]["epochs"]:
        if state["phase"] == "train":
            indices = sampler.next(recipe["training"]["effective_batch_size"])
            if indices:
                result = update(
                    model, optimizer, scheduler, context.views["train"], indices,
                    context.collator, args.microbatch_size,
                    recipe["training"]["clip_norm"],
                )
                state["global_step"] += 1
                state["examples_seen"] += len(indices)
                state["last_update"] = result
                import json
                with trace_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"step": state["global_step"], **result}) + "\n")
                if state["global_step"] <= 5 or state["global_step"] % 100 == 0:
                    print(f"P5_UPDATE epoch={state['epoch']} step={state['global_step']} "
                          f"loss={result['loss']:.6f} cursor={sampler.cursor}", flush=True)
                boundary = (
                    state["global_step"] == 1 or STOP_REQUESTED
                    or time.monotonic() - last_save >= recipe["training"]["checkpoint_every_seconds"]
                    or args.stop_steps is not None and state["global_step"] >= args.stop_steps
                )
                if boundary:
                    save_checkpoint(args.output, model, optimizer, scheduler, sampler, state, context.contract)
                    last_save = time.monotonic()
                if STOP_REQUESTED:
                    print("P5_DRAINED_AT_EXACT_OPTIMIZER_BOUNDARY", flush=True)
                    return
                if args.stop_steps is not None and state["global_step"] >= args.stop_steps:
                    return
                continue
            if args.gate_subset:
                sampler.advance()
                state["epoch"] += 1
                continue
            state["phase"] = "validate"
            save_checkpoint(args.output, model, optimizer, scheduler, sampler, state, context.contract)
        if state["phase"] == "validate":
            try:
                metrics = evaluate(
                    model, context.views["valid"], context.users, context.interactions, "valid",
                    context.collator, context.tokenizer, context.catalog, context.native, recipe,
                    args.output / "validation" / f"epoch-{state['epoch'] + 1:02d}.jsonl",
                )
            except InterruptedError:
                print("P5_DRAINED_VALIDATION_WILL_RESTART_FROM_SEALED_STATE", flush=True)
                return
            metric = metrics["primary"]["recall@10"]
            selected = state["best_metric"] is None or metric > state["best_metric"]
            if selected:
                state["best_metric"] = metric
            state["validation_history"].append(metrics)
            state["epoch"] += 1
            state["phase"] = "train" if state["epoch"] < recipe["training"]["epochs"] else "frozen"
            sampler.advance()
            save_checkpoint(
                args.output, model, optimizer, scheduler, sampler, state, context.contract, selected=selected,
            )
            last_save = time.monotonic()
            write_json(args.output / "validation-history.json", state["validation_history"])
    if args.gate_subset:
        return
    selected = pointer_path(args.output, "selected")
    verify_checkpoint(selected, context.contract)
    write_json(args.output / "frozen.json", {
        "status": "frozen-before-test", "selected": selected.name,
        "selected_seal_sha256": sha256(selected / "p5-seal.json"),
        "best_validation_recall10": state["best_metric"], "epochs": state["epoch"],
    })
    from p5_bank import wait_for_bank
    wait_for_bank(args.output, selected, timeout_seconds=3600)
    wait_for_bank(args.output, pointer_path(args.output), timeout_seconds=3600)
    selected_sha = sha256(selected / "p5-seal.json")
    if (args.output / "test-started.json").exists():
        if read_json(args.output / "test-started.json")["selected_seal_sha256"] != selected_sha:
            raise RuntimeError("Test already started for a different frozen model")
        if (args.output / "test-result.json").exists():
            existing = read_json(args.output / "test-result.json")
            if (existing["selected_seal_sha256"] != selected_sha
                    or existing["predictions_sha256"] != sha256(args.output / "test-predictions.jsonl")):
                raise ValueError("Committed P5 test result does not match its frozen model/predictions")
            write_json(args.output / "completed.json", {"status": "complete", "test_evaluations": 1})
            print("P5_TEST_ALREADY_COMMITTED", flush=True)
            return
        print("P5_RESUMING_COMMITTED_TEST_CHUNKS_WITH_SAME_FROZEN_MODEL", flush=True)
    model.load_state_dict(load_file(str(selected / "model.safetensors")), strict=True)
    write_json(args.output / "test-started.json", {
        "selected_seal_sha256": selected_sha, "started_after_bank_ack": True,
    })
    try:
        metrics = evaluate(
            model, context.views["test"], context.users, context.interactions, "test", context.collator,
            context.tokenizer, context.catalog, context.native, recipe, args.output / "test-predictions.jsonl",
        )
    except InterruptedError:
        print("P5_DRAINED_TEST_WILL_CONTINUE_COMMITTED_CHUNKS_WITH_SAME_MODEL", flush=True)
        return
    write_json(args.output / "test-result.json", {
        "status": "complete", "kind": "declared-P5-reference", **metrics,
        "selected_seal_sha256": sha256(selected / "p5-seal.json"),
        "exact_letter_recipe_claim": False,
    })
    write_json(args.output / "completed.json", {"status": "complete", "test_evaluations": 1})


def parser():
    result = argparse.ArgumentParser()
    for name in ("native", "upstream", "manifest", "recipe", "model", "corpus", "output"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--gate", type=Path)
    result.add_argument("--resume", type=Path)
    result.add_argument("--gate-subset", type=int, default=0)
    result.add_argument("--stop-steps", type=int)
    result.add_argument("--microbatch-size", type=int, default=64, choices=(1, 2, 4, 8, 16, 32, 64))
    return result


if __name__ == "__main__":
    train(parser().parse_args())
