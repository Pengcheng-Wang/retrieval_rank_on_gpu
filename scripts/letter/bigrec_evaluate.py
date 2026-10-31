"""Native top-one title query and frozen-base full-catalog L2 grounding."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import signal
import time
import uuid
from collections import Counter
from contextlib import contextmanager, nullcontext
from pathlib import Path

from bigrec_bank import require_ack, seal_payload, stage_payload, verify_payload
from bigrec_data import digest_json, read_rows, verify_admission, verify_user_rows, write_rows
from bigrec_source import load_tokenizer, verify_model
from protocol import bind_run_config, file_sha256, write_json


def last_token_pool(hidden, attention_mask):
    import torch

    if (hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]
            or not torch.all(attention_mask[:, -1] == 1)
            or torch.any(attention_mask[:, 1:] < attention_mask[:, :-1])):
        raise ValueError("Final-token pooling requires nonempty left-padded sequences")
    return hidden[:, -1, :]


def rank_l2(queries, catalog):
    import torch

    if (queries.ndim != 2 or catalog.ndim != 2 or queries.shape[1] != catalog.shape[1]
            or not len(catalog) or not torch.isfinite(queries).all() or not torch.isfinite(catalog).all()):
        raise ValueError("Full-catalog L2 requires finite compatible embeddings")
    distances = torch.cdist(queries.float(), catalog.float(), p=2,
                            compute_mode="donot_use_mm_for_euclid_dist")
    if not torch.isfinite(distances).all():
        raise ValueError("Full-catalog L2 distances are not finite")
    return torch.argsort(distances, dim=1, stable=True), distances


def target_metrics(ranking: list[int], target: int) -> dict:
    from protocol import rank_metrics

    return rank_metrics(ranking, target, ks=(1, 5, 10))


def title_diagnostics(ranking: list[int], target: int, titles: list[str], counts=None) -> dict:
    title = titles[target]
    first = next(rank for rank, item in enumerate(ranking) if titles[item] == title)
    metrics = {}
    for k in (1, 5, 10):
        metrics[f"recall@{k}"] = float(first < k)
        metrics[f"ndcg@{k}"] = 1 / math.log2(first + 2) if first < k else 0.0
    return {
        "target_title_group_size": counts[title] if counts is not None else titles.count(title),
        "best_same_title_rank_zero_based": first, "catalog_title_equivalence_metrics": metrics,
    }


def grounding_query(text: str) -> str:
    return text.strip('"')


def decode_tail(sequence: list[int], prompt_width: int, tokenizer, max_new_tokens: int) -> dict:
    tail = list(sequence[prompt_width:])
    if tokenizer.eos_token_id in tail:
        tail = tail[:tail.index(tokenizer.eos_token_id) + 1]
        finish = "eos"
    elif len(tail) == max_new_tokens:
        finish = "length"
    else:
        raise ValueError("Generated sequence has an unexplained finish without EOS or token budget")
    if not 0 < len(tail) <= max_new_tokens:
        raise ValueError("Generated sequence exceeds its frozen token budget")
    return {
        "text": tokenizer.decode(tail, skip_special_tokens=True),
        "token_ids": tail, "generated_tokens": len(tail), "finish_reason": finish,
    }


def load_frozen_base(model_path: Path):
    import torch
    from transformers import LlamaForCausalLM

    model = LlamaForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, use_safetensors=True,
        torch_dtype=torch.float16, device_map={"": 0},
    ).eval()
    model.requires_grad_(False)
    model.config.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    return model


def encode_title_batch(model, tokenizer, titles: list[str]):
    import torch

    if any("lora_" in name for name, _ in model.named_parameters()):
        raise ValueError("Grounding must use the frozen unadapted backbone, never a LoRA model")
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Grounding backbone must be frozen and in evaluation mode")
    inputs = tokenizer(titles, return_tensors="pt", padding=True, truncation=False)
    if inputs["input_ids"].shape[1] > model.config.max_position_embeddings:
        raise ValueError("A full grounding title exceeds the base context; do not truncate it")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        output = model(**inputs, output_hidden_states=True, use_cache=False)
        pooled = last_token_pool(output.hidden_states[-1], inputs["attention_mask"]).detach().cpu()
    if not torch.isfinite(pooled).all():
        raise ValueError("Frozen-base title embeddings must be finite")
    return pooled


def compact_generation_output(module, inputs, output):
    if getattr(output, "loss", None) is not None:
        raise ValueError("Output compaction is only valid during label-free generation")
    if output.logits.shape[1] > 1:
        output.logits = output.logits[:, -1:, :].clone()
    return output


def reorder_generation_cache(past_key_values, beam_indices):
    import torch

    if torch.is_grad_enabled():
        raise ValueError("In-place beam cache permutation is only valid for generation without gradients")
    for layer in past_key_values:
        for value in layer:
            value.copy_(value.index_select(0, beam_indices.to(value.device)))
    return past_key_values


@contextmanager
def generation_memory_scope(model):
    if model.training:
        raise ValueError("Generation output lifetimes require an evaluation-mode model")
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    previous = {}

    def release_previous(module, inputs):
        output = previous.pop("output", None)
        if output is not None:
            # ModelOutput's attribute setter retains dict values when assigned None.
            if isinstance(output, dict):
                output["past_key_values"] = None
            else:
                output.past_key_values = None

    def retain_only_needed_output(module, inputs, output):
        output = compact_generation_output(module, inputs, output)
        previous["output"] = output
        return output

    before = base.register_forward_pre_hook(release_previous)
    after = base.register_forward_hook(retain_only_needed_output)
    has_reorder = hasattr(base, "_reorder_cache")
    own_reorder = base.__dict__.get("_reorder_cache")
    if has_reorder:
        base._reorder_cache = reorder_generation_cache
    try:
        yield
    finally:
        before.remove()
        after.remove()
        if has_reorder:
            if own_reorder is None:
                del base._reorder_cache
            else:
                base._reorder_cache = own_reorder
        previous.clear()


def generate_batch(
    model, tokenizer, prompts: list[list[int]], recipe: dict, manage_output_memory: bool = True,
) -> list[list[dict]]:
    import torch

    generation = recipe["generation"]
    inputs = tokenizer.pad({
        "input_ids": prompts, "attention_mask": [[1] * len(ids) for ids in prompts],
    }, padding=True, return_tensors="pt")
    width = inputs["input_ids"].shape[1]
    if width + generation["max_new_tokens"] > recipe["data"]["context_tokens"]:
        raise ValueError("Generation cannot exceed the frozen base context")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    memory_scope = generation_memory_scope(model) if manage_output_memory else nullcontext()
    with torch.inference_mode(), memory_scope:
        output = model.generate(
            **inputs, do_sample=False, num_beams=generation["num_beams"],
            num_return_sequences=generation["num_return_sequences"],
            max_new_tokens=generation["max_new_tokens"],
            length_penalty=generation["length_penalty"], early_stopping=generation["early_stopping"],
            return_dict_in_generate=True, output_scores=True,
            pad_token_id=0, bos_token_id=1, eos_token_id=2, use_cache=True,
        )
    expected = len(prompts) * generation["num_return_sequences"]
    if len(output.sequences) != expected or output.sequences_scores is None:
        raise ValueError("Native four-beam generation returned the wrong number of candidates")
    result = []
    for i, sequence in enumerate(output.sequences.detach().cpu().tolist()):
        row = decode_tail(sequence, width, tokenizer, generation["max_new_tokens"])
        row["sequence_score"] = float(output.sequences_scores[i].detach().cpu())
        if not math.isfinite(row["sequence_score"]):
            raise ValueError("A generated beam has a non-finite score")
        result.append(row)
    return [result[i:i + 4] for i in range(0, len(result), 4)]


def verify_generation_shard(path: Path, binding: dict, start: int, users: list[str]) -> list[dict]:
    seal = verify_payload(path)
    if (seal["kind"] != "bigrec_four_beam_generated_titles"
            or seal["binding"] != {**binding, "start": start, "users": users}):
        raise ValueError("Generation shard binding differs from the selected model and fixed user inventory")
    rows = read_rows(path / "rows.jsonl")
    verify_user_rows(users, rows)
    for row in rows:
        if len(row["beams"]) != 4:
            raise ValueError("A saved user is missing native generated beams")
        for beam in row["beams"]:
            ids = beam["token_ids"]
            finish = beam["finish_reason"]
            valid = (finish == "eos" and ids and ids[-1] == 2 and 2 not in ids[:-1])
            valid = valid or (finish == "length" and len(ids) == 128 and 2 not in ids)
            if (not valid or not 0 < len(ids) <= 128 or beam["generated_tokens"] != len(ids)
                    or not isinstance(beam["text"], str) or not math.isfinite(beam["sequence_score"])):
                raise ValueError("A saved beam lacks valid token-observed finish metadata")
    return rows


def check_selection(admission: Path, run: Path, outbox: Path) -> tuple[dict, dict]:
    report = verify_admission(admission)
    selected = run / "selected"
    receipt = verify_payload(selected)
    if (receipt["kind"] != "bigrec_selected_adapter"
            or receipt["binding"]["admission_sha256"] != file_sha256(admission / "admission.json")
            or receipt["binding"]["recipe_sha256"] != report["recipe_sha256"]
            or receipt["binding"]["model"] != report["model"]):
        raise ValueError("Selected model and admitted data/recipe do not match")
    require_ack(outbox, f"{run.name}-selected", selected)
    selection = json.loads((selected / "selection.json").read_text(encoding="utf-8"))
    from bigrec_train import source_identity
    if selection["source"] != source_identity():
        raise ValueError("Execution implementation changed after the model was frozen")
    return report, receipt


def start_evaluation(args) -> tuple[dict, dict, list[dict], object, dict]:
    report, selection = check_selection(args.admission, args.run, args.outbox)
    model = verify_model(args.model)
    if model != report["model"]:
        raise ValueError("Generation/grounding backbone differs from admission")
    recipe = json.loads((args.admission / "recipe.json").read_text(encoding="utf-8"))
    rows = read_rows(args.admission / "test.jsonl")
    if len(rows) != report["splits"]["test"]["records"]:
        raise ValueError("All fixed test users are required")
    from bigrec_train import deterministic_seed, source_identity
    deterministic_seed()
    binding = {
        "track": recipe["track"], "dataset": report["dataset"],
        "admission_sha256": file_sha256(args.admission / "admission.json"),
        "selected_sha256": file_sha256(args.run / "selected/complete.json"),
        "recipe_sha256": report["recipe_sha256"], "model": model, "source": source_identity(),
        "test_users": len(rows), "catalog_items": report["catalog"]["items"],
    }
    bind_run_config(args.output, binding)
    return report, recipe, rows, load_tokenizer(args.model), binding


def generation(args) -> None:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    report, recipe, rows, tokenizer, binding = start_evaluation(args)
    import torch
    from peft import PeftModel

    base = load_frozen_base(args.model)
    selected = args.run / "selected"
    if list(selected.glob("*.bin")):
        raise ValueError("Only safe adapter tensors may be used for generation")
    model = PeftModel.from_pretrained(
        base, str(selected), is_trainable=False, torch_dtype=torch.float16, device_map={"": 0},
    ).half().eval()
    model.requires_grad_(False)
    draining = {"requested": False}
    signal.signal(signal.SIGTERM, lambda sig, frame: draining.update(requested=True))
    destination = args.output / "generation"
    destination.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(rows), 64):
        subset = rows[start:start + 64]
        target = destination / f"chunk-{start:08d}"
        chunk_binding = {**binding, "start": start, "users": [r["user"] for r in subset]}
        if target.exists():
            verify_generation_shard(target, binding, start, chunk_binding["users"])
        else:
            pending = destination / f".partial-{start:08d}-{uuid.uuid4().hex}"
            pending.mkdir()
            result = []
            for offset in range(0, len(subset), recipe["generation"]["batch_size"]):
                batch = subset[offset:offset + recipe["generation"]["batch_size"]]
                beams = generate_batch(model, tokenizer, [r["prompt_ids"] for r in batch], recipe)
                result.extend({
                    "user": r["user"], "target": r["target"], "beams": b,
                    "prompt_sha256": digest_json(r["prompt_ids"]), "status": "generated",
                } for r, b in zip(batch, beams))
                write_rows(pending / "rows.jsonl", result)
            verify_user_rows(chunk_binding["users"], result)
            seal_payload(pending, "bigrec_four_beam_generated_titles", chunk_binding)
            pending.rename(target)
        request = stage_payload(target, args.outbox, f"{args.run.name}-generation-{start:08d}")
        write_json(args.output / "generation-progress.json", {
            "users": min(start + 64, len(rows)), "expected_users": len(rows),
            "staged": request["identity"], "unix": time.time(),
        })
        print(f"BIGREC_GENERATED {report['dataset']} {min(start + 64, len(rows))}/{len(rows)}", flush=True)
        if draining["requested"]:
            return
    write_json(args.output / "generation-completed.json", {
        "status": "complete", "users": len(rows), "beams_per_user": 4,
        "selected_sha256": binding["selected_sha256"], "binding": binding,
    })
    del model, base
    gc.collect()
    torch.cuda.empty_cache()


def catalog_embeddings(args, report, recipe, tokenizer, model):
    import torch
    from safetensors.torch import load_file, save_file

    rows = read_rows(args.admission / "catalog.jsonl")
    titles = [r["title"] for r in rows]
    unique = list(dict.fromkeys(titles))
    title_to_unique = {t: i for i, t in enumerate(unique)}
    mapping = [title_to_unique[t] for t in titles]
    identity = {
        "catalog_sha256": file_sha256(args.admission / "catalog.jsonl"),
        "model": report["model"], "pooling": recipe["grounding"]["pooling"],
        "encoder_source_sha256": file_sha256(Path(__file__)),
        "dtype": "float16", "bos": 1, "automatic_eos": False,
        "items": len(rows), "unique_titles": len(unique),
    }
    final = args.catalog / "catalog"
    if final.exists():
        if verify_payload(final)["binding"] != identity:
            raise ValueError("Catalog embeddings came from a different backbone/tokenizer/pooling/catalog")
        return load_file(str(final / "catalog.safetensors"), device="cpu")["embeddings"]
    parts = args.catalog / "unique-parts"
    parts.mkdir(parents=True, exist_ok=True)
    vectors = []
    for start in range(0, len(unique), 256):
        target = parts / f"chunk-{start:08d}"
        part_identity = {**identity, "start": start, "count": min(256, len(unique) - start)}
        if target.exists():
            if verify_payload(target)["binding"] != part_identity:
                raise ValueError("Unique-title embedding shard binding changed")
        else:
            values = []
            for offset in range(start, min(start + 256, len(unique)), recipe["grounding"]["embedding_batch_size"]):
                values.append(encode_title_batch(model, tokenizer, unique[offset:min(offset + 16, start + 256)]))
            pending = parts / f".partial-{start:08d}-{uuid.uuid4().hex}"
            pending.mkdir()
            save_file({"embeddings": torch.cat(values).contiguous()}, str(pending / "embeddings.safetensors"))
            seal_payload(pending, "bigrec_frozen_unique_title_embeddings", part_identity)
            pending.rename(target)
        vectors.append(load_file(str(target / "embeddings.safetensors"), device="cpu")["embeddings"])
        stage_payload(target, args.outbox, f"bigrec-{report['dataset']}-catalog-unique-{start:08d}")
    unique_embeddings = torch.cat(vectors)
    embeddings = unique_embeddings[torch.tensor(mapping, dtype=torch.long)].contiguous()
    if embeddings.shape != (len(rows), model.config.hidden_size) or not torch.isfinite(embeddings).all():
        raise ValueError("Full-catalog embedding inventory is incomplete")
    pending = args.catalog / f".partial-catalog-{uuid.uuid4().hex}"
    pending.mkdir()
    save_file({"embeddings": embeddings}, str(pending / "catalog.safetensors"))
    write_json(pending / "layout.json", {"original_item_to_unique_title": mapping, "original_ids": list(range(len(rows)))})
    seal_payload(pending, "bigrec_frozen_full_catalog_embeddings", identity)
    pending.rename(final)
    stage_payload(final, args.outbox, f"bigrec-{report['dataset']}-catalog")
    return embeddings


def grounding(args) -> None:
    report, recipe, rows, tokenizer, binding = start_evaluation(args)
    if not (args.output / "generation-completed.json").is_file():
        raise ValueError("Grounding requires completed generation for every fixed test user")
    generation_receipt = json.loads((args.output / "generation-completed.json").read_text(encoding="utf-8"))
    if generation_receipt["binding"] != binding or generation_receipt["users"] != len(rows):
        raise ValueError("Generation identity/user coverage differs")
    import torch
    from safetensors.torch import save_file

    model = load_frozen_base(args.model)
    catalog = catalog_embeddings(args, report, recipe, tokenizer, model).to(model.device)
    catalog_titles = [r["title"] for r in read_rows(args.admission / "catalog.jsonl")]
    title_counts = Counter(catalog_titles)
    destination = args.output / "grounding"
    destination.mkdir(parents=True, exist_ok=True)
    all_rows, finish_all, finish_top = [], Counter(), Counter()
    draining = {"requested": False}
    signal.signal(signal.SIGTERM, lambda sig, frame: draining.update(requested=True))
    for start in range(0, len(rows), 64):
        generation_path = args.output / "generation" / f"chunk-{start:08d}"
        expected = rows[start:start + 64]
        generated = verify_generation_shard(generation_path, binding, start, [r["user"] for r in expected])
        for truth, prediction in zip(expected, generated):
            if (truth["user"] != prediction["user"] or truth["target"] != prediction["target"]
                    or prediction["prompt_sha256"] != digest_json(truth["prompt_ids"]) or len(prediction["beams"]) != 4):
                raise ValueError("Generated rows changed target, order, prompt or native beam budget")
            finish_top.update([prediction["beams"][0]["finish_reason"]])
            finish_all.update(b["finish_reason"] for b in prediction["beams"])
        target = destination / f"chunk-{start:08d}"
        part_binding = {
            **binding, "start": start, "generation_sha256": file_sha256(generation_path / "complete.json"),
            "catalog_sha256": file_sha256(args.catalog / "catalog/complete.json"),
        }
        if target.exists():
            if verify_payload(target)["binding"] != part_binding:
                raise ValueError("Grounding shard changed its frozen inputs")
            scored = read_rows(target / "rows.jsonl")
        else:
            texts = [grounding_query(r["beams"][0]["text"]) for r in generated]
            embeddings = torch.cat([
                encode_title_batch(model, tokenizer, texts[i:i + 16]) for i in range(0, len(texts), 16)
            ])
            ranking, distances = rank_l2(embeddings.to(model.device), catalog)
            ranking = ranking.cpu().tolist()
            scored = []
            for prediction, ranked in zip(generated, ranking):
                original_target = prediction["target"]
                scored.append({
                    "user": prediction["user"], "target": original_target,
                    "target_rank_zero_based": ranked.index(original_target),
                    "top20_original_items": ranked[:20], "metrics": target_metrics(ranked, original_target),
                    "generation_finish_reason": prediction["beams"][0]["finish_reason"],
                    "literal_generated_title_match": grounding_query(prediction["beams"][0]["text"])
                                                     == catalog_titles[original_target],
                    "title_diagnostics": title_diagnostics(ranked, original_target, catalog_titles, title_counts),
                })
            pending = destination / f".partial-{start:08d}-{uuid.uuid4().hex}"
            pending.mkdir()
            save_file({"embeddings": embeddings.contiguous()}, str(pending / "query-embeddings.safetensors"))
            write_rows(pending / "rows.jsonl", scored)
            seal_payload(pending, "bigrec_full_catalog_grounding", part_binding)
            pending.rename(target)
        verify_user_rows([r["user"] for r in expected], scored)
        all_rows.extend(scored)
        stage_payload(target, args.outbox, f"{args.run.name}-grounding-{start:08d}")
        write_json(args.output / "grounding-progress.json", {
            "users": len(all_rows), "expected_users": len(rows), "unix": time.time(),
        })
        print(f"BIGREC_GROUNDED {report['dataset']} {len(all_rows)}/{len(rows)}", flush=True)
        if draining["requested"]:
            return
    verify_user_rows([r["user"] for r in rows], all_rows)
    metrics = {key: sum(r["metrics"][key] for r in all_rows) / len(all_rows) for key in all_rows[0]["metrics"]}
    title_metrics = {
        key: sum(r["title_diagnostics"]["catalog_title_equivalence_metrics"][key] for r in all_rows) / len(all_rows)
        for key in metrics
    }
    strata = {}
    for name, ambiguous in (("ambiguous_title_targets", True), ("unique_title_targets", False)):
        subset = [r for r in all_rows if (r["title_diagnostics"]["target_title_group_size"] > 1) == ambiguous]
        strata[name] = {
            "users": len(subset),
            "original_item_metrics": {
                key: sum(r["metrics"][key] for r in subset) / len(subset) for key in metrics
            } if subset else {},
        }
    summary = args.output / "summary"
    summary.mkdir(exist_ok=True)
    write_json(summary / "result.json", {
        "status": "complete", "track": recipe["track"], "dataset": report["dataset"],
        "exact_original_weights_reproduction": False, "users": len(rows),
        "catalog_items": len(catalog), "metrics": metrics,
        "metric_identity": "original_LETTER_item_id",
        "title_diagnostics_not_item_metrics": {
            "catalog_title_equivalence_metrics": title_metrics,
            "literal_generated_title_match_rate": sum(r["literal_generated_title_match"] for r in all_rows) / len(all_rows),
            "original_item_metrics_by_ambiguity_stratum": strata,
        },
        "primary_finish_reasons": dict(finish_top), "all_beam_finish_reasons": dict(finish_all),
        "failed_or_length_filtered_users": 0, "seen_item_exclusion": False,
        "generation_beams": 4, "grounding_queries_per_user": 1,
        "distance": recipe["grounding"]["distance"], "binding": binding,
    })
    seal_payload(summary, "bigrec_aggregate_evaluation", binding)
    request = stage_payload(summary, args.outbox, f"{args.run.name}-evaluation")
    write_json(args.output / "completed.json", {
        "status": "complete", "users": len(rows), "summary_sha256": file_sha256(summary / "complete.json"),
        "staged": request["identity"],
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("generate", "ground"))
    for name in ("admission", "model", "run", "output", "outbox", "catalog"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    for name in ("admission", "model", "run", "output", "outbox", "catalog"):
        setattr(args, name, getattr(args, name).resolve())
    try:
        generation(args) if args.command == "generate" else grounding(args)
    except Exception as error:
        write_json(args.output / f"{args.command}-failed.json", {
            "status": "fail", "type": type(error).__name__, "error": str(error), "unix": time.time(),
            "policy": "No beam reduction, filtered users, substituted titles or partial-denominator score",
        })
        raise


if __name__ == "__main__":
    main()
