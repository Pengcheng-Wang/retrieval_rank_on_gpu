"""All-record title/length admission without held-out-target-dependent prompts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from collections import Counter
from pathlib import Path

from bigrec_source import SOURCE_HASHES, load_pure_renderer, load_tokenizer, verify_model, verify_native
from prepare import read_release
from protocol import file_sha256, require_hash, sequence_records, verify_coverage, write_json


DOMAINS = {
    "Instruments": ("musical products", "musical product", "bought"),
    "Beauty": ("beauty products", "beauty product", "bought"),
    "Yelp": ("businesses", "business", "visited"),
}


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def catalog_rows(items: dict) -> list[dict]:
    if set(items) != set(map(str, range(len(items)))):
        raise ValueError("The catalog must have contiguous original item IDs 0..N-1")
    rows = []
    for item in range(len(items)):
        title = items[str(item)].get("title")
        if not isinstance(title, str):
            raise ValueError("Every original item needs its released string title, including empty strings")
        rows.append({"item": item, "title": title})
    return rows


def render_fields(dataset: str, history: list[int], titles: list[str], target: int | None) -> dict:
    plural, singular, verb = DOMAINS[dataset]
    return {
        "instruction": f"Given a list of {plural} the user has {verb} before, please recommend a new "
                       f"{singular} that the user likes to the user.",
        "input": f"The user has {verb} the following {plural} before:"
                 + ", ".join('"' + titles[i] + '"' for i in history) + "\n ",
        "output": "" if target is None else '"' + titles[target] + '"',
    }


def token_ids(tokenizer, text: str) -> list[int]:
    return tokenizer(text, truncation=False, padding=False, add_special_tokens=True)["input_ids"]


def catalog_reserve(dataset, titles, renderer, tokenizer, context: int, generation_tokens: int) -> dict:
    prefix = renderer(render_fields(dataset, [], titles, None))
    prefix_ids = token_ids(tokenizer, prefix)
    suffix_lengths, title_lengths = [], []
    for item, title in enumerate(titles):
        full = token_ids(tokenizer, renderer(render_fields(dataset, [], titles, item)))
        if full[:len(prefix_ids)] != prefix_ids:
            raise ValueError("Native SentencePiece catalog target/prompt boundary does not match")
        suffix_lengths.append(len(full) - len(prefix_ids) + 1)
        title_lengths.append(len(token_ids(tokenizer, title)))
    reserve = max(generation_tokens, max(suffix_lengths))
    if len(prefix_ids) + reserve > context or max(title_lengths) > context:
        raise ValueError("A catalog target cannot fit the native model context even without history")
    return {
        "answer_reserve_tokens": reserve, "prompt_limit": context - reserve,
        "empty_history_prompt_tokens": len(prefix_ids),
        "maximum_quoted_target_tokens_including_eos": max(suffix_lengths),
        "maximum_grounding_title_tokens": max(title_lengths),
        "quoted_targets_longer_than_generation_budget": sum(n - 1 > generation_tokens for n in suffix_lengths),
        "uses_heldout_targets": False,
    }


def tokenize_record(record, dataset, titles, renderer, tokenizer, context: int, reserve: int) -> dict:
    history = list(record["history"])
    raw = token_ids(tokenizer, renderer(render_fields(dataset, history, titles, record["target"])))
    while True:
        prompt = renderer(render_fields(dataset, history, titles, None))
        prompt_ids = token_ids(tokenizer, prompt)
        if len(prompt_ids) <= context - reserve:
            break
        if not history:
            raise ValueError("Static native prompt cannot fit the declared context/reserve")
        history.pop(0)
    full_prompt = renderer(render_fields(dataset, history, titles, record["target"]))
    if not full_prompt.startswith(prompt) or full_prompt[len(prompt):] != '"' + titles[record["target"]] + '"':
        raise ValueError("Canonical renderer did not preserve the complete quoted target")
    full = raw if history == record["history"] else token_ids(tokenizer, full_prompt)
    if full[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Canonical training/generation token prefixes differ")
    if not full or full[0] != tokenizer.bos_token_id:
        raise ValueError("Native prompt lost BOS")
    if len(full) - len(prompt_ids) + 1 > reserve:
        raise ValueError("Catalog-only reserve does not cover a complete target and EOS")
    full = full + [tokenizer.eos_token_id]
    if len(full) > context:
        raise ValueError("Complete quoted target and EOS exceed the model context")
    return {
        **record, "history": history, "prompt_ids": prompt_ids, "input_ids": full,
        "attention_mask": [1] * len(full), "labels": list(full),
        "raw_tokens_without_eos": len(raw),
        "dropped_history_items": len(record["history"]) - len(history),
        "finish_reason": "N/A",
    }


def native_cutoff_diagnostic(lengths, cutoff: int) -> dict:
    return {
        "records": len(lengths), "missing_eos": sum(n >= cutoff for n in lengths),
        "right_truncated": sum(n > cutoff for n in lengths),
    }


def training_selection(count: int, sample: int, seed: int) -> list[int]:
    from datasets import Dataset

    if not 0 < sample <= count:
        raise ValueError("Invalid native few-shot training sample")
    return list(Dataset.from_dict({"row": list(range(count))}).shuffle(seed=seed)
                .select(range(sample)).shuffle(seed=seed)["row"])


def validation_selection(count: int, sample: int, seed: int) -> list[int]:
    import pandas as pd

    if not 0 < sample <= count:
        raise ValueError("Invalid fixed validation sample")
    return pd.Series(range(count)).sample(n=sample, random_state=seed).tolist()


def verify_user_rows(expected: list[str], rows: list[dict]) -> None:
    verify_coverage(expected, [r["user"] for r in rows])


def write_rows(path: Path, rows) -> None:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".part")
    with pending.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def verify_admission(path: Path) -> dict:
    report = json.loads((path / "admission.json").read_text(encoding="utf-8"))
    if report["status"] != "pass":
        raise ValueError("Full-corpus BIGRec admission did not pass")
    for name, digest in report["files"].items():
        require_hash(path / name, digest)
    for name in ("bigrec_source.py", "bigrec_data.py"):
        require_hash(Path(__file__).parent / name, report["format_source_sha256"][name])
    recipe = json.loads((path / "recipe.json").read_text(encoding="utf-8"))
    if report["recipe_sha256"] != file_sha256(path / "recipe.json"):
        raise ValueError("Admission recipe binding differs")
    if (report["selected_train"]["records"] != recipe["training"]["sample"]
            or report["selected_validation"]["records"] != recipe["training"]["validation_sample"]
            or any(s["strict_clean"] != s["records"] for s in report["splits"].values())):
        raise ValueError("Incomplete full-corpus or selection admission")
    return report


def audit(args) -> dict:
    import numpy as np

    start = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "admission.json").exists():
        raise FileExistsError("Admission receipts are immutable; choose a new directory")
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    native = verify_native(args.native)
    model = verify_model(args.model)
    interactions, items, _ = read_release(args.upstream, manifest, args.dataset)
    if any(not str(user).isdecimal() for user in interactions):
        raise ValueError("Expected numeric original user IDs")
    interactions = dict(sorted(interactions.items(), key=lambda p: int(p[0])))
    catalog = catalog_rows(items)
    titles = [r["title"] for r in catalog]
    tokenizer = load_tokenizer(args.model)
    renderer = load_pure_renderer(args.native / "train.py", SOURCE_HASHES["train.py"])
    native_infer = load_pure_renderer(args.native / "inference.py", SOURCE_HASHES["inference.py"])
    context = recipe["data"]["context_tokens"]
    reserve = catalog_reserve(args.dataset, titles, renderer, tokenizer, context,
                              recipe["generation"]["max_new_tokens"])
    expected = manifest["datasets"][args.dataset]
    report = {
        "status": "checking", "dataset": args.dataset, "track": recipe["track"],
        "exact_original_weights_reproduction": False,
        "inputs": expected["sha256"], "release_manifest_sha256": file_sha256(args.manifest),
        "native_source": native, "model": model, "context": context, "reserve": reserve,
        "format_source_sha256": {
            n: file_sha256(Path(__file__).parent / n) for n in ("bigrec_source.py", "bigrec_data.py")
        },
        "catalog": {
            "items": len(catalog), "empty_titles": sum(not t for t in titles),
            "distinct_titles": len(set(titles)), "duplicate_title_excess": len(titles) - len(set(titles)),
            "maximum_title_multiplicity": max(Counter(titles).values()),
            "original_ids_preserved": True,
        },
        "finish_reason": "N/A", "splits": {},
        "versions": {n: importlib.metadata.version(n) for n in
                     ("transformers", "sentencepiece", "datasets", "numpy", "pandas")},
    }
    selected = {}
    train_choice = training_selection(expected["training_prefix_examples"],
                                      recipe["training"]["sample"], recipe["training"]["selection_seed"])
    valid_choice = validation_selection(expected["validation_users"],
                                         recipe["training"]["validation_sample"], recipe["training"]["validation_seed"])
    for split in ("train", "valid", "test"):
        records = sequence_records(interactions, split, recipe["data"]["history_items"])
        wanted = set(train_choice if split == "train" else valid_choice) if split != "test" else set(range(len(records)))
        rows = {}
        lengths, retained_lengths, prompt_lengths, drops = [], [], [], []
        raw_full_digest = hashlib.sha256()
        first = None
        for index, record in enumerate(records):
            row = tokenize_record(record, args.dataset, titles, renderer, tokenizer,
                                  context, reserve["answer_reserve_tokens"])
            lengths.append(row["raw_tokens_without_eos"])
            retained_lengths.append(len(row["input_ids"]))
            prompt_lengths.append(len(row["prompt_ids"]))
            drops.append(row["dropped_history_items"])
            raw_full_digest.update(digest_json({
                "user": row["user"], "position": row["position"],
                "input_ids": row["input_ids"], "target": row["target"],
            }).encode())
            if index in wanted:
                rows[index] = row if split != "test" else {
                    k: row[k] for k in ("user", "position", "history", "target", "prompt_ids", "dropped_history_items")
                }
            if first is None:
                fields = render_fields(args.dataset, row["history"], titles, None)
                native_ids = token_ids(tokenizer, native_infer(fields["instruction"], fields["input"]))
                first = {
                    "canonical_prefix_token_sha256": digest_json(row["prompt_ids"]),
                    "native_inference_token_sha256": digest_json(native_ids),
                    "native_inference_differs": native_ids != row["prompt_ids"],
                    "canonical_train_eval_prefix_equal": True,
                }
            if index and index % 25000 == 0:
                print(f"BIGREC_AUDIT_PROGRESS {args.dataset} {split} {index}/{len(records)}", flush=True)
        stats = {
            "records": len(records), "strict_clean": len(records),
            "raw_tokens_without_eos_p50_p90_p99_max": np.quantile(lengths, [.5, .9, .99, 1]).tolist(),
            "native_512": native_cutoff_diagnostic(lengths, 512),
            "raw_over_2048_with_eos": sum(n + 1 > context for n in lengths),
            "final_tokens_p50_p90_p99_max": np.quantile(retained_lengths, [.5, .9, .99, 1]).tolist(),
            "prompt_tokens_p50_p90_p99_max": np.quantile(prompt_lengths, [.5, .9, .99, 1]).tolist(),
            "shortened_records": sum(n > 0 for n in drops),
            "whole_history_items_removed": sum(drops), "maximum_removed": max(drops),
            "missing_target_or_eos": 0, "prefix_mismatches": 0,
            "post_tokenization_stream_sha256": raw_full_digest.hexdigest(), "prefix_probe": first,
        }
        report["splits"][split] = stats
        order = train_choice if split == "train" else valid_choice if split == "valid" else list(range(len(records)))
        selected[split] = [rows[i] for i in order]
        if split == "test":
            verify_user_rows(list(interactions), selected[split])
        print(f"BIGREC_AUDITED {args.dataset} {split} {len(records)} strict-clean", flush=True)
    if report["splits"]["train"]["records"] != expected["training_prefix_examples"]:
        raise ValueError("Training-prefix count differs from the frozen LETTER release")
    for key, split, choice in (("selected_train", "train", train_choice),
                                ("selected_validation", "valid", valid_choice)):
        report[key] = {
            "records": len(selected[split]), "selection_indices_sha256": digest_json(choice),
            "maximum_tokens": max(len(r["input_ids"]) for r in selected[split]),
            "shortened_records": sum(r["dropped_history_items"] > 0 for r in selected[split]),
            "complete_quoted_target_and_eos": True,
        }
    write_rows(args.output / "train.jsonl", selected["train"])
    write_rows(args.output / "validation.jsonl", selected["valid"])
    write_rows(args.output / "test.jsonl", selected["test"])
    write_rows(args.output / "catalog.jsonl", catalog)
    write_json(args.output / "selection.json", {"train": train_choice, "validation": valid_choice})
    (args.output / "recipe.json").write_bytes(args.recipe.read_bytes())
    write_json(args.output / "source-model.json", {"source": native, "model": model})
    report.update(
        status="pass", elapsed_seconds=time.monotonic() - start,
        recipe_sha256=file_sha256(args.output / "recipe.json"),
        files={p.name: file_sha256(p) for p in args.output.iterdir() if p.is_file()},
    )
    write_json(args.output / "admission.json", report)
    verify_admission(args.output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("upstream", "manifest", "native", "model", "recipe", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(DOMAINS), required=True)
    args = parser.parse_args()
    try:
        report = audit(args)
        print(json.dumps({"status": report["status"], "dataset": report["dataset"],
                          "admission_sha256": file_sha256(args.output / "admission.json")}))
    except Exception as error:
        write_json(args.output / "failed.json", {
            "status": "fail", "dataset": args.dataset, "type": type(error).__name__, "error": str(error),
        })
        raise


if __name__ == "__main__":
    main()
