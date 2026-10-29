"""Exact LETTER split views, native P5 rendering/collation, and exhaustive admission."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from p5_assets import verify_assets
from p5_index import MetadataUnavailable, identifiers, validate_sequences
from p5_source import SOURCE_HASHES, digest_json, load_native, load_recipe, native_args
from p5_source import read_json, require_hash, runtime_versions, sha256, write_json
from prepare import read_release


class IndexedUsers(list):
    def __init__(self, values):
        super().__init__(values)
        self.positions = {value: i for i, value in enumerate(values)}
        if len(self.positions) != len(self):
            raise ValueError("Duplicate user in native P5 dataset")

    def index(self, value, *args):
        return super().index(value, *args) if args else self.positions[value]


def datasets(native, interactions, codes, args):
    users = IndexedUsers(validate_sequences(interactions, len(codes)))
    train = [[codes[i] for i in interactions[u][:-2]] for u in users]
    valid = [[codes[i] for i in interactions[u][:-1]] for u in users]
    test = [[codes[i] for i in interactions[u]] for u in users]
    templates = native.prompt.task_subgroup_1
    if len(templates) != 10:
        raise ValueError("Native sequential prompt family changed")

    class CachedNativeTrain(native.data.sequential_item_dataset):
        def __len__(self):
            return len(self.user_interaction_code) * self.num_template

    return users, {
        "train": CachedNativeTrain(args, users, train, templates),
        "valid": native.data.evaluation_sequential_item_dataset(args, users, valid, templates, "validation"),
        "test": native.data.evaluation_sequential_item_dataset(args, users, test, templates, "test"),
    }


class CompleteTokenizer:
    """Change only the native collator's hard-coded truncation; reject over-bound rows."""

    def __init__(self, tokenizer, limit):
        self.tokenizer, self.limit = tokenizer, limit

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def batch_encode_plus(self, texts, **kwargs):
        kwargs.pop("max_length", None)
        kwargs["truncation"] = False
        result = self.tokenizer.batch_encode_plus(texts, **kwargs)
        if any(len(ids) > self.limit for ids in result["input_ids"]):
            raise ValueError("Unadmitted complete tokenized record; never silently truncate")
        return result


class CompleteCollator:
    def __init__(self, native, args, tokenizer, input_limit, target_limit):
        self.tokenizer, self.target_limit = tokenizer, target_limit
        self.native = native.data.Collator(args, CompleteTokenizer(tokenizer, input_limit))

    def __call__(self, rows):
        import torch

        batch = list(self.native(rows))
        inputs, attention, whole, labels, target_attention = batch
        eos, pad = self.tokenizer.eos_token_id, self.tokenizer.pad_token_id
        whole[(inputs == eos) | (inputs == pad)] = 0
        if torch.any(whole < 0) or torch.any(whole >= 512):
            raise ValueError("Native whole-word table would be indexed out of range")
        for ids, mask, bound in ((inputs, attention, inputs.shape[1]),
                                 (labels, target_attention, self.target_limit)):
            lengths = mask.sum(1)
            if torch.any(lengths <= 0) or torch.any(lengths > bound):
                raise ValueError("Invalid complete record length")
            if torch.any(ids.gather(1, (lengths - 1).unsqueeze(1)).flatten() != eos):
                raise ValueError("Complete P5 record lost terminal EOS")
            if torch.any((ids == eos).sum(1) != 1):
                raise ValueError("Embedded EOS changes title/prompt semantics")
            if torch.any((ids == pad) & mask.bool()):
                raise ValueError("Embedded padding would be a real ignored token")
            if torch.any(ids[~mask.bool()] != pad):
                raise ValueError("Non-padding token masked by collation")
        return tuple(batch)


def tokenized_catalog(tokenizer, codes):
    texts = ["item_" + code for code in codes]
    token_ids = tokenizer(texts, truncation=False, add_special_tokens=True)["input_ids"]
    canonical = tokenizer.batch_decode(
        token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True,
    )
    groups = defaultdict(list)
    for original, (ids, text) in enumerate(zip(token_ids, canonical)):
        if (not ids or ids[-1] != tokenizer.eos_token_id
                or ids.count(tokenizer.eos_token_id) != 1
                or tokenizer.pad_token_id in ids):
            raise ValueError(f"Unreachable catalog target / embedded EOS or padding: {original}")
        if not text or not text.startswith("item_"):
            raise ValueError(f"Native output prefix disappeared: {original}")
        groups[text].append(original)
    lengths = np.asarray(list(map(len, token_ids)))
    return {
        "original_item_codes": codes, "target_texts": texts,
        "token_ids": token_ids, "canonical_targets": canonical,
        "canonical_to_original_ids": dict(groups),
        "diagnostics": {
            "items": len(codes), "unique_raw_codes": len(set(codes)),
            "unique_token_sequences": len({tuple(ids) for ids in token_ids}),
            "unique_canonical_targets": len(groups),
            "canonical_collision_groups": sum(len(ids) > 1 for ids in groups.values()),
            "canonical_excess_collision_items": sum(len(ids) - 1 for ids in groups.values()),
            "raw_target_not_equal_canonical": sum(a != b for a, b in zip(texts, canonical)),
            "targets_with_unknown_tokens": sum(tokenizer.unk_token_id in ids for ids in token_ids),
            "token_length_p50_p90_p99_max": np.quantile(lengths, [.5, .9, .99, 1]).tolist(),
            "native_generation_max_length10_would_be_incomplete": int((lengths + 1 > 10).sum()),
            "native_collator512_would_truncate": int((lengths > 512).sum()),
        },
    }


def length_bounds(tokenizer, catalog, templates, users):
    target_limit = max(map(len, catalog["token_ids"]))
    prompt_limit = max(len(tokenizer.encode(t["source"])) for t in templates)
    user_limit = max(len(tokenizer.encode(u)) for u in users)
    separator = len(tokenizer.encode(" , "))
    # Depends on the catalog/prompt vocabulary and user IDs, never heldout item assignments.
    input_limit = 20 * (target_limit + separator) + prompt_limit + user_limit + 64
    return input_limit, target_limit


def all_supported_rows(native, datasets_, interactions, users):
    train = datasets_["train"]
    old_choice = native.data.random.choice
    selected = 1

    def chosen_position(values):
        if selected not in values:
            raise ValueError("Audit attempted a target outside native training support")
        return selected

    try:
        native.data.random.choice = chosen_position
        offset = 0
        for user in users:
            sequence = interactions[user][:-2]
            for position in range(1, len(sequence)):
                selected = position
                for template in range(10):
                    row = train[offset * 10 + template]
                    metadata = {
                        "user": user, "position": position, "template": template,
                        "history": sequence[max(0, position - 20):position],
                        "target": sequence[position],
                    }
                    yield "train", row, metadata
            offset += len(sequence) - 1
    finally:
        native.data.random.choice = old_choice
    for split in ("valid", "test"):
        for index, user in enumerate(users):
            sequence = interactions[user]
            target_position = len(sequence) - (2 if split == "valid" else 1)
            metadata = {
                "user": user, "position": target_position, "template": 0,
                "history": sequence[max(0, target_position - 20):target_position],
                "target": sequence[target_position],
            }
            yield split, datasets_[split][index], metadata


def audit(native, datasets_, interactions, users, tokenizer, catalog, args, output):
    input_limit, target_limit = length_bounds(
        tokenizer, catalog, native.prompt.task_subgroup_1, users,
    )
    state = {s: {"input_lengths": [], "target_lengths": [], "digest": hashlib.sha256(),
                 "whole_word_max": 0, "unknown_input_rows": 0, "native_eos_corrections": 0}
             for s in ("train", "valid", "test")}
    examples, longest, batch, sequence_number = [], [], [], 0
    started = time.monotonic()
    word_function = (native.data.calculate_whole_word_ids_title if args.item_representation == "title"
                     else native.data.calculate_whole_word_ids)

    def consume():
        nonlocal sequence_number
        encoded = tokenizer([row[1][0] for row in batch], truncation=False)
        ids_batch = encoded["input_ids"]
        for entry, ids in zip(batch, ids_batch):
            split, row, metadata = entry
            expected_target = catalog["target_texts"][metadata["target"]]
            if row[1] != expected_target:
                raise ValueError("Native target disagrees with exact original-item split mapping")
            template = native.prompt.task_subgroup_1[metadata["template"]]
            history = " , ".join("item_" + catalog["original_item_codes"][i] for i in metadata["history"])
            expected = template["source"].format(
                *((metadata["user"], history) if template["input_first"] == "user"
                  else (history, metadata["user"]))
            )
            if row[0] != expected:
                raise ValueError("Native source disagrees with original history/prompt")
            if (len(ids) > input_limit or ids[-1] != tokenizer.eos_token_id
                    or ids.count(tokenizer.eos_token_id) != 1 or tokenizer.pad_token_id in ids):
                raise ValueError("Input tokens exceed the fixed bound or contain embedded special controls")
            words = word_function(tokenizer.convert_ids_to_tokens(ids), ids)
            if len(words) != len(ids) or words[-1] != 0 or min(words) < 0 or max(words) >= 512:
                raise ValueError("Native whole-word IDs do not align or exceed the native table")
            target = catalog["token_ids"][metadata["target"]]
            s = state[split]
            s["input_lengths"].append(len(ids))
            s["target_lengths"].append(len(target))
            s["whole_word_max"] = max(s["whole_word_max"], max(words))
            s["unknown_input_rows"] += tokenizer.unk_token_id in ids
            s["digest"].update((digest_json([metadata, row, ids, words, target]) + "\n").encode())
            evidence = {"split": split, **metadata, "input": row[0], "target_text": row[1],
                        "input_length": len(ids), "target_length": len(target)}
            if sum(e["split"] == split for e in examples) < 2:
                examples.append(evidence)
            sequence_number += 1
            heapq.heappush(longest, (len(ids), sequence_number, evidence))
            if len(longest) > 100:
                heapq.heappop(longest)

    for record in all_supported_rows(native, datasets_, interactions, users):
        batch.append(record)
        if len(batch) == 512:
            consume()
            batch.clear()
            if sequence_number % 32768 == 0:
                write_json(Path(output) / "audit-progress.json", {
                    "records": sequence_number, "elapsed_seconds": time.monotonic() - started,
                })
                print(f"CORPUS_AUDIT records={sequence_number}", flush=True)
    if batch:
        consume()
    summary = {}
    for split, s in state.items():
        inputs = np.asarray(s["input_lengths"])
        targets = np.asarray(s["target_lengths"])
        if len(inputs) != len(datasets_[split]):
            raise ValueError("Incomplete native prompt/prefix/user coverage")
        summary[split] = {
            "records": len(inputs), "strict_clean": len(inputs), "over_bound": 0,
            "input_tokens_p50_p90_p99_max": np.quantile(inputs, [.5, .9, .99, 1]).tolist(),
            "target_tokens_p50_p90_p99_max": np.quantile(targets, [.5, .9, .99, 1]).tolist(),
            "native512_input_truncation_rows": int((inputs > 512).sum()),
            "native512_target_truncation_rows": int((targets > 512).sum()),
            "whole_word_max": s["whole_word_max"], "unknown_input_rows": s["unknown_input_rows"],
            "complete_record_sha256": s["digest"].hexdigest(),
        }
    write_json(Path(output) / "examples.json", examples)
    write_json(Path(output) / "longest-records.json", [e for _, _, e in sorted(longest, reverse=True)])
    return {
        "status": "pass", "splits": summary, "input_bound": input_limit,
        "target_bound_including_eos": target_limit, "generation_max_length": target_limit + 1,
        "generation_finish_reason": "N/A: static targets",
        "elapsed_seconds": time.monotonic() - started,
        "complete_support_not_only_sampled_training_rows": True,
    }


def prepare(args):
    from transformers import AutoTokenizer

    runtime = runtime_versions()
    recipe = load_recipe(args.recipe)
    assets = verify_assets(args.model)
    native = load_native(args.native)
    manifest = read_json(args.manifest)
    interactions, items, _ = read_release(args.upstream, manifest, args.dataset)
    output = args.output
    if (output / "admission.json").exists():
        return verify_admission(output, args.recipe)
    output.mkdir(parents=True, exist_ok=True)
    try:
        if (output / "catalog-index.json").exists():
            index = read_json(output / "catalog-index.json")
            if (index["recipe_sha256"] != sha256(args.recipe)
                    or index["release_inputs"] != manifest["datasets"][args.dataset]["sha256"]):
                raise ValueError("Cannot reuse a differently bound identifier construction")
            codes, vocabulary, index_report = index["codes"], index["vocabulary"], index["report"]
        else:
            codes, vocabulary, index_report = identifiers(
                native, args.dataset, args.representation, interactions, items, output,
            )
            write_json(output / "catalog-index.json", {
                "codes": codes, "vocabulary": vocabulary, "report": index_report,
                "recipe_sha256": sha256(args.recipe),
                "release_inputs": manifest["datasets"][args.dataset]["sha256"],
            })
    except MetadataUnavailable as error:
        write_json(output / "blocked.json", {
            "status": "blocked", "dataset": args.dataset, "representation": args.representation,
            "reason": str(error), "recipe_sha256": sha256(args.recipe),
            "release_inputs": manifest["datasets"][args.dataset]["sha256"],
        })
        print(f"P5_BLOCKED {args.dataset} {args.representation}: {error}", flush=True)
        return
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, use_fast=True)
    tokenizer.model_max_length = 1_000_000
    tokenizer.add_tokens(sorted(vocabulary))
    for token in vocabulary:
        if tokenizer.encode(token, add_special_tokens=False) != [tokenizer.convert_tokens_to_ids(token)]:
            raise ValueError(f"Native added identifier token is not atomic: {token}")
    catalog = tokenized_catalog(tokenizer, codes)
    if args.representation != "TID" and len(catalog["canonical_to_original_ids"]) != len(items):
        raise ValueError("Unique native code identities collapsed after tokenization")
    args_ = native_args(args.dataset, args.representation, len(items))
    users, views = datasets(native, interactions, codes, args_)
    if len(views["train"]) != 10 * manifest["datasets"][args.dataset]["training_prefix_examples"]:
        raise ValueError("Native random-target support differs from LETTER training-prefix support")
    tokenizer.save_pretrained(output / "tokenizer")
    write_json(output / "catalog.json", catalog)
    report = audit(native, views, interactions, users, tokenizer, catalog, args_, output)
    report.update(
        dataset=args.dataset, representation=args.representation, runtime=runtime,
        recipe_sha256=sha256(args.recipe), model_receipt_sha256=sha256(args.model / "asset-receipt.json"),
        pretrained_weights_sha256=assets["files_sha256"]["model.safetensors"],
        source_sha256=SOURCE_HASHES, release_inputs=manifest["datasets"][args.dataset]["sha256"],
        indexing=index_report, catalog=catalog["diagnostics"], added_tokens=sorted(vocabulary),
        tokenizer_vocab_sha256=digest_json(tokenizer.get_vocab()),
        added_token_policy={"special": False, "normalized": True, "lstrip": False, "rstrip": False},
    )
    write_json(output / "audit.json", report)
    files = {
        p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob("*"))
        if p.is_file() and p.name not in ("admission.json", "audit-progress.json", "blocked.json")
    }
    write_json(output / "admission.json", {
        "status": "pass", "dataset": args.dataset, "representation": args.representation,
        "recipe_sha256": sha256(args.recipe), "files_sha256": files,
    })
    if (output / "blocked.json").exists():
        (output / "blocked.json").replace(output / "prior-blocked.json")
    print(f"P5_ADMITTED {args.dataset} {args.representation} {digest_json(files)}", flush=True)
    return report


def verify_admission(output, recipe):
    admission = read_json(Path(output) / "admission.json")
    if admission.get("status") != "pass" or admission["recipe_sha256"] != sha256(recipe):
        raise ValueError("P5 complete corpus has not been admitted for this frozen recipe")
    for name, digest in admission["files_sha256"].items():
        require_hash(Path(output) / name, digest)
    report = read_json(Path(output) / "audit.json")
    if (report["status"] != "pass" or any(
        s["strict_clean"] != s["records"] or s["over_bound"] != 0 for s in report["splits"].values()
    )):
        raise ValueError("Incomplete P5 corpus admission")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("native", "upstream", "model", "manifest", "recipe", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--dataset", choices=("Instruments", "Beauty", "Yelp"), required=True)
    parser.add_argument("--representation", choices=("TID", "CID", "SemID"), required=True)
    prepare(parser.parse_args())
