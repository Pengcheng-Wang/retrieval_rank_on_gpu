"""Read-only, exhaustive admission of the released-ID LC-Rec compatibility hybrid.

This is not a trainer, a data cleaner, or an end-to-end LETTER reproduction.
Only explicitly selected private output directories under .local/letter are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from string import Formatter


HYBRID = "RELEASED-ID COMPATIBILITY/BRINGUP HYBRID"
COMMIT = "8d0154e28de37dbb6e24871c508ad8ddb1921cda"


def _inspect_tokens(prompt, full, target, *, response, limit, bos, eos, unk,
                    expected_code):
    common = 0
    for left, right in zip(prompt, full):
        if left != right:
            break
        common += 1
    defects = []
    checks = {
        "prefix_mismatch": common != len(prompt),
        "prompt_over_cap": len(prompt) > limit,
        "full_over_cap": len(full) > limit,
        "target_over_cap": len(target) > limit,
        "empty_response": not response.strip(),
        "prompt_bos_policy": not prompt or prompt[0] != bos or prompt.count(bos) != 1,
        "full_bos_policy": not full or full[0] != bos or full.count(bos) != 1,
        "prompt_eos_policy": eos in prompt,
        "full_eos_policy": not full or full[-1] != eos or full.count(eos) != 1,
        "target_special_token_policy": (
            not target or target[-1] != eos or target.count(eos) != 1 or bos in target
        ),
        "unknown_prompt": unk in prompt,
        "unknown_full": unk in full,
        "unknown_target": unk in target,
    }
    if expected_code is not None:
        checks["code_format"] = (
            len(expected_code) != 4 or target != expected_code + [eos]
            or full[len(prompt):] != expected_code + [eos]
        )
    defects.extend(name for name, failed in checks.items() if failed)
    return {
        "defects": defects,
        "lengths": {
            "prompt": len(prompt), "full": len(full), "target": len(target),
            "contextual_target": len(full) - common,
        },
        "diagnostics": {
            "prefix_common_tokens": common,
            "prompt_unk_tokens": prompt.count(unk), "full_unk_tokens": full.count(unk),
            "target_unk_tokens": target.count(unk),
            "prompt_bos_tokens": prompt.count(bos), "full_bos_tokens": full.count(bos),
            "target_bos_tokens": target.count(bos),
            "prompt_eos_tokens": prompt.count(eos), "full_eos_tokens": full.count(eos),
            "target_eos_tokens": target.count(eos),
            "full_tokens_beyond_cap": max(0, len(full) - limit),
            "contextual_target_tokens_beyond_cap": max(0, len(full) - max(common, limit)),
        },
    }


def _finish_row(row):
    row["defects"] = sorted(set(row["defects"]))
    row["strict_clean"] = not row["defects"]
    return row


def inspect_training_row(prompt, full, target, native, *, response, limit,
                         bos=1, eos=2, pad=0, unk=0, expected_code=None):
    """Compare native tensors both to its algorithm and to lossless response masking."""
    row = _inspect_tokens(
        prompt, full, target, response=response, limit=limit, bos=bos, eos=eos,
        unk=unk, expected_code=expected_code,
    )
    width = len(native["input_ids"])
    clipped = full[:limit]
    padded = clipped + [pad] * max(0, width - len(clipped))
    attention = [1] * len(clipped) + [0] * max(0, width - len(clipped))
    algorithm = [-100 if token == pad else token for token in padded]
    for position, token in enumerate(prompt[:limit]):
        if position < len(algorithm) and token != pad:
            algorithm[position] = -100
    parity = (
        native["input_ids"] == padded and native["attention_mask"] == attention
        and native["labels"] == algorithm
    )
    row["diagnostics"]["native_algorithm_parity"] = int(parity)
    if not parity:
        row["defects"].append("native_algorithm_mismatch")
    common = row["diagnostics"]["prefix_common_tokens"]
    intended = [-100] * common + full[common:]
    intended += [-100] * max(0, width - len(full))
    full_padded = full + [pad] * max(0, width - len(full))
    intended_parity = (
        common == len(prompt) and native["labels"] == intended
        and native["input_ids"] == full_padded
    )
    row["diagnostics"]["intended_labels_parity"] = int(intended_parity)
    if not intended_parity:
        row["defects"].append("intended_labels_mismatch")
    supervised = [token for token in native["labels"] if token != -100]
    row["diagnostics"]["native_supervised_tokens"] = len(supervised)
    row["diagnostics"]["native_supervised_eos_tokens"] = supervised.count(eos)
    if not supervised:
        row["defects"].append("all_masked")
    if not supervised or supervised[-1] != eos:
        row["defects"].append("eos_lost")
    if expected_code is not None and supervised != expected_code + [eos]:
        row["defects"].append("incomplete_code")
    return _finish_row(row)


def inspect_test_row(prompt, full, target, native, native_target, *, response, limit,
                     bos=1, eos=2, pad=0, unk=0, expected_code=None):
    row = _inspect_tokens(
        prompt, full, target, response=response, limit=limit, bos=bos, eos=eos,
        unk=unk, expected_code=expected_code,
    )
    clipped = prompt[:limit]
    padding = max(0, len(native["input_ids"]) - len(clipped))
    parity = (
        native["input_ids"] == [pad] * padding + clipped
        and native["attention_mask"] == [0] * padding + [1] * len(clipped)
    )
    if not parity:
        row["defects"].append("native_test_input_mismatch")
    if native_target != response:
        row["defects"].append("native_test_target_mismatch")
    row["diagnostics"]["native_test_parity"] = int(parity and native_target == response)
    return _finish_row(row)


class Coverage:
    """Ordered Cartesian coverage detects omissions/duplicates even at equal totals."""

    def __init__(self, first, stop, prompt_ids):
        if first < 0 or stop < first or not prompt_ids or len(set(prompt_ids)) != len(prompt_ids):
            raise ValueError("Invalid coverage interval or prompt inventory")
        self.first, self.stop, self.prompt_ids = first, stop, list(prompt_ids)
        self.observed = self.order_mismatches = 0
        self.digest = hashlib.sha256()

    def observe(self, record, prompt):
        count = len(self.prompt_ids)
        expected = (self.first + self.observed // count, self.prompt_ids[self.observed % count])
        self.order_mismatches += int((record, prompt) != expected or record >= self.stop)
        self.observed += 1
        self.digest.update(f"{record}:{prompt}\n".encode())

    def finish(self):
        expected = (self.stop - self.first) * len(self.prompt_ids)
        return {
            "status": "pass" if expected == self.observed and not self.order_mismatches else "fail",
            "first_record": self.first, "stop_record": self.stop,
            "prompt_ids": self.prompt_ids, "expected": expected, "observed": self.observed,
            "order_mismatches": self.order_mismatches, "ordered_keys_sha256": self.digest.hexdigest(),
        }


def iter_rendered(dataset, task, split, record_ids, first=0, stop=None):
    records = dataset.inter_data if task == "seqrec" else dataset.feat_data
    if len(records) != len(record_ids):
        raise ValueError("Native/canonical base-record coverage mismatch")
    stop = len(records) if stop is None else stop
    prompt_ids = [0] if split == "test" else list(range(len(dataset.prompts)))
    response_fields = {
        i: [name for _, name, _, _ in Formatter().parse(dataset.prompts[i]["response"]) if name]
        for i in prompt_ids
    }
    for number in range(first, stop):
        data = records[number]
        for prompt_id in prompt_ids:
            template = dataset.prompts[prompt_id]
            prompt, labels = dataset._get_text_data(data, template)
            response = template["response"].format(**data)
            yield {
                "record_number": number, "record_id": record_ids[number],
                "prompt_id": prompt_id, "task": task, "split": split,
                "prompt": prompt, "full": prompt + labels if split == "test" else labels,
                "response": response, "native": {"input_ids": prompt, "labels": labels},
                "empty_response_fields": [
                    name for name in response_fields[prompt_id]
                    if isinstance(data[name], str) and not data[name].strip()
                ],
            }


def length_quantiles(histogram):
    counts = sorted((int(value), count) for value, count in histogram.items() if count)
    total = sum(count for _, count in counts)
    if not total:
        return {"p50": None, "p90": None, "p99": None, "max": None}

    def at(index):
        seen = 0
        for value, count in counts:
            seen += count
            if seen > index:
                return value
        raise AssertionError("Invalid quantile rank")

    result = {}
    for name, quantile in (("p50", .5), ("p90", .9), ("p99", .99)):
        rank = (total - 1) * quantile
        lower = math.floor(rank)
        result[name] = at(lower) + (at(math.ceil(rank)) - at(lower)) * (rank - lower)
    return {**result, "max": counts[-1][0]}


def memoize_tokenizer(tokenizer, size):
    """Cache only native tokenize() results, never padding, truncation or labels."""
    original = tokenizer.tokenize

    @lru_cache(maxsize=size)
    def cached(text, keywords):
        return tuple(original(text, **dict(keywords)))

    def tokenize(text, **kwargs):
        keywords = tuple(sorted(kwargs.items()))
        try:
            hash(keywords)
        except TypeError:
            return original(text, **kwargs)
        return list(cached(text, keywords))

    tokenizer.tokenize = tokenize
    return cached.cache_info


def new_statistics():
    return {
        "total": 0, "strict_clean": 0, "defects": Counter(), "diagnostics": Counter(),
        "lengths": {name: Counter() for name in ("prompt", "full", "target", "contextual_target")},
    }


def add_statistics(statistics, row):
    statistics["total"] += 1
    statistics["strict_clean"] += int(row["strict_clean"])
    statistics["defects"].update(row["defects"])
    statistics["diagnostics"].update(row["diagnostics"])
    for kind, length in row["lengths"].items():
        statistics["lengths"][kind][length] += 1


def merge_statistics(parts):
    result = new_statistics()
    for part in parts:
        for key in ("total", "strict_clean"):
            result[key] += part[key]
        for key in ("defects", "diagnostics"):
            result[key].update(part[key])
        for kind, histogram in part["lengths"].items():
            result["lengths"][kind].update({int(k): v for k, v in histogram.items()})
    return result


def summarize_statistics(statistics, expected):
    return {
        "status": "pass" if statistics["total"] == statistics["strict_clean"] == expected else "fail",
        "expected": expected, "total": statistics["total"], "strict_clean": statistics["strict_clean"],
        "missing_or_extra_records": statistics["total"] - expected,
        "over_cap": statistics["defects"].get("full_over_cap", 0),
        "defects": dict(statistics["defects"]), "diagnostics": dict(statistics["diagnostics"]),
        "length_tokens": {
            kind: length_quantiles(histogram) for kind, histogram in statistics["lengths"].items()
        },
    }


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_once_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def validate_audit_spec(spec, manifest):
    from download_openllama import TOKENIZER_SHA, validate_track

    validate_track(spec)
    shared = manifest["author_lc_rec_recipes"]["shared"]
    tokenizer = spec["tokenizer"]
    if (
        spec["official_letter_commit"] != COMMIT or manifest["sources"]["letter"]["commit"] != COMMIT
        or spec["claim_type"] != "compatible_backbone_method_replication"
        or spec["model"]["context"] != 2048 or shared["model_max_length"] != 2048
        or shared["only_train_response"] is not True
        or shared["sample_valid"] is not True or shared["valid_prompt_sample_num"] != 2
        or shared["test_prompt_id"] != 0
        or manifest["paper_protocol"]["max_generative_history_items"] != 20
        or any(tokenizer.get(key) != value for key, value in {
            "use_fast": False, "legacy": True, "bos": 1, "eos": 2, "padding": 0,
            "automatic_eos": False, "apply_chat_template": False, "model_sha256": TOKENIZER_SHA,
        }.items())
    ):
        raise ValueError("Audit requires the pinned compatible track and unchanged native LC recipe")
    for dataset in ("Instruments", "Beauty", "Yelp"):
        expected = ["seqrec", "item2index", "index2item"] if dataset == "Beauty" else ["seqrec"]
        if manifest["author_lc_rec_recipes"][dataset]["tasks"] != expected:
            raise ValueError("LC task selection differs from the declared author recipe")


def verify_sequence_rows(dataset, canonical, indices):
    if len(dataset.inter_data) != len(canonical):
        raise ValueError("Native/canonical sequence coverage mismatch")
    identities = []
    for row, record in zip(dataset.inter_data, canonical):
        expected = {
            "inters": ", ".join("".join(indices[str(item)]) for item in record["history"]),
            "item": "".join(indices[str(record["target"])]),
        }
        if row != expected:
            raise ValueError("Native sequence differs from the independent canonical split")
        identities.append(f'{record["user"]}:{record["position"]}')
    return identities


def verify_metadata_rows(dataset, items, indices):
    identities = list(items)
    if list(dataset.item_feat) != identities or len(dataset.feat_data) != len(items):
        raise ValueError("Native metadata item coverage/order mismatch")
    for item, actual in zip(identities, dataset.feat_data):
        expected = {
            **items[item], "item": "".join(indices[item]),
            "title": items[item]["title"].strip().strip(".!?,;:`"),
        }
        if actual != expected:
            raise ValueError("Native metadata changed raw fields beyond the author's title stripping")
    return identities


def _native_args(author, args, manifest):
    from author import default_args

    native = default_args(author, args.upstream, args.dataset)
    shared = manifest["author_lc_rec_recipes"]["shared"]
    native.only_train_response = shared["only_train_response"]
    native.model_max_length = shared["model_max_length"]
    native.valid_prompt_sample_num = shared["valid_prompt_sample_num"]
    tasks = manifest["author_lc_rec_recipes"][args.dataset]["tasks"]
    native.tasks = ",".join(tasks)
    native.train_prompt_sample_num = ",".join("1" for _ in tasks)
    native.train_data_sample_num = ",".join("0" for _ in tasks)
    if native.his_sep != ", " or native.add_prefix or native.max_his_len != 20:
        raise ValueError("Native history rendering defaults changed")
    return native


def _groups(author, native_args, manifest, interactions, items, indices):
    from protocol import sequence_records

    groups = []
    for task in manifest["author_lc_rec_recipes"][native_args.dataset]["tasks"][1:]:
        dataset = author.data.ItemFeatDataset(native_args, task=task, prompt_sample_num=1, sample_num=0)
        keys = verify_metadata_rows(dataset, items, indices)
        groups.append((f"train/{task}", dataset, task, "train", keys))
    for split in ("train", "valid", "test"):
        dataset = author.data.SeqRecDataset(
            native_args, mode=split, prompt_sample_num=2 if split == "valid" else 1,
            prompt_id=0, sample_num=0,
        )
        canonical = sequence_records(interactions, split)
        keys = verify_sequence_rows(dataset, canonical, indices)
        expected = manifest["datasets"][native_args.dataset][
            "training_prefix_examples" if split == "train" else "users"
        ]
        if len(keys) != expected or len(dataset.prompts) != 17:
            raise ValueError("Native record or sequential template count differs from the frozen contract")
        groups.append((f"{split}/seqrec", dataset, "seqrec", split, keys))
    return groups


def _tokenizer(model, indices, limit):
    from transformers import LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained(
        model, legacy=True, local_files_only=True, model_max_length=limit,
    )
    tokenizer.pad_token_id = 0
    if (
        tokenizer.is_fast or tokenizer.legacy is not True or len(tokenizer) != 32000
        or tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2
        or not tokenizer.add_bos_token or tokenizer.add_eos_token
        or tokenizer.padding_side != "right" or tokenizer.truncation_side != "right"
        or getattr(tokenizer, "chat_template", None)
    ):
        raise ValueError("Unexpected slow tokenizer, special-token, padding or truncation policy")
    tokens = sorted({token for code in indices.values() for token in code})
    if tokenizer.add_tokens(tokens) != len(tokens):
        raise ValueError("Not every SID token was newly added to the native vocabulary")
    return tokenizer


def _inventory(tokenizer, dataset, indices):
    import torch

    tokens = sorted({token for code in indices.values() for token in code})
    atomic_failures = [
        token for token in tokens
        if tokenizer.encode(token, add_special_tokens=False) != [tokenizer.convert_tokens_to_ids(token)]
    ]
    code_map = {
        "".join(code): [tokenizer.convert_tokens_to_ids(token) for token in code]
        for code in indices.values()
    }
    catalog_failures = []
    for item, code in indices.items():
        expected = code_map["".join(code)]
        if (
            len(code) != 4 or tokenizer.encode("".join(code), add_special_tokens=False) != expected
            or tokenizer.encode("".join(code)) != [tokenizer.bos_token_id] + expected
        ):
            catalog_failures.append(item)
    trigger = tokenizer("Response:")["input_ids"][1:]
    first_prompt, _ = dataset._get_text_data(dataset.inter_data[0], dataset.prompts[0])
    prompt_ids = tokenizer(first_prompt)["input_ids"]
    callback = dataset.get_prefix_allowed_tokens_fn(tokenizer)
    expected_sets = [set(code[position] for code in code_map.values()) for position in range(4)]
    expected_sets.append({tokenizer.eos_token_id})
    transition_failures = []
    for code, ids in code_map.items():
        for position in range(5):
            try:
                allowed = callback(0, torch.tensor(prompt_ids + ids[:position]))
                matches = allowed is not None and set(allowed) == expected_sets[position]
            except Exception:
                matches = False
            if not matches:
                transition_failures.append([code, position])
    return code_map, trigger, {
        "status": "pass" if not (atomic_failures or catalog_failures or transition_failures) else "fail",
        "sorted_unique_sid_tokens": len(tokens), "atomic_sid_tokens": len(tokens) - len(atomic_failures),
        "atomic_failures": atomic_failures, "catalog_items_checked": len(indices),
        "four_token_catalog_failures": catalog_failures,
        "response_trigger_ids": trigger, "native_transition_checks": len(code_map) * 5,
        "native_transition_failures": transition_failures,
        "constraint_scope": "Author per-position allowed sets, not a catalog trie; no generation performed.",
    }


def _batch_results(rows, tokenizer, collator, test_collator, code_map, trigger):
    prompts = tokenizer([row["prompt"] for row in rows], truncation=False)["input_ids"]
    full = tokenizer(
        [row["full"] + tokenizer.eos_token for row in rows], truncation=False,
    )["input_ids"]
    targets = tokenizer(
        [row["response"] + tokenizer.eos_token for row in rows],
        truncation=False, add_special_tokens=False,
    )["input_ids"]
    if not len(rows) == len(prompts) == len(full) == len(targets):
        raise ValueError("Tokenizer changed batch coverage")
    is_test = rows[0]["split"] == "test"
    native_rows = [row["native"] for row in rows]
    errors = [None] * len(rows)
    selected_collator = test_collator if is_test else collator
    try:
        batched = selected_collator(native_rows)
        tensors, native_targets = batched if is_test else (batched, [None] * len(rows))
        arrays = {key: value.tolist() for key, value in tensors.items()}
        native = [{key: value[i] for key, value in arrays.items()} for i in range(len(rows))]
    except Exception as error:
        # A bad batch is evidence, not permission to drop its rows.
        native, native_targets = [], []
        for i, raw in enumerate(native_rows):
            errors[i] = f"{type(error).__name__}: {error}"
            try:
                one = selected_collator([raw])
                tensors, labels = one if is_test else (one, [None])
                native.append({key: value[0].tolist() for key, value in tensors.items()})
                native_targets.append(labels[0])
            except Exception as row_error:
                errors[i] += f"; singleton: {type(row_error).__name__}: {row_error}"
                native.append({"input_ids": [], "attention_mask": [], "labels": []})
                native_targets.append(None)
    for i, raw in enumerate(rows):
        expected_code = None if raw["task"] == "index2item" else code_map.get(raw["response"])
        settings = {
            "response": raw["response"], "limit": tokenizer.model_max_length,
            "bos": tokenizer.bos_token_id, "eos": tokenizer.eos_token_id,
            "pad": tokenizer.pad_token_id, "unk": tokenizer.unk_token_id,
            "expected_code": expected_code,
        }
        if is_test:
            result = inspect_test_row(
                prompts[i], full[i], targets[i], native[i], native_targets[i], **settings,
            )
        else:
            result = inspect_training_row(prompts[i], full[i], targets[i], native[i], **settings)
        if raw["full"] != raw["prompt"] + raw["response"]:
            result["defects"].append("rendered_prefix_mismatch")
        if raw["task"] != "index2item" and expected_code is None:
            result["defects"].append("noncatalog_code_response")
        if raw["empty_response_fields"]:
            result["defects"].append("empty_response_field")
        result["diagnostics"]["response_trigger_checks"] = 1
        if not trigger or prompts[i][-len(trigger):] != trigger:
            result["defects"].append("response_trigger_mismatch")
        clipped_prompt = prompts[i][:tokenizer.model_max_length]
        if not trigger or clipped_prompt[-len(trigger):] != trigger:
            result["defects"].append("response_trigger_lost_to_truncation")
        if errors[i] is not None:
            result["defects"].append("native_collator_exception")
            result["native_error"] = errors[i]
        _finish_row(result)
        yield raw, result, {
            "prompt_ids": prompts[i], "full_ids": full[i], "standalone_target_ids": targets[i],
            "native": native[i], "native_test_target": native_targets[i],
        }


def _runtime():
    import importlib.metadata
    import platform
    import torch

    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"} or torch.cuda.is_available():
        raise RuntimeError("CPU-only audit requires CUDA_VISIBLE_DEVICES='' or '-1'")
    versions = {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in (
            "torch", "transformers", "peft", "sentencepiece", "tokenizers",
        )},
    }
    if (
        not versions["python"].startswith("3.10.") or not versions["torch"].startswith("1.13.1")
        or versions["transformers"] != "4.31.0" or versions["peft"] != "0.4.0"
    ):
        raise RuntimeError("Use the existing pinned legacy LETTER runtime, not a replacement environment")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    return versions


def _worker(args, rank, run_config_sha256):
    import traceback
    from author import load_backend
    from prepare import read_release
    from protocol import require_hash

    output = args.output / f"worker-{rank}"
    output.mkdir()
    result = {"rank": rank, "pid": os.getpid(), "groups": {}}
    try:
        _runtime()
        require_hash(args.output / "run-config.json", run_config_sha256)
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        interactions, items, indices = read_release(args.upstream, manifest, args.dataset)
        author = load_backend(args.upstream, "lc-rec")
        native_args = _native_args(author, args, manifest)
        author.utils.set_seed(42)
        groups = _groups(author, native_args, manifest, interactions, items, indices)
        train_tokenizer = _tokenizer(args.model, indices, native_args.model_max_length)
        test_tokenizer = _tokenizer(args.model, indices, native_args.model_max_length)
        collator = author.collator.Collator(native_args, train_tokenizer)
        test_collator = author.collator.TestCollator(native_args, test_tokenizer)
        if train_tokenizer.padding_side != "right" or test_tokenizer.padding_side != "left":
            raise ValueError("Training/test tokenizer padding instances are not safely isolated")
        code_map = {
            "".join(code): [train_tokenizer.convert_tokens_to_ids(token) for token in code]
            for code in indices.values()
        }
        trigger = train_tokenizer("Response:")["input_ids"][1:]
        cache_info = [] if args.no_memoize else [
            memoize_tokenizer(tokenizer, 3 * args.batch_size + 256)
            for tokenizer in (train_tokenizer, test_tokenizer)
        ]
        examples = {}
        started = time.monotonic()
        with (output / "failures.jsonl").open("x", encoding="utf-8", newline="\n") as failures, (
            output / "progress.jsonl"
        ).open("x", encoding="utf-8", newline="\n") as progress:
            for name, dataset, task, split, record_ids in groups:
                total_records = len(record_ids)
                if args.probe_base_records is not None:
                    total_records = min(total_records, args.probe_base_records)
                first = total_records * rank // args.workers
                stop = total_records * (rank + 1) // args.workers
                prompt_ids = [0] if split == "test" else list(range(len(dataset.prompts)))
                coverage = Coverage(first, stop, prompt_ids)
                statistics = new_statistics()
                per_prompt = {str(i): new_statistics() for i in prompt_ids}
                batch = []

                def process_batch():
                    for raw, row, tokens in _batch_results(
                        batch, train_tokenizer, collator, test_collator, code_map, trigger,
                    ):
                        coverage.observe(raw["record_number"], raw["prompt_id"])
                        add_statistics(statistics, row)
                        add_statistics(per_prompt[str(raw["prompt_id"])], row)
                        baseline_key = f'{name}/template-{raw["prompt_id"]}'
                        maximum_key = f"{name}/longest"
                        sample_keys = []
                        if baseline_key not in examples:
                            sample_keys.append(baseline_key)
                        if (
                            maximum_key not in examples
                            or row["lengths"]["full"] > examples[maximum_key]["audit"]["lengths"]["full"]
                        ):
                            sample_keys.append(maximum_key)
                        if not row["strict_clean"]:
                            evidence = {
                                key: raw[key] for key in ("record_number", "record_id", "prompt_id", "task", "split")
                            }
                            evidence.update(
                                audit=row, empty_response_fields=raw["empty_response_fields"],
                                prompt_sha256=hashlib.sha256(raw["prompt"].encode()).hexdigest(),
                                full_text_sha256=hashlib.sha256(raw["full"].encode()).hexdigest(),
                            )
                            failures.write(json.dumps(evidence, ensure_ascii=False) + "\n")
                            for defect in row["defects"]:
                                key = f"{name}/defect-{defect}"
                                if key not in examples:
                                    sample_keys.append(key)
                        for key in sample_keys:
                            examples[key] = {"rendered": raw, "audit": row, "tokens": tokens}
                    if train_tokenizer.padding_side != "right":
                        raise RuntimeError("Test collation mutated the training tokenizer")

                for rendered in iter_rendered(dataset, task, split, record_ids, first, stop):
                    batch.append(rendered)
                    if len(batch) == args.batch_size:
                        process_batch()
                        batch.clear()
                        if statistics["total"] // args.batch_size % 20 == 0:
                            progress.write(json.dumps({
                                "group": name, "total": statistics["total"],
                                "strict_clean": statistics["strict_clean"],
                                "elapsed_seconds": time.monotonic() - started, "pid": os.getpid(),
                            }) + "\n")
                            progress.flush()
                            failures.flush()
                if batch:
                    process_batch()
                result["groups"][name] = {
                    "coverage": coverage.finish(), "statistics": statistics,
                    "per_prompt": per_prompt,
                }
                summary = summarize_statistics(statistics, coverage.finish()["expected"])
                write_once_json(output / (name.replace("/", "-") + ".json"), {
                    **summary, "coverage": coverage.finish(),
                })
                failures.flush()
                print(
                    f"AUDITED {args.dataset} worker={rank} {name} rows={statistics['total']} "
                    f"clean={statistics['strict_clean']} over_cap={summary['over_cap']}", flush=True,
                )
        result["tokenize_cache"] = [info()._asdict() for info in cache_info]
        result["status"] = "complete"
        write_once_json(output / "examples.json", examples)
    except Exception as error:
        result.update(status="error", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    write_once_json(output / "partition.json", result)
    return result


def _verify_inputs(args):
    from author import SOURCE_HASHES, verify_backend_source
    from download_openllama import validate_architecture
    from encode import verify_backbone
    from prepare import read_release
    from protocol import file_sha256

    spec = json.loads(args.method_spec.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_audit_spec(spec, manifest)
    if args.manifest.name != spec["input_protocol"]:
        raise ValueError("Method spec points to a different input protocol")
    receipt = verify_backbone(args, spec)
    files = receipt["converted_files_sha256"]
    required = {
        "LICENSE", "README.md", "config.json", "generation_config.json",
        "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
        "model.safetensors.index.json", "tokenizer.model", "tokenizer_config.json",
        "special_tokens_map.json",
    }
    if set(files) != required or {p.name for p in args.model.iterdir() if p.is_file()} != required:
        raise ValueError("Converted model/tokenizer asset inventory differs from the verified snapshot")
    if files["tokenizer.model"] != spec["tokenizer"]["model_sha256"]:
        raise ValueError("Model receipt names a different tokenizer")
    validate_architecture(json.loads((args.model / "config.json").read_text(encoding="utf-8")))
    source = verify_backend_source(args.upstream, "lc-rec")
    interactions, items, indices = read_release(args.upstream, manifest, args.dataset)
    verified = {
        str(args.model / name): digest for name, digest in files.items()
    }
    verified.update({
        str(source / name): digest for name, digest in SOURCE_HASHES["lc-rec"].items()
    })
    verified.update({
        str(args.upstream / "data" / args.dataset / f"{args.dataset}.{kind}.json"): digest
        for kind, digest in manifest["datasets"][args.dataset]["sha256"].items()
    })
    for path in (args.manifest, args.method_spec, args.model_receipt):
        verified[str(path)] = file_sha256(path)
    for name in ("lc_audit.py", "author.py", "prepare.py", "protocol.py", "encode.py",
                 "download_openllama.py", "env.sh"):
        path = Path(__file__).parent / name
        verified[str(path.resolve())] = file_sha256(path)
    return spec, manifest, receipt, verified, (interactions, items, indices)


def run(args):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from author import SOURCE_HASHES, load_backend
    from prepare import tokenizer_digest
    from protocol import collision_summary, file_sha256, require_hash

    private_root = Path(__file__).resolve().parents[2] / ".local" / "letter"
    args.output.resolve().relative_to(private_root.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        "schema_version": 1, "status": "fail", "dataset": args.dataset, "claim": HYBRID,
        "corpus_admission": "FAIL", "exact_original_llama1_reproduction": False,
        "generation_finish_reason": "N/A (static author-rendered targets; no generation)",
        "template": "Native sft_prompt only; no chat wrapper and no thinking mode.",
        "length_definition": {
            "full": "BOS + tokenized complete native prompt/response + explicitly appended EOS",
            "prompt": "Native prompt with BOS, without automatic EOS",
            "target": "Standalone native response + EOS, without BOS (diagnostic for text)",
            "contextual_target": "Full sequence after longest common prompt/full token prefix",
        },
        "no_filtering_or_recipe_changes": True, "cpu_only": True, "groups": {},
        "probe": args.probe_base_records is not None,
    }
    try:
        report["runtime"] = _runtime()
        spec, manifest, receipt, verified, corpus = _verify_inputs(args)
        author = load_backend(args.upstream, "lc-rec")
        native_args = _native_args(author, args, manifest)
        author.utils.set_seed(42)
        groups = _groups(author, native_args, manifest, *corpus)
        tokenizer = _tokenizer(args.model, corpus[2], native_args.model_max_length)
        seq_train = next(group[1] for group in groups if group[0] == "train/seqrec")
        _, _, inventory = _inventory(tokenizer, seq_train, corpus[2])
        plan = {}
        for name, dataset, task, split, keys in groups:
            count = len(keys)
            if args.probe_base_records is not None:
                count = min(count, args.probe_base_records)
            prompt_ids = [0] if split == "test" else list(range(len(dataset.prompts)))
            plan[name] = {
                "base_records": count, "full_corpus_base_records": len(keys),
                "prompt_ids": prompt_ids, "expected_variants": count * len(prompt_ids),
                "target_kind": "text" if task == "index2item" else "four-token SID plus EOS",
                "templates_sha256": fingerprint(dataset.prompts),
            }
        config = {
            "claim": HYBRID, "track_id": spec["track_id"], "dataset": args.dataset,
            "official_letter_commit": COMMIT, "model_id": receipt["model_id"],
            "model_revision": receipt["revision"], "runtime": report["runtime"],
            "index_origin": "Pinned author-released indices, NOT fresh compatible-variant indices",
            "verified_files_sha256": verified, "source_sha256": SOURCE_HASHES["lc-rec"],
            "corpus_sha256": manifest["datasets"][args.dataset]["sha256"],
            "model_receipt_sha256": file_sha256(args.model_receipt),
            "tokenizer_vocab_sha256": tokenizer_digest(tokenizer),
            "tokenizer_settings": spec["tokenizer"], "context": native_args.model_max_length,
            "only_train_response": True, "history": 20, "history_separator": ", ",
            "train_padding": "right", "test_padding": "left, distinct tokenizer instance",
            "workers": args.workers, "threads_per_worker": 1, "batch_size": args.batch_size,
            "memoization": (
                "disabled" if args.no_memoize else
                "Exact native tokenize() results only; padding/truncation/collators always run natively"
            ),
            "plan": plan, "probe_base_records": args.probe_base_records,
        }
        write_once_json(args.output / "run-config.json", config)
        identity = file_sha256(args.output / "run-config.json")
        report.update(
            run_config_sha256=identity, config_fingerprint=fingerprint(config),
            atomic_sid_and_native_constraints=inventory, model_max_length=native_args.model_max_length,
            collision_diagnostic=collision_summary(corpus[2]),
        )
        # Spawn avoids sharing the mutable test tokenizer or the author's generic modules.
        del groups, seq_train, tokenizer, corpus
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(_worker, args, rank, identity) for rank in range(args.workers)]
            partitions = [future.result() for future in futures]
        report["workers"] = [
            {key: value for key, value in part.items() if key != "groups"} for part in partitions
        ]
        all_statistics = []
        coverage_pass = all(part["status"] == "complete" for part in partitions)
        for name, planned in plan.items():
            parts = [part["groups"][name] for part in partitions if name in part["groups"]]
            statistics = merge_statistics([part["statistics"] for part in parts])
            all_statistics.append(statistics)
            coverage = [part["coverage"] for part in parts]
            group_coverage = (
                len(parts) == args.workers and all(c["status"] == "pass" for c in coverage)
                and sum(c["observed"] for c in coverage) == planned["expected_variants"]
            )
            coverage_pass = coverage_pass and group_coverage
            summary = summarize_statistics(statistics, planned["expected_variants"])
            summary.update(
                **planned, coverage=coverage, coverage_status="pass" if group_coverage else "fail",
                per_prompt={
                    str(prompt): summarize_statistics(
                        merge_statistics([part["per_prompt"][str(prompt)] for part in parts]),
                        planned["base_records"],
                    ) for prompt in planned["prompt_ids"]
                },
            )
            report["groups"][name] = summary
        report["totals"] = summarize_statistics(
            merge_statistics(all_statistics), sum(p["expected_variants"] for p in plan.values()),
        )
        for path, digest in verified.items():
            require_hash(Path(path), digest)
        require_hash(args.output / "run-config.json", identity)
        report["all_fingerprints_reverified_after_audit"] = True
        report["coverage_status"] = "pass" if coverage_pass else "fail"
        passed = report["totals"]["status"] == "pass" and coverage_pass and inventory["status"] == "pass"
        report["status"] = "pass" if passed else "fail"
        report["corpus_admission"] = (
            "NOT_ADMITTED_PROBE" if report["probe"] else ("PASS" if passed else "FAIL")
        )
        report["options_if_not_admitted"] = [
            "Keep this track blocked; preserve the untouched author corpus/recipe and seek author clarification.",
            "A tokenizer/model change requires its own licensed, identified track and a fresh full audit.",
            "Any prompt boundary, masking, metadata, length, or task change is a separately named "
            "method deviation requiring explicit approval; do not silently filter, truncate or drop tasks.",
        ] if not passed else []
    except Exception as error:
        import traceback

        report.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        report["coverage_status"] = "incomplete"
    report["elapsed_seconds"] = time.monotonic() - started
    write_once_json(args.output / "audit.json", report)
    artifacts = {
        str(path.relative_to(args.output)): file_sha256(path)
        for path in sorted(args.output.rglob("*")) if path.is_file()
    }
    write_once_json(args.output / "completed.json", {
        "status": report["status"], "corpus_admission": report["corpus_admission"],
        "claim": HYBRID, "files_sha256": artifacts,
    })
    print(json.dumps({
        "dataset": args.dataset, "status": report["status"],
        "corpus_admission": report["corpus_admission"],
        "totals": report.get("totals"), "output": str(args.output),
        "error": report.get("error"), "elapsed_seconds": report["elapsed_seconds"],
    }), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("upstream", "manifest", "method-spec", "model", "model-receipt", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--dataset", choices=("Instruments", "Beauty", "Yelp"), required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--no-memoize", action="store_true")
    parser.add_argument(
        "--probe-base-records", type=int,
        help="Inspect at most this many base records per group; never admits a corpus.",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 2048:
        parser.error("--batch-size must be between 1 and 2048")
    if args.probe_base_records is not None and args.probe_base_records < 1:
        parser.error("--probe-base-records must be positive")
    for name in ("upstream", "manifest", "method_spec", "model", "model_receipt", "output"):
        setattr(args, name, getattr(args, name).resolve())
    report = run(args)
    sys.exit(0 if report["status"] == "pass" else 2)


if __name__ == "__main__":
    main()
