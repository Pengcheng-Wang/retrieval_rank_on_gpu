"""Read-only, fail-closed Beauty prompt-support study; not a sampler or trainer.

The original exhaustive FAIL remains authoritative. This study reconstructs
item-keyed support from its complete failure ledger and independently rechecks
every native variant of every zero-support item/task on CPU. It never admits a
training variant, even if a different corpus has nonempty support.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys


CLAIM = "conditioned-native-prompt-support-v1 admission study"
AUDITOR_SHA256 = "dc67b3127c3d396878900596ed194936ed84cf23e84bbf5743a56566190494e0"
NATIVE_AUDIT_SHA256 = "16c5cdb3976d6f469d7e16687438409e6c16588cbc502ae6f61dbe964bc00b78"
AUXILIARY_TASKS = ("item2index", "index2item")


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path, expected):
    if file_sha256(path) != expected:
        raise ValueError(f"Evidence checksum mismatch: {path}")


def verify_native_bundle(directory, expected_audit_sha256):
    require_hash(directory / "audit.json", expected_audit_sha256)
    receipt = json.loads((directory / "completed.json").read_text(encoding="utf-8"))
    files = receipt["files_sha256"]
    if not {"audit.json", "run-config.json"} <= files.keys():
        raise ValueError("Incomplete native receipt")
    for name in files:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe native receipt path")
        try:
            (directory / path).resolve().relative_to(directory.resolve())
        except ValueError as error:
            raise ValueError("Unsafe native receipt path") from error
    for name, digest in files.items():
        require_hash(directory / name, digest)
    audit = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
    config = json.loads((directory / "run-config.json").read_text(encoding="utf-8"))
    require_hash(directory / "run-config.json", audit["run_config_sha256"])
    if (
        receipt["status"] != audit["status"] or audit["status"] != "fail"
        or receipt["corpus_admission"] != audit["corpus_admission"]
        or audit["corpus_admission"] != "FAIL" or config["dataset"] != "Beauty"
    ):
        raise ValueError("This study must preserve the untouched Beauty FAIL")
    return audit, config, receipt


def compile_support(item_ids, prompt_ids_by_task, failures):
    """Keep original item order/identity, including explicit empty support entries."""
    if not item_ids or len(set(item_ids)) != len(item_ids):
        raise ValueError("Missing or duplicate original item identities")
    if any(not isinstance(item, str) for item in item_ids):
        raise ValueError("Original item IDs must be strings, never SID strings or integers")
    if set(prompt_ids_by_task) != set(AUXILIARY_TASKS):
        raise ValueError("Both original auxiliary tasks are required")
    for prompts in prompt_ids_by_task.values():
        if (
            not prompts or len(set(prompts)) != len(prompts)
            or any(type(prompt) is not int or prompt < 0 for prompt in prompts)
        ):
            raise ValueError("Invalid native prompt inventory")
    rejected = {task: [set() for _ in item_ids] for task in AUXILIARY_TASKS}
    for row in failures:
        task, number, prompt = row["task"], row["record_number"], row["prompt_id"]
        if (
            task not in rejected or row["split"] != "train"
            or type(number) is not int or not 0 <= number < len(item_ids)
            or row["record_id"] != item_ids[number]
            or type(prompt) is not int or prompt not in prompt_ids_by_task[task]
            or row["audit"]["strict_clean"] is not False or not row["audit"]["defects"]
        ):
            raise ValueError("Unknown or contradictory native failure evidence")
        if prompt in rejected[task][number]:
            raise ValueError("Duplicate native failure pair")
        rejected[task][number].add(prompt)
    result = {}
    for task, prompts in prompt_ids_by_task.items():
        allowed = [[prompt for prompt in prompts if prompt not in bad] for bad in rejected[task]]
        sizes = Counter(map(len, allowed))
        total = len(item_ids) * len(prompts)
        admitted = sum(map(len, allowed))
        result[task] = {
            "item_ids": list(item_ids), "allowed_prompt_ids": allowed,
            "base_items": len(item_ids), "native_prompt_ids": list(prompts),
            "total_variants": total, "admissible_variants": admitted,
            "excluded_variants": total - admitted,
            "support_size_counts": {str(size): sizes[size] for size in sorted(sizes)},
            "unsupported_item_ids": [
                item for item, support in zip(item_ids, allowed) if not support
            ],
        }
    return result


def require_complete_support(support):
    if any(group["unsupported_item_ids"] for group in support.values()):
        raise ValueError("At least one original item/task has zero admissible native prompts")


def reconcile_failures(failures, report):
    counts, defects = Counter(), defaultdict(Counter)
    for row in failures:
        key = (f'{row["split"]}/{row["task"]}', str(row["prompt_id"]))
        counts[key] += 1
        defects[key].update(row["audit"]["defects"])
    expected_keys = set()
    for name, group in report["groups"].items():
        for prompt, statistics in group["per_prompt"].items():
            key = (name, prompt)
            expected_keys.add(key)
            if (
                counts[key] != statistics["total"] - statistics["strict_clean"]
                or defects[key] != Counter(statistics["defects"])
            ):
                raise ValueError("Incomplete or inconsistent native failure ledger")
    if set(counts) - expected_keys:
        raise ValueError("Unknown group in native failure ledger")


def cpu_runtime():
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise ValueError("Use scoped CPU-only CUDA_VISIBLE_DEVICES=-1")
    versions = {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in (
            "torch", "transformers", "peft", "sentencepiece", "tokenizers",
        )},
    }
    if (
        not versions["python"].startswith("3.10.")
        or not versions["torch"].startswith("1.13.1")
        or versions["transformers"] != "4.31.0" or versions["peft"] != "0.4.0"
    ):
        raise ValueError("Use the existing pinned legacy CPU runtime")
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    return versions


def run(args):
    private_root = Path(__file__).resolve().parents[2] / ".local" / "letter"
    args.output.resolve().relative_to(private_root.resolve())
    if args.output.exists():
        raise FileExistsError("Study receipts are write-once; choose a fresh private output")
    native_report, native_config, native_receipt = verify_native_bundle(
        args.native_audit, NATIVE_AUDIT_SHA256,
    )
    for path, digest in native_config["verified_files_sha256"].items():
        require_hash(Path(path), digest)
    require_hash(Path(__file__).with_name("lc_audit.py"), AUDITOR_SHA256)

    import lc_audit as auditor
    from author import load_backend
    from prepare import tokenizer_digest
    from protocol import collision_summary
    import numpy as np

    runtime = cpu_runtime()
    if runtime != native_config["runtime"]:
        raise ValueError("Runtime differs from the original audit")
    args.dataset = "Beauty"
    spec, manifest, _, verified, corpus = auditor._verify_inputs(args)
    if verified != native_config["verified_files_sha256"]:
        raise ValueError("Model/tokenizer/source/corpus binding differs from native audit")
    if native_report["probe"] or native_report["coverage_status"] != "pass":
        raise ValueError("Support cannot be inferred from partial native coverage")
    parts = [
        json.loads((args.native_audit / f"worker-{rank}" / "partition.json").read_text())
        for rank in range(native_config["workers"])
    ]
    for name, group in native_report["groups"].items():
        for part in parts:
            if (
                part["status"] != "complete"
                or part["groups"][name]["coverage"]["status"] != "pass"
            ):
                raise ValueError("Incomplete native worker coverage")
        statistics = auditor.merge_statistics([part["groups"][name]["statistics"] for part in parts])
        summary = auditor.summarize_statistics(statistics, group["expected"])
        if summary != {key: group[key] for key in summary}:
            raise ValueError("Native worker/group statistics disagree")
        for prompt, expected in group["per_prompt"].items():
            statistics = auditor.merge_statistics([
                part["groups"][name]["per_prompt"][prompt] for part in parts
            ])
            if auditor.summarize_statistics(statistics, group["base_records"]) != expected:
                raise ValueError("Native worker/prompt statistics disagree")
    failures = [
        json.loads(line)
        for rank in range(native_config["workers"])
        for line in (args.native_audit / f"worker-{rank}" / "failures.jsonl").read_text().splitlines()
    ]
    reconcile_failures(failures, native_report)
    inventory = {task: native_config["plan"][f"train/{task}"]["prompt_ids"] for task in AUXILIARY_TASKS}
    support = compile_support(list(corpus[1]), inventory, failures)
    for task, group in support.items():
        if group["admissible_variants"] != native_report["groups"][f"train/{task}"]["strict_clean"]:
            raise ValueError("Support totals disagree with native exhaustive audit")

    random.seed(42)
    np.random.seed(42)
    author = load_backend(args.upstream, "lc-rec")
    native_args = auditor._native_args(author, args, manifest)
    groups = auditor._groups(author, native_args, manifest, *corpus)
    tokenizer = auditor._tokenizer(args.model, corpus[2], native_args.model_max_length)
    if tokenizer_digest(tokenizer) != native_config["tokenizer_vocab_sha256"]:
        raise ValueError("Added-token vocabulary differs from the native audit")
    collator = author.collator.Collator(native_args, tokenizer)
    trigger = tokenizer("Response:")["input_ids"][1:]
    code_map = {
        "".join(code): [tokenizer.convert_tokens_to_ids(token) for token in code]
        for code in corpus[2].values()
    }
    original_failures = {
        (row["task"], row["record_id"], row["prompt_id"]): row for row in failures
    }
    witnesses, witness_statistics = [], auditor.new_statistics()
    verified_dataset_lengths = {}
    for name, dataset, task, split, identities in groups:
        verified_dataset_lengths[name] = len(dataset)
        if auditor.fingerprint(dataset.prompts) != native_config["plan"][name]["templates_sha256"]:
            raise ValueError("Native template inventory changed")
        if task not in support:
            continue
        for item in support[task]["unsupported_item_ids"]:
            number = identities.index(item)
            rows = list(auditor.iter_rendered(dataset, task, split, identities, number, number + 1))
            for raw, check, tokens in auditor._batch_results(
                rows, tokenizer, collator, None, code_map, trigger,
            ):
                original = original_failures[(task, item, raw["prompt_id"])]
                _, singleton, single_tokens = next(auditor._batch_results(
                    [raw], tokenizer, collator, None, code_map, trigger,
                ))
                width = min(len(tokens["full_ids"]), tokenizer.model_max_length)
                if (
                    check["strict_clean"] or singleton["strict_clean"]
                    or check != singleton or check["defects"] != original["audit"]["defects"]
                    or file_text_sha(raw["prompt"]) != original["prompt_sha256"]
                    or file_text_sha(raw["full"]) != original["full_text_sha256"]
                    or any(
                        tokens["native"][key][:width] != single_tokens["native"][key][:width]
                        for key in ("input_ids", "attention_mask", "labels")
                    )
                ):
                    raise ValueError("Fresh native failure witness disagrees with exhaustive evidence")
                auditor.add_statistics(witness_statistics, check)
                witnesses.append({
                    "rendered": raw, "audit": check, "tokens": tokens,
                    "matches_original_failure": True, "singleton_batch_parity": True,
                })
    expected_witnesses = sum(
        len(group["unsupported_item_ids"]) * len(group["native_prompt_ids"])
        for group in support.values()
    )
    if len(witnesses) != expected_witnesses:
        raise ValueError("Incomplete zero-support witness coverage")
    unsupported = sum(len(group["unsupported_item_ids"]) for group in support.values())
    status = "FAIL_ZERO_SUPPORT" if unsupported else "NOT_ADMITTED_REQUIRES_WRAPPER_AUDIT"
    if unsupported:
        try:
            require_complete_support(support)
        except ValueError:
            pass
        else:
            raise AssertionError("Zero-support admission did not fail closed")

    blank_descriptions = {item for item, row in corpus[1].items() if not row["description"].strip()}
    title_failures = {
        row["record_id"] for row in failures
        if row["task"] == "index2item" and row["prompt_id"] == 0
    }
    binding = {
        "native_audit_sha256": NATIVE_AUDIT_SHA256,
        "native_completed_sha256": file_sha256(args.native_audit / "completed.json"),
        "native_run_config_sha256": file_sha256(args.native_audit / "run-config.json"),
        "verified_files_sha256": verified,
        "tokenizer_vocab_sha256": tokenizer_digest(tokenizer),
        "tokenizer_settings": spec["tokenizer"], "context": 2048,
        "original_item_order_sha256": auditor.fingerprint(list(corpus[1])),
        "study_script_sha256": file_sha256(Path(__file__)),
        "study_tests_sha256": file_sha256(
            Path(__file__).resolve().parents[2] / "tests" / "test_letter_beauty_prompt_support.py",
        ),
    }
    report = {
        "schema_version": 1, "claim": CLAIM, "candidate_admission": status,
        "native_corpus_admission": "FAIL", "candidate_implemented": False,
        "exact_original_llama1_reproduction": False, "training_performed": False,
        "cpu_only": True, "cpu_workers": 1, "runtime": runtime,
        "generation_finish_reason": "N/A (static author-rendered targets)",
        "binding": binding,
        "original_artifacts_verified": len(native_receipt["files_sha256"]),
        "original_totals": native_report["totals"],
        "original_groups": native_report["groups"],
        "support": {
            task: {key: value for key, value in group.items()
                   if key not in {"item_ids", "allowed_prompt_ids"}}
            for task, group in support.items()
        },
        "collision_diagnostic": collision_summary(corpus[2]),
        "native_dataset_lengths": verified_dataset_lengths,
        "required_training_rows_per_epoch": sum(
            count for name, count in verified_dataset_lengths.items() if name.startswith("train/")
        ),
        "native_stripped_empty_titles": sum(
            not row["title"].strip().strip(".!?,;:`").strip() for row in corpus[1].values()
        ),
        "blank_description_items": len(blank_descriptions),
        "blank_description_items_with_clean_title_response": len(blank_descriptions - title_failures),
        "fresh_zero_support_witnesses": auditor.summarize_statistics(
            witness_statistics, expected_witnesses,
        ),
        "fresh_witness_verification": "pass",
        "post_wrapper_full_audit": "N/A: no wrapper; zero-support prerequisite failed",
        "sampler_resume_gate": "N/A: no sampler implemented or admitted",
        "empty_input_field_limitation": (
            "Original strict-clean checks empty response fields, not empty instruction fields. "
            "Requiring nonempty fields used in instructions can only shrink these support sets; "
            "it cannot rescue a zero-support item/task."
        ),
        "recommendation": (
            "Reject the all-item conditioned-native-prompt-support hypothesis; retain the "
            "original FAIL. Any string/mask/metadata/task/item/context amendment is a new "
            "scientific decision, not an exact paper reproduction."
        ),
    }
    for path, digest in verified.items():
        require_hash(Path(path), digest)
    verify_native_bundle(args.native_audit, NATIVE_AUDIT_SHA256)
    report["all_input_and_native_artifact_hashes_reverified_after_witnesses"] = True
    args.output.mkdir(parents=True, exist_ok=False)
    auditor.write_once_json(args.output / "report.json", report)
    auditor.write_once_json(args.output / "support-diagnostic.json", {
        "claim": CLAIM, "candidate_admission": status, "not_an_admitted_sampler": True,
        "binding": binding, "support": support,
    })
    auditor.write_once_json(args.output / "zero-support-witnesses.json", witnesses)
    auditor.write_once_json(args.output / "completed.json", {
        "status": "verified_rejection" if unsupported else "unadmitted_study",
        "native_corpus_admission": "FAIL", "candidate_admission": status,
        "files_sha256": {
            path.name: file_sha256(path) for path in sorted(args.output.iterdir()) if path.is_file()
        },
    })
    print(json.dumps({
        "candidate_admission": status, "unsupported_item_tasks": unsupported,
        "fresh_witnesses": len(witnesses), "fresh_strict_clean": witness_statistics["strict_clean"],
        "original_FAIL_preserved": True, "output": str(args.output),
    }), flush=True)
    return report


def file_text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native-audit", "upstream", "manifest", "method-spec",
                 "model", "model-receipt", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    for name, value in vars(args).items():
        setattr(args, name, value.resolve())
    report = run(args)
    sys.exit(2 if report["candidate_admission"] == "FAIL_ZERO_SUPPORT" else 3)


if __name__ == "__main__":
    main()
