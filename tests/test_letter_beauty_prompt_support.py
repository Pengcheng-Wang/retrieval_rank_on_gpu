"""Fail-closed contracts for the Beauty prompt-support admission study."""

import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "letter"
sys.path.insert(0, str(SCRIPTS))


def study():
    assert (SCRIPTS / "beauty_prompt_support.py").is_file(), "Admission study is missing"
    return importlib.import_module("beauty_prompt_support")


def failure(item, number, prompt, task="index2item", defects=None):
    return {
        "task": task, "split": "train", "record_id": item, "record_number": number,
        "prompt_id": prompt,
        "audit": {"strict_clean": False, "defects": defects or ["prefix_mismatch"]},
    }


def test_zero_support_is_a_failed_item_task_not_a_dropped_item():
    module = study()
    result = module.compile_support(
        ["0", "1"], {"item2index": [0, 1], "index2item": [0, 1]},
        [failure("0", 0, 0), failure("0", 0, 1)],
    )
    assert result["index2item"]["item_ids"] == ["0", "1"]
    assert result["index2item"]["allowed_prompt_ids"] == [[], [0, 1]]
    assert result["index2item"]["unsupported_item_ids"] == ["0"]
    assert result["index2item"]["support_size_counts"] == {"0": 1, "2": 1}
    with pytest.raises(ValueError, match="zero admissible"):
        module.require_complete_support(result)


def test_original_item_identity_is_preserved_even_if_sid_strings_collide():
    module = study()
    indices = {"3": ["same", "SID"], "7": ["same", "SID"]}
    result = module.compile_support(
        list(indices), {"item2index": [0, 1], "index2item": [0, 1]},
        [failure("3", 0, 0)],
    )
    assert result["index2item"]["item_ids"] == ["3", "7"]
    assert result["index2item"]["allowed_prompt_ids"] == [[1], [0, 1]]
    assert result["index2item"]["base_items"] == 2
    assert result["index2item"]["admissible_variants"] == 3
    module.require_complete_support(result)


def test_only_the_failed_prompt_pair_is_removed_not_other_tasks_or_responses():
    result = study().compile_support(
        ["0"], {"item2index": [0, 1], "index2item": [0, 1]},
        [failure("0", 0, 1, defects=["empty_response", "empty_response_field"])],
    )
    assert result["item2index"]["allowed_prompt_ids"] == [[0, 1]]
    assert result["index2item"]["allowed_prompt_ids"] == [[0]]
    assert result["index2item"]["excluded_variants"] == 1


@pytest.mark.parametrize("change", [
    {"record_id": "unknown"}, {"record_number": 1}, {"prompt_id": 99},
    {"task": "seqrec"}, {"split": "test"},
    {"audit": {"strict_clean": True, "defects": []}},
])
def test_unknown_or_contradictory_evidence_fails_instead_of_blanket_filtering(change):
    row = {**failure("0", 0, 0), **change}
    with pytest.raises(ValueError):
        study().compile_support(
            ["0"], {"item2index": [0, 1], "index2item": [0, 1]}, [row],
        )


def test_duplicate_failure_pairs_cannot_disguise_incomplete_evidence():
    row = failure("0", 0, 0)
    with pytest.raises(ValueError, match="Duplicate"):
        study().compile_support(
            ["0"], {"item2index": [0, 1], "index2item": [0, 1]}, [row, row],
        )


def test_missing_failure_ledger_rows_are_rejected():
    report = {
        "groups": {
            "train/index2item": {
                "per_prompt": {
                    "0": {"total": 2, "strict_clean": 1, "defects": {"prefix_mismatch": 1}},
                },
            },
        },
    }
    with pytest.raises(ValueError, match="failure ledger"):
        study().reconcile_failures([], report)
    study().reconcile_failures([failure("0", 0, 0)], report)


@pytest.mark.parametrize("kind", ["prefix", "truncation", "empty_target"])
def test_original_strict_criteria_reject_invalid_variants_without_repair(kind):
    from lc_audit import inspect_training_row

    prompt, full, response, limit = [1, 7], [1, 7, 10, 11, 2], "answer", 32
    target = [10, 11, 2]
    if kind == "prefix":
        prompt = [1, 7, 9]
    elif kind == "truncation":
        limit = 3
    else:
        full, target, response = [1, 7, 2], [2], ""
    ids = full[:limit]
    labels = [-100 if i < len(prompt[:limit]) else token for i, token in enumerate(ids)]
    row = inspect_training_row(
        prompt, full, target,
        {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)},
        response=response, limit=limit,
    )
    assert row["strict_clean"] is False
    result = study().compile_support(
        ["0"], {"item2index": [0, 1], "index2item": [0, 1]},
        [{**failure("0", 0, 0), "audit": row}],
    )
    assert result["index2item"]["allowed_prompt_ids"] == [[1]]


def bundle(directory):
    audit = {"status": "fail", "corpus_admission": "FAIL"}
    config = {"dataset": "Beauty"}
    config_path = directory / "run-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    audit["run_config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    audit_path = directory / "audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    digest = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    (directory / "completed.json").write_text(json.dumps({
        "status": "fail", "corpus_admission": "FAIL",
        "files_sha256": {
            "audit.json": digest, "run-config.json": audit["run_config_sha256"],
        },
    }), encoding="utf-8")
    return digest


def test_bad_or_stale_audit_support_hash_is_rejected(tmp_path):
    digest = bundle(tmp_path)
    study().verify_native_bundle(tmp_path, digest)
    with pytest.raises(ValueError, match="checksum"):
        study().verify_native_bundle(tmp_path, "0" * 64)
    (tmp_path / "run-config.json").write_text('{"dataset":"other"}', encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        study().verify_native_bundle(tmp_path, digest)


def test_receipt_paths_cannot_escape_the_original_audit_directory(tmp_path):
    digest = bundle(tmp_path)
    path = tmp_path / "completed.json"
    receipt = json.loads(path.read_text())
    receipt["files_sha256"]["../outside.json"] = "0" * 64
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt path"):
        study().verify_native_bundle(tmp_path, digest)


def test_this_read_only_study_cannot_be_mistaken_for_a_training_admission():
    module = study()
    assert module.CLAIM == "conditioned-native-prompt-support-v1 admission study"
    assert not hasattr(module, "ConditionedItemFeatDataset")
    assert not (ROOT / "configs" / "letter-beauty-prompt-support-v1.json").exists()
