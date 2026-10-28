"""CPU contracts for exhaustive, non-filtering LC-Rec corpus admission."""

import importlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "letter"
sys.path.insert(0, str(SCRIPTS))


def audit():
    assert (SCRIPTS / "lc_audit.py").is_file(), "Exhaustive LC-Rec audit is missing"
    return importlib.import_module("lc_audit")


def inspect(prompt, full, labels, *, limit=32, response="answer", target=None,
            native_ids=None, attention=None, code=None):
    native_ids = list(full if native_ids is None else native_ids)
    return audit().inspect_training_row(
        prompt, full, target or [10, 11, 2],
        {"input_ids": native_ids, "attention_mask": attention or [1] * len(native_ids),
         "labels": labels},
        response=response, limit=limit, bos=1, eos=2, pad=0, unk=0,
        expected_code=code,
    )


def test_boundary_change_is_failure_even_when_native_algorithm_matches():
    row = inspect([1, 7, 9], [1, 7, 10, 11, 2], [-100, -100, -100, 11, 2])
    assert row["strict_clean"] is False
    assert "prefix_mismatch" in row["defects"]
    assert "intended_labels_mismatch" in row["defects"]
    assert row["diagnostics"]["native_algorithm_parity"] == 1
    assert row["diagnostics"]["prefix_common_tokens"] == 2


def test_truncated_response_and_eos_cannot_be_admitted():
    row = inspect(
        [1, 7], [1, 7, 10, 11, 12, 13, 2], [-100, -100, 10, 11],
        native_ids=[1, 7, 10, 11], limit=4, target=[10, 11, 12, 13, 2],
        code=[10, 11, 12, 13],
    )
    assert {"full_over_cap", "eos_lost", "incomplete_code", "intended_labels_mismatch"} <= set(row["defects"])
    assert row["diagnostics"]["native_algorithm_parity"] == 1
    assert row["lengths"] == {"prompt": 2, "full": 7, "target": 5, "contextual_target": 5}


def test_exact_cap_preserves_complete_code_and_eos():
    row = inspect(
        [1, 7], [1, 7, 10, 11, 12, 13, 2], [-100, -100, 10, 11, 12, 13, 2],
        limit=7, target=[10, 11, 12, 13, 2], code=[10, 11, 12, 13],
    )
    assert row["strict_clean"] is True
    assert row["defects"] == []


def test_real_unknown_zero_is_not_padding():
    row = inspect(
        [1, 7], [1, 7, 10, 0, 2], [-100, -100, 10, -100, 2, -100],
        native_ids=[1, 7, 10, 0, 2, 0], attention=[1, 1, 1, 1, 1, 0],
        target=[10, 0, 2],
    )
    assert {"unknown_full", "unknown_target", "intended_labels_mismatch"} <= set(row["defects"])
    assert row["diagnostics"]["full_unk_tokens"] == 1
    assert row["diagnostics"]["native_algorithm_parity"] == 1
    clean = inspect(
        [1, 7], [1, 7, 10, 11, 2], [-100, -100, 10, 11, 2, -100],
        native_ids=[1, 7, 10, 11, 2, 0], attention=[1, 1, 1, 1, 1, 0],
    )
    assert clean["strict_clean"] is True
    assert clean["diagnostics"]["full_unk_tokens"] == 0


def test_unknown_prompt_and_all_masked_rows_are_reported_separately():
    row = inspect(
        [1, 0, 7, 8], [1, 0, 7, 8, 10, 11, 2], [-100, -100, -100],
        native_ids=[1, 0, 7], limit=3,
    )
    assert {"unknown_prompt", "prompt_over_cap", "full_over_cap", "all_masked", "eos_lost"} <= set(row["defects"])
    assert row["diagnostics"]["native_algorithm_parity"] == 1


def test_empty_text_target_is_not_a_valid_eos_only_example():
    row = inspect([1, 7], [1, 7, 2], [-100, -100, 2], response=" \n", target=[2])
    assert "empty_response" in row["defects"]
    assert row["strict_clean"] is False


def test_bos_and_eos_policy_is_checked_without_truncation():
    row = inspect(
        [1, 7], [1, 7, 1, 10, 2, 2], [-100, -100, 1, 10, 2, 2],
        target=[1, 10, 2, 2],
    )
    assert {"full_bos_policy", "full_eos_policy", "target_special_token_policy"} <= set(row["defects"])


def test_native_masking_mismatch_is_not_hidden_by_a_clean_token_sequence():
    row = inspect([1, 7], [1, 7, 10, 11, 2], [-100, 7, 10, 11, 2])
    assert {"native_algorithm_mismatch", "intended_labels_mismatch"} <= set(row["defects"])
    assert row["diagnostics"]["native_algorithm_parity"] == 0


class RenderDataset:
    def __init__(self, mode="train", templates=17):
        self.mode = mode
        self.inter_data = [{"item": "A"}, {"item": "B"}]
        self.prompts = [{"instruction": str(i), "response": "{item}"} for i in range(templates)]

    def __getitem__(self, index):
        raise AssertionError("A random native __getitem__ draw cannot certify all variants")

    def _get_text_data(self, record, prompt):
        prefix = prompt["instruction"] + ":"
        response = prompt["response"].format(**record)
        return prefix, response if self.mode == "test" else prefix + response


@pytest.mark.parametrize("split", ["train", "valid"])
def test_all_seventeen_variants_are_counted_for_every_base_record(split):
    rows = list(audit().iter_rendered(RenderDataset(split), "seqrec", split, ["u:1", "v:1"]))
    assert len(rows) == 34
    assert Counter(row["record_id"] for row in rows) == {"u:1": 17, "v:1": 17}
    assert {row["prompt_id"] for row in rows} == set(range(17))
    assert all(row["full"] == row["prompt"] + row["response"] for row in rows)


def test_primary_test_uses_prompt_zero_and_retains_native_response_only_labels():
    rows = list(audit().iter_rendered(RenderDataset("test"), "seqrec", "test", ["u:1", "v:1"]))
    assert len(rows) == 2
    assert {row["prompt_id"] for row in rows} == {0}
    assert rows[0]["native"]["labels"] == "A"
    assert rows[0]["full"] == "0:A"


def test_auxiliary_template_count_comes_from_author_list_not_comments():
    dataset = RenderDataset(templates=19)
    dataset.feat_data = dataset.inter_data
    rows = list(audit().iter_rendered(dataset, "item2index", "train", ["0", "1"]))
    assert len(rows) == 38


def test_missing_base_record_is_rejected_before_rendering():
    with pytest.raises(ValueError, match="coverage"):
        list(audit().iter_rendered(RenderDataset(), "seqrec", "train", ["u:1"]))


def test_equal_total_does_not_hide_missing_or_duplicate_variant():
    coverage = audit().Coverage(0, 2, [0, 1])
    for pair in [(0, 0), (0, 1), (0, 1), (1, 1)]:
        coverage.observe(*pair)
    result = coverage.finish()
    assert result["expected"] == result["observed"] == 4
    assert result["status"] == "fail"
    assert result["order_mismatches"] > 0


def test_missing_partition_records_fail_coverage():
    coverage = audit().Coverage(10, 12, [0, 1])
    coverage.observe(10, 0)
    coverage.observe(10, 1)
    assert coverage.finish()["status"] == "fail"


def test_complete_partition_coverage_passes():
    coverage = audit().Coverage(10, 12, [0, 1])
    for pair in [(10, 0), (10, 1), (11, 0), (11, 1)]:
        coverage.observe(*pair)
    assert coverage.finish()["status"] == "pass"


def test_histogram_quantiles_are_exact_linear_quantiles():
    result = audit().length_quantiles({4: 1, 8: 1})
    assert result == {"p50": 6.0, "p90": 7.6, "p99": 7.96, "max": 8}


def test_memoization_retains_native_results_and_does_not_share_mutable_lists():
    calls = []

    def tokenize(text, **kwargs):
        calls.append((text, kwargs))
        return [text, str(kwargs.get("prefix", False))]

    tokenizer = SimpleNamespace(tokenize=tokenize)
    audit().memoize_tokenizer(tokenizer, 4)
    first = tokenizer.tokenize("x", prefix=True)
    first.append("mutation")
    assert tokenizer.tokenize("x", prefix=True) == ["x", "True"]
    assert tokenizer.tokenize("x", prefix=False) == ["x", "False"]
    assert len(calls) == 2


def test_eval_left_padding_is_checked_by_attention_not_unknown_id():
    row = audit().inspect_test_row(
        [1, 7], [1, 7, 10, 11, 12, 13, 2], [10, 11, 12, 13, 2],
        {"input_ids": [0, 1, 7], "attention_mask": [0, 1, 1]}, "code",
        response="code", limit=32, bos=1, eos=2, pad=0, unk=0,
        expected_code=[10, 11, 12, 13],
    )
    assert row["strict_clean"] is True
    assert row["diagnostics"]["native_test_parity"] == 1


def test_eval_truncation_and_target_changes_fail():
    row = audit().inspect_test_row(
        [1, 7, 8, 9], [1, 7, 8, 9, 10, 11, 12, 13, 2], [10, 11, 12, 13, 2],
        {"input_ids": [1, 7, 8], "attention_mask": [1, 1, 1]}, "wrong",
        response="code", limit=3, bos=1, eos=2, pad=0, unk=0,
        expected_code=[10, 11, 12, 13],
    )
    assert {"prompt_over_cap", "native_test_target_mismatch"} <= set(row["defects"])
    assert row["strict_clean"] is False


def test_statistics_keep_failed_rows_and_separate_algorithm_parity():
    module = audit()
    statistics = module.new_statistics()
    module.add_statistics(statistics, inspect(
        [1, 7], [1, 7, 10, 11, 2], [-100, -100, 10, 11, 2],
    ))
    module.add_statistics(statistics, inspect(
        [1, 7, 9], [1, 7, 10, 11, 2], [-100, -100, -100, 11, 2],
    ))
    report = module.summarize_statistics(statistics, expected=2)
    assert report["status"] == "fail"
    assert report["total"] == 2 and report["strict_clean"] == 1
    assert report["diagnostics"]["native_algorithm_parity"] == 2
    assert report["defects"]["prefix_mismatch"] == 1


def test_missing_clean_record_fails_even_with_no_token_defects():
    module = audit()
    statistics = module.new_statistics()
    module.add_statistics(statistics, inspect(
        [1, 7], [1, 7, 10, 11, 2], [-100, -100, 10, 11, 2],
    ))
    report = module.summarize_statistics(statistics, expected=2)
    assert report["status"] == "fail"
    assert report["missing_or_extra_records"] == -1


def test_partition_histograms_merge_without_averaging_quantiles():
    module = audit()
    parts = []
    for repetitions, width in [(9, 2), (1, 10)]:
        part = module.new_statistics()
        for _ in range(repetitions):
            module.add_statistics(part, {
                "strict_clean": True, "defects": [], "diagnostics": {},
                "lengths": {"prompt": width, "full": width + 1, "target": 1,
                            "contextual_target": 1},
            })
        parts.append(part)
    merged = module.merge_statistics(parts)
    report = module.summarize_statistics(merged, expected=10)
    assert report["status"] == "pass"
    assert report["length_tokens"]["prompt"] == {"p50": 2.0, "p90": pytest.approx(2.8),
                                               "p99": pytest.approx(9.28), "max": 10}


def test_audit_recipe_requires_native_response_only_and_unchanged_context():
    module = audit()
    spec = json.loads((ROOT / "configs" / "letter-openllama-v2-method-v1.json").read_text())
    manifest = json.loads((ROOT / "configs" / "letter-reproduction-v1.json").read_text())
    module.validate_audit_spec(spec, manifest)
    manifest["author_lc_rec_recipes"]["shared"]["only_train_response"] = False
    with pytest.raises(ValueError):
        module.validate_audit_spec(spec, manifest)


def test_receipts_are_write_once_and_config_fingerprints_are_order_independent(tmp_path):
    module = audit()
    path = tmp_path / "failure.json"
    module.write_once_json(path, {"status": "fail", "defects": {"full_over_cap": 1}})
    with pytest.raises(FileExistsError):
        module.write_once_json(path, {"status": "pass"})
    assert json.loads(path.read_text())["status"] == "fail"
    assert module.fingerprint({"a": 1, "b": 2}) == module.fingerprint({"b": 2, "a": 1})
    assert module.fingerprint({"a": 1}) != module.fingerprint({"a": 2})


def test_native_sequence_rows_must_match_canonical_history_and_targets():
    module = audit()
    indices = {"0": ["a"], "1": ["b"], "2": ["c"]}
    records = [{"user": "u", "position": 2, "history": [0, 1], "target": 2}]
    native = SimpleNamespace(inter_data=[{"inters": "a, b", "item": "c"}])
    assert module.verify_sequence_rows(native, records, indices) == ["u:2"]
    native.inter_data[0]["inters"] = "b, a"
    with pytest.raises(ValueError, match="canonical"):
        module.verify_sequence_rows(native, records, indices)
    native.inter_data = []
    with pytest.raises(ValueError, match="coverage"):
        module.verify_sequence_rows(native, records, indices)


def test_native_metadata_rows_preserve_raw_description_not_semantic_cleaning():
    module = audit()
    long_description = "raw " * 1000
    items = {"0": {"title": " title! ", "description": long_description}}
    indices = {"0": ["code"]}
    native = SimpleNamespace(
        item_feat={"0": {}},
        feat_data=[{"title": "title", "description": long_description, "item": "code"}],
    )
    assert module.verify_metadata_rows(native, items, indices) == ["0"]
    native.feat_data[0]["description"] = ""
    with pytest.raises(ValueError, match="metadata"):
        module.verify_metadata_rows(native, items, indices)


def test_cli_exposes_explicit_track_and_probe_is_not_corpus_admission():
    process = subprocess.run(
        [sys.executable, str(SCRIPTS / "lc_audit.py"), "--help"],
        text=True, capture_output=True, check=True,
    )
    assert "--method-spec" in process.stdout
    assert "--model-receipt" in process.stdout
    assert "--probe-base-records" in process.stdout
    assert "never admits a corpus" in process.stdout


@pytest.mark.skipif(not os.environ.get("LETTER_LC_MODEL"), reason="Pinned CPU native runtime required")
def test_native_slow_tokenizer_and_collator_are_identical_with_and_without_memoization():
    from author import default_args, load_backend

    module = audit()
    upstream = Path(os.environ["LETTER_UPSTREAM"])
    author = load_backend(upstream, "lc-rec")
    args = default_args(author, upstream, "Beauty")
    args.only_train_response = True
    dataset = author.data.SeqRecDataset(args, mode="train")
    tokenizer = module._tokenizer(Path(os.environ["LETTER_LC_MODEL"]), dataset.indices, 2048)
    rows = []
    for template in dataset.prompts:
        prompt, full = dataset._get_text_data(dataset.inter_data[0], template)
        rows.append({"input_ids": prompt, "labels": full})
    collator = author.collator.Collator(args, tokenizer)
    before = collator(rows)
    module.memoize_tokenizer(tokenizer, 256)
    after = collator(rows)
    repeated = collator(rows)
    for key in before:
        assert before[key].equal(after[key])
        assert before[key].equal(repeated[key])
    codes, trigger, inventory = module._inventory(tokenizer, dataset, dataset.indices)
    assert inventory["status"] == "pass"
    assert inventory["catalog_items_checked"] == 12101
    assert len(codes) == 12088
    assert trigger == tokenizer(rows[0]["input_ids"])["input_ids"][-len(trigger):]


@pytest.mark.skipif(not os.environ.get("LETTER_LC_MODEL"), reason="Pinned CPU native runtime required")
def test_native_boundary_unknown_empty_and_eos_truncation_are_detected():
    from author import default_args, load_backend

    module = audit()
    upstream = Path(os.environ["LETTER_UPSTREAM"])
    author = load_backend(upstream, "lc-rec")
    args = default_args(author, upstream, "Beauty")
    args.only_train_response = True
    dataset = author.data.SeqRecDataset(args, mode="train")
    tokenizer = module._tokenizer(Path(os.environ["LETTER_LC_MODEL"]), dataset.indices, 2048)
    prompt, _ = dataset._get_text_data(dataset.inter_data[0], dataset.prompts[0])
    prompt_ids = tokenizer(prompt, truncation=False)["input_ids"]
    code = "".join(dataset.indices["0"])
    expected_code = tokenizer.encode(code, add_special_tokens=False)
    cases = [
        ("<synthetic>", 2048, None, {"prefix_mismatch", "intended_labels_mismatch"}),
        ("<unk>synthetic", 2048, None, {"unknown_full", "unknown_target", "intended_labels_mismatch"}),
        ("", 2048, None, {"empty_response"}),
        (code, len(prompt_ids) + 4, expected_code, {"full_over_cap", "eos_lost", "incomplete_code"}),
    ]
    for response, limit, code_ids, expected_defects in cases:
        tokenizer.model_max_length = limit
        full = prompt + response
        full_ids = tokenizer(full + tokenizer.eos_token, truncation=False)["input_ids"]
        target_ids = tokenizer(
            response + tokenizer.eos_token, truncation=False, add_special_tokens=False,
        )["input_ids"]
        tensors = author.collator.Collator(args, tokenizer)([{"input_ids": prompt, "labels": full}])
        row = module.inspect_training_row(
            prompt_ids, full_ids, target_ids,
            {name: tensor[0].tolist() for name, tensor in tensors.items()},
            response=response, limit=limit, expected_code=code_ids,
        )
        assert expected_defects <= set(row["defects"])
        assert row["diagnostics"]["native_algorithm_parity"] == 1
        assert row["strict_clean"] is False
