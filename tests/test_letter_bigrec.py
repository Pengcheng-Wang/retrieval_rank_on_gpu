"""Contracts for the separately labelled, title-grounded BIGRec reference."""

import importlib
import json
import shutil
import sys
import uuid
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "letter"))


@pytest.fixture
def work():
    path = ROOT / ".local" / "letter" / "bigrec-tests" / uuid.uuid4().hex
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path)


def module(name):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        pytest.fail(f"BIGRec implementation is missing: {error.name}")


class CharacterTokenizer:
    bos_token_id, eos_token_id, pad_token_id = 1, 2, 0

    def __call__(self, text, **kwargs):
        assert not kwargs.get("truncation", False)
        ids = [ord(c) + 10 for c in text]
        if kwargs.get("add_special_tokens", True):
            ids.insert(0, self.bos_token_id)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def decode(self, ids, **kwargs):
        return "".join(chr(i - 10) for i in ids if i > 2)


def renderer(row):
    return "Instruction:\n" + row["instruction"] + "\nInput:\n" + row["input"] + "\nResponse:\n" + row["output"]


def test_recipe_freezes_native_budgets_and_declared_deviations():
    path = ROOT / "configs" / "letter-bigrec-compatible-v1.json"
    assert path.is_file(), "A frozen compatible BIGRec recipe must exist"
    recipe = json.loads(path.read_text())
    assert recipe["exact_original_weights_reproduction"] is False
    assert recipe["training"]["sample"] == 1024
    assert recipe["training"]["selection_seed"] == recipe["training"]["initialization_seed"] == 42
    assert recipe["training"]["epochs"] == 50
    assert recipe["training"]["micro_batch"] == 4
    assert recipe["training"]["accumulation"] == 32
    assert recipe["training"]["warmup_steps"] == 20
    assert recipe["training"]["lora"]["target_modules"] == ["q_proj", "v_proj"]
    assert recipe["training"]["train_on_inputs"] is True
    assert recipe["generation"]["num_beams"] == recipe["generation"]["num_return_sequences"] == 4
    assert recipe["generation"]["max_new_tokens"] == 128
    assert recipe["generation"]["do_sample"] is False
    assert recipe["grounding"]["adapted_base"] is False
    assert recipe["grounding"]["distance"] == "unnormalized_l2_float32"
    assert recipe["grounding"]["seen_item_exclusion"] is False
    assert recipe["data"]["context_tokens"] == 2048


def test_native_prompt_is_loaded_without_executing_top_level_code(work):
    source = module("bigrec_source")
    path = work / "unsafe.py"
    path.write_text('raise RuntimeError("do not execute")\n'
                    'def generate_prompt(row):\n    return row["output"]\n')
    import hashlib
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert source.load_pure_renderer(path, digest)({"output": "safe"}) == "safe"
    with pytest.raises(ValueError, match="checksum"):
        source.load_pure_renderer(path, "0" * 64)


def test_verified_native_training_renderer_removes_inference_space_drift():
    source = module("bigrec_source")
    native = ROOT / ".local" / "letter" / "bigrec-native"
    if not (native / "train.py").is_file():
        pytest.skip("Separately acquired private BIGRec source required")
    train = source.load_pure_renderer(native / "train.py", source.SOURCE_HASHES["train.py"])
    infer = source.load_pure_renderer(native / "inference.py", source.SOURCE_HASHES["inference.py"])
    row = {"instruction": "recommend", "input": 'history:"a"', "output": '"b"'}
    prompt = train({**row, "output": ""})
    assert train(row) == prompt + '"b"'
    assert "request. \n" in prompt
    assert "request.  \n" in infer(row["instruction"], row["input"])
    assert prompt != infer(row["instruction"], row["input"])


def test_catalog_keeps_duplicate_and_empty_title_ids():
    data = module("bigrec_data")
    catalog = data.catalog_rows({
        "2": {"title": ""}, "0": {"title": "same"}, "1": {"title": "same"},
    })
    assert catalog == [
        {"item": 0, "title": "same"}, {"item": 1, "title": "same"}, {"item": 2, "title": ""},
    ]
    with pytest.raises(ValueError, match="title"):
        data.catalog_rows({"0": {"description": "not a substitute"}})
    with pytest.raises(ValueError, match="contiguous"):
        data.catalog_rows({"1": {"title": "gap"}})


def test_whole_item_shortening_is_target_blind_and_preserves_full_labels():
    data = module("bigrec_data")
    tokenizer = CharacterTokenizer()
    titles = ["old" * 80, 'a"b', "", "new"]
    empty = data.render_fields("Instruments", [], titles, None)
    base_len = len(tokenizer(renderer(empty))["input_ids"])
    reserve = 50
    context = base_len + reserve + 22
    record = {"user": "7", "position": 4, "history": [0, 1, 2], "target": 3}
    row = data.tokenize_record(record, "Instruments", titles, renderer, tokenizer, context, reserve)
    second = data.tokenize_record({**record, "target": 1}, "Instruments", titles,
                                  renderer, tokenizer, context, reserve)
    assert row["history"] == second["history"] == [1, 2]
    assert row["dropped_history_items"] == 1
    assert row["prompt_ids"] == second["prompt_ids"]
    assert row["input_ids"][:-1][:len(row["prompt_ids"])] == row["prompt_ids"]
    assert row["input_ids"] == row["labels"]
    assert row["input_ids"][-1] == tokenizer.eos_token_id
    assert tokenizer.decode(row["input_ids"]).endswith('"new"')
    assert len(row["input_ids"]) <= context
    assert row["finish_reason"] == "N/A"
    assert '""' in tokenizer.decode(row["prompt_ids"])
    assert row["target"] == 3


def test_overlong_target_fails_instead_of_truncating_or_substituting():
    data = module("bigrec_data")
    tokenizer = CharacterTokenizer()
    titles = ["a", "x" * 2200]
    with pytest.raises(ValueError, match="catalog target.*context"):
        data.catalog_reserve("Beauty", titles, renderer, tokenizer, 2048, 128)


def test_native_cutoff_counts_missing_eos_at_exact_boundary():
    data = module("bigrec_data")
    assert data.native_cutoff_diagnostic([511, 512, 513], 512) == {
        "records": 3, "missing_eos": 2, "right_truncated": 1,
    }


def test_training_selection_matches_two_native_huggingface_shuffles():
    data = module("bigrec_data")
    from datasets import Dataset
    expected = Dataset.from_dict({"row": list(range(41))}).shuffle(seed=42).select(range(13)).shuffle(seed=42)["row"]
    assert data.training_selection(41, 13, 42) == list(expected)
    assert data.validation_selection(41, 13, 42) == __import__("pandas").Series(range(41)).sample(n=13, random_state=42).tolist()


def test_all_user_coverage_is_exact_not_intersection():
    data = module("bigrec_data")
    data.verify_user_rows(["0", "1"], [{"user": "0"}, {"user": "1"}])
    with pytest.raises(ValueError, match="coverage"):
        data.verify_user_rows(["0", "1"], [{"user": "0"}, {"user": "0"}])


def test_full_catalog_l2_retains_duplicate_targets_and_stable_original_id_ties():
    evaluation = module("bigrec_evaluate")
    import torch
    catalog = torch.tensor([[1., 0.], [1., 0.], [0., 3.], [0., 0.]], dtype=torch.float16)
    queries = torch.tensor([[1., 0.], [0., 0.]], dtype=torch.float16)
    ranks, distances = evaluation.rank_l2(queries, catalog)
    assert ranks.tolist() == [[0, 1, 3, 2], [3, 0, 1, 2]]
    assert distances.dtype == torch.float32
    assert evaluation.target_metrics(ranks[0].tolist(), 1)["ndcg@1"] == 0
    assert evaluation.target_metrics(ranks[0].tolist(), 1)["recall@5"] == 1
    expected = torch.linalg.vector_norm(queries.float()[:, None] - catalog.float()[None], dim=-1)
    torch.testing.assert_close(distances, expected, rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError, match="finite"):
        evaluation.rank_l2(torch.tensor([[float("nan"), 0.]]), catalog)


def test_pooling_selects_final_real_token_with_left_padding():
    evaluation = module("bigrec_evaluate")
    import torch
    hidden = torch.arange(24).reshape(2, 4, 3).float()
    mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    torch.testing.assert_close(evaluation.last_token_pool(hidden, mask), hidden[:, -1])
    with pytest.raises(ValueError, match="left"):
        evaluation.last_token_pool(hidden, torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]]))


def test_finish_reason_uses_generated_tokens_not_decoded_text():
    evaluation = module("bigrec_evaluate")
    tokenizer = CharacterTokenizer()
    eos = evaluation.decode_tail([1, 100, 77, 2, 0], 2, tokenizer, 4)
    budget = evaluation.decode_tail([1, 100, 30, 31, 32, 33], 2, tokenizer, 4)
    assert eos["finish_reason"] == "eos" and eos["generated_tokens"] == 2
    assert eos["token_ids"] == [77, 2]
    assert budget["finish_reason"] == "length" and budget["generated_tokens"] == 4
    with pytest.raises(ValueError, match="unexplained"):
        evaluation.decode_tail([1, 100, 30], 2, tokenizer, 4)
    assert evaluation.grounding_query('""a""') == "a"
    assert evaluation.grounding_query(' "a"\n') == ' "a"\n'


def test_checkpoint_requires_scaler_and_detects_tampering(work):
    checkpoint = module("bigrec_checkpoint")
    files = {
        "adapter_model.safetensors": b"private-adapter",
        "adapter_config.json": b"{}",
        "optimizer.pt": b"trusted-local",
        "scheduler.pt": b"trusted-local",
        "rng_state.pth": b"trusted-local",
        "trainer_state.json": b'{"global_step": 5, "epoch": 0.625}',
        "continuation.json": b'{"sampler": {}, "early_stopping_patience_counter": 0, "evaluations":0,'
                             b'"microbatches_per_epoch":256,"accumulation":32,"microbatches":160}',
    }
    for name, content in files.items():
        (work / name).write_bytes(content)
    with pytest.raises(ValueError, match="scaler"):
        checkpoint.seal(work)
    (work / "scaler.pt").write_bytes(b"trusted-local")
    checkpoint.seal(work)
    assert checkpoint.verify(work)["global_step"] == 5
    (work / "scaler.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        checkpoint.verify(work)


def test_cached_native_sampler_matches_fresh_order_and_cold_continuation():
    execution = module("bigrec_train")
    import torch
    from transformers.trainer_pt_utils import LengthGroupedSampler
    lengths = list(range(1, 34))
    torch.manual_seed(42)
    expected0 = list(LengthGroupedSampler(8, lengths=lengths))
    expected1 = list(LengthGroupedSampler(8, lengths=lengths))
    expected_rng = torch.get_rng_state()
    torch.manual_seed(42)
    sampler = execution.recording_sampler(8, lengths)
    assert list(sampler) == expected0
    assert list(sampler) == expected1
    assert torch.equal(torch.get_rng_state(), expected_rng)
    restored = execution.recording_sampler(8, lengths, sampler.export_state())
    torch.manual_seed(444)
    assert list(restored) == expected0
    assert list(restored) == expected1
    assert restored.export_state()["epoch_indices"] == sampler.export_state()["epoch_indices"]


def test_training_options_do_not_shorten_the_gate_schedule():
    execution = module("bigrec_train")
    recipe = json.loads((ROOT / "configs" / "letter-bigrec-compatible-v1.json").read_text())
    assert hasattr(execution, "training_options"), "Native training options are not implemented"
    options = execution.training_options(recipe)
    assert options["num_train_epochs"] == 50 and "max_steps" not in options
    assert options["learning_rate"] == 1e-4
    assert options["lr_scheduler_type"] == "linear"
    assert options["warmup_steps"] == 20 and options["fp16"]
    assert options["gradient_accumulation_steps"] == 32
    assert options["evaluation_strategy"] == options["save_strategy"] == "epoch"
    assert options["seed"] == 42 and options["group_by_length"]
    assert options["weight_decay"] == 0 and options["load_best_model_at_end"]


def test_artifact_export_is_hash_bound_and_idempotent(work):
    banking = module("bigrec_bank")
    source = work / "payload"
    source.mkdir()
    (source / "data.json").write_text('{"private":"fixture"}')
    banking.seal_payload(source, "fixture", {"track": "bigrec"})
    destination = work / "durable"
    first = banking.export_payload(source, destination)
    assert first == banking.export_payload(source, destination)
    assert (destination / "bank-receipt.json").is_file()
    (source / "data.json").write_text('{"private":"changed"}')
    with pytest.raises(ValueError, match="checksum"):
        banking.export_payload(source, destination)


def test_staging_payload_is_immutable_and_acknowledgment_binds_its_seal(work):
    banking = module("bigrec_bank")
    source = work / "source"
    source.mkdir()
    (source / "data.json").write_text("{}")
    banking.seal_payload(source, "fixture", {"track": "bigrec"})
    request = banking.stage_payload(source, work / "outbox", "test-fixture")
    assert banking.verify_payload(Path(request["payload"]))["kind"] == "fixture"
    assert request["complete_sha256"] == __import__("hashlib").sha256(
        (Path(request["payload"]) / "complete.json").read_bytes()).hexdigest()
    (source / "data.json").write_text('{"changed":true}')
    assert (Path(request["payload"]) / "data.json").read_text() == "{}"


def test_model_gate_refuses_gpu_work_without_full_corpus_admission(work):
    gate = module("bigrec_gate")
    with pytest.raises((ValueError, FileNotFoundError)):
        gate.model_gate(work / "not-admitted", work / "no-model", work / "output")


def test_interrupted_durable_copy_can_resume_without_acknowledging_partial_files(work):
    banking = module("bigrec_bank")
    source, destination = work / "payload", work / "durable"
    source.mkdir()
    (source / "large.dat").write_bytes(b"complete private fixture")
    banking.seal_payload(source, "fixture", {})
    destination.mkdir()
    (destination / "large.dat").write_bytes(b"partial")
    banking.export_payload(source, destination)
    assert (destination / "large.dat").read_bytes() == b"complete private fixture"
    assert (destination / "bank-receipt.json").is_file()


def test_pipeline_restarts_only_its_drained_or_preempted_stage():
    pipeline = module("bigrec_pipeline")
    assert pipeline.retryable_exit(-15, False)
    assert pipeline.retryable_exit(-9, False)
    assert pipeline.retryable_exit(0, False)
    assert not pipeline.retryable_exit(1, False)
    assert not pipeline.retryable_exit(-15, True)


def test_generation_logit_retention_preserves_values_without_retaining_prefill_storage():
    evaluation = module("bigrec_evaluate")
    assert hasattr(evaluation, "compact_generation_output"), "Generation must not retain unused prefill logits"
    import torch
    from types import SimpleNamespace

    original = torch.arange(2 * 17 * 5).reshape(2, 17, 5).float()
    cache = object()
    output = SimpleNamespace(logits=original, past_key_values=cache, loss=None)
    result = evaluation.compact_generation_output(None, (), output)
    torch.testing.assert_close(result.logits, original[:, -1:, :], rtol=0, atol=0)
    assert result.logits.data_ptr() != original[:, -1:, :].data_ptr()
    assert result.logits.numel() == 10 and result.past_key_values is cache
    with pytest.raises(ValueError, match="generation"):
        evaluation.compact_generation_output(None, (), SimpleNamespace(logits=original, loss=torch.tensor(1.)))


def test_title_equivalence_diagnostics_do_not_replace_original_item_hits():
    evaluation = module("bigrec_evaluate")
    assert hasattr(evaluation, "title_diagnostics"), "Title ambiguity needs explicitly separate diagnostics"
    ranking = list(range(149))
    titles = ["same business"] * 149
    original = evaluation.target_metrics(ranking, 148)
    diagnostic = evaluation.title_diagnostics(ranking, 148, titles)
    assert original["recall@10"] == 0
    assert diagnostic["catalog_title_equivalence_metrics"]["recall@10"] == 1
    assert diagnostic["target_title_group_size"] == 149
    assert diagnostic["best_same_title_rank_zero_based"] == 0
    assert len(ranking) == 149


def test_generation_shard_rejects_another_selected_model(work):
    evaluation = module("bigrec_evaluate")
    banking = module("bigrec_bank")
    assert hasattr(evaluation, "verify_generation_shard"), "Generation chunks must bind the selected model"
    payload = work / "generation"
    payload.mkdir()
    rows = [{"user": "0", "target": 1, "prompt_sha256": "test", "beams": [{}, {}, {}, {}]}]
    (payload / "rows.jsonl").write_text(json.dumps(rows[0]) + "\n")
    banking.seal_payload(payload, "bigrec_four_beam_generated_titles", {
        "selected_sha256": "other-model", "start": 0, "users": ["0"],
    })
    with pytest.raises(ValueError, match="binding"):
        evaluation.verify_generation_shard(payload, {"selected_sha256": "selected-model"}, 0, ["0"])


def test_generation_memory_scope_releases_only_obsolete_output_references():
    evaluation = module("bigrec_evaluate")
    assert hasattr(evaluation, "generation_memory_scope"), "Generation output lifetimes need a scoped guard"
    import torch
    from transformers.modeling_outputs import CausalLMOutputWithPast

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_cache = None

        def forward(self, ids, past_key_values=None):
            self.seen_cache = past_key_values
            return CausalLMOutputWithPast(logits=ids[..., None].repeat(1, 1, 3).float(),
                                          past_key_values=(ids.clone(),), loss=None)

        @staticmethod
        def _reorder_cache(past, indices):
            return tuple(tuple(v.index_select(0, indices) for v in layer) for layer in past)

    model = Model().eval()
    ids = torch.tensor([[1, 2, 3], [1, 4, 5]])
    with evaluation.generation_memory_scope(model):
        first = model(ids)
        cache = first.past_key_values
        second = model(ids[:, -1:], past_key_values=cache)
        assert first.past_key_values is None
        assert first["past_key_values"] is None
        assert model.seen_cache is cache
        torch.testing.assert_close(cache[0], ids, rtol=0, atol=0)
        assert second.past_key_values is not None
        assert first.logits.shape == (2, 1, 3)
    assert not model._forward_hooks and not model._forward_pre_hooks
    assert "_reorder_cache" not in model.__dict__
    assert model(ids).logits.shape == (2, 3, 3)


def test_resume_evidence_reuse_allows_only_separately_gated_evaluation_changes():
    gate = module("bigrec_gate")
    assert hasattr(gate, "same_training_source"), "Training proof reuse must be dependency-scoped"
    old = {
        "bigrec_train.py": "train", "bigrec_checkpoint.py": "state",
        "bigrec_source.py": "source", "bigrec_data.py": "data", "bigrec_evaluate.py": "old-eval",
    }
    assert gate.same_training_source(old, {**old, "bigrec_evaluate.py": "memory-lifetime-only"})
    assert not gate.same_training_source(old, {**old, "bigrec_train.py": "changed-train"})
    assert not gate.same_training_source(old, {**old, "bigrec_data.py": "changed-data"})


def test_in_place_beam_cache_permutation_matches_native_with_repeated_parents():
    evaluation = module("bigrec_evaluate")
    assert hasattr(evaluation, "reorder_generation_cache"), "Beam reordering must avoid a third complete KV copy"
    import torch

    past = tuple((torch.arange(4 * 2 * 3 * 5).reshape(4, 2, 3, 5).float() + layer * 1000,
                  torch.arange(4 * 2 * 3 * 5).reshape(4, 2, 3, 5).float() - layer * 1000)
                 for layer in range(3))
    indices = torch.tensor([2, 2, 0, 1])
    expected = tuple(tuple(t.index_select(0, indices) for t in layer) for layer in past)
    pointers = [[t.data_ptr() for t in layer] for layer in past]
    with torch.inference_mode():
        result = evaluation.reorder_generation_cache(past, indices)
    assert result is past
    assert pointers == [[t.data_ptr() for t in layer] for layer in result]
    for actual_layer, expected_layer in zip(result, expected):
        for actual, target in zip(actual_layer, expected_layer):
            torch.testing.assert_close(actual, target, rtol=0, atol=0)


def test_epoch_boundary_checkpoint_waits_for_validation_before_publication():
    execution = module("bigrec_train")
    assert hasattr(execution, "save_on_step_end"), "Epoch-boundary snapshots must wait for validation"
    assert execution.save_on_step_end(4, False, 4, 8)
    assert execution.save_on_step_end(5, True, 4, 8)
    assert not execution.save_on_step_end(8, False, 4, 8)
    assert not execution.save_on_step_end(8, True, 4, 8)
    assert not execution.save_on_step_end(400, True, 4, 8)


def test_prevalidation_checkpoint_is_not_a_complete_continuation_state(work):
    checkpoint = module("bigrec_checkpoint")
    for name in checkpoint.REQUIRED:
        (work / name).write_bytes(b"fixture")
    (work / "trainer_state.json").write_text('{"global_step":8,"epoch":1.0}')
    (work / "continuation.json").write_text(
        '{"sampler":{},"early_stopping_patience_counter":0,"evaluations":0,'
        '"microbatches_per_epoch":256,"accumulation":32,"microbatches":256}')
    with pytest.raises(ValueError, match="validation"):
        checkpoint.seal(work)


def test_pipeline_preserves_but_quarantines_an_interrupted_checkpoint_seal(work):
    pipeline = module("bigrec_pipeline")
    broken = work / "checkpoint-8"
    broken.mkdir()
    (broken / "complete.json").write_text("{}")
    pipeline.quarantine_unsealed_checkpoints(work)
    assert not broken.exists()
    assert len(list(work.glob(".incomplete-checkpoint-8-*/complete.json"))) == 1


def test_chained_cold_processes_cover_every_frozen_interruption_position(work):
    gate = module("bigrec_gate")
    assert hasattr(gate, "continuation_cases"), "The production multi-preemption chain must be explicit"
    (work / "chained").mkdir()
    cases = gate.continuation_cases(work)
    assert cases == [(work / "chained", (0, 5, 8, 9), (5, 8, 9))]


def test_metadata_payload_keeps_nested_completion_receipts_in_its_hash_inventory(work):
    banking = module("bigrec_bank")
    nested = work / "nested"
    nested.mkdir()
    (nested / "complete.json").write_text('{"status":"fixture"}')
    (work / "summary.json").write_text("{}")
    receipt = banking.seal_payload(work, "metadata", {})
    assert "nested/complete.json" in receipt["files"]
    assert banking.verify_payload(work) == receipt


def test_exporter_prioritizes_checkpoints_ahead_of_catalog_fragments(work):
    banking = module("bigrec_bank")
    for identity, name in (
        ("a-catalog", "bigrec-Beauty-catalog-unique-00000000"),
        ("z-state", "chained-checkpoint-5"),
    ):
        folder = work / identity
        folder.mkdir()
        (folder / "ready.json").write_text(json.dumps({
            "identity": identity, "name": name, "complete_sha256": "0" * 64,
        }))
    assert banking.pending_requests(work)[0]["identity"] == "z-state"


def test_pipeline_waits_for_every_own_expensive_export_not_only_summary(work):
    pipeline = module("bigrec_pipeline")
    assert hasattr(pipeline, "pending_run_exports"), "A summary acknowledgment cannot hide unbanked predictions"
    for identity, name in (("summary", "run-A-evaluation"), ("predictions", "run-A-generation-00000000"),
                           ("unrelated", "run-B-generation-00000000")):
        folder = work / identity
        folder.mkdir()
        (folder / "ready.json").write_text(json.dumps({
            "identity": identity, "name": name, "complete_sha256": "0" * 64,
        }))
    (work / "summary" / "banked.json").write_text(json.dumps({
        "status": "banked", "identity": "summary", "complete_sha256": "0" * 64,
        "bank_receipt_sha256": "1" * 64,
    }))
    assert pipeline.pending_run_exports(work, "run-A") == ["predictions"]
