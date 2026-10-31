import copy
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "letter"
RECIPE = ROOT / "configs" / "letter-conventional-baselines-v1.json"
sys.path.insert(0, str(SCRIPTS))


def baseline():
    assert (SCRIPTS / "baselines.py").is_file(), "Native baseline adapter is missing"
    return importlib.import_module("baselines")


@pytest.fixture
def workspace():
    path = ROOT / ".local" / "letter" / "baseline-unit-tests" / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path)


def small_release():
    return baseline().release_from_interactions(
        {"7": [0, 1, 0, 2, 9, 10], "3": [2, 3, 4, 8, 11]}, 12
    )


def caser_source():
    path = os.environ.get("LETTER_CASER")
    if not path:
        pytest.skip("Set LETTER_CASER to the private pinned original Caser source")
    return Path(path)


def test_mapping_preserves_original_users_repeats_and_future_isolation():
    data = small_release()
    assert data.users == ["3", "7"]
    assert data.train == {1: [3, 4, 5], 2: [1, 2, 1, 3]}
    assert data.valid == {1: 9, 2: 10}
    assert data.test == {1: 12, 2: 11}
    assert data.histories("valid") == data.train
    assert data.histories("test") == {1: [3, 4, 5, 9], 2: [1, 2, 1, 3, 10]}
    changed = baseline().release_from_interactions(
        {"7": [0, 1, 0, 2, 4, 5], "3": [2, 3, 4, 0, 1]}, 12
    )
    assert changed.train == data.train


@pytest.mark.parametrize("interactions,items", [
    ({}, 12), ({"a": [0, 1, 2, 3, 4]}, 12),
    ({"1": [0, 1, 2, 3]}, 12), ({"1": [0, 1, 2, 3, 12]}, 12),
    ({"1": [0, 1, 2, 3, -1]}, 12), ({"1": [0, 1, 2, 3, True]}, 12),
    ({"1": [0, 1, 2, 3, 4.0]}, 12), ({"1": [0, 1, 2, 3, 4]}, 0),
    ({"1": [0, 1, 2, 3, 4], "01": [0, 1, 2, 3, 4]}, 12),
])
def test_malformed_release_is_rejected(interactions, items):
    with pytest.raises(ValueError):
        baseline().release_from_interactions(interactions, items)


def test_train_view_is_binary_and_contains_first_events_not_heldouts():
    data = small_release()
    view = baseline().TrainView(data.train, data.items)
    assert view.num("user_id") == 3 and view.num("item_id") == 13
    graph = view.inter_matrix(form="coo")
    assert graph.shape == (3, 13)
    assert set(zip(graph.row, graph.col)) == {
        (1, 3), (1, 4), (1, 5), (2, 1), (2, 2), (2, 3)
    }
    np.testing.assert_array_equal(graph.data, np.ones(6))
    with pytest.raises(KeyError):
        view.num("made_up_field")


def test_negative_sampling_only_excludes_train_and_restores_draws():
    data = small_release()
    pool = baseline().TrainingNegatives(data.train, data.items)
    rng = np.random.RandomState(42)
    state = rng.get_state()
    draws = pool.draw(rng, 1, 1000)
    assert set(draws) == set(range(1, 13)) - set(data.train[1])
    assert data.valid[1] in draws and data.test[1] in draws
    rng.set_state(state)
    np.testing.assert_array_equal(draws, pool.draw(rng, 1, 1000))
    with pytest.raises(ValueError, match="negative"):
        baseline().TrainingNegatives({1: [1, 2, 3]}, 3)


def test_training_records_keep_event_multiplicity_and_complete_bert_prefixes():
    data = small_release()
    module = baseline()
    pairs = module.training_records(data, "bpr", 0)
    assert len(pairs) == 7
    assert [r["target"] for r in pairs if r["user_id"] == 2] == [1, 2, 1, 3]
    hgn = module.training_records(data, "hgn", 20)
    bert = module.training_records(data, "bert4rec", 20)
    assert len(hgn) == len(bert) == 5
    assert hgn[-1] == {"user_id": 2, "history": [1, 2, 1], "target": 3}
    assert bert[-1] == {"user_id": 2, "history": [1, 2, 1, 3], "target": 3}
    for row in bert:
        assert row["history"][-1] == row["target"]
        assert set(row["history"]) <= set(data.train[row["user_id"]])


def test_bert_terminal_masks_align_with_native_inference_positions():
    module = baseline()
    data = small_release()
    records = module.training_records(data, "bert4rec", 20)
    batch = module.bert_mask_batch(
        [r["history"] for r in records], [r["user_id"] for r in records],
        module.TrainingNegatives(data.train, data.items), np.random.RandomState(42),
        ft_ratio=1.0,
    )
    assert set(batch) == {"Mask_item_id_list", "Pos_item_id", "Neg_item_id", "MASK_INDEX"}
    assert batch["Mask_item_id_list"].shape == (5, 21)
    assert batch["MASK_INDEX"].shape == (5, 4)
    for row, record in enumerate(records):
        effective = batch["MASK_INDEX"][row] > 0
        assert effective.sum() == 1
        index = batch["MASK_INDEX"][row, effective][0]
        assert index == len(record["history"]) - 1
        assert batch["Pos_item_id"][row, effective][0] == record["target"]
        assert batch["Mask_item_id_list"][row, index] == data.items + 1
        assert batch["Mask_item_id_list"][row, 0] == record["history"][0]
        negative = batch["Neg_item_id"][row, effective][0]
        assert negative not in data.train[record["user_id"]]


def test_bert_training_retains_twenty_context_items_plus_terminal_target():
    module = baseline()
    data = module.release_from_interactions({"0": list(range(29))}, 32)
    records = module.training_records(data, "bert4rec", 20)
    assert records[-1]["history"] == list(range(7, 28))
    batch = module.bert_mask_batch(
        [records[-1]["history"]], [1], module.TrainingNegatives(data.train, data.items),
        np.random.RandomState(42), ft_ratio=1.0,
    )
    assert batch["Mask_item_id_list"].tolist() == [list(range(7, 27)) + [33]]
    assert batch["MASK_INDEX"][0, 0] == 20
    assert batch["Pos_item_id"][0, 0] == 27


def test_bert_zero_draws_get_an_effective_terminal_mask_and_bound_capacity():
    module = baseline()
    pool = module.TrainingNegatives({1: list(range(1, 27))}, 32)
    batch = module.bert_mask_batch(
        [[1, 2], list(range(1, 27))], [1, 1], pool, np.random.RandomState(4),
        mask_ratio=0.05, ft_ratio=0.0,
    )
    assert np.all((batch["MASK_INDEX"] > 0).sum(axis=1) == 1)
    assert np.all(batch["MASK_INDEX"] < 21)
    assert batch["Pos_item_id"][0, 0] == 2
    with pytest.raises(ValueError, match="mask"):
        module.bert_mask_batch([[1, 2]], [1], pool, np.random.RandomState(0),
                               mask_ratio=0.001)
    with pytest.raises(ValueError):
        module.bert_mask_batch([[]], [1], pool, np.random.RandomState(0))


def test_bert_inference_sentinel_retains_twenty_history_items_and_mask_slot():
    inputs = baseline().bert_inference_inputs([[1, 2, 3], list(range(1, 27))])
    assert inputs["item_id_list"].shape == (2, 21)
    assert inputs["item_length"].tolist() == [4, 21]
    assert inputs["item_id_list"][0, :5].tolist() == [0, 1, 2, 3, 0]
    assert inputs["item_id_list"][1].tolist() == [0] + list(range(7, 27))
    for row, length in zip(inputs["item_id_list"], inputs["item_length"]):
        native_input = np.append(row, 0)
        native_input[length] = 33
        rebuilt = native_input[1:]
        assert rebuilt[length - 1] == 33
    with pytest.raises(ValueError):
        baseline().bert_inference_inputs([[]])


def test_original_caser_windows_are_one_based_newest_first_and_left_padded():
    source = caser_source()
    records = baseline().caser_records(
        {1: list(range(1, 12)), 2: list(range(1, 8)), 3: [2, 3, 4]}, source
    )
    assert len(records) == 6
    assert records[0] == {"user_id": 1, "history": [4, 5, 6, 7, 8],
                          "targets": [9, 10, 11]}
    assert records[3]["history"] == [1, 2, 3, 4, 5]
    assert records[4] == {"user_id": 2, "history": [0, 1, 2, 3, 4],
                          "targets": [5, 6, 7]}
    assert records[5] == {"user_id": 3, "history": [0, 0, 0, 0, 0],
                          "targets": [2, 3, 4]}
    with pytest.raises(ValueError, match="three"):
        baseline().caser_records({1: [1, 2]}, source)


def test_caser_inference_uses_actual_last_five_observed_items():
    rows = baseline().padded_histories([[1, 2], list(range(1, 12))], 5, left=True)
    assert rows.tolist() == [[0, 0, 0, 1, 2], [7, 8, 9, 10, 11]]


def test_sampled_bce_is_sum_of_separate_means_and_is_numerically_stable():
    positive = torch.tensor([[2.0, -3.0, 1000.0]], requires_grad=True)
    negative = torch.tensor([[-2.0, 1.0]], requires_grad=True)
    loss = baseline().sampled_bce(positive, negative)
    expected = torch.nn.functional.softplus(-positive).mean()
    expected += torch.nn.functional.softplus(negative).mean()
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    loss.backward()
    assert torch.isfinite(positive.grad).all() and torch.isfinite(negative.grad).all()
    with pytest.raises(ValueError):
        baseline().sampled_bce(positive[:, :0], negative)


def test_ranking_only_excludes_padding_and_uses_original_id_ties():
    raw = torch.tensor([[1e6, 4.0, 7.0, 7.0, 7.0, 3.0]])
    scores = baseline().mask_padding(raw, 5)
    assert torch.isneginf(scores[:, 0]).all()
    assert raw[0, 0] == 1e6
    assert baseline().top_items(scores, 5).tolist() == [[1, 2, 3, 0, 4]]
    assert baseline().top_items(baseline().mask_padding(torch.zeros(1, 6), 5), 3).tolist() == [
        [0, 1, 2]
    ]


@pytest.mark.parametrize("scores", [
    torch.tensor([[0.0, float("nan")]]),
    torch.tensor([[float("inf"), 1.0]]),
    torch.tensor([[0.0, float("-inf")]]),
])
def test_nonfinite_catalog_scores_fail_before_padding_mask(scores):
    with pytest.raises(ValueError, match="finite"):
        baseline().mask_padding(scores, 1)


def test_coverage_and_original_item_metrics_include_repeat_targets():
    rows = [{"user": "7", "top10": list(range(10))},
            {"user": "3", "top10": list(reversed(range(10)))}]
    summary = baseline().aggregate_predictions(["3", "7"], {"3": 9, "7": 0}, rows, 12)
    assert summary == {"recall@5": 1.0, "ndcg@5": 1.0, "recall@10": 1.0, "ndcg@10": 1.0}
    for invalid in [rows[:1], rows + rows[:1], [rows[0], {"user": "8", "top10": list(range(10))}]]:
        with pytest.raises(ValueError, match="coverage"):
            baseline().aggregate_predictions(["3", "7"], {"3": 9, "7": 0}, invalid, 12)
    with pytest.raises(ValueError):
        baseline().aggregate_predictions(
            ["7"], {"7": 0}, [{"user": "7", "top10": [0] * 10}], 12
        )


@pytest.mark.parametrize("name", ["bpr", "lightgcn", "hgn", "bert4rec", "sasrec"])
def test_epoch_sampler_continuation_repeats_all_next_epoch_batches(name):
    module = baseline()
    recipe = module.load_recipe(RECIPE)
    settings = copy.deepcopy(recipe["models"][name])
    settings["batch_size"] = 2
    sampler = module.EpochSampler(small_release(), name, settings, 17)
    first = list(sampler.batches())
    assert first
    state = copy.deepcopy(sampler.state())
    expected = list(sampler.batches())
    sampler.restore(state)
    actual = list(sampler.batches())
    assert len(actual) == len(expected)
    for left, right in zip(expected, actual):
        assert left.keys() == right.keys()
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])
    assert sampler.state()["epochs"] == 2


def test_recipe_declares_reconstruction_and_exact_original_model_contracts():
    recipe = baseline().load_recipe(RECIPE)
    assert recipe["status"] == "declared-reconstruction-not-recovered-letter-settings"
    assert recipe["models"]["sasrec"]["architecture"]["hidden_units"] == 50
    assert recipe["models"]["caser"]["architecture"]["T"] == 3
    assert recipe["models"]["hgn"]["architecture"]["loss_type"] == "CE"
    assert recipe["models"]["bert4rec"]["architecture"]["ft_ratio"] == 0.5
    assert recipe["models"]["lightgcn"]["architecture"]["require_pow"] is False
    assert recipe["sources"]["recbole"]["commit"] == "362d31f00af801d7d99bc635c902d1df1405e79d"
    assert recipe["sources"]["caser"]["license"] == "LGPL-3.0"
    assert recipe["training"]["selection_metric"] == "ndcg@10"
    assert recipe["protocol"]["seen_item_exclusion"] is False


@pytest.mark.parametrize("mutation", [
    "teacher32", "hgn_bpr", "source", "unknown", "nan", "budget", "schema_bool", "no_declarations"
])
def test_invalid_or_drifting_recipe_is_rejected(workspace, mutation):
    recipe = json.loads(RECIPE.read_text(encoding="utf-8")) if RECIPE.exists() else {}
    assert recipe, "Frozen reconstruction recipe is missing"
    if mutation == "teacher32":
        recipe["models"]["sasrec"]["architecture"]["hidden_units"] = 32
    elif mutation == "hgn_bpr":
        recipe["models"]["hgn"]["architecture"]["loss_type"] = "BPR"
    elif mutation == "source":
        recipe["sources"]["recbole"]["commit"] = "main"
    elif mutation == "unknown":
        recipe["training"]["invented_option"] = 1
    elif mutation == "nan":
        recipe["training"]["learning_rate"] = float("nan")
    elif mutation == "schema_bool":
        recipe["schema_version"] = True
    elif mutation == "no_declarations":
        recipe["declarations"] = []
    else:
        recipe["training"]["max_epochs"] = 0
    path = workspace / "invalid.json"
    path.write_text(json.dumps(recipe), encoding="utf-8")
    with pytest.raises(ValueError):
        baseline().load_recipe(path)


def test_source_hash_verification_rejects_mutation_and_missing_file(workspace):
    source = workspace / "model.py"
    source.write_text("original\n", encoding="utf-8")
    expected = {"model.py": hashlib.sha256(source.read_bytes()).hexdigest()}
    assert baseline().verify_source_files(workspace, expected) == expected
    source.write_text("modified\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        baseline().verify_source_files(workspace, expected)
    source.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        baseline().verify_source_files(workspace, expected)


def checkpoint_state(module, epoch, selection=None):
    return {
        "schema_version": 1, "config_sha256": "a" * 64, "epoch": epoch,
        "model": {"weight": torch.tensor([float(epoch)])},
        "optimizer": {"state": {}, "param_groups": []},
        "sampler": {"epochs": epoch}, "rng": module.capture_rng(),
        "selection": selection, "bad_checks": 0,
        "history": [{"epoch": i} for i in range(1, epoch + 1)],
    }


def test_checkpoint_commit_marker_and_corrupt_latest_fail_closed(workspace):
    module = baseline()
    module._save_checkpoint(workspace, checkpoint_state(module, 1))
    incomplete = workspace / "checkpoints" / "epoch-00002.pt"
    incomplete.write_bytes(b"unfinished transaction")
    state, _ = module._read_checkpoint(workspace, "a" * 64)
    assert state["epoch"] == 1
    module._save_checkpoint(workspace, checkpoint_state(module, 2))
    state, _ = module._read_checkpoint(workspace, "a" * 64)
    assert state["epoch"] == 2
    with incomplete.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        module._read_checkpoint(workspace, "a" * 64)


def test_checkpoint_retention_preserves_selected_and_two_latest_epochs(workspace):
    module = baseline()
    prediction = workspace / "validation-top10-00002.json"
    module.write_json(prediction, [])
    selection = {"epoch": 2, "validation": {"ndcg@10": 0.1},
                 "predictions": prediction.name,
                 "predictions_sha256": module.file_sha256(prediction)}
    for epoch in range(1, 7):
        module._save_checkpoint(
            workspace, checkpoint_state(module, epoch, selection if epoch >= 2 else None)
        )
        if epoch == 1:
            module.write_json(prediction, [])
    assert sorted(p.stem for p in (workspace / "checkpoints").glob("*.pt")) == [
        "epoch-00002", "epoch-00005", "epoch-00006"
    ]
    assert prediction.is_file()
    assert module._read_checkpoint(workspace, "a" * 64, 2)[0]["epoch"] == 2


def test_checkpoint_receipt_epoch_must_match_its_filename(workspace):
    module = baseline()
    module._save_checkpoint(workspace, checkpoint_state(module, 2))
    folder = workspace / "checkpoints"
    (folder / "epoch-00002.pt").rename(folder / "epoch-00001.pt")
    (folder / "epoch-00002.json").rename(folder / "epoch-00001.json")
    with pytest.raises(ValueError, match="identity"):
        module._read_checkpoint(workspace, "a" * 64)


def test_run_config_binding_never_accepts_a_changed_recipe(workspace):
    baseline().bind_run_config(workspace, {"recipe": "first"})
    baseline().bind_run_config(workspace, {"recipe": "first"})
    with pytest.raises(ValueError, match="configuration"):
        baseline().bind_run_config(workspace, {"recipe": "changed"})


def test_resume_never_rewrites_a_damaged_original_user_map(workspace):
    module = baseline()
    assert hasattr(module, "bind_user_map"), "Immutable user mapping binding is missing"
    module.bind_user_map(workspace, ["3", "7"], resume=False)
    original = (workspace / "user-map.json").read_bytes()
    module.bind_user_map(workspace, ["3", "7"], resume=True)
    assert (workspace / "user-map.json").read_bytes() == original
    (workspace / "user-map.json").write_text('["7", "3"]', encoding="utf-8")
    with pytest.raises(ValueError, match="user mapping"):
        module.bind_user_map(workspace, ["3", "7"], resume=True)
    assert json.loads((workspace / "user-map.json").read_text()) == ["7", "3"]
    (workspace / "user-map.json").unlink()
    with pytest.raises(ValueError, match="user mapping"):
        module.bind_user_map(workspace, ["3", "7"], resume=True)


def test_complete_rng_capture_restores_python_numpy_torch_and_sampler():
    import random

    module = baseline()
    module.seed_everything(42, "cpu")
    state = module.capture_rng()
    expected = (random.random(), np.random.random(3), torch.rand(3))
    module.restore_rng(state)
    actual = (random.random(), np.random.random(3), torch.rand(3))
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    assert set(state) == {"python", "numpy", "torch", "cuda"}


def test_gradient_gate_rejects_nan_and_empty_backward():
    model = torch.nn.Linear(2, 1)
    with pytest.raises(ValueError, match="gradient"):
        baseline().check_gradients(model)
    model(torch.ones(1, 2)).sum().backward()
    baseline().check_gradients(model)
    model.weight.grad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="gradient"):
        baseline().check_gradients(model)


def test_loss_gate_accepts_native_singleton_regularizer_but_not_vector_or_nan():
    module = baseline()
    assert hasattr(module, "scalar_loss"), "Native singleton loss normalization is missing"
    value = torch.tensor([2.0], requires_grad=True)
    normalized = module.scalar_loss(value)
    assert normalized.shape == () and normalized.item() == 2.0
    normalized.backward()
    assert value.grad.tolist() == [1.0]
    for invalid in [torch.ones(2), torch.tensor([float("nan")]), torch.empty(0)]:
        with pytest.raises(ValueError, match="finite scalar"):
            module.scalar_loss(invalid)


def test_cli_separates_frozen_test_evaluation_and_epoch_boundary_drain():
    baseline()
    for command, expected in [
        ([], ("train", "evaluate")),
        (["train"], ("--resume", "--stop-after-epoch", "--gate")),
        (["evaluate"], ("--output", "--manifest")),
    ]:
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / "baselines.py"), *command, "--help"],
            check=True, capture_output=True, text=True,
        )
        assert all(text in result.stdout for text in expected)
