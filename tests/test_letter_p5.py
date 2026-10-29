import copy
import importlib
import os
import random
import shutil
import socket
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import sparse
from sklearn.cluster import SpectralClustering


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "letter"))
from p5_checkpoint import IndexSampler, pointer_path, restore_checkpoint, save_checkpoint
from p5_checkpoint import verify_checkpoint
from p5_index import MetadataUnavailable, metadata_view, training_graph, validate_sequences
from p5_source import digest_json, load_native, load_recipe, native_cid_format, read_json, write_json


@pytest.fixture
def workspace():
    path = ROOT / ".local" / "letter" / "p5-unit-tests" / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path)


def interactions():
    return {
        "3": [0, 1, 0, 4, 5], "7": [0, 1, 2, 4, 5],
        "11": [1, 0, 2, 4, 5], "23": [0, 1, 2, 4, 5],
    }


@pytest.fixture
def native():
    source = os.environ.get("LETTER_P5_SOURCE")
    if not source:
        pytest.skip("Set LETTER_P5_SOURCE to the private pinned native P5 checkout")
    return load_native(Path(source))


@pytest.fixture
def tokenizer():
    model = os.environ.get("LETTER_P5_MODEL")
    if not model:
        pytest.skip("Set LETTER_P5_MODEL to the verified private T5-small snapshot")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model, local_files_only=True, use_fast=True)


def test_recipe_is_source_pinned_and_not_claimed_as_letter_values():
    recipe = load_recipe(ROOT / "configs" / "letter-p5-reference-v1.json")
    assert recipe["training"]["microbatch_size"] * recipe["training"]["gradient_accumulation"] == 128
    assert recipe["training"]["epochs"] == 20
    assert recipe["evaluation"]["seen_item_exclusion"] is False
    assert recipe["evaluation"]["beams"] == recipe["evaluation"]["returns"] == 20
    assert recipe["model"]["initialization"] == "pretrained-native-P5"
    assert "BLOCKED" in recipe["indexing"]["semid_beauty"]
    assert "BLOCKED" in recipe["indexing"]["semid_yelp"]


def test_graph_retains_cold_zero_nodes_and_original_zero():
    graph, _, report = training_graph(interactions(), 6)
    assert graph.shape == (6, 6)
    assert graph[0, 1] == 2
    assert graph[0, 0] == 1
    assert graph[3:].nnz == 0
    assert report["cold_original_ids"] == [3, 4, 5]
    assert report["native_matrix_size_would_be"] == 3
    assert report["cold_zero_rows"]


def test_graph_keeps_native_overwrite_not_undirected_sum():
    graph, pairs, report = training_graph(interactions(), 6)
    assert dict(pairs)[0, 1] == 3
    assert dict(pairs)[1, 0] == 2
    assert graph[0, 1] == graph[1, 0] == 2
    assert graph[0, 1] != 5
    assert report["unequal_reverse_counts"] >= 1


def test_graph_equals_native_dense_where_native_domain_assumption_holds():
    graph, pairs, _ = training_graph(interactions(), 6)
    native_dense = np.zeros((3, 3), dtype=np.float64)
    for (a, b), frequency in pairs:
        native_dense[a, b] = native_dense[b, a] = frequency
    np.testing.assert_array_equal(graph.toarray()[:3, :3], native_dense)


def test_no_heldout_labels_enter_any_fitting_edges():
    changed = {u: seq[:-2] + [3, 0] for u, seq in interactions().items()}
    a, pairs_a, report_a = training_graph(interactions(), 6)
    b, pairs_b, report_b = training_graph(changed, 6)
    assert (a != b).nnz == 0
    assert pairs_a == pairs_b
    assert report_a["fitting_sequences_sha256"] == report_b["fitting_sequences_sha256"]


def test_sparse_spectral_partition_matches_dense_source_compatible_fixture():
    rng = np.random.RandomState(4)
    blocks = []
    for _ in range(3):
        block = rng.uniform(.8, 1.2, (10, 10))
        blocks.append((block + block.T) / 2)
    graph = sparse.block_diag(blocks, format="csr").toarray() + .001
    np.fill_diagonal(graph, 0.)
    settings = dict(n_clusters=3, assign_labels="cluster_qr", random_state=0, affinity="precomputed")
    a = SpectralClustering(**settings).fit_predict(graph)
    b = SpectralClustering(**settings).fit_predict(sparse.csr_matrix(graph))
    c = SpectralClustering(**settings).fit_predict(sparse.csr_matrix(graph))
    np.testing.assert_array_equal(a[:, None] == a, b[:, None] == b)
    np.testing.assert_array_equal(b, c)


@pytest.mark.parametrize("sequences", [
    {"0": [0, 1, 2, 3]}, {"0": [0, 1, -1, 3, 4]}, {"0": [0, 1, 6, 3, 4]},
    {"0": [0, True, 2, 3, 4]}, {"01": [0, 1, 2, 3, 4], "1": [0, 1, 2, 3, 4]},
])
def test_invalid_or_ambiguous_release_domain_rejected(sequences):
    with pytest.raises((ValueError, TypeError)):
        validate_sequences(sequences, 6)


def test_metadata_mapping_keeps_raw_title_semantics_and_one_based_view():
    source = {
        "0": {"title": "  A > B  ", "description": "not used"},
        "1": {"title": "", "categories": "Musical Instruments,Root Child"},
    }
    saved = copy.deepcopy(source)
    rows, meta_dict, mapping = metadata_view(source)
    assert mapping == {"1": "0", "2": "1"}
    assert rows[meta_dict[mapping["1"]]]["title"] == "  A > B  "
    assert rows[meta_dict[mapping["2"]]]["title"] == ""
    assert source == saved


def test_categories_are_explicit_release_path_adaptation():
    items = {"0": {"title": "x", "categories": "Musical Instruments,Studio,Audio Interfaces"}}
    rows, _, _ = metadata_view(items, categories=True)
    assert rows[0]["categories"] == [["Musical Instruments", "Studio", "Audio Interfaces"]]
    assert items["0"]["categories"] == "Musical Instruments,Studio,Audio Interfaces"
    with pytest.raises(MetadataUnavailable, match="absent"):
        metadata_view({"0": {"title": "x"}}, categories=True)
    empty, _, _ = metadata_view({"0": {"title": "x", "categories": ""}}, categories=True)
    assert empty[0]["categories"] == [[]]
    with pytest.raises(MetadataUnavailable):
        metadata_view({"0": {"title": "x", "categories": None}}, categories=True)


def test_sampler_preserves_tail_permutation_and_cold_state():
    continuous = IndexSampler(131, 42)
    first = continuous.next(128)
    state = continuous.state_dict()
    restored = IndexSampler(131, 999)
    restored.load_state_dict(state)
    assert restored.next(128) == continuous.next(128)
    assert len(first) == 128
    assert continuous.next(128) == []
    continuous.advance()
    restored.advance()
    assert continuous.next(128) == restored.next(128)
    assert continuous.state_dict()["cursor"] == 128


def test_checkpoint_restores_parameters_optimizer_scheduler_rng_sampler(workspace):
    from p5_checkpoint import restore_rng, rng_state

    torch.manual_seed(1)
    random.seed(2)
    np.random.seed(3)
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda n: 1. / (n + 1))
    sampler = IndexSampler(131, 42)
    sampler.next(128)
    loss = model(torch.ones(4, 3)).square().mean()
    loss.backward()
    opt.step()
    sched.step()
    opt.zero_grad(set_to_none=True)
    state = {"epoch": 0, "global_step": 1, "phase": "train", "best_metric": .2,
             "validation_history": [{"epoch": 0, "recall@10": .2, "users": 131}]}
    contract = {"unit_test": True, "seed": 42}
    artifact, seal = save_checkpoint(workspace, model, opt, sched, sampler, state, contract, selected=True)
    assert pointer_path(workspace) == pointer_path(workspace, "selected") == artifact
    assert seal["kind"] == "p5-full-state-v1"
    expected = (random.random(), np.random.random(), torch.rand(3))
    with torch.no_grad():
        model.weight.add_(17)
    restored = restore_checkpoint(artifact, model, opt, sched, sampler, contract)
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])
    assert restored["selected_artifact"] == artifact.name
    assert restored["best_metric"] == .2
    assert restored["validation_history"] == state["validation_history"]
    assert sampler.next(128) == [i for i in sampler.order[128:].tolist()]
    assert sched.last_epoch == 1
    with pytest.raises(ValueError, match="different"):
        verify_checkpoint(artifact, {"unit_test": False})
    with (artifact / "training-state.pt").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        verify_checkpoint(artifact)


def test_collision_metrics_keep_beam_rank_and_separate_item_diagnostic():
    from p5_run import per_user_metrics
    groups = {"same title": [0, 2], "other": [1], "third": [3]}
    m = per_user_metrics(["other", "same title", "same title", "third"],
                         "same title", 2, groups)
    assert m["primary_rank"] == 2
    assert m["item_rank"] == 3
    assert m["top_original_items"] == [1, 0, 2, 3]
    assert m["duplicate_beams"] == 1


def test_native_loss_is_actual_example_mean_with_only_padding_masked():
    from types import SimpleNamespace
    from p5_run import native_loss
    values = torch.tensor([2., 4., 1000., 5., 7., 9.], requires_grad=True)
    model = lambda **kw: SimpleNamespace(loss=values)
    ids = torch.tensor([[3, 1, 0], [4, 3, 1]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    loss = native_loss(model, (ids, mask, ids * 0, ids, mask))
    assert loss.item() == 5.
    loss.backward()
    assert values.grad[2] == 0
    assert torch.all(values.grad[[0, 1, 3, 4, 5]] > 0)


def test_resume_reconciles_only_unsaved_trace_and_preserves_forensics(workspace):
    from p5_run import reconcile_trace
    path = workspace / "updates.jsonl"
    path.write_text('{"step": 1}\n{"step": 2}\n{"ste', encoding="utf-8")
    reconcile_trace(path, 1)
    assert path.read_text() == '{"step": 1}\n'
    copies = list(workspace.glob("updates-uncommitted-*.jsonl"))
    assert len(copies) == 1
    assert copies[0].read_text().endswith('{"ste')


def test_test_continuation_reuses_committed_chunks_without_reselection(workspace, monkeypatch):
    from types import SimpleNamespace
    import p5_run
    output = workspace / "test-predictions.jsonl"
    write_json(workspace / "test-started.json", {"selected_seal_sha256": "same-frozen-model"})
    calls, allow = [], [False]

    def fake_generation(model, batch, tokenizer, catalog, native, recipe):
        index = int(batch[0][0, 0])
        calls.append(index)
        if index == 1 and not allow[0]:
            raise InterruptedError("interrupted before committing user1")
        return SimpleNamespace(
            sequences=torch.tensor([[index]] * 20), sequences_scores=torch.zeros(20),
        )

    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **kw: (
        self if a == ("cuda",) else original_to(self, *a, **kw)
    ))
    monkeypatch.setattr(p5_run, "generation", fake_generation)
    tokenizer = SimpleNamespace(batch_decode=lambda ids, **kw: ["item_" + str(int(r[0])) for r in ids])
    model = SimpleNamespace(eval=lambda: None)
    catalog = {"canonical_targets": ["item_0", "item_1"],
               "canonical_to_original_ids": {"item_0": [0], "item_1": [1]}}
    source = {"0": [0, 1, 0, 1, 0], "1": [0, 1, 0, 1, 1]}
    kwargs = dict(
        model=model, data=[0, 1], users=["0", "1"], interactions=source, split="test",
        collator=lambda batch: (torch.tensor([[batch[0]]]),),
        tokenizer=tokenizer, catalog=catalog, native=None,
        recipe={"evaluation": {"batch_size": 1}}, output=output,
    )
    with pytest.raises(InterruptedError):
        p5_run.evaluate(**kwargs)
    allow[0] = True
    result = p5_run.evaluate(**kwargs)
    assert calls == [0, 1, 1]
    assert result["users"] == 2 and result["resumed_frozen_test_chunks"] == 1
    assert len(output.read_text().splitlines()) == 2
    write_json(workspace / "test-started.json", {"selected_seal_sha256": "different-model"})
    with pytest.raises(ValueError, match="different frozen"):
        p5_run.evaluate(**kwargs)


def test_native_cid_format_not_arbitrary_letter_tokens(native):
    mapping, vocabulary = native_cid_format(native.source, {
        0: ["0", "2"], 1: ["0", "2"], 2: ["1"],
    })
    assert mapping == {
        "0": "<A0><A0-2><A00>", "1": "<A0><A0-2><A01>", "2": "<A1><A00>",
    }
    assert len(set(mapping.values())) == 3
    assert native.item_rep_method.CF_representation("1", mapping) == mapping["0"]
    assert all(t.startswith("<A") for t in vocabulary)


def test_native_train_distribution_and_complete_split_support(native):
    from p5_data import all_supported_rows, datasets
    from p5_source import native_args
    source = {"7": [0, 1, 2, 0, 4, 5], "3": [1, 2, 3, 4, 5]}
    codes = [f"[title {i}]" for i in range(6)]
    args = native_args("Instruments", "TID", 6)
    users, views = datasets(native, source, codes, args)
    original = native.data.sequential_item_dataset(
        args, users, [[codes[i] for i in source[u][:-2]] for u in users], native.prompt.task_subgroup_1,
    )
    assert len(original) == len(views["train"]) == 50
    for index in range(len(original)):
        random.seed(index)
        expected = original[index]
        random.seed(index)
        assert views["train"][index] == expected
    rows = list(all_supported_rows(native, views, source, users))
    train = [r for r in rows if r[0] == "train"]
    assert len(train) == 50
    assert {r[2]["template"] for r in train} == set(range(10))
    for split, (_, target), meta in rows:
        assert target == "item_" + codes[meta["target"]]
        if split == "test":
            assert meta["history"][-1] == source[meta["user"]][-2]
            assert meta["target"] == source[meta["user"]][-1]
        if split == "valid":
            assert meta["target"] == source[meta["user"]][-2]


def test_native_title_mapping_preserves_empty_and_duplicate_titles(native):
    from p5_index import identifiers
    source = {"0": {"title": "  A > B  "}, "1": {"title": "same"}, "2": {"title": "same"},
              "3": {"title": ""}, "4": {"title": "last"}, "5": {"title": "cold"}}
    codes, vocab, report = identifiers(native, "Instruments", "TID", interactions(), source, Path("."))
    assert codes[:4] == ["[  A > B  ]", "[same]", "[same]", "[]"]
    assert not vocab
    assert report["raw_empty_titles"] == report["raw_duplicate_groups"] == 1
    with pytest.raises(MetadataUnavailable):
        identifiers(native, "Beauty", "SemID", interactions(), source, Path("."))


def test_native_semid_partial_empty_metadata_is_named_id_fallback(native, workspace):
    from p5_index import identifiers
    items = {str(i): {"title": "x", "categories": "Musical Instruments,Strings"} for i in range(6)}
    items["0"]["categories"] = ""
    items["1"]["categories"] = "Musical Instruments"
    codes, _, report = identifiers(native, "Instruments", "SemID", interactions(), items, workspace)
    assert codes[:2] == ["<category_1>", "<category_2>"]
    assert report["empty_released_paths"] == 1
    assert report["root_only_paths"] == 1
    assert report["native_item_id_fallback_rows"] == 2
    assert len(set(codes)) == 6
    altered = {u: sequence[:-2] + [3, 0] for u, sequence in interactions().items()}
    other, _, other_report = identifiers(native, "Instruments", "SemID", altered, items, workspace)
    assert other == codes and report["mapping_sha256"] == other_report["mapping_sha256"]


def test_full_targets_atomic_tokens_wholewords_padding_and_eos(native, tokenizer):
    from p5_data import CompleteCollator, tokenized_catalog
    from p5_source import native_args
    tokenizer.add_tokens(["<A0>", "<A00>"])
    assert tokenizer.encode("<A00>", add_special_tokens=False) == [tokenizer.convert_tokens_to_ids("<A00>")]
    catalog = tokenized_catalog(tokenizer, ["[same]", "[same]", "[" + "long title " * 300 + "]"])
    assert catalog["canonical_to_original_ids"][catalog["canonical_targets"][0]] == [0, 1]
    assert len(catalog["token_ids"][2]) > 512
    rows = [
        ("User_0 has purchased item_[same] , predict next possible item ?", "item_[same]"),
        ("User_1 has purchased item_[" + "long title " * 300 + "] , predict next possible item ?",
         catalog["target_texts"][2]),
    ]
    collator = CompleteCollator(native, native_args("Instruments", "TID", 3), tokenizer, 4096, 4096)
    batch = collator(rows)
    single = collator(rows[:1])
    n = single[0].shape[1]
    assert torch.equal(single[2][0], batch[2][0, :n])
    assert batch[2][0, n - 1] == 0
    assert batch[2][0, n:].sum() == 0
    assert batch[0].shape[1] > 512 and batch[3].shape[1] > 512
    assert batch[3][1, batch[4][1].sum() - 1] == tokenizer.eos_token_id
    with pytest.raises(ValueError, match="truncate"):
        CompleteCollator(native, native_args("Instruments", "TID", 3), tokenizer, 512, 512)(rows)
    with pytest.raises(ValueError, match="Unreachable"):
        tokenized_catalog(tokenizer, ["[a </s> b]"])


@pytest.mark.skipif(os.name == "nt", reason="The shared bank uses node0 POSIX locking")
def test_p5_bank_receipt_is_immutable_and_corruption_is_rejected(workspace):
    from p5_bank import bank_checkpoint, verify_ack
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    sampler = IndexSampler(5, 42)
    run = workspace / "run"
    path, _ = save_checkpoint(run, model, optimizer, scheduler, sampler, {
        "epoch": 0, "global_step": 0, "phase": "train",
    }, {"unit_test": True})
    destination = workspace / "durable" / path.name
    bank_checkpoint(run, path, destination, workspace / "bank.lock", socket.gethostname())
    assert verify_ack(run, path)["status"] == "verified"
    with (destination / "model.safetensors").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        bank_checkpoint(run, path, destination, workspace / "bank.lock", socket.gethostname())
