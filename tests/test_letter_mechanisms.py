import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "letter"))


def module():
    assert (ROOT / "scripts" / "letter" / "mechanisms.py").exists()
    return importlib.import_module("mechanisms")


def test_cf_neighbor_excludes_self_and_uses_original_id_ties():
    embeddings = np.asarray([[1, 0], [2, 0], [0, 1], [0, 1]], dtype=np.float32)
    neighbors = module().nearest_neighbors(embeddings, "cosine", block_size=2)
    assert neighbors.tolist() == [1, 0, 3, 2]
    np.testing.assert_array_equal(
        neighbors, module().nearest_neighbors(embeddings, "cosine", block_size=3),
    )
    with pytest.raises(ValueError):
        module().nearest_neighbors(np.zeros((3, 2)), "cosine")
    with pytest.raises(ValueError):
        module().nearest_neighbors(embeddings, "guessed_metric")


def test_code_similarity_uses_level_qualified_tokens():
    codes = np.asarray([[1, 2, 3, 4], [1, 9, 3, 7], [2, 1, 4, 3]])
    result = module().code_similarity(codes, np.asarray([1, 0, 0]))
    assert result["position_overlap_mean"] == pytest.approx(1 / 3)
    assert result["jaccard_mean"] == pytest.approx(2 / 9)
    assert result["prefix_overlap_mean"] == pytest.approx(1 / 6)
    with pytest.raises(ValueError):
        module().code_similarity(codes, np.asarray([0, 1, 2]))


def test_code_assignment_groups_include_unused_codes_and_last_partial_group():
    codes = np.asarray([[0, 0], [0, 1], [1, 1], [1, 1], [0, 0]])
    result = module().assignment_statistics(codes, [4, 3], group_size=3)
    first = result["levels"][0]
    assert first["counts_by_code"] == [3, 2, 0, 0]
    assert first["sorted_code_ids"] == [0, 1, 2, 3]
    assert first["grouped_assignment_counts"] == [5, 0]
    assert first["utilized_codes"] == 2
    assert sum(first["counts_by_code"]) == 5
    with pytest.raises(ValueError):
        module().assignment_statistics(codes, [1, 3])


def test_quantized_vectors_sum_codebooks_in_residual_order():
    books = [
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
    ]
    actual = module().quantized_vectors(books, np.asarray([[0, 1], [1, 0]]))
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, [[8.0, 10.0], [8.0, 10.0]])
    with pytest.raises(ValueError):
        module().quantized_vectors(books, np.asarray([[2, 0]]))


def test_pca_retains_centering_and_reports_explained_variance():
    points = np.asarray([[1, 0, 0], [-1, 0, 0], [0, 2, 0], [0, -2, 0]], dtype=np.float32)
    result = module().codebook_pca(points)
    coordinates = np.asarray(result["coordinates"])
    np.testing.assert_allclose(coordinates.mean(axis=0), 0, atol=1e-15)
    assert result["explained_variance_ratio"] == pytest.approx([0.8, 0.2, 0.0])
    assert result["normalization"] == "center_only_no_spherical_rescaling"


def test_item_replacement_preserves_padding_and_does_not_rescale():
    import torch
    from types import SimpleNamespace

    model = SimpleNamespace(item_emb=torch.nn.Embedding(4, 2))
    before = model.item_emb.weight[0].detach().clone()
    values = np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    module().replace_item_embeddings(model, values)
    assert torch.equal(model.item_emb.weight[0], before)
    np.testing.assert_array_equal(model.item_emb.weight[1:].detach().numpy(), values)
    with pytest.raises(ValueError):
        module().replace_item_embeddings(model, values[:2])


def analysis_fixture(tmp_path):
    import torch
    from protocol import file_sha256, write_json
    from tokenizer_run import format_codes

    dataset = "Instruments"
    source = tmp_path / "upstream"
    inter = {"0": [0, 1, 2, 3, 4], "1": [1, 2, 3, 4, 5]}
    items = {str(i): {"title": "fixture", "description": "fixture"} for i in range(16)}
    codes = np.asarray([[i % 4, (i // 4) % 4, i % 3, i % 2] for i in range(16)])
    data = {"inter": inter, "item": items, "index": format_codes(codes)}
    hashes = {}
    for kind, value in data.items():
        p = source / "data" / dataset / f"{dataset}.{kind}.json"
        write_json(p, value)
        hashes[kind] = file_sha256(p)
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"datasets": {dataset: {
        "items": 16, "users": 2, "interactions": 10, "sha256": hashes,
    }}})
    teacher = tmp_path / "teacher"
    write_json(teacher / "run-config.json", {
        "dataset": dataset, "input_sha256": hashes["inter"],
        "kind": "training-only-BCE-teacher-reconstruction",
    })
    torch.manual_seed(3)
    torch.save(torch.randn(16, 32), teacher / "item-embeddings.pt")
    write_json(teacher / "completed.json", {
        "status": "complete", "smoke": False, "test_evaluated": False,
        "config_sha256": file_sha256(teacher / "run-config.json"),
        "item_embeddings_sha256": file_sha256(teacher / "item-embeddings.pt"),
    })
    run = tmp_path / "rq"
    config = {
        "dataset": dataset, "synthetic_fixture": False, "epochs": 20000,
        "track_id": "letter-openllama7b-v2-method-replication", "codes": [4] * 4,
        "selection": "fixture minimum collision", "cf_sha256": file_sha256(teacher / "item-embeddings.pt"),
    }
    write_json(run / "run-config.json", config)
    for epoch in (2000, 20000):
        p = run / "checkpoints" / f"epoch-{epoch:05d}"
        p.mkdir(parents=True)
        torch.save({
            "epoch": epoch - 1, "spec": config,
            "model": {f"rq.vq_layers.{i}.embedding.weight": torch.randn(4, 32) for i in range(4)},
            "optimizer": {}, "initted": [True] * 4, "selected_checkpoint": "checkpoints/epoch-02000",
            "python_rng": (), "numpy_rng": (), "torch_rng": torch.get_rng_state(), "cuda_rng": [],
            "best_loss": 1, "best_collision_rate": 0,
        }, p / "state.pt")
        write_json(p / "completed.json", {
            "status": "complete", "kind": "tokenizer_epoch_checkpoint", "epoch": epoch - 1,
            "state_sha256": file_sha256(p / "state.pt"),
        })
    write_json(run / "latest-state.json", {
        "epoch": 20000, "checkpoint": str(run / "checkpoints/epoch-20000"),
    })
    write_json(run / "export/index.json", format_codes(codes))
    np.save(run / "export/codes.npy", codes, allow_pickle=False)
    export = {"index_sha256": file_sha256(run / "export/index.json")}
    write_json(run / "export/export.json", export)
    write_json(run / "completed.json", {
        "status": "complete", "synthetic_fixture": False, "track_id": config["track_id"],
        "selected_epoch": 2000, "selected_checkpoint": "checkpoints/epoch-02000", "export": export,
        "run_config_sha256": file_sha256(run / "run-config.json"),
    })
    return SimpleNamespace(output=tmp_path / "analysis", manifest=manifest, upstream=source,
                           dataset=dataset, tokenizer_run=run, teacher=teacher,
                           rank_replacement=False, sasrec=None, device="cpu")


def test_mechanism_cli_reads_real_artifact_shapes_without_refitting(tmp_path):
    args = analysis_fixture(tmp_path)
    module().analyze(args)
    report = json.loads((args.output / "completed.json").read_text())
    assert report["status"] == "complete"
    assert report["assignment_distribution"]["items"] == 16
    assert set(report["cf_neighbor_overlap"]) == {"cosine", "dot"}
    assert "cf_quantized_replacement" not in report


def test_mechanism_admission_requires_the_actual_final_checkpoint(tmp_path):
    args = analysis_fixture(tmp_path)
    (args.tokenizer_run / "checkpoints/epoch-20000/state.pt").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        module().analyze(args)


def test_cf_transfer_persists_every_user_with_stable_item_ties(tmp_path):
    import torch

    class ConstantFeatures(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.item_emb = torch.nn.Embedding(13, 2)

        def log2feats(self, sequences):
            return torch.zeros((*sequences.shape, 2))

    interactions = {"4": [0, 1, 2, 3, 4], "9": [5, 6, 7, 8, 9]}
    output = tmp_path / "predictions.json"
    report = module().rank_cf(ConstantFeatures(), interactions, 50, "cpu", output)
    rows = json.loads(output.read_text())
    assert [r["user"] for r in rows] == ["4", "9"]
    assert all(r["top10"] == list(range(10)) for r in rows)
    assert report["metrics"]["recall@5"] == 0.5
    assert report["metrics"]["recall@10"] == 1.0
    assert report["predictions_sha256"] == module().file_sha256(output)


def test_harness_teacher_checkpoint_float_metadata_is_hash_bound(tmp_path):
    import torch

    path = tmp_path / "best.pt"
    weights = {"item_emb.weight": torch.arange(8).reshape(4, 2).float()}
    torch.save({"model": weights, "epoch": 2, "validation": {"ndcg@10": 0.25}}, path)
    digest = module().file_sha256(path)
    loaded = module().load_teacher_model_state(path, digest)
    assert torch.equal(loaded["item_emb.weight"], weights["item_emb.weight"])
    path.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError):
        module().load_teacher_model_state(path, digest)
