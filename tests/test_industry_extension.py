from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rrbench.industry_data import (
    prepare_ranking_data,
    hard_negative_ids_and_ranks,
    rank_interleaved_union,
    select_hard_negatives,
)
from rrbench.industry_models import RetrievalResidualRanker, make_industry_model


def _write_fixture(path: Path) -> tuple[Path, Path]:
    sequences = path / "sequences.csv"
    catalog = path / "movies.csv"
    with sequences.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "user_id",
                "sequence_item_ids",
                "sequence_ratings",
                "sequence_timestamps",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "user_id": 10,
                "sequence_item_ids": "[1, 2, 3, 4, 5, 6]",
                "sequence_ratings": "[5, 4, 3, 2, 1, 5]",
                "sequence_timestamps": "[10, 20, 30, 40, 50, 60]",
            }
        )
        writer.writerow(
            {
                "user_id": 20,
                "sequence_item_ids": "[6, 5, 4, 3, 2, 1]",
                "sequence_ratings": "[1, 2, 3, 4, 5, 1]",
                "sequence_timestamps": "[11, 21, 31, 41, 51, 61]",
            }
        )
    with catalog.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["movie_id"])
        writer.writeheader()
        for item in range(1, 9):
            writer.writerow({"movie_id": item})
    return sequences, catalog


def test_prepared_data_has_chronological_holdouts_and_fixed_negatives(
    tmp_path: Path,
) -> None:
    sequences, catalog = _write_fixture(tmp_path)
    output = tmp_path / "ranking.npz"
    receipt = prepare_ranking_data(
        sequence_csv=sequences,
        catalog_csv=catalog,
        output=output,
        max_history=4,
        maximum_examples=4,
        per_user_cap=2,
        negatives=2,
        seed=7,
    )
    data = np.load(output)
    assert receipt["validation_examples"] == 2
    assert data["validation_targets"].tolist() == [5, 2]
    assert data["test_targets"].tolist() == [6, 1]
    assert sorted(data["test_user_partition_validation_indices"].tolist()) == [0, 1]
    assert data["validation_histories"][0].tolist() == [4, 3, 2, 1]
    for history, target, negatives in zip(
        data["train_histories"],
        data["train_targets"],
        data["train_negatives"],
    ):
        excluded = set(int(item) for item in history if item)
        excluded.add(int(target))
        assert not excluded.intersection(int(item) for item in negatives)


def test_rank_interleaved_union_deduplicates_without_labels() -> None:
    first = np.asarray([[1, 2, 3, 4], [4, 3, 2, 1]])
    second = np.asarray([[1, 5, 6, 7], [4, 8, 9, 10]])
    union = rank_interleaved_union([first, second], cap=4)
    assert union.tolist() == [[1, 2, 5, 3], [4, 3, 8, 2]]


def test_hard_negative_selection_removes_target_and_duplicates() -> None:
    retrieved = np.asarray([[4, 2, 4, 5, 6], [9, 8, 7, 6, 5]])
    targets = np.asarray([4, 7])
    negatives = select_hard_negatives(retrieved, targets, count=3)
    assert negatives.tolist() == [[2, 5, 6], [9, 8, 6]]
    negatives, negative_ranks, target_ranks = hard_negative_ids_and_ranks(
        retrieved, targets, count=3
    )
    assert negatives.tolist() == [[2, 5, 6], [9, 8, 6]]
    assert negative_ranks.tolist() == [[2, 4, 5], [1, 2, 4]]
    assert target_ranks.tolist() == [1, 3]


def test_epoch_zero_residual_exactly_preserves_retrieval_rank() -> None:
    class ConstantRanker(torch.nn.Module):
        def forward(self, **_: torch.Tensor) -> torch.Tensor:
            return torch.ones((1, 3))

    model = RetrievalResidualRanker(ConstantRanker())
    ranks = torch.tensor([[1.0, 2.0, 4.0]])
    assert torch.allclose(model(candidate_ranks=ranks), -torch.log(ranks))
    with torch.no_grad():
        model.residual_weight.fill_(2.0)
    assert torch.allclose(
        model(candidate_ranks=ranks),
        -torch.log(ranks) + 2.0,
    )


@pytest.mark.parametrize("name", ["din", "twin", "transact"])
def test_sequence_rankers_have_finite_forward_and_backward(name: str) -> None:
    torch.manual_seed(3)
    model = make_industry_model(
        name,
        users=4,
        items=32,
        device=torch.device("cpu"),
        seed=3,
        maximum_gap_seconds=1000,
    )
    with torch.no_grad():
        model.residual_weight.fill_(1.0)
    batch = 2
    candidates = 3
    history = 50
    histories = torch.zeros((batch, history), dtype=torch.long)
    histories[:, :5] = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    ratings = torch.full((batch, history), -1, dtype=torch.long)
    ratings[:, :5] = torch.tensor([[5, 4, 3, 2, 1], [1, 2, 3, 4, 5]])
    timestamps = torch.zeros((batch, history), dtype=torch.long)
    timestamps[:, :5] = torch.tensor(
        [[100, 90, 80, 70, 60], [110, 100, 90, 80, 70]]
    )
    logits = model(
        users=torch.tensor([0, 1]),
        candidates=torch.tensor([[11, 12, 13], [14, 15, 16]]),
        dense=torch.rand(batch, candidates, 4),
        histories=histories,
        ratings=ratings,
        timestamps=timestamps,
        lengths=torch.tensor([5, 5]),
        request_times=torch.tensor([120, 130]),
        candidate_ranks=torch.tensor([[1, 2, 3], [1, 2, 3]]),
    )
    assert logits.shape == (batch, candidates)
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_official_transact_snapshot_matches_manifest() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "third_party" / "transact-source-sha256.json").read_text()
    )
    assert manifest["commit"] == "2c8b6c93b679949b9581cf592c46ddcba55f3d09"
    assert manifest["license"] == "Apache-2.0"
    for record in manifest["files"]:
        path = (root / "third_party" / "transact").joinpath(*record["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


def test_committed_industry_results_are_valid_and_checksummed() -> None:
    root = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "2026-11-08-industry-ranking"
    )
    checksums = json.loads((root / "checksums.json").read_text())
    for name, expected in checksums.items():
        path = root / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        if path.suffix == ".json":
            json.loads(path.read_text())
    summary = json.loads((root / "summary.json").read_text())
    assert summary["candidate_pool"]["target_injection"] is False
    assert summary["training_negatives"]["search_k"] == 200
    assert all(
        row["passed_seeds"] == 0 for row in summary["learning_gates"]
    )
    assert (
        summary["models"]["torchrec-dlrm"]["upstream_class"]
        == "torchrec.models.dlrm.DLRM"
    )
    assert (
        summary["models"]["torchrec-dcn"]["upstream_class"]
        == "torchrec.models.dlrm.DLRM_DCN"
    )
    assert summary["models"]["transact"]["upstream_class"] == "transact.TransAct"
