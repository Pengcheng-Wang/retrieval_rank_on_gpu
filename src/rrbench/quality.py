from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from rrbench.io import append_jsonl, bind_run_config, sha256, write_json
from rrbench.metrics import single_positive_metrics
from rrbench.models import MatrixFactorization, make_model
from rrbench.movielens import RecommendationData, load_movielens
from rrbench.runtime import configure, environment_manifest, require_cuda


def save_checkpoint(path: Path, state: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class Features:
    def __init__(self, data: RecommendationData, device: torch.device):
        users = np.log1p(data.user_counts).astype(np.float32)
        items = np.log1p(data.item_counts).astype(np.float32)
        self.users = torch.tensor(users / max(float(users.max()), 1.0), device=device)
        self.items = torch.tensor(items / max(float(items.max()), 1.0), device=device)

    def get(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return torch.stack((self.users[users], self.items[items]), dim=1)


def epoch_examples(
    data: RecommendationData, negatives: int, seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if negatives < 0:
        raise ValueError("Negative sample count cannot be negative.")
    positive_users = data.train_users[data.train_labels.astype(bool)]
    if negatives and any(data.train_seen[user, data.catalog].all() for user in np.unique(positive_users)):
        raise ValueError("A user has no eligible training negatives.")
    generator = torch.Generator(device=device).manual_seed(seed)
    users = torch.tensor(data.train_users, device=device)
    items = torch.tensor(data.train_items, device=device)
    labels = torch.tensor(data.train_labels, device=device)
    if negatives:
        negative_users = torch.tensor(positive_users, device=device).repeat_interleave(negatives)
        catalog = torch.tensor(data.catalog, device=device)
        seen = torch.tensor(data.train_seen, device=device)
        sampled = catalog[torch.randint(len(catalog), (len(negative_users),), generator=generator, device=device)]
        invalid = seen[negative_users, sampled]
        while invalid.any():
            count = int(invalid.sum().item())
            sampled[invalid] = catalog[torch.randint(len(catalog), (count,), generator=generator, device=device)]
            invalid = seen[negative_users, sampled]
        users = torch.cat((users, negative_users))
        items = torch.cat((items, sampled))
        labels = torch.cat((labels, torch.zeros(len(sampled), device=device)))
    permutation = torch.randperm(len(users), generator=generator, device=device)
    return users[permutation], items[permutation], labels[permutation]


@torch.inference_mode()
def rank_catalog(
    model: nn.Module, features: Features, users: torch.Tensor,
    catalog: torch.Tensor, seen: torch.Tensor, batch_users: int = 16,
    pair_chunk: int = 65536,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_scores = []
    all_rankings = []
    for start in range(0, len(users), batch_users):
        batch = users[start : start + batch_users]
        pair_users = batch.repeat_interleave(len(catalog))
        pair_items = catalog.repeat(len(batch))
        pieces = []
        for offset in range(0, len(pair_users), pair_chunk):
            user = pair_users[offset : offset + pair_chunk]
            item = pair_items[offset : offset + pair_chunk]
            pieces.append(model(user, item, features.get(user, item)))
        scores = torch.cat(pieces).reshape(len(batch), len(catalog))
        if not torch.isfinite(scores).all():
            raise RuntimeError("Ranker generated non-finite catalog scores.")
        scores.masked_fill_(seen[batch[:, None], catalog[None, :]], -torch.inf)
        positions = scores.argsort(dim=1, descending=True, stable=True)
        ranked = catalog[positions]
        ranked.masked_fill_(~torch.isfinite(scores.gather(1, positions)), -1)
        all_scores.append(scores.cpu().numpy())
        all_rankings.append(ranked.cpu().numpy())
    return np.concatenate(all_rankings), np.concatenate(all_scores)


@torch.inference_mode()
def evaluate_ranker(
    model: nn.Module, features: Features, data: RecommendationData,
    selected: np.ndarray, phase: str, device: torch.device,
    retriever: MatrixFactorization | None = None, candidate_counts: list[int] | None = None,
) -> dict[str, object]:
    users_array = data.eval_users[selected]
    targets = (data.validation_targets if phase == "validation" else data.test_targets)[selected]
    history = data.validation_seen if phase == "validation" else data.test_seen
    users = torch.tensor(users_array, device=device)
    catalog = torch.tensor(data.catalog, device=device)
    seen = torch.tensor(history, device=device)
    rankings, model_scores = rank_catalog(model, features, users, catalog, seen)
    covered = np.isin(targets, data.catalog) & ~history[users_array, targets]
    result: dict[str, object] = {
        "phase": phase, "users": len(selected), "eligible_target_coverage": float(covered.mean()),
        "full_catalog": single_positive_metrics(rankings, targets, [10, 20, 100]),
        "user_ids": users_array.tolist(),
        "targets_covered": covered.tolist(),
    }
    if retriever is not None and candidate_counts:
        retrieval_rankings, _ = rank_catalog(retriever, features, users, catalog, seen)
        column_for_item = np.full(data.items, -1, dtype=np.int64)
        column_for_item[data.catalog] = np.arange(len(data.catalog))
        candidates_result = {}
        for candidate_count in candidate_counts:
            candidates = retrieval_rankings[:, :candidate_count]
            columns = column_for_item[candidates.clip(min=0)]
            valid = candidates >= 0
            scores = np.take_along_axis(model_scores, columns.clip(min=0), axis=1)
            scores[~valid] = -np.inf
            positions = np.argsort(-scores, axis=1, kind="stable")
            ranked = np.take_along_axis(candidates, positions, axis=1)
            candidate_recall = (candidates == targets[:, None]).any(axis=1)
            candidates_result[str(candidate_count)] = {
                "retrieval_hit_rate": float(candidate_recall.mean()),
                "quality": single_positive_metrics(ranked, targets, [10, 20, 100]),
            }
        result["two_stage"] = candidates_result
    return result


def run_quality(args: Any) -> None:
    configure(args.seed, args.threads)
    device = require_cuda()
    if min(args.epochs, args.batch_size, args.eval_users) <= 0 or args.learning_rate <= 0:
        raise ValueError("Epochs, batch size, eval users, and learning rate must be positive.")
    output, directory = Path(args.output), Path(args.data)
    output.mkdir(parents=True, exist_ok=True)
    data = load_movielens(directory)
    bind_run_config(output, {
        "model": args.model, "seed": args.seed, "epochs": args.epochs,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "negatives": args.negatives, "eval_users": args.eval_users,
        "candidates": args.candidates, "threads": args.threads,
        "dataset_sha256": json.loads((directory / "manifest.json").read_text())["prepared_sha256"],
        "retriever_sha256": sha256(Path(args.mf_checkpoint)) if args.mf_checkpoint else None,
    })
    if (output / "completed.json").exists():
        print("Quality run already completed; test evaluation is not repeated.", flush=True)
        return
    selection_rng = np.random.default_rng(20261010)
    selected = np.sort(selection_rng.choice(
        len(data.eval_users), min(args.eval_users, len(data.eval_users)), replace=False
    ))
    features = Features(data, device)
    model = make_model(args.model, data.users, data.items, device, args.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.BCEWithLogitsLoss()
    protocol = {
        "model": args.model, "seed": args.seed, "epochs": args.epochs,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "negatives_per_positive": args.negatives, "evaluation_users": len(selected),
        "feature_budget": "user ID, item ID, log-normalized training user/item counts",
        "selection_metric": "validation full-catalog NDCG@10",
        "test_policy": "held-out test evaluated once after validation selection",
        "dataset_manifest": json.loads((directory / "manifest.json").read_text()),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "model_class": getattr(model, "model_class", f"{type(model).__module__}.{type(model).__name__}"),
        "environment": environment_manifest(),
    }
    if (output / "protocol.json").exists():
        existing = json.loads((output / "protocol.json").read_text())
        if existing != protocol:
            raise ValueError("Existing quality run has a different protocol.")
    else:
        write_json(output / "protocol.json", protocol)
    start_epoch, best = 0, -1.0
    latest = output / "latest.pt"
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=True)
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, best = state["epoch"] + 1, state["best"]
    for epoch in range(start_epoch, args.epochs):
        model.train()
        started = time.perf_counter()
        users, items, labels = epoch_examples(data, args.negatives, args.seed + 10000 + epoch, device)
        total_loss = 0.0
        for start in range(0, len(users), args.batch_size):
            user, item, label = (value[start : start + args.batch_size] for value in (users, items, labels))
            optimizer.zero_grad(set_to_none=True)
            prediction = model(user, item, features.get(user, item))
            loss = loss_function(prediction, label)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss in epoch {epoch}.")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(user)
        validation = evaluate_ranker(model, features, data, selected, "validation", device)
        score = validation["full_catalog"]["ndcg@10"]
        if score > best:
            best = score
            save_checkpoint(output / "best.pt", {
                "state_dict": model.state_dict(), "epoch": epoch, "model": args.model,
                "seed": args.seed, "validation_ndcg@10": best,
            })
            write_json(output / "best.json", {
                "epoch": epoch, "validation_ndcg@10": best,
                "checkpoint_sha256": sha256(output / "best.pt"), "model": args.model,
            })
        save_checkpoint(latest, {
            "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch, "best": best,
        })
        append_jsonl(output / "epochs.jsonl", {
            "epoch": epoch, "training_loss": total_loss / len(users),
            "examples": len(users), "seconds": time.perf_counter() - started,
            "validation": validation["full_catalog"],
            "validation_coverage": validation["eligible_target_coverage"],
        })
        print(f"{args.model} seed={args.seed} epoch={epoch} loss={total_loss/len(users):.5f} val_ndcg10={score:.5f}", flush=True)
    state = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(state["state_dict"])
    retriever = None
    if args.model == "mf":
        retriever = model
    elif args.mf_checkpoint:
        retriever = make_model("mf", data.users, data.items, device, args.seed)
        retrieval_state = torch.load(args.mf_checkpoint, map_location=device, weights_only=True)
        if retrieval_state["model"] != "mf":
            raise ValueError("The retrieval checkpoint is not a matrix-factorization model.")
        retriever.load_state_dict(retrieval_state["state_dict"])
    result = evaluate_ranker(
        model, features, data, selected, "test", device, retriever, args.candidates
    )
    result["selected_epoch"] = state["epoch"]
    result["seed"] = args.seed
    result["model"] = args.model
    write_json(output / "quality.json", result)
    write_json(output / "completed.json", {"complete": True, "experiment": "recommendation quality"})
    print(f"QUALITY_COMPLETE {args.model} seed={args.seed} {result['full_catalog']}", flush=True)
