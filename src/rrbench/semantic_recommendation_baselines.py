from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sqlite3
import time
from typing import Any, Iterable
import uuid

import numpy as np
import torch

from rrbench.exact import exact_topk
from rrbench.io import bind_run_config, sha256, write_json
from rrbench.models import MatrixFactorization
from rrbench.runtime import configure, environment_manifest, require_cuda
from rrbench.semantic_ir import judged_metrics


def validate_catalog_mapping(catalog: Path, manifest: dict[str, Any]) -> np.ndarray:
    path = catalog / "original_ids.npy"
    if sha256(path) != manifest["ids_sha256"]:
        raise ValueError("MF catalog item-row mapping checksum changed.")
    values = np.load(path, mmap_mode="r")
    if (
        values.dtype != np.int64 or values.shape != (manifest["rows"],) or (values < 0).any()
        or len(np.unique(values)) != len(values)
    ):
        raise ValueError("MF catalog item-row mapping has invalid shape, types or duplicate IDs.")
    return values


def validate_mf_resume(state: dict[str, Any], expected_epochs: int) -> None:
    epochs = state["epochs"]
    if (
        not isinstance(epochs, int) or not 0 <= epochs <= expected_epochs
        or len(state["losses"]) != epochs or not all(math.isfinite(value) for value in state["losses"])
        or not math.isfinite(state["elapsed_seconds"]) or state["elapsed_seconds"] < 0
    ):
        raise ValueError("MF resume epoch/loss/runtime metadata is inconsistent with completed training steps.")
    for values in state.get("model", {}).values():
        if not isinstance(values, torch.Tensor) or not torch.isfinite(values).all():
            raise ValueError("MF resume model contains invalid parameter tensors.")
    for optimizer_state in state.get("optimizer", {}).get("state", {}).values():
        for value in optimizer_state.values():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise ValueError("MF resume optimizer contains nonfinite state tensors.")


def validate_prepared_logged_data(output: Path) -> dict[str, Any]:
    complete = json.loads((output / "completed.json").read_text(encoding="utf-8"))
    if not complete["complete"]:
        raise ValueError("Logged training preparation is not complete.")
    for filename, key in (
        ("pairs.npy", "pairs_sha256"), ("users.json", "users_sha256"),
        ("training-positive-popularity.npy", "positive_popularity_sha256"),
    ):
        if sha256(output / filename) != complete[key]:
            raise ValueError("Prepared logged baseline payload checksum changed.")
    rows = np.load(output / "pairs.npy", mmap_mode="r")
    if rows.shape != (complete["pairs"], 3) or rows.dtype != np.int64:
        raise ValueError("Prepared logged pairs have invalid row count or dtype.")
    if (
        (rows[:, 0] < 0).any() or (rows[:, 0] >= complete["users"]).any()
        or (rows[:, 1] < 0).any() or (rows[:, 1] >= complete["items"]).any()
        or not np.isin(rows[:, 2], [0, 1]).all()
    ):
        raise ValueError("Prepared logged pairs have invalid user/item/label values.")
    return complete


def validate_final_mf(checkpoint: Path) -> dict[str, Any]:
    completed = json.loads((checkpoint.parent / "completed.json").read_text(encoding="utf-8"))
    if not completed["complete"] or sha256(checkpoint) != completed["model_sha256"]:
        raise ValueError("MF evaluation requires the verified final trained model, not an intermediate checkpoint.")
    if completed["completed_epochs"] != completed["requested_epochs"] or len(completed["losses"]) != completed["requested_epochs"]:
        raise ValueError("MF final receipt does not attest all requested training epochs.")
    if not all(math.isfinite(value) for value in completed["losses"]):
        raise ValueError("MF final training loss history is nonfinite.")
    return completed


def publish_mf_epoch(output: Path, state: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / f"epoch-{state['epochs']:04d}-{uuid.uuid4().hex}.pt"
    temporary = checkpoint.with_suffix(".partial.pt")
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(checkpoint)
    receipt = {
        "checkpoint": checkpoint.name, "checkpoint_sha256": sha256(checkpoint),
        "completed_epochs": state["epochs"],
    }
    write_json(output / "training-state.json", receipt)


def mapped_logged_pairs(
    rows: Iterable[tuple[int, int, int]], users: dict[int, int], items: dict[int, int],
) -> np.ndarray:
    result = []
    for user, item, label in rows:
        if user not in users:
            raise ValueError("Logged training user is outside the training-derived user mapping.")
        if label not in (0, 1):
            raise ValueError("Logged long-view target is not binary.")
        if item in items:
            result.append((users[user], items[item], label))
    if not result:
        raise ValueError("No logged training exposures join the frozen semantic candidate catalog.")
    return np.asarray(result, dtype=np.int64)


def prepare_logged_pairs(catalog: Path, output: Path) -> dict[str, Any]:
    manifest = json.loads((catalog / "corpus.json").read_text(encoding="utf-8"))
    original = validate_catalog_mapping(catalog, manifest)
    mapped_items = {int(item): row for row, item in enumerate(original)}
    database = catalog / "private-catalog.sqlite"
    binding = {
        "catalog_sha256": manifest["catalog_sha256"], "database_sha256": sha256(database),
        "label": "long_view", "phase": "Admitted training recorded-clock blocks only.",
        "negative_policy": "Observed logged long_view=0 exposures, not synthetic unexposed negatives.",
        "catalog_ids_sha256": manifest["ids_sha256"], "catalog_items": manifest["rows"],
    }
    bind_run_config(output, binding)
    if (output / "completed.json").exists():
        return validate_prepared_logged_data(output)
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        users = {
            user: index for index, (user,) in enumerate(connection.execute(
                "SELECT DISTINCT e.user FROM events e JOIN blocks b USING(user,timestamp) "
                "WHERE b.phase=1 ORDER BY e.user"
            ))
        }
        rows = connection.execute(
            "SELECT e.user,e.item,e.label FROM events e JOIN blocks b USING(user,timestamp) "
            "JOIN semantic s USING(item) WHERE b.phase=1 ORDER BY e.user,e.timestamp,e.item,e.label"
        )
        pairs = mapped_logged_pairs(rows, users, mapped_items)
    np.save(output / "pairs.npy", pairs)
    write_json(output / "users.json", {str(user): row for user, row in users.items()})
    positive_popularity = np.bincount(pairs[pairs[:, 2] == 1, 1], minlength=len(original))
    np.save(output / "training-positive-popularity.npy", positive_popularity)
    complete = {
        "complete": True, "pairs": len(pairs), "users": len(users), "items": len(original),
        "positives": int(pairs[:, 2].sum()), "pairs_sha256": sha256(output / "pairs.npy"),
        "users_sha256": sha256(output / "users.json"),
        "positive_popularity_sha256": sha256(output / "training-positive-popularity.npy"),
        "input_contract": binding,
    }
    write_json(output / "completed.json", complete)
    return complete


def train_logged_mf(data: Path, output: Path, seed: int = 17, dimension: int = 128, epochs: int = 3) -> None:
    if min(dimension, epochs) <= 0:
        raise ValueError("MF embedding dimension and requested training epochs must be positive.")
    configure(seed, 4)
    device = require_cuda()
    source = validate_prepared_logged_data(data)
    pairs = np.load(data / "pairs.npy", mmap_mode="r")
    binding = {
        "prepared_data_sha256": sha256(data / "completed.json"), "seed": seed,
        "dimension": dimension, "epochs": epochs, "batch_size": 8192,
        "learning_rate": 0.01, "model": "Logged long-view user/item matrix factorization.",
        "study_role": source.get("study_role", "full_control"),
    }
    bind_run_config(output, binding)
    if (output / "completed.json").exists():
        validate_final_mf(output / "model.pt")
        return
    model = MatrixFactorization(source["users"], source["items"], dimension).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    losses, start_epoch, previous_seconds = [], 0, 0.0
    if (output / "training-state.json").exists():
        receipt = json.loads((output / "training-state.json").read_text(encoding="utf-8"))
        checkpoint = output / receipt["checkpoint"]
        if checkpoint.parent != output or not checkpoint.name.startswith("epoch-") or checkpoint.suffix != ".pt":
            raise ValueError("MF checkpoint pointer escapes its immutable epoch snapshot set.")
        if sha256(checkpoint) != receipt["checkpoint_sha256"]:
            raise ValueError("MF partial checkpoint bytes differ from their completed-epoch receipt.")
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if state["binding"] != binding:
            raise ValueError("Logged MF resume binding changed.")
        validate_mf_resume(state, epochs)
        if receipt["completed_epochs"] != state["epochs"]:
            raise ValueError("MF resume state epoch count differs from its checkpoint receipt.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, losses, previous_seconds = state["epochs"], state["losses"], state["elapsed_seconds"]
    started = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        order = np.random.default_rng(seed + epoch).permutation(len(pairs))
        total, count = 0.0, 0
        for start in range(0, len(pairs), 8192):
            rows = np.array(pairs[order[start:start + 8192]], copy=True)
            user = torch.as_tensor(rows[:, 0], device=device)
            item = torch.as_tensor(rows[:, 1], device=device)
            label = torch.as_tensor(rows[:, 2], device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            values = model(user, item, torch.empty(len(rows), 0, device=device))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(values, label)
            if not torch.isfinite(loss):
                raise RuntimeError("Logged MF fitting loss is nonfinite.")
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(rows)
            count += len(rows)
        losses.append(total / count)
        publish_mf_epoch(output, {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "binding": binding,
            "epochs": epoch + 1, "losses": losses, "elapsed_seconds": previous_seconds + time.perf_counter() - started,
        })
        print(f"LOGGED_LONGVIEW_MF_EPOCH seed={seed} epoch={epoch + 1} loss={losses[-1]:.6f}", flush=True)
    torch.save({"model": model.state_dict(), "binding": binding}, output / "model.partial.pt")
    (output / "model.partial.pt").replace(output / "model.pt")
    write_json(output / "completed.json", {
        "complete": True, "model_sha256": sha256(output / "model.pt"), "losses": losses,
        "training_seconds": previous_seconds + time.perf_counter() - started,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "users": source["users"], "items": source["items"], "dimension": dimension,
        "completed_epochs": epochs, "requested_epochs": epochs,
        "training_pairs": source["pairs"], "study_role": source.get("study_role", "full_control"),
        "label_scope": "Training logged binarylong-view exposures, fixed postguardcatalog; nofuturehistoryorstatistics.",
        "environment": environment_manifest(),
    })


def evaluate_logged_mf(data: Path, checkpoint: Path, panel_file: Path, output: Path) -> None:
    configure(17, 4)
    device = require_cuda()
    panel = json.loads(panel_file.read_text(encoding="utf-8"))
    source = validate_prepared_logged_data(data)
    users = json.loads((data / "users.json").read_text(encoding="utf-8"))
    if sha256(data / "users.json") != source["users_sha256"]:
        raise ValueError("MF training user mapping bytes changed.")
    if (
        panel.get("catalog_sha256") != source["input_contract"]["catalog_sha256"]
        or panel.get("catalog_ids_sha256") != source["input_contract"]["catalog_ids_sha256"]
        or panel.get("catalog_items") != source["items"]
    ):
        raise ValueError("MF evaluation panel belongs to a different frozen catalog/item-row mapping.")
    final = validate_final_mf(checkpoint)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if state["binding"]["prepared_data_sha256"] != sha256(data / "completed.json"):
        raise ValueError("MF evaluation data does not match the fitted training catalog/user mapping.")
    if final["completed_epochs"] != state["binding"]["epochs"] or final["dimension"] != state["binding"]["dimension"]:
        raise ValueError("MF final checkpoint/receipt training budget differs.")
    binding = {
        "checkpoint_sha256": sha256(checkpoint), "panel_sha256": sha256(panel_file),
        "data_sha256": sha256(data / "completed.json"), "oracle": "Whole1.8McatalogFP32MFinnerproduct",
    }
    bind_run_config(output, binding)
    model = MatrixFactorization(source["users"], source["items"], state["binding"]["dimension"]).to(device)
    model.load_state_dict(state["model"])
    unique_users = sorted(set(panel["users"]))
    if any(str(user) not in users for user in unique_users):
        raise ValueError("MF warm-query panel contains an unfitted user.")
    query = model.user(torch.tensor([users[str(user)] for user in unique_users], device=device)).detach()
    started = time.perf_counter()
    _, found = exact_topk(model.item.weight.detach(), query, 1000, chunk_size=65536)
    rows = {user: found[index].cpu().numpy() for index, user in enumerate(unique_users)}
    predictions = np.stack([rows[user] for user in panel["users"]])
    np.save(output / "mf-ranked.npy", predictions)
    popularity = np.load(data / "training-positive-popularity.npy")
    if sha256(data / "training-positive-popularity.npy") != source["positive_popularity_sha256"]:
        raise ValueError("Training-only popularity array changed.")
    ordered = np.lexsort((np.arange(len(popularity)), -popularity))[:1000]
    popular = np.tile(ordered, (len(predictions), 1))
    qrels = {query: {int(item): int(grade) for item, grade in values.items()} for query, values in panel["qrels"].items()}
    write_json(output / "completed.json", {
        "complete": True, "mf_prediction_sha256": sha256(output / "mf-ranked.npy"),
        "mf_relevance": judged_metrics(panel["query_ids"], predictions, qrels, 1),
        "training_popularity_relevance": judged_metrics(panel["query_ids"], popular, qrels, 1),
        "full_catalog_items": source["items"], "elapsed_seconds": time.perf_counter() - started,
        "model_contract": "StaticfittedMFpreferenceprofile; originalMFvectors, notsemanticLMandnotunit-normalizedsurrogate.",
        "quality_condition": panel["target_condition"],
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Real logged-feedback MF and training-only popularity recommendation baselines.")
    commands = parser.add_subparsers(dest="operation", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--catalog", required=True)
    prepare.add_argument("--output", required=True)
    train = commands.add_parser("train")
    train.add_argument("--data", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--seed", type=int, default=17)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--panel", required=True)
    evaluate.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.operation == "prepare":
        prepare_logged_pairs(Path(args.catalog), Path(args.output))
    elif args.operation == "train":
        train_logged_mf(Path(args.data), Path(args.output), args.seed)
    else:
        evaluate_logged_mf(Path(args.data), Path(args.checkpoint), Path(args.panel), Path(args.output))


if __name__ == "__main__":
    main()
