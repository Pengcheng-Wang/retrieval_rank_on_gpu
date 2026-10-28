"""Identity, metric and artifact contracts for the author-code LETTER runs."""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from rrbench.io import bind_run_config, sha256 as file_sha256, write_json  # noqa: E402


def require_hash(path: Path, expected: str) -> None:
    if file_sha256(path) != expected:
        raise ValueError(f"Source/artifact checksum mismatch: {path}")


def sequence_records(
    interactions: dict[str, list[int]], split: str, max_history: int = 20,
) -> list[dict[str, Any]]:
    if split not in {"train", "valid", "test"} or max_history <= 0:
        raise ValueError("Invalid split or history limit")
    records = []
    for user, items in interactions.items():
        if len(items) < 3 or any(type(i) is not int or i < 0 for i in items):
            raise ValueError(f"Invalid sequence for user {user}")
        positions = range(1, len(items) - 2) if split == "train" else [
            len(items) - (2 if split == "valid" else 1)
        ]
        for position in positions:
            records.append({
                "user": user, "position": position,
                "history": items[max(0, position - max_history):position],
                "target": items[position],
            })
    return records


def code_to_items(indices: dict[str, list[str]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for item, tokens in indices.items():
        groups["".join(tokens)].append(int(item))
    return {code: sorted(items) for code, items in groups.items()}


def collision_summary(indices: dict[str, list[str]]) -> dict[str, int]:
    groups = code_to_items(indices)
    sizes = [len(items) for items in groups.values()]
    if not sizes:
        raise ValueError("Empty semantic-ID inventory")
    return {
        "items": len(indices), "unique_sids": len(groups),
        "excess_collision_items": len(indices) - len(groups),
        "collision_groups": sum(n > 1 for n in sizes),
        "items_in_collision_groups": sum(n for n in sizes if n > 1),
        "largest_group": max(sizes),
    }


def expand_item_predictions(
    predictions: list[str], indices: dict[str, list[str]], k: int,
) -> list[int]:
    groups = code_to_items(indices)
    items = []
    for code in dict.fromkeys(predictions):
        items.extend(groups.get(code, []))
    return list(dict.fromkeys(items))[:k]


def rank_metrics(
    predictions: list, target: Any, ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    rank = predictions.index(target) + 1 if target in predictions else None
    result = {}
    for k in ks:
        hit = rank is not None and rank <= k
        result[f"recall@{k}"] = float(hit)
        result[f"ndcg@{k}"] = 1 / math.log2(rank + 1) if hit else 0.0
    return result


def score_prediction(
    predictions: list[str], target: int, indices: dict[str, list[str]],
) -> dict[str, Any]:
    target_code = "".join(indices[str(target)])
    groups = code_to_items(indices)
    return {
        "code": rank_metrics(predictions, target_code),
        "item": rank_metrics(expand_item_predictions(predictions, indices, 10), target),
        "target_collision_size": len(groups[target_code]),
        "invalid_codes": sum(code not in groups for code in predictions),
    }


class PrefixTrie:
    def __init__(self, sequences: Iterable[list[int]]):
        self.root: dict = {}
        for sequence in sequences:
            node = self.root
            for token in sequence:
                node = node.setdefault(token, {})

    def allowed(self, prefix: list[int]) -> list[int]:
        node = self.root
        for token in prefix:
            if token not in node:
                return []
            node = node[token]
        return sorted(node)


def shard_positions(count: int, rank: int, world_size: int) -> list[int]:
    if count < 0 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid shard specification")
    return list(range(rank, count, world_size))


def verify_coverage(expected: list[str], observed: list[str]) -> None:
    if len(set(expected)) != len(expected) or Counter(expected) != Counter(observed):
        raise ValueError("Evaluation user coverage mismatch (missing, duplicated or unknown users)")


def accumulation_steps(effective: int, micro_batch: int, world_size: int) -> int:
    if min(effective, micro_batch, world_size) < 1 or effective % (micro_batch * world_size):
        raise ValueError("Effective batch must be a positive multiple of worker micro-batches")
    return effective // (micro_batch * world_size)


def _checkpoint_files(directory: Path) -> list[Path]:
    required = ("trainer_state.json", "optimizer.pt", "scheduler.pt")
    weights = any((directory / name).is_file() for name in (
        "pytorch_model.bin", "model.safetensors", "adapter_model.bin", "adapter_model.safetensors",
    ))
    if not weights or not all((directory / name).is_file() for name in required):
        raise ValueError(f"Checkpoint is incomplete, not an exact resume: {directory}")
    if not list(directory.glob("rng_state*.pth")):
        raise ValueError(f"Checkpoint is incomplete (missing RNG state): {directory}")
    return sorted(p for p in directory.iterdir() if p.is_file() and p.name != "complete.json")


def seal_checkpoint(directory: Path) -> None:
    files = _checkpoint_files(directory)
    state = json.loads((directory / "trainer_state.json").read_text(encoding="utf-8"))
    write_json(directory / "complete.json", {
        "global_step": state["global_step"],
        "files": {p.name: file_sha256(p) for p in files},
    })


def verify_checkpoint(directory: Path) -> int:
    _checkpoint_files(directory)
    marker = directory / "complete.json"
    if not marker.is_file():
        raise ValueError(f"Checkpoint is incomplete (missing completion marker): {directory}")
    receipt = json.loads(marker.read_text(encoding="utf-8"))
    actual_files = {p.name for p in _checkpoint_files(directory)}
    if actual_files != set(receipt["files"]):
        raise ValueError(f"Checkpoint checksum manifest/file-set mismatch: {directory}")
    for name, digest in receipt["files"].items():
        require_hash(directory / name, digest)
    return int(receipt["global_step"])
