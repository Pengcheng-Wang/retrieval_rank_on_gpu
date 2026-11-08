from __future__ import annotations

import ast
import csv
import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


field_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(field_limit)
        break
    except OverflowError:
        field_limit //= 10


@dataclass(frozen=True)
class SequenceRecord:
    user_index: int
    user_id: int
    items: tuple[int, ...]
    ratings: tuple[int, ...]
    timestamps: tuple[int, ...]


@dataclass(frozen=True)
class RankingExample:
    user_index: int
    history_items: tuple[int, ...]
    history_ratings: tuple[int, ...]
    history_timestamps: tuple[int, ...]
    target_item: int
    request_time: int


def _parse_ints(value: str) -> tuple[int, ...]:
    parsed = ast.literal_eval(value)
    if isinstance(parsed, (int, float)):
        return (int(parsed),)
    return tuple(int(item) for item in parsed)


def read_sequences(path: Path) -> list[SequenceRecord]:
    records = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for user_index, row in enumerate(reader):
            items = _parse_ints(row["sequence_item_ids"])
            ratings = _parse_ints(row["sequence_ratings"])
            timestamps = _parse_ints(row["sequence_timestamps"])
            if not (len(items) == len(ratings) == len(timestamps)):
                raise ValueError(f"Mismatched sequence lengths for row {user_index}")
            if len(items) < 3:
                raise ValueError(f"Need train/validation/test events for row {user_index}")
            if any(left > right for left, right in zip(timestamps, timestamps[1:])):
                raise ValueError(f"Nonchronological timestamps for row {user_index}")
            records.append(
                SequenceRecord(
                    user_index=user_index,
                    user_id=int(row["user_id"]),
                    items=items,
                    ratings=ratings,
                    timestamps=timestamps,
                )
            )
    return records


def read_catalog(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        ids = sorted({int(row["movie_id"]) for row in csv.DictReader(handle)})
    if not ids or ids[0] <= 0:
        raise ValueError("Catalog IDs must be positive.")
    return np.asarray(ids, dtype=np.int64)


def _hash_key(seed: int, user_index: int, position: int) -> str:
    return hashlib.sha256(f"{seed}:{user_index}:{position}".encode()).hexdigest()


def _example(record: SequenceRecord, position: int, max_history: int) -> RankingExample:
    start = max(0, position - max_history)
    return RankingExample(
        user_index=record.user_index,
        history_items=record.items[start:position],
        history_ratings=record.ratings[start:position],
        history_timestamps=record.timestamps[start:position],
        target_item=record.items[position],
        request_time=record.timestamps[position],
    )


def select_training_examples(
    records: Sequence[SequenceRecord],
    *,
    maximum_examples: int,
    per_user_cap: int,
    max_history: int,
    seed: int,
) -> list[RankingExample]:
    selected: list[tuple[str, RankingExample]] = []
    for record in records:
        positions = list(range(1, len(record.items) - 2))
        positions.sort(key=lambda position: _hash_key(seed, record.user_index, position))
        for position in positions[:per_user_cap]:
            selected.append(
                (
                    _hash_key(seed + 1, record.user_index, position),
                    _example(record, position, max_history),
                )
            )
    selected.sort(key=lambda row: row[0])
    return [example for _, example in selected[:maximum_examples]]


def evaluation_examples(
    records: Sequence[SequenceRecord],
    *,
    split: str,
    max_history: int,
) -> list[RankingExample]:
    if split == "validation":
        offset = 2
    elif split == "test":
        offset = 1
    else:
        raise ValueError(f"Unsupported split: {split}")
    return [_example(record, len(record.items) - offset, max_history) for record in records]


def training_item_popularity(
    records: Sequence[SequenceRecord],
    num_items: int,
) -> np.ndarray:
    counts = np.zeros(num_items, dtype=np.int64)
    for record in records:
        for item in record.items[:-2]:
            counts[item] += 1
    return counts


def _pad_examples(
    examples: Sequence[RankingExample],
    max_history: int,
) -> dict[str, np.ndarray]:
    count = len(examples)
    histories = np.zeros((count, max_history), dtype=np.int32)
    ratings = np.full((count, max_history), -1, dtype=np.int8)
    timestamps = np.zeros((count, max_history), dtype=np.int64)
    lengths = np.zeros(count, dtype=np.int16)
    users = np.zeros(count, dtype=np.int32)
    targets = np.zeros(count, dtype=np.int32)
    request_times = np.zeros(count, dtype=np.int64)
    for row, example in enumerate(examples):
        length = min(len(example.history_items), max_history)
        # Official TransAct consumes the latest actions first.
        item_values = list(reversed(example.history_items[-length:]))
        rating_values = list(reversed(example.history_ratings[-length:]))
        time_values = list(reversed(example.history_timestamps[-length:]))
        histories[row, :length] = item_values
        ratings[row, :length] = rating_values
        timestamps[row, :length] = time_values
        lengths[row] = length
        users[row] = example.user_index
        targets[row] = example.target_item
        request_times[row] = example.request_time
    return {
        "histories": histories,
        "ratings": ratings,
        "timestamps": timestamps,
        "lengths": lengths,
        "users": users,
        "targets": targets,
        "request_times": request_times,
    }


def fixed_negatives(
    examples: Sequence[RankingExample],
    catalog_ids: np.ndarray,
    *,
    count: int,
    seed: int,
) -> np.ndarray:
    output = np.zeros((len(examples), count), dtype=np.int32)
    catalog = [int(item) for item in catalog_ids]
    for row, example in enumerate(examples):
        excluded = set(example.history_items)
        excluded.add(example.target_item)
        candidates = [item for item in catalog if item not in excluded]
        if len(candidates) < count:
            raise ValueError("Not enough negative candidates.")
        digest = hashlib.sha256(
            f"{seed}:{example.user_index}:{example.target_item}".encode()
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        output[row] = rng.choice(candidates, size=count, replace=False)
    return output


def selection_indices(users: np.ndarray, size: int, seed: int) -> np.ndarray:
    keys = [
        (hashlib.sha256(f"{seed}:{int(user)}".encode()).hexdigest(), index)
        for index, user in enumerate(users)
    ]
    keys.sort()
    return np.asarray([index for _, index in keys[:size]], dtype=np.int32)


def prepare_ranking_data(
    *,
    sequence_csv: Path,
    catalog_csv: Path,
    output: Path,
    max_history: int,
    maximum_examples: int,
    per_user_cap: int,
    negatives: int,
    seed: int,
) -> dict[str, object]:
    records = read_sequences(sequence_csv)
    catalog_ids = read_catalog(catalog_csv)
    num_items = int(catalog_ids.max()) + 1
    popularity = training_item_popularity(records, num_items)
    train_examples = select_training_examples(
        records,
        maximum_examples=maximum_examples,
        per_user_cap=per_user_cap,
        max_history=max_history,
        seed=seed,
    )
    validation = evaluation_examples(records, split="validation", max_history=max_history)
    test = evaluation_examples(records, split="test", max_history=max_history)
    train_arrays = _pad_examples(train_examples, max_history)
    validation_arrays = _pad_examples(validation, max_history)
    test_arrays = _pad_examples(test, max_history)
    train_arrays["negatives"] = fixed_negatives(
        train_examples, catalog_ids, count=negatives, seed=seed
    )
    validation_arrays["selection_indices"] = selection_indices(
        validation_arrays["users"], min(1000, len(validation)), seed
    )
    test_arrays["user_partition_validation_indices"] = selection_indices(
        test_arrays["users"], min(1000, len(test)), seed
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        catalog_ids=catalog_ids,
        popularity=popularity,
        **{f"train_{key}": value for key, value in train_arrays.items()},
        **{f"validation_{key}": value for key, value in validation_arrays.items()},
        **{f"test_{key}": value for key, value in test_arrays.items()},
    )
    maximum_gap = int(
        max(
            example.request_time - example.history_timestamps[-1]
            for example in train_examples + validation + test
        )
    )
    return {
        "sequence_csv_sha256": hashlib.sha256(sequence_csv.read_bytes()).hexdigest(),
        "catalog_csv_sha256": hashlib.sha256(catalog_csv.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "users": len(records),
        "catalog_rows": len(catalog_ids),
        "num_item_embeddings": num_items,
        "training_examples": len(train_examples),
        "validation_examples": len(validation),
        "test_examples": len(test),
        "negative_count": negatives,
        "maximum_history": max_history,
        "maximum_request_gap_seconds": maximum_gap,
        "maximum_popularity": int(popularity.max()),
        "training_target_cold_count": int(
            sum(popularity[example.target_item] == 0 for example in train_examples)
        ),
    }


def dense_features(
    *,
    histories: np.ndarray,
    ratings: np.ndarray,
    lengths: np.ndarray,
    candidates: np.ndarray,
    request_times: np.ndarray,
    timestamps: np.ndarray,
    popularity: np.ndarray,
    max_history: int,
    maximum_popularity: int,
    maximum_gap: int,
) -> np.ndarray:
    batch, candidate_count = candidates.shape
    safe_lengths = np.maximum(lengths.astype(np.float32), 1.0)
    rating_sum = np.where(ratings > 0, ratings, 0).sum(axis=1, dtype=np.float64)
    mean_rating = (rating_sum / safe_lengths / 5.0).astype(np.float32)
    history_ratio = (lengths.astype(np.float32) / max_history).clip(0.0, 1.0)
    last_times = timestamps[:, 0]
    gap = np.maximum(request_times - last_times, 0)
    gap_feature = (
        np.log1p(gap.astype(np.float64)) / math.log1p(max(maximum_gap, 1))
    ).astype(np.float32)
    popularity_feature = (
        np.log1p(popularity[candidates].astype(np.float64))
        / math.log1p(max(maximum_popularity, 1))
    ).astype(np.float32)
    dense = np.empty((batch, candidate_count, 4), dtype=np.float32)
    dense[:, :, 0] = history_ratio[:, None]
    dense[:, :, 1] = mean_rating[:, None]
    dense[:, :, 2] = gap_feature[:, None]
    dense[:, :, 3] = popularity_feature
    return dense


def fingerprint_arrays(arrays: Iterable[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def rank_interleaved_union(inputs: Sequence[np.ndarray], cap: int) -> np.ndarray:
    if not inputs or cap <= 0:
        raise ValueError("At least one candidate matrix and a positive cap are required.")
    shape = inputs[0].shape
    if len(shape) != 2 or any(source.shape != shape for source in inputs):
        raise ValueError("Candidate matrices must have the same two-dimensional shape.")
    if cap > shape[1]:
        raise ValueError("The cap cannot exceed each source's candidate depth.")
    output = np.empty((shape[0], cap), dtype=np.int32)
    for row in range(shape[0]):
        seen: set[int] = set()
        merged: list[int] = []
        for rank in range(shape[1]):
            for source in inputs:
                item = int(source[row, rank])
                if item > 0 and item not in seen:
                    seen.add(item)
                    merged.append(item)
                    if len(merged) == cap:
                        break
            if len(merged) == cap:
                break
        if len(merged) != cap:
            raise RuntimeError(f"Only {len(merged)} unique candidates for row {row}")
        output[row] = merged
    return output


def select_hard_negatives(
    retrieved: np.ndarray,
    targets: np.ndarray,
    count: int,
) -> np.ndarray:
    if retrieved.ndim != 2 or targets.shape != (retrieved.shape[0],):
        raise ValueError("Retrieved candidates and targets have incompatible shapes.")
    if count <= 0 or retrieved.shape[1] <= count:
        raise ValueError("Retrieval depth must exceed the requested negative count.")
    output = np.empty((retrieved.shape[0], count), dtype=np.int32)
    for row, (candidates, target) in enumerate(zip(retrieved, targets)):
        selected = []
        seen: set[int] = set()
        for value in candidates:
            item = int(value)
            if item > 0 and item != int(target) and item not in seen:
                seen.add(item)
                selected.append(item)
                if len(selected) == count:
                    break
        if len(selected) != count:
            raise RuntimeError(f"Only {len(selected)} hard negatives for row {row}")
        output[row] = selected
    return output


def hard_negative_ids_and_ranks(
    retrieved: np.ndarray,
    targets: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    negatives = select_hard_negatives(retrieved, targets, count)
    negative_ranks = np.empty_like(negatives)
    target_ranks = np.full(
        retrieved.shape[0], retrieved.shape[1] + 1, dtype=np.int32
    )
    for row, (candidates, target) in enumerate(zip(retrieved, targets)):
        rank_by_item: dict[int, int] = {}
        for rank, item in enumerate(candidates, start=1):
            value = int(item)
            if value > 0:
                rank_by_item.setdefault(value, rank)
        target_ranks[row] = rank_by_item.get(int(target), retrieved.shape[1] + 1)
        negative_ranks[row] = [rank_by_item[int(item)] for item in negatives[row]]
    return negatives, negative_ranks, target_ranks
