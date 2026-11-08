from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any


EPOCH_RE = re.compile(
    r"eval @ epoch (?P<epoch>\d+).*?"
    r"NDCG@10 (?P<ndcg10>[0-9.]+), NDCG@50 (?P<ndcg50>[0-9.]+), "
    r"HR@10 (?P<hr10>[0-9.]+), HR@50 (?P<hr50>[0-9.]+), MRR (?P<mrr>[0-9.]+)"
)

field_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(field_limit)
        break
    except OverflowError:
        field_limit //= 10
TRAIN_RE = re.compile(
    r"batch-stat \(train\): step (?P<step>\d+) .*?: (?P<loss>[-+0-9.eE]+)"
)


def _percentile(values: list[int], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    offset = fraction * (len(ordered) - 1)
    lower = math.floor(offset)
    upper = math.ceil(offset)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] * (upper - offset) + ordered[upper] * (offset - lower)


def _parse_sequence(value: str) -> list[int]:
    parsed = ast.literal_eval(value)
    if isinstance(parsed, int):
        return [parsed]
    return [int(item) for item in parsed]


def summarize_csv(path: Path, catalog_path: Path | None = None) -> dict[str, Any]:
    users = 0
    events = 0
    unique_items: set[int] = set()
    training_items: set[int] = set()
    target_items: list[int] = []
    history_lengths: list[int] = []
    nonmonotonic_timestamps = 0
    empty_or_singleton_users = 0
    target_in_history = 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "user_id",
            "sequence_item_ids",
            "sequence_ratings",
            "sequence_timestamps",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        for row in reader:
            item_ids = _parse_sequence(row["sequence_item_ids"])
            ratings = _parse_sequence(row["sequence_ratings"])
            timestamps = _parse_sequence(row["sequence_timestamps"])
            if not (len(item_ids) == len(ratings) == len(timestamps)):
                raise ValueError(f"Mismatched sequence lengths for user {row['user_id']}")
            users += 1
            events += len(item_ids)
            unique_items.update(item_ids)
            if item_ids:
                training_items.update(item_ids[:-1])
                target_items.append(item_ids[-1])
            history_lengths.append(max(0, len(item_ids) - 1))
            if len(item_ids) < 2:
                empty_or_singleton_users += 1
            if any(left > right for left, right in zip(timestamps, timestamps[1:])):
                nonmonotonic_timestamps += 1
            if item_ids and item_ids[-1] in set(item_ids[:-1]):
                target_in_history += 1
    result = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "users": users,
        "events": events,
        "unique_items": len(unique_items),
        "minimum_item_id": min(unique_items) if unique_items else None,
        "maximum_item_id": max(unique_items) if unique_items else None,
        "history_length": {
            "minimum": min(history_lengths) if history_lengths else None,
            "mean": statistics.fmean(history_lengths) if history_lengths else None,
            "p50": _percentile(history_lengths, 0.50),
            "p95": _percentile(history_lengths, 0.95),
            "maximum": max(history_lengths) if history_lengths else None,
        },
        "empty_or_singleton_users": empty_or_singleton_users,
        "nonmonotonic_timestamp_users": nonmonotonic_timestamps,
        "heldout_target_already_in_history_users": target_in_history,
        "split_coverage": {
            "training_events": events - len(target_items),
            "distinct_training_items": len(training_items),
            "heldout_events": len(target_items),
            "distinct_heldout_items": len(set(target_items)),
            "warm_heldout_events": sum(
                target in training_items for target in target_items
            ),
            "cold_heldout_events": sum(
                target not in training_items for target in target_items
            ),
            "cold_heldout_item_ids_sha256": hashlib.sha256(
                "\n".join(
                    str(target)
                    for target in sorted(
                        {
                            target
                            for target in target_items
                            if target not in training_items
                        }
                    )
                ).encode()
            ).hexdigest(),
            "warm_heldout_fraction": (
                sum(target in training_items for target in target_items)
                / len(target_items)
                if target_items
                else None
            ),
        },
    }
    if catalog_path is not None:
        with catalog_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            catalog_ids = {int(row["movie_id"]) for row in reader}
        result["evaluation_catalog"] = {
            "path": str(catalog_path),
            "sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
            "rows": len(catalog_ids),
            "rated_items": len(unique_items),
            "unrated_catalog_rows": len(catalog_ids.difference(unique_items)),
            "rated_items_missing_from_catalog": len(unique_items.difference(catalog_ids)),
        }
    return result


def select_smoke_users(source: Path, destination: Path, users: int, seed: int) -> dict[str, Any]:
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if fieldnames is None:
        raise ValueError("Input CSV has no header")
    if users <= 0 or users > len(rows):
        raise ValueError(f"Requested {users} users from {len(rows)} rows")

    def key(row: dict[str, str]) -> str:
        payload = f"{seed}:{row['user_id']}".encode()
        return hashlib.sha256(payload).hexdigest()

    selected = sorted(rows, key=key)[:users]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    selected_user_ids = [str(row["user_id"]) for row in selected]
    return {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "destination": str(destination),
        "destination_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "users": users,
        "seed": seed,
        "selected_user_ids_sha256": hashlib.sha256(
            "\n".join(selected_user_ids).encode()
        ).hexdigest(),
    }


def parse_log(path: Path) -> dict[str, Any]:
    epochs: list[dict[str, Any]] = []
    train_points: list[dict[str, Any]] = []
    for line in path.read_text(errors="replace").splitlines():
        if match := EPOCH_RE.search(line):
            epochs.append(
                {
                    "epoch": int(match.group("epoch")),
                    "ndcg@10": float(match.group("ndcg10")),
                    "ndcg@50": float(match.group("ndcg50")),
                    "hr@10": float(match.group("hr10")),
                    "hr@50": float(match.group("hr50")),
                    "mrr": float(match.group("mrr")),
                }
            )
        if match := TRAIN_RE.search(line):
            train_points.append(
                {
                    "step": int(match.group("step")),
                    "loss": float(match.group("loss")),
                }
            )
    if not epochs:
        raise ValueError(f"No completed epoch evaluations found in {path}")
    return {
        "log": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "completed_epochs": len(epochs),
        "final": epochs[-1],
        "best_observed_ndcg10_diagnostic": max(epochs, key=lambda row: row["ndcg@10"]),
        "epochs": epochs,
        "train_points": train_points,
    }


def parse_events(path: Path) -> dict[str, Any]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    files = sorted(path.rglob("events.out.tfevents.*"))
    if not files:
        raise ValueError(f"No TensorBoard event files found under {path}")
    runs = []
    for event_file in files:
        accumulator = EventAccumulator(str(event_file))
        accumulator.Reload()
        scalar_tags = set(accumulator.Tags().get("scalars", []))
        epoch_tags = sorted(tag for tag in scalar_tags if tag.startswith("eval_epoch/"))
        if not epoch_tags:
            continue
        values = {
            tag.removeprefix("eval_epoch/"): [
                {"step": event.step, "value": event.value}
                for event in accumulator.Scalars(tag)
            ]
            for tag in epoch_tags
        }
        final_step = max(event["step"] for events in values.values() for event in events)
        final = {
            metric: next(
                event["value"] for event in reversed(events) if event["step"] == final_step
            )
            for metric, events in values.items()
            if any(event["step"] == final_step for event in events)
        }
        losses = [
            {"step": event.step, "value": event.value}
            for event in accumulator.Scalars("losses/ar_loss")
        ] if "losses/ar_loss" in scalar_tags else []
        runs.append(
            {
                "event_file": str(event_file),
                "sha256": hashlib.sha256(event_file.read_bytes()).hexdigest(),
                "final_epoch": final_step,
                "final": final,
                "epoch_values": values,
                "training_losses": losses,
            }
        )
    if not runs:
        raise ValueError(f"No completed epoch metrics found under {path}")
    return {"event_root": str(path), "runs": runs}


def summarize_memory(path: Path, total_memory_mib: int) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No memory samples found in {path}")
    used = [int(row["memory_used_mib"].strip()) for row in rows]
    utilization = [int(row["utilization_gpu_percent"].strip()) for row in rows]
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "samples": len(rows),
        "maximum_memory_used_mib": max(used),
        "minimum_free_mib": total_memory_mib - max(used),
        "maximum_utilization_percent": max(utilization),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize = subparsers.add_parser("summarize-csv")
    summarize.add_argument("--input", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    summarize.add_argument("--catalog", type=Path)

    smoke = subparsers.add_parser("select-smoke")
    smoke.add_argument("--input", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--receipt", type=Path, required=True)
    smoke.add_argument("--users", type=int, default=100)
    smoke.add_argument("--seed", type=int, default=42)

    log = subparsers.add_parser("parse-log")
    log.add_argument("--input", type=Path, required=True)
    log.add_argument("--output", type=Path, required=True)

    events = subparsers.add_parser("parse-events")
    events.add_argument("--input", type=Path, required=True)
    events.add_argument("--output", type=Path, required=True)

    memory = subparsers.add_parser("summarize-memory")
    memory.add_argument("--input", type=Path, required=True)
    memory.add_argument("--output", type=Path, required=True)
    memory.add_argument("--total-memory-mib", type=int, required=True)

    args = parser.parse_args()
    if args.command == "summarize-csv":
        result = summarize_csv(args.input, args.catalog)
        output = args.output
    elif args.command == "select-smoke":
        result = select_smoke_users(args.input, args.output, args.users, args.seed)
        output = args.receipt
    elif args.command == "parse-log":
        result = parse_log(args.input)
        output = args.output
    elif args.command == "parse-events":
        result = parse_events(args.input)
        output = args.output
    else:
        result = summarize_memory(args.input, args.total_memory_mib)
        output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
