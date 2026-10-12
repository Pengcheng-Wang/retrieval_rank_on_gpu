from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from rrbench.io import bind_run_config, sha256, write_json
from rrbench.semantic_encoding import validate_completed
from rrbench.semantic_recommendation_baselines import validate_prepared_logged_data


def validate_profile_user_map(users: dict[str, int], count: int) -> None:
    if (
        len(users) != count or any(type(value) is not int for value in users.values())
        or set(users.values()) != set(range(count))
        or any(not key.isdecimal() for key in users)
    ):
        raise ValueError("Profile user map must be a correctly typed bijection onto frozen training rows.")


def validate_profile_item_bank(item_file: Path, approved_ann: Path, rows: int) -> dict[str, Any]:
    source = json.loads((item_file.parent / "completed.json").read_text(encoding="utf-8"))
    items = np.load(item_file, mmap_mode="r")
    if items.dtype != np.float16 or items.shape != (rows, 1024):
        raise ValueError("Static profile bank actual shape/dtype differs from the approved1024D catalog.")
    original = json.loads((approved_ann / "run-config.json").read_text(encoding="utf-8"))
    if original["items_sha256"] != sha256(item_file) or source["vectors_sha256"] != original["items_sha256"]:
        raise ValueError("Static profile bank differs from the already approved original ANN item vectors.")
    validate_completed(item_file.parent, rows, 0)
    if source["representation"]["model"] != "Qwen/Qwen3-Embedding-8B" or source["stored_dimension"] != 1024:
        raise ValueError("Static profile diagnostic must preserve the approved Qwen8MRL1024representation.")
    return source


def validate_profile_inventory(
    output: Path, binding: dict[str, Any], users: int, dimension: int = 1024,
) -> dict[str, Any]:
    complete = json.loads((output / "completed.json").read_text(encoding="utf-8"))
    if complete.get("complete") is not True or complete.get("binding") != binding:
        raise ValueError("Static profile artifact is not complete or uses a different binding.")
    if set(complete["panels"]) != {"validation", "test"}:
        raise ValueError("Static profile artifact lacks the complete held-out phase set.")
    arrays = {}
    for name, key in (
        ("profiles.npy", "profiles_sha256"), ("supported_training_user_rows.npy", "supported_users_sha256"),
        ("training_positive_row_counts.npy", "positive_counts_sha256"),
    ):
        if sha256(output / name) != complete[key]:
            raise ValueError("Static profile/support array checksum changed.")
        arrays[name] = np.load(output / name, mmap_mode="r")
    supported, counts, profiles = (
        arrays["supported_training_user_rows.npy"], arrays["training_positive_row_counts.npy"], arrays["profiles.npy"],
    )
    if (
        supported.dtype != np.int64 or counts.dtype != np.int64 or counts.shape != (users,)
        or not np.array_equal(supported, np.flatnonzero(counts > 0))
        or profiles.dtype != np.float16 or profiles.shape != (len(supported), dimension)
        or not np.isfinite(profiles).all()
        or not np.allclose(np.linalg.norm(profiles.astype(np.float32), axis=1), 1, atol=0.002)
    ):
        raise ValueError("Static profile/support matrix shapes, user rows or normalized values are inconsistent.")
    for phase, details in complete["panels"].items():
        panel_file, receipt_file = output / (phase + ".json"), output / phase / "completed.json"
        if sha256(panel_file) != details["panel_sha256"] or sha256(receipt_file) != details["query_receipt_sha256"]:
            raise ValueError("Static profile held-out panel/receipt checksum changed.")
        panel = json.loads(panel_file.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
        values = np.load(output / phase / "vectors.npy", mmap_mode="r")
        if (
            not receipt["complete"] or receipt["query_ids"] != panel["query_ids"]
            or receipt["rows"] != len(panel["query_ids"]) or values.shape != (receipt["rows"], dimension)
            or values.dtype != np.float16 or sha256(output / phase / "vectors.npy") != receipt["vectors_sha256"]
        ):
            raise ValueError("Static profile query mapping/payload is incomplete or corrupt.")
    return complete


def training_positive_profiles(
    pairs: np.ndarray, items: np.ndarray, users: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if pairs.ndim != 2 or pairs.shape[1] != 3 or users <= 0 or items.ndim != 2:
        raise ValueError("Static profiles require aligned logged pairs and item vectors.")
    if (pairs[:, 0] < 0).any() or (pairs[:, 0] >= users).any():
        raise ValueError("Static profile training user row is outside the frozen training mapping.")
    if (
        (pairs[:, 1] < 0).any() or (pairs[:, 1] >= len(items)).any()
        or not np.isin(pairs[:, 2], [0, 1]).all()
    ):
        raise ValueError("Static profile training item IDs or binary labels are invalid.")
    positives = pairs[pairs[:, 2] == 1]
    counts = np.bincount(positives[:, 0], minlength=users)
    supported = np.flatnonzero(counts > 0)
    if not len(supported):
        raise ValueError("No admitted training-positive rows support a static profile.")
    profiles = np.empty((len(supported), items.shape[1]), dtype=np.float16)
    ordered = positives[np.argsort(positives[:, 0], kind="stable")]
    offsets = np.r_[0, np.cumsum(counts)]
    for output_row, user in enumerate(supported):
        identifiers = ordered[offsets[user]:offsets[user + 1], 1]
        values = np.asarray(items[identifiers], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Static profile item features are nonfinite.")
        mean = values.mean(axis=0, dtype=np.float32)
        norm = np.linalg.norm(mean)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Static positive profile has a zero or invalid norm; no default query is published.")
        profiles[output_row] = (mean / norm).astype(np.float16)
    return profiles, supported.astype(np.int64), counts


def prepare_profiles(data: Path, item_file: Path, panels: Path, output: Path, approved_ann: Path) -> None:
    complete_data = validate_prepared_logged_data(data)
    source = validate_profile_item_bank(item_file, approved_ann, complete_data["items"])
    items = np.load(item_file, mmap_mode="r")
    if (
        not source["complete"] or sha256(item_file) != source["vectors_sha256"]
        or source["rows"] != complete_data["items"] or source["stored_dimension"] != 1024
    ):
        raise ValueError("Static semantic profile requires the unchanged complete1024D real item bank.")
    binding = {
        "data_receipt_sha256": sha256(data / "completed.json"), "items_sha256": sha256(item_file),
        "users_sha256": complete_data["users_sha256"], "profile": "mean of all catalog-covered positive training exposure rows",
        "repeated_event_weight": True, "dtype": "FP32mean and normalization, then canonicalFP16",
        "status": "Test-informed exploratory follow-up; not a confirmatory exposure-selection ablation.",
        "source_panel_sha256": {phase: sha256(panels / (phase + ".json")) for phase in ("validation", "test")},
        "approved_original_ann_sha256": sha256(approved_ann / "run-config.json"),
        "source_item_receipt_sha256": sha256(item_file.parent / "completed.json"),
    }
    bind_run_config(output, binding)
    if (output / "completed.json").exists():
        validate_profile_inventory(output, binding, complete_data["users"])
        return
    started = time.perf_counter()
    pairs = np.load(data / "pairs.npy", mmap_mode="r")
    profiles, supported, counts = training_positive_profiles(pairs, items, complete_data["users"])
    users = json.loads((data / "users.json").read_text(encoding="utf-8"))
    validate_profile_user_map(users, complete_data["users"])
    indexed = {int(user): row for row, user in enumerate(supported)}
    np.save(output / "profiles.npy", profiles)
    np.save(output / "supported_training_user_rows.npy", supported)
    np.save(output / "training_positive_row_counts.npy", counts)
    prepared = {}
    for phase in ("validation", "test"):
        panel = json.loads((panels / (phase + ".json")).read_text(encoding="utf-8"))
        contract = complete_data["input_contract"]
        if (
            panel["catalog_sha256"] != contract["catalog_sha256"]
            or panel["catalog_ids_sha256"] != contract["catalog_ids_sha256"]
        ):
            raise ValueError("Static profile panel uses a different catalog row mapping.")
        rows, mapped = [], []
        for original_row, user in enumerate(panel["users"]):
            if str(user) not in users:
                raise ValueError("Static semantic profile encountered an unfitted raw user ID.")
            training_row = users[str(user)]
            if training_row in indexed:
                rows.append(original_row)
                mapped.append(indexed[training_row])
            elif counts[training_row] != 0:
                raise ValueError("Supported profile user is missing from its frozen profile index.")
        if not rows:
            raise ValueError("Static profile panel has no training-positive-supported queries.")
        ids = [panel["query_ids"][row] for row in rows]
        new_panel = {
            **panel, "name": "kuairand-static-positive-profile-" + phase,
            "query_ids": ids,
            "users": [panel["users"][row] for row in rows],
            "qrels": {query: panel["qrels"][query] for query in ids},
            "original_query_rows": rows,
            "query_representation": "Offline training-positive event-weighted semantic profile; no onlineLMencoding.",
            "diagnostic_status": binding["status"],
        }
        new_panel.pop("queries")
        for field in ("history_item_rows", "target_clock_lower"):
            if field in new_panel:
                new_panel[field] = [panel[field][row] for row in rows]
        panel_path = output / (phase + ".json")
        write_json(panel_path, new_panel)
        destination = output / phase
        destination.mkdir(parents=True, exist_ok=True)
        np.save(destination / "vectors.npy", profiles[np.asarray(mapped)])
        write_json(destination / "completed.json", {
            "complete": True, "rows": len(rows), "query_ids": ids, "panel_sha256": sha256(panel_path),
            "native_dimension": source["native_dimension"], "stored_dimension": 1024,
            "vectors_sha256": sha256(destination / "vectors.npy"),
            "query_model": "training-derived-static-semantic-profile",
            "item_representation": source["representation"],
            "online_query_encoder": None, "original_query_rows": rows, "profile_binding": binding,
        })
        prepared[phase] = {
            "original_queries": len(panel["query_ids"]), "retained_queries": len(rows),
            "users": len(set(new_panel["users"])), "missing_training_positive_profile_queries": len(panel["query_ids"]) - len(rows),
            "panel_sha256": sha256(panel_path),
            "query_receipt_sha256": sha256(destination / "completed.json"),
        }
    write_json(output / "completed.json", {
        "complete": True, "profiles_sha256": sha256(output / "profiles.npy"),
        "supported_users_sha256": sha256(output / "supported_training_user_rows.npy"),
        "positive_counts_sha256": sha256(output / "training_positive_row_counts.npy"),
        "supported_training_users": len(supported), "panels": prepared,
        "positive_training_rows": int(counts.sum()),
        "offline_seconds": time.perf_counter() - started, "binding": binding,
        "interpretation": "Joint feedback-selection/history-length/aggregation/query-encoder intervention; not an isolated causal effect.",
    })
    validate_profile_inventory(output, binding, complete_data["users"])
    print("TRAINING_POSITIVE_STATIC_SEMANTIC_DIAGNOSTIC_PREPARED", prepared, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed exploratory training-positive semantic user profiles.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--items", required=True)
    parser.add_argument("--panels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--approved-ann", required=True)
    args = parser.parse_args()
    prepare_profiles(
        Path(args.data), Path(args.items), Path(args.panels), Path(args.output), Path(args.approved_ann),
    )


if __name__ == "__main__":
    main()
