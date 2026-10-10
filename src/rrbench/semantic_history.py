from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np

from rrbench.io import bind_run_config, sha256, write_json
from rrbench.semantic_ir import PassageStore

HISTORY_INSTRUCTION = (
    "Given the user's previously exposed video content, retrieve other videos the user is likely to watch for a long time."
)


def frozen_exposure_history(
    item_rows: np.ndarray, upper_bounds: np.ndarray, target_lower: int, maximum: int,
) -> np.ndarray:
    if (
        item_rows.ndim != 1 or item_rows.shape != upper_bounds.shape or maximum <= 0
        or not np.issubdtype(item_rows.dtype, np.integer) or not np.issubdtype(upper_bounds.dtype, np.integer)
        or (np.diff(upper_bounds) < 0).any()
    ):
        raise ValueError("Frozen histories require aligned ordered integer items/clock envelopes and a positive limit.")
    boundary = np.searchsorted(upper_bounds, target_lower, side="right")
    return np.array(item_rows[max(0, boundary - maximum):boundary], dtype=np.int64)


def validate_history_panels(output: Path) -> dict[str, Any]:
    completed = json.loads((output / "completed.json").read_text(encoding="utf-8"))
    if not completed["complete"] or set(completed["panel_sha256"]) != {"fit", "validation", "test"}:
        raise ValueError("Recommendation completion lacks the required fit/validation/test panel set.")
    for name, digest in completed["panel_sha256"].items():
        if sha256(output / (name + ".json")) != digest:
            raise ValueError("Frozen recommendation query/history panel bytes changed.")
    if sha256(output / "training-history.json") != completed["training_history_sha256"]:
        raise ValueError("Frozen training exposure-history bytes changed.")
    return completed


def prepare_history_panels(
    catalog: Path, output: Path, maximum_history: int = 12, blocks_per_user: int = 12, seed: int = 20261010,
) -> dict[str, Any]:
    manifest = json.loads((catalog / "corpus.json").read_text(encoding="utf-8"))
    if manifest["clock_policy_version"] != 2 or not manifest["million_item_gate_pass"]:
        raise ValueError("Frozen history construction requires the approved dual-clock million-item catalog.")
    binding = {
        "catalog_sha256": manifest["catalog_sha256"], "catalog_manifest_sha256": sha256(catalog / "corpus.json"),
        "database_sha256": sha256(catalog / "private-catalog.sqlite"),
        "maximum_history": maximum_history, "blocks_per_user_per_phase": blocks_per_user, "seed": seed,
        "history": "Admitted training exposure items only; upper<=target lower; no label filtering or heldoutappend.",
        "history_text_order": "Most recently exposed first.",
        "reranker_query_token_cap": 160,
        "required_phase_names": ["fit", "validation", "test"], "panel_schema_version": 3,
    }
    bind_run_config(output, binding)
    if (output / "completed.json").exists():
        return validate_history_panels(output)
    original = np.load(catalog / "original_ids.npy", mmap_mode="r")
    mapped = {int(item): row for row, item in enumerate(original)}
    store = PassageStore(catalog / "catalog.tsv", catalog / "catalog_offsets.npy")
    uri = (catalog / "private-catalog.sqlite").resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        histories = defaultdict(list)
        for user, item, upper, timestamp in connection.execute(
            "SELECT e.user,e.item,b.upper,e.timestamp FROM events e JOIN blocks b USING(user,timestamp) "
            "JOIN semantic s USING(item) WHERE b.phase=1 ORDER BY e.user,b.upper,e.timestamp,e.item"
        ):
            histories[user].append((mapped[item], upper, timestamp))
        history_arrays = {
            user: (np.asarray([row[0] for row in values], dtype=np.int64),
                   np.asarray([row[1] for row in values], dtype=np.int64))
            for user, values in histories.items()
        }
        del histories
        panels, counters = {}, {}
        for phase, name in ((1, "fit"), (2, "validation"), (3, "test")):
            selected = defaultdict(list)
            source_blocks = 0
            for user, timestamp, lower in connection.execute(
                "SELECT user,timestamp,lower FROM blocks WHERE phase=? ORDER BY user,timestamp", (phase,),
            ):
                digest = hashlib.sha256(f"{seed}:{phase}:{user}:{timestamp}".encode()).digest()
                selected[user].append((digest, timestamp, lower))
                source_blocks += 1
            chosen = {
                (user, timestamp): lower
                for user, blocks in selected.items()
                for _, timestamp, lower in sorted(blocks)[:blocks_per_user]
            }
            positives, all_targets = defaultdict(set), defaultdict(set)
            for user, timestamp, item, label in connection.execute(
                "SELECT e.user,e.timestamp,e.item,e.label FROM events e JOIN blocks b USING(user,timestamp) "
                "WHERE b.phase=? ORDER BY e.user,e.timestamp", (phase,),
            ):
                key = (user, timestamp)
                if key in chosen and label:
                    all_targets[key].add(item)
                    if item in mapped:
                        positives[key].add(mapped[item])
            ids, queries, qrels, users, histories_used, query_lower = [], [], {}, [], [], []
            skipped_no_history, skipped_no_positive = 0, 0
            for (user, timestamp), lower in sorted(chosen.items()):
                item_rows, upper_bounds = history_arrays.get(user, (np.array([], dtype=np.int64), np.array([], dtype=np.int64)))
                history = frozen_exposure_history(item_rows, upper_bounds, lower, maximum_history)
                if not len(history):
                    skipped_no_history += 1
                    continue
                if not positives[(user, timestamp)]:
                    skipped_no_positive += 1
                    continue
                permitted = upper_bounds <= lower
                if not permitted.any():
                    raise RuntimeError("A frozen history query was published without prior training exposure blocks.")
                text = "\n".join(f"Previously exposed video: {value}" for value in store.fetch(history[::-1]))
                identifier = f"{phase}:{user}:{timestamp}"
                ids.append(identifier)
                queries.append(text)
                qrels[identifier] = {row: 1 for row in sorted(positives[(user, timestamp)])}
                users.append(user)
                histories_used.append(history.tolist())
                query_lower.append(lower)
            if not ids:
                raise ValueError("No warm-catalog positive queries with permitted frozen histories remain.")
            panel = {
                "name": "kuairand-" + name, "query_ids": ids, "queries": queries,
                "qrels": qrels, "binary_threshold": 1, "users": users,
                "history_item_rows": histories_used, "target_clock_lower": query_lower,
                "history_instruction": HISTORY_INSTRUCTION,
                "history_text_order": "Most recently exposed first; prefix truncation retains recent rather than oldest selected items.",
                "reranker_query_token_cap": 160,
                "history_contract": binding["history"],
                "target_condition": "Positive-target membership in the frozen catalog and nonempty admissible training exposure history.",
                "no_seen_item_exclusion": "Repeated-video targets remain valid; no unseen-only recommendation claim.",
            }
            write_json(output / (name + ".json"), panel)
            counters[name] = {
                "source_timestamp_blocks": source_blocks, "label_independently_sampled_blocks": len(chosen),
                "users_sampled": len(selected), "evaluated_positive_warm_history_queries": len(ids),
                "sampled_blocks_without_history": skipped_no_history,
                "sampled_blocks_without_covered_positive_after_history_gate": skipped_no_positive,
                "sampled_unique_positive_target_count": sum(map(len, all_targets.values())),
                "sampled_catalog_covered_positive_target_count": sum(map(len, positives.values())),
                "evaluated_users": len(set(users)), "full_positive_event_coverage": manifest["coverage_counts"],
            }
            panels[name] = sha256(output / (name + ".json"))
        sparse_users = sorted(history_arrays)
        write_json(output / "training-history.json", {
            "users": sparse_users,
            "history_item_rows": [
                frozen_exposure_history(*history_arrays[user], target_lower=1650556800000, maximum=maximum_history).tolist()
                for user in sparse_users
            ],
            "history_contract": binding["history"], "query_cutoff": 1650556800000,
        })
    completed = {
        "complete": True, "catalog_items": len(original), "panel_counts": counters,
        "panel_sha256": panels, "training_history_sha256": sha256(output / "training-history.json"),
        "selection": "Per-user recorded-clock blocks hashed independently of labels; positive queries filtered afterward with attritionreported.",
        "feedback_availability": "No feedback-conditioned histories; retrospective recorded-clock separation, not verifiedonline replay.",
    }
    write_json(output / "completed.json", completed)
    print("FROZEN_RECOMMENDATION_EXPOSURE_HISTORIES_VERIFIED", {name: value["evaluated_positive_warm_history_queries"] for name, value in counters.items()}, flush=True)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare label-independent frozen-training exposure-history query panels.")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare_history_panels(Path(args.catalog), Path(args.output))


if __name__ == "__main__":
    main()
