from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from rrbench.io import bind_run_config, sha256, write_json
from rrbench.semantic_candidates import common_candidate_pool
from rrbench.semantic_rankers import query_representation
from rrbench.semantic_ir import panel_metrics


def stable_rank(domain: str, seed: int, identifier: str) -> tuple[bytes, str]:
    payload = json.dumps([domain, seed, identifier], ensure_ascii=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).digest(), identifier


def latency_query_rows(panel: dict[str, Any], batch_size: int, seed: int = 20261010) -> list[list[int]]:
    identifiers = panel["query_ids"]
    if batch_size <= 0 or len(identifiers) < batch_size * 4 or len(identifiers) != len(set(identifiers)):
        raise ValueError("Latency sampling requires enough unique real queries for four disjoint batches.")
    order = sorted(
        range(len(identifiers)),
        key=lambda row: stable_rank("online-latency-batch-" + str(batch_size), seed, identifiers[row]),
    )[:batch_size * 4]
    return [order[start:start + batch_size] for start in range(0, len(order), batch_size)]


def select_text_rows(panel: dict[str, Any], seed: int = 20261010, maximum_dev: int = 1000) -> list[int]:
    identifiers = panel["query_ids"]
    if len(identifiers) != len(set(identifiers)) or not identifiers or maximum_dev <= 0:
        raise ValueError("Text sampling needs unique nonempty query identity and a positive cohort cap.")
    if "evaluation_groups" in panel:
        groups = panel["evaluation_groups"]
        if len(groups) != len(identifiers) or set(groups) - {"dev-small", "trec19", "trec20"}:
            raise ValueError("Text IR population has a missing or unrecognized official group.")
        dev = [index for index, group in enumerate(groups) if group == "dev-small"]
        selected = sorted(dev, key=lambda row: stable_rank("text-quality-ir-dev", seed, identifiers[row]))[:maximum_dev]
        selected += [index for index, group in enumerate(groups) if group != "dev-small"]
    elif "users" in panel:
        if len(panel["users"]) != len(identifiers):
            raise ValueError("Recommendation text sampling user identities do not align with queries.")
        best = {}
        for row, user in enumerate(panel["users"]):
            key = stable_rank("text-quality-recommendation-" + str(user), seed, identifiers[row])
            if user not in best or key < best[user][0]:
                best[user] = (key, row)
        selected = [value[1] for value in best.values()]
    else:
        raise ValueError("Text sampling requires an explicit official IR or conditional-user population.")
    return sorted(selected)


def freeze_text_panel(source: Path, output: Path) -> dict[str, Any]:
    panel = json.loads(source.read_text(encoding="utf-8"))
    rows = select_text_rows(panel)
    binding = {
        "source_panel_sha256": sha256(source), "seed": 20261010, "selected_rows": rows,
        "status": "Post-test compute-budget adjustment, fixed before bulk expensive-text inference.",
        "selection": "SHA256domain-separated IDs only; qrels, lengths and scores do not determine selection.",
    }
    bind_run_config(output, binding)
    if (output / "completed.json").exists():
        complete = json.loads((output / "completed.json").read_text(encoding="utf-8"))
        if (
            not complete["complete"] or complete["query_ids"] != [panel["query_ids"][row] for row in rows]
            or sha256(output / "panel.json") != complete["panel_sha256"]
            or sha256(output / "source_rows.npy") != complete["row_mapping_sha256"]
        ):
            raise ValueError("Frozen expensive-text query cohort changed.")
        return complete
    selected = {**panel}
    for field in ("query_ids", "queries", "users", "evaluation_groups", "history_item_rows", "target_clock_lower"):
        if field in selected:
            if len(selected[field]) != len(panel["query_ids"]):
                raise ValueError("Frozen source per-query attributes do not align with the full query population.")
            selected[field] = [panel[field][row] for row in rows]
    selected["qrels"] = {query: panel["qrels"][query] for query in selected["query_ids"]}
    selected["sampling"] = binding
    selected["source_query_rows"] = rows
    write_json(output / "panel.json", selected)
    np.save(output / "source_rows.npy", np.asarray(rows, dtype=np.int64))
    complete = {
        "complete": True, "rows": len(rows), "source_rows": len(panel["query_ids"]),
        "panel_sha256": sha256(output / "panel.json"), "row_mapping_sha256": sha256(output / "source_rows.npy"),
        "query_ids": selected["query_ids"], "binding": binding,
    }
    write_json(output / "completed.json", complete)
    return complete


def prepare_common_text_inputs(
    full_panel: Path, frozen_panel: Path, candidates: list[Path], query_vectors: Path, output: Path,
) -> None:
    source = json.loads(full_panel.read_text(encoding="utf-8"))
    selected = json.loads(frozen_panel.read_text(encoding="utf-8"))
    rows = np.asarray(selected["source_query_rows"], dtype=np.int64)
    if [source["query_ids"][row] for row in rows] != selected["query_ids"]:
        raise ValueError("Expensive text cohort IDs do not map to the exact full source panel.")
    predictions = []
    for path in candidates:
        complete = json.loads((path.parent / "completed.json").read_text(encoding="utf-8"))
        if (
            not complete["complete"] or complete["query_ids"] != source["query_ids"]
            or complete["binding"]["panel_sha256"] != sha256(full_panel)
            or sha256(path) != complete["ids_sha256"]
        ):
            raise ValueError("Common-pool retriever outputs do not match the full frozen query panel.")
        values = np.load(path, mmap_mode="r")
        if values.dtype != np.int64 or len(values) != len(source["query_ids"]):
            raise ValueError("Common-pool source candidate shapes/dtypes are invalid.")
        predictions.append(np.array(values[rows], copy=True))
    representation = query_representation(query_vectors)
    original_receipt = json.loads((query_vectors.parent / "completed.json").read_text(encoding="utf-8"))
    if (
        original_receipt["panel_sha256"] != sha256(full_panel)
        or original_receipt["query_ids"] != source["query_ids"]
    ):
        raise ValueError("Cached comparator query vectors do not align with the exact full common-pool query panel.")
    binding = {
        "source_panel_sha256": sha256(full_panel), "frozen_panel_sha256": sha256(frozen_panel),
        "candidate_source_sha256": [sha256(path) for path in candidates], "query_vectors_sha256": sha256(query_vectors),
        "source_order": "Qwen06thenQwen8MRL", "pool": "Rankinterleaveduniqueuniontop1000,labelsneverinjected.",
    }
    bind_run_config(output, binding)
    if (output / "completed.json").exists():
        complete = json.loads((output / "completed.json").read_text(encoding="utf-8"))
        query_complete = json.loads((output / "query-vectors" / "completed.json").read_text(encoding="utf-8"))
        if (
            not complete["complete"] or complete["query_ids"] != selected["query_ids"]
            or sha256(output / "ids.npy") != complete["ids_sha256"]
            or not query_complete["complete"] or query_complete["query_ids"] != selected["query_ids"]
            or sha256(output / "query-vectors" / "vectors.npy") != query_complete["vectors_sha256"]
        ):
            raise ValueError("Completed common-pool/scorer-controlled input payloads changed.")
        return
    ids = common_candidate_pool(predictions, 1000)
    np.save(output / "ids.npy", ids)
    query = np.array(np.load(query_vectors, mmap_mode="r")[rows], copy=True)
    vectors = output / "query-vectors"
    vectors.mkdir(parents=True, exist_ok=True)
    np.save(vectors / "vectors.npy", query)
    write_json(vectors / "completed.json", {
        "complete": True, "rows": len(rows), "query_ids": selected["query_ids"], "panel_sha256": sha256(frozen_panel),
        "native_dimension": original_receipt["native_dimension"], "stored_dimension": representation["stored_dimension"],
        "model": representation["model"], "revision": representation["revision"],
        "specification_sha256": representation["specification_sha256"],
        "query_instruction": representation["query_instruction"], "mrl_representations": {},
        "vectors_sha256": sha256(vectors / "vectors.npy"),
    })
    ceilings = {}
    for cap in (10, 50, 100, 500):
        prefix = ids[:, :cap]
        oracle = np.empty_like(prefix)
        eligible_hits = 0
        grouped = {}
        for index, query in enumerate(selected["query_ids"]):
            grades = {int(item): grade for item, grade in selected["qrels"][query].items()}
            if "evaluation_groups" in selected:
                threshold = selected["threshold_by_group"][selected["evaluation_groups"][index]]
            else:
                threshold = selected["binary_threshold"]
            hit = int(any(grades.get(int(item), 0) >= threshold for item in prefix[index] if item >= 0))
            eligible_hits += hit
            group = selected["evaluation_groups"][index] if "evaluation_groups" in selected else "recommendation"
            values = grouped.setdefault(group, {"queries": 0, "queries_with_known_positive": 0, "candidate_shortfall_queries": 0})
            values["queries"] += 1
            values["queries_with_known_positive"] += hit
            values["candidate_shortfall_queries"] += int((prefix[index] >= 0).sum() < cap)
            order = sorted(range(cap), key=lambda position: -grades.get(int(prefix[index, position]), 0))
            oracle[index] = prefix[index, order]
        ceilings[str(cap)] = {
            "queries_with_known_positive": eligible_hits, "known_judgment_oracle_metrics": panel_metrics(selected, oracle),
            "scope": "Fixed-prefix known-qrel ceiling, not exhaustive true relevance or injected positives.",
            "counts_by_group": grouped,
        }
    write_json(output / "completed.json", {
        "complete": True, "rows": len(rows), "query_ids": selected["query_ids"], "ids_sha256": sha256(output / "ids.npy"),
        "binding": {"panel_sha256": sha256(frozen_panel)}, "positive_injection": False,
        "scored_prefix": 500, "source_rows": rows.tolist(), "inputs": binding,
        "candidate_prefix_ceilings": ceilings,
    })
    print("COMMON_POOL_EXPENSIVE_TEXT_INPUTS_FROZEN", len(rows), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze label-independent expensive ranking cohorts and identical common pools.")
    commands = parser.add_subparsers(dest="operation", required=True)
    panel = commands.add_parser("panel")
    panel.add_argument("--source", required=True)
    panel.add_argument("--output", required=True)
    common = commands.add_parser("common")
    common.add_argument("--source-panel", required=True)
    common.add_argument("--frozen-panel", required=True)
    common.add_argument("--candidate", action="append", required=True)
    common.add_argument("--query-vectors", required=True)
    common.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.operation == "panel":
        freeze_text_panel(Path(args.source), Path(args.output))
    else:
        prepare_common_text_inputs(
            Path(args.source_panel), Path(args.frozen_panel), [Path(value) for value in args.candidate],
            Path(args.query_vectors), Path(args.output),
        )


if __name__ == "__main__":
    main()
