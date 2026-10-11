from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import unicodedata

import numpy as np

from rrbench.io import bind_run_config, sha256, write_json
from rrbench.semantic_data import download_file

TREC_SOURCES = {
    "topics.dl19-passage.txt": "topics/topics.dl19-passage.txt",
    "topics.dl20.txt": "topics/topics.dl20.txt",
    "qrels.dl19-passage.txt": "qrels/qrels.dl19-passage.txt",
    "qrels.dl20-passage.txt": "qrels/qrels.dl20-passage.txt",
}
TREC_MIRROR = "https://raw.githubusercontent.com/castorini/anserini-tools/master/"


def canonical_query(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def split_fitting_queries(
    queries: dict[str, str], excluded_ids: set[str], excluded_texts: set[str],
    train_count: int, validation_count: int, seed: int,
) -> tuple[list[str], list[str]]:
    if min(train_count, validation_count) <= 0:
        raise ValueError("Fitting and validation query-group counts must be positive.")
    excluded = {canonical_query(text) for text in excluded_texts}
    groups = {}
    for identifier in sorted(queries):
        text = canonical_query(queries[identifier])
        if not text:
            raise ValueError("IR fitting query is empty after canonical text grouping.")
        if identifier in excluded_ids or text in excluded:
            continue
        groups.setdefault(text, identifier)
    ordered = sorted(
        groups.values(),
        key=lambda identifier: hashlib.sha256(f"{seed}:{canonical_query(queries[identifier])}".encode()).digest(),
    )
    if len(ordered) < train_count + validation_count:
        raise ValueError("Not enough disjoint fitting query-text groups outside final evaluation.")
    return ordered[:train_count], ordered[train_count:train_count + validation_count]


def load_queries(path: Path) -> dict[str, str]:
    queries = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            identifier, separator, text = line.rstrip("\n").partition("\t")
            if not separator or not text.strip() or identifier in queries:
                raise ValueError("IR query file has an invalid, empty or duplicate query.")
            queries[identifier] = text
    return queries


def load_qrels(path: Path, catalog_rows: int = 8841823) -> dict[str, dict[int, int]]:
    qrels = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            fields = line.split()
            if len(fields) != 4:
                raise ValueError("Judgments require the four-field official TREC qrel schema.")
            query, _, item, relevance = fields
            item, relevance = int(item), int(relevance)
            if not 0 <= item < catalog_rows or not 0 <= relevance <= 3:
                raise ValueError("Judgment item ID or graded relevance is outside the declared benchmark.")
            values = qrels.setdefault(query, {})
            if item in values and values[item] != relevance:
                raise ValueError("Judgment contains contradictory relevance for one query/item.")
            values[item] = relevance
    return qrels


def judged_metrics(
    query_ids: list[str], rankings: np.ndarray, qrels: dict[str, dict[int, int]],
    binary_threshold: int = 1,
) -> dict[str, Any]:
    if rankings.ndim != 2 or len(rankings) != len(query_ids) or len(set(query_ids)) != len(query_ids):
        raise ValueError("Judged rankings require aligned unique query rows.")
    if not 1 <= binary_threshold <= 3 or not len(rankings):
        raise ValueError("A positive relevance threshold and nonempty evaluation panel are required.")
    rows = []
    for query, ranked in zip(query_ids, rankings):
        if query not in qrels:
            raise ValueError("Evaluation query has no declared judgment set.")
        candidates = [int(value) for value in ranked if value >= 0]
        if len(candidates) != len(set(candidates)):
            raise ValueError("Retrieved duplicate candidate IDs would inflate relevance gain.")
        grades = qrels[query]
        relevant = {item for item, grade in grades.items() if grade >= binary_threshold}
        if not relevant:
            raise ValueError("Judged panel contains a query without threshold-qualified relevant items.")
        row = {"query_id": query}
        for cutoff in (1, 10, 100, 1000):
            selected = candidates[:cutoff]
            hits = [item in relevant for item in selected]
            row[f"recall@{cutoff}"] = len(set(selected) & relevant) / len(relevant)
            row[f"mrr@{cutoff}"] = next((1 / (rank + 1) for rank, hit in enumerate(hits) if hit), 0.0)
            discount = lambda values: sum(value / math.log2(rank + 2) for rank, value in enumerate(values))
            gain = discount([grades.get(item, 0) for item in selected])
            ideal = discount(sorted(grades.values(), reverse=True)[:cutoff])
            row[f"ndcg@{cutoff}"] = gain / ideal if ideal else 0.0
            row[f"judged_fraction@{cutoff}"] = sum(item in grades for item in selected) / len(selected) if selected else 0.0
        rows.append(row)
    values = {
        name: float(np.mean([row[name] for row in rows])) for name in rows[0] if name != "query_id"
    }
    return {
        **values, "queries": len(rows), "binary_relevance_threshold": binary_threshold,
        "ndcg_gain": "Raw graded relevance, matching default trec_eval gain.",
        "unjudged_policy": "Unjudged items contribute zero gain; no negative-truth completeness claim.",
        "per_query": rows,
    }


def panel_metrics(panel: dict[str, Any], rankings: np.ndarray) -> dict[str, Any]:
    qrels = {
        query: {int(item): int(grade) for item, grade in labels.items()}
        for query, labels in panel["qrels"].items()
    }
    if "evaluation_groups" not in panel:
        return judged_metrics(panel["query_ids"], rankings, qrels, panel["binary_threshold"])
    groups = panel["evaluation_groups"]
    if len(groups) != len(rankings) or len(panel["query_ids"]) != len(rankings):
        raise ValueError("Combined official evaluation groups do not align with query rows.")
    results = {}
    for name in sorted(set(groups)):
        indices = [index for index, group in enumerate(groups) if group == name]
        queries = [panel["query_ids"][index] for index in indices]
        if name not in panel["threshold_by_group"]:
            raise ValueError("Official evaluation group lacks its explicit binary relevance threshold.")
        results[name] = judged_metrics(
            queries, rankings[indices], {query: qrels[query] for query in queries},
            panel["threshold_by_group"][name],
        )
    return {
        "by_group": results, "queries": len(rankings),
        "policy": "Separate official judged query populations and thresholds; no misleading combined relevance mean.",
    }


class PassageStore:
    def __init__(self, source: Path, offsets: Path) -> None:
        self.source = source
        self.offsets = np.load(offsets, mmap_mode="r")
        if (
            self.offsets.ndim != 1 or len(self.offsets) <= 1 or self.offsets[0] != 0
            or self.offsets[-1] != source.stat().st_size or not np.all(np.diff(self.offsets.astype(np.int64)) > 0)
        ):
            raise ValueError("Passage byte-offset mapping does not cover contiguous nonempty corpus rows.")

    def fetch(self, identifiers: np.ndarray) -> list[str]:
        if identifiers.ndim != 1 or not np.issubdtype(identifiers.dtype, np.integer):
            raise ValueError("Text access requires a one-dimensional integer item-ID list.")
        if ((identifiers < 0) | (identifiers >= len(self.offsets) - 1)).any():
            raise ValueError("Invalid passage ID cannot be silently remapped to another item.")
        texts = []
        with self.source.open("rb") as stream:
            for identifier in identifiers:
                stream.seek(int(self.offsets[identifier]))
                line = stream.read(int(self.offsets[identifier + 1] - self.offsets[identifier]))
                found, separator, text = line.partition(b"\t")
                if not separator or int(found) != int(identifier) or not text.strip():
                    raise ValueError("Passage lookup changed the exact item-ID/text mapping.")
                texts.append(text.decode("utf-8").rstrip("\n"))
        return texts


def validate_prepared_panels(output: Path) -> dict[str, Any]:
    completed = json.loads((output / "completed.json").read_text(encoding="utf-8"))
    if not completed["complete"]:
        raise ValueError("IR prepared panels are not complete.")
    for name, expected in completed["panel_sha256"].items():
        if sha256(output / (name + ".json")) != expected:
            raise ValueError("Frozen IR query/judgment panel bytes changed.")
    return completed


def prepare_ir(data: Path, output: Path, fitting_count: int = 100000, validation_count: int = 5000) -> None:
    downloads = data / "downloads" / "trec"
    receipts = [
        download_file(downloads, name, {"url": TREC_MIRROR + relative})
        for name, relative in TREC_SOURCES.items()
    ]
    raw = data / "raw" / "msmarco"
    dev = load_queries(raw / "queries.dev.small.tsv")
    dev_qrels = load_qrels(raw / "qrels.dev.small.tsv")
    if len(dev) != 6980 or set(dev) != set(dev_qrels):
        raise ValueError("Official dev-small queries/judgments do not match the6980-query contract.")
    dl19 = load_queries(downloads / "topics.dl19-passage.txt")
    dl20 = load_queries(downloads / "topics.dl20.txt")
    labels19 = load_qrels(downloads / "qrels.dl19-passage.txt")
    labels20 = load_qrels(downloads / "qrels.dl20-passage.txt")
    for queries, labels, expected in ((dl19, labels19, 43), (dl20, labels20, 54)):
        if len(labels) != expected or not set(labels).issubset(queries):
            raise ValueError("TREC judged-query panel differs from its declared official subset.")
        if any(not any(grade >= 2 for grade in values.values()) for values in labels.values()):
            raise ValueError("TREC panel contains a query without answer-bearing grade>=2 passages.")
    train = load_queries(raw / "queries.train.tsv")
    train_qrels = load_qrels(data / "downloads" / "qrels.train.tsv")
    fitting = {query: text for query, text in train.items() if query in train_qrels}
    excluded_ids = set(dev) | set(dl19) | set(dl20)
    excluded_texts = set(dev.values()) | set(dl19.values()) | set(dl20.values())
    selected_train, selected_validation = split_fitting_queries(
        fitting, excluded_ids, excluded_texts, fitting_count, validation_count, 20261010,
    )
    binding = {
        "query_split_seed": 20261010, "fitting_queries": selected_train,
        "validation_queries": selected_validation, "sources": receipts,
        "train_queries_sha256": sha256(raw / "queries.train.tsv"),
        "train_qrels_sha256": sha256(data / "downloads" / "qrels.train.tsv"),
        "dev_small_queries_sha256": sha256(raw / "queries.dev.small.tsv"),
        "dev_small_qrels_sha256": sha256(raw / "qrels.dev.small.tsv"),
        "final_query_group_policy": "All dev-small and all TREC19/20 topic IDs/canonicaltexts excluded before fitting.",
    }
    bind_run_config(output, binding)
    if (output / "completed.json").exists():
        validate_prepared_panels(output)
        print("FROZEN_IR_PANELS_REUSED_WITH_VERIFIED_BYTES", flush=True)
        return
    panels = {
        "fit": (selected_train, fitting, train_qrels, 1),
        "validation": (selected_validation, fitting, train_qrels, 1),
        "dev-small": (sorted(dev), dev, dev_qrels, 1),
        "trec19": (sorted(labels19), dl19, labels19, 2),
        "trec20": (sorted(labels20), dl20, labels20, 2),
    }
    for name, (identifiers, queries, labels, threshold) in panels.items():
        write_json(output / (name + ".json"), {
            "name": name, "query_ids": identifiers, "queries": [queries[query] for query in identifiers],
            "qrels": {query: labels[query] for query in identifiers}, "binary_threshold": threshold,
        })
    write_json(output / "completed.json", {
        "complete": True, "panel_queries": {name: len(row[0]) for name, row in panels.items()},
        "fitting_split_sha256": sha256(output / "run-config.json"),
        "panel_sha256": {name: sha256(output / (name + ".json")) for name in panels},
        "sources": receipts, "corpus_items": 8841823,
        "judgment_source": "NIST TREC judgments mirrored by verified Anserini-tools repository paths.",
        "pretraining_exposure": "Pinned pretrained models may have seen MS MARCO; not untouched zero-shot evaluation.",
    })
    print("IR_FITTING_AND_OFFICIAL_FINAL_PANELS_FROZEN", {name: len(row[0]) for name, row in panels.items()}, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze real IR query groups and official judged panels.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare_ir(Path(args.data), Path(args.output))


if __name__ == "__main__":
    main()
