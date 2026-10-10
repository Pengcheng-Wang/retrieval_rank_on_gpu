from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from rrbench.io import sha256, write_json

JOURNALS = ("ann", "stages", "filter", "pipeline", "rank_timing", "quality_ann")


def reject_private_fields(value: Any) -> None:
    forbidden = {
        "private_key", "subscription", "resource_group", "workspace", "job_name",
        "ssh_command", "access_token", "password", "credentials", "connection_string",
        "sas_token", "user_ids", "targets",
    }
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in forbidden:
                raise ValueError(f"Unsafe operational or dataset identifier field: {key}")
            reject_private_fields(child)
    elif isinstance(value, list):
        for child in value:
            reject_private_fields(child)


def aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for key, group in sorted(grouped.items()):
        result = dict(zip(keys, key))
        result["measurements"] = len(group)
        if "effective_effort" in group[0]:
            result["effort_parameter"] = group[0]["effort_parameter"]
            result["effective_effort"] = group[0]["effective_effort"]
        for field in (
            "latency_mean_ms", "recall", "candidate_recall", "candidate_fill_fraction",
            "candidate_geometric_recall", "candidate_hit_rate", "candidate_fill",
            "gpu_resident_delta_bytes", "torch_incremental_peak_bytes",
        ):
            if field in group[0]:
                values = np.asarray([row[field] for row in group], dtype=np.float64)
                result[field] = float(values.mean())
                result[field + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        if "quality" in group[0]:
            result["quality"] = {
                metric: float(np.mean([row["quality"][metric] for row in group]))
                for metric in group[0]["quality"]
            }
        output.append(result)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (dict, list))})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ])


def publish(input_root: Path, output: Path, tables_path: Path) -> dict[str, Any]:
    if not input_root.is_dir():
        raise ValueError("A collected numerical-results directory is required.")
    output.mkdir(parents=True, exist_ok=True)
    raw = {name: [] for name in JOURNALS}
    sources = []
    environments = {}
    quality_runs = []
    audits = []
    profiles = []
    for path in sorted(input_root.rglob("*.jsonl")):
        name = path.stem
        if name not in raw or "smoke" in path.parent.name:
            continue
        if name != "stages" and not (path.parent / "completed.json").exists():
            raise ValueError(f"Refusing to publish incomplete experiment: {path.parent.name}")
        experiment = path.parent.name
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for row in rows:
            row["experiment"] = experiment
            if experiment == "pipeline_1m_s17" and row.get("backend", "").startswith("faiss-"):
                row["status"] = "superseded"
                row["superseded_reason"] = "Re-measured on the quiescent node alongside the optimized pipeline implementation."
            if (
                row.get("backend", "").startswith("silver-")
                and name == "pipeline"
                and row.get("verification_implementation") != "batched-exact-predicate-v2"
            ):
                row["status"] = "superseded"
                row["superseded_reason"] = "Unvectorized exact verification replaced by correctness-equivalent batched evaluation."
            if (
                experiment == "pipeline_1m_s17"
                and row.get("backend") == "silver-int8-global-full"
                and "filtering_implementation" not in row
            ):
                row["status"] = "superseded"
                row["superseded_reason"] = "Initial full-to-partial adapter replaced by measured native GPU full-mask baseline."
            reject_private_fields(row)
        raw[name].extend(rows)
        sources.append({"experiment": experiment, "journal": path.name, "sha256": sha256(path), "rows": len(rows)})
        manifest = path.parent / "manifest.json"
        if manifest.exists():
            metadata = json.loads(manifest.read_text())
            if "environment" in metadata:
                environments[experiment] = metadata["environment"]
    for path in sorted(input_root.rglob("quality.json")):
        if not (path.parent / "completed.json").exists():
            raise ValueError("An incomplete quality run cannot be published.")
        data = json.loads(path.read_text())
        protocol = json.loads((path.parent / "protocol.json").read_text())
        quality_runs.append({
            "model": data["model"], "seed": data["seed"], "selected_epoch": data["selected_epoch"],
            "eligible_target_coverage": data["eligible_target_coverage"],
            "full_catalog": data["full_catalog"], "two_stage": data.get("two_stage", {}),
            "parameters": protocol["parameters"], "model_class": protocol["model_class"],
            "validation_best_ndcg10": json.loads((path.parent / "best.json").read_text())["validation_ndcg@10"],
            "dataset": protocol["dataset_manifest"]["dataset"],
            "dataset_archive_sha256": protocol["dataset_manifest"]["archive_sha256"],
        })
    for path in input_root.rglob("upstream_audit.json"):
        audits.append(json.loads(path.read_text()))
    for path in input_root.rglob("pipeline_profile.json"):
        profiles.append(json.loads(path.read_text()))
    summary = {
        "ann": aggregate(raw["ann"], ("experiment", "backend", "items", "distribution", "k", "probes", "batch_size")),
        "stages": aggregate(raw["stages"], ("experiment", "backend", "representation", "items", "probes", "batch_size")),
        "pipeline": aggregate(raw["pipeline"], ("experiment", "backend", "ranker", "items", "candidates", "probes", "batch_size")),
        "trained_ann": aggregate(raw["quality_ann"], ("backend", "ranker", "probes", "candidates")),
        "quality_runs": quality_runs,
        "audits": audits,
        "pipeline_profiles": profiles,
        "environments": environments,
        "sources": sources,
        "interpretation": "Local offline measurements and a public-data adaptation, not Meta production QPS/TCO.",
        "privacy": "No job access, credentials, user/item labels, or per-user dataset identifiers published.",
    }
    reject_private_fields(summary)
    for name, rows in raw.items():
        content = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
        (output / f"{name}-raw.jsonl.gz").write_bytes(gzip.compress(content.encode(), mtime=0))
    write_json(output / "summary.json", summary)
    write_json(output / "quality.json", quality_runs)
    for name in ("ann", "stages", "pipeline", "trained_ann"):
        write_csv(output / f"{name}.csv", summary[name])
    write_csv(output / "filter.csv", raw["filter"])
    write_csv(output / "rank-timing.csv", raw["rank_timing"])
    write_json(output / "upstream-audit.json", audits)
    write_json(output / "pipeline-profile.json", profiles)
    content = [
        "# Measured benchmark tables", "",
        "_Generated from checksummed numerical journals. See results.md for interpretation._", "", "---", "",
        "## 📊 Representative ANN operating points", "",
        "Batch 16, k=2048; IVF probes=64 at 1M/20M and probes=24 at 80M. HNSW uses efSearch=2048. "
        "Values are means across repeats; SD is repeat variation, not a serving-tail interval.", "",
    ]
    selected = [
        row for row in summary["ann"]
        if row["k"] == 2048 and row["batch_size"] == 16
        and row["probes"] == (24 if row["items"] == 80000000 else 64)
    ]
    content.append(table(
        ["Experiment", "Backend", "Batch ms", "Repeat SD ms", "ANN recall", "Repeats"],
        [[
            row["experiment"], row["backend"], f"{row['latency_mean_ms']:.4f}",
            f"{row['latency_mean_ms_std']:.4f}", f"{row['recall']:.4f}", str(row["measurements"])
        ] for row in selected],
    ))
    content.extend(["", "## ⚖️ Recall-matched ANN frontier", "",
        "20M clustered items, batch 16, k=2048. For each arm, choose the fastest "
        "measured probe setting meeting the target. A speedup at mismatched recall is not treated as comparable.", ""])
    frontier = []
    eligible_rows = [
        row for row in summary["ann"]
        if row["items"] == 20000000 and row["batch_size"] == 16 and row["k"] == 2048
    ]
    for target in (0.5, 0.8, 0.95):
        for backend in sorted({row["backend"] for row in eligible_rows}):
            choices = [row for row in eligible_rows if row["backend"] == backend and row["recall"] >= target]
            if choices:
                best = min(choices, key=lambda row: row["latency_mean_ms"])
                frontier.append([f"{target:.2f}", backend, str(best["probes"]),
                                 f"{best['recall']:.4f}", f"{best['latency_mean_ms']:.4f}"])
            else:
                frontier.append([f"{target:.2f}", backend, "-", "not attained", "-"])
    content.append(table(["Recall target", "Backend", "Chosen probes", "Measured recall", "Batch ms"], frontier))
    content.extend(["", "## 🔬 Isolated scoring and scratch", "",
        "20M items, batch 16, probes=64. Query quantization, centroid probing, and top-k are excluded.", ""])
    content.append(table(
        ["Representation", "Scoring implementation", "Batch ms", "Incremental PyTorch peak MiB"],
        [[row["representation"], row["backend"], f"{row['latency_mean_ms']:.4f}",
          f"{row['torch_incremental_peak_bytes'] / (1 << 20):.1f}"]
         for row in summary["stages"] if row["probes"] == 64],
    ))
    content.extend(["", "## 🛡️ Exact filtering versus Bloom approximation", "",
        "10M items, positive query `1:1`, Bloom multiplier 5. Raw Bloom is approximate; the exact row pays verification cost.", ""])
    filter_rows = [row for row in raw["filter"] if row["items"] == 10000000 and row["expression"] == "1:1" and row["multiplier"] == 5]
    content.append(table(
        ["Backend", "Mean batch ms", "Output false positives", "Output false negatives"],
        [[row["backend"], f"{row['latency_mean_ms']:.4f}",
          str(row["conservative_false_positives"] if row["backend"] == "silver-bloom-packed" else 0),
          str(row["conservative_false_negatives"] if row["backend"] == "silver-bloom-packed" else 0)]
         for row in filter_rows],
    ))
    content.extend(["", "## 🧠 Trained full-catalog recommendation", "",
        "Same 2,000 users, three training seeds, validation-selected checkpoints. SD below is across seeds.", ""])
    quality_table = []
    for name in sorted({row["model"] for row in quality_runs}):
        group = [row for row in quality_runs if row["model"] == name]
        ndcg = [row["full_catalog"]["ndcg@10"] for row in group]
        quality_table.append([
            name, str(len(group)), f"{np.mean(ndcg):.5f}", f"{np.std(ndcg, ddof=1):.5f}" if len(group) > 1 else "n/a",
            f"{np.mean([row['full_catalog']['recall@10'] for row in group]):.5f}",
            f"{np.mean([row['full_catalog']['recall@100'] for row in group]):.5f}",
        ])
    for row in quality_table:
        group = [run for run in quality_runs if run["model"] == row[0]]
        row.append(str(group[0]["parameters"]))
    content.append(table(["Model", "Seeds", "NDCG@10", "Seed SD", "Recall@10", "Recall@100", "Parameters"], quality_table))
    content.extend(["", "## 🔗 Complete local pipeline runtime", "",
        "1M synthetic items, 512 candidates, probes=24, batch=16. "
        "Reranker weights are random; these are runtime and candidate-geometry results only.", ""])
    content.append(table(
        ["ANN/filter backend", "Actual reranker", "Batch ms", "Candidate recall", "Fill"],
        [[row["backend"], row["ranker"], f"{row['latency_mean_ms']:.4f}",
          f"{row['candidate_recall']:.4f}", f"{row['candidate_fill_fraction']:.4f}"]
         for row in summary["pipeline"] if row["candidates"] == 512 and row["probes"] == 24 and row["batch_size"] == 16],
    ))
    forty_m = [
        row for row in raw["filter"]
        if row["items"] == 40000000 and row["expression"] == "1:1"
    ]
    if forty_m:
        content.extend(["", "## 🧮 Paper-sized filtering adaptation", "",
            "40M synthetic items with six two-valued attributes per item, one query per batch. "
            "Multipliers control adaptive Bloom width; this is not the private production attribute distribution.", ""])
        content.append(table(
            ["Multiplier", "Backend", "Batch ms", "Bloom index GiB", "Output false positives"],
            [[str(row["multiplier"]), row["backend"], f"{row['latency_mean_ms']:.4f}",
              f"{row['index_bytes'] / (1 << 30):.3f}",
              str(row["conservative_false_positives"] if row["backend"] == "silver-bloom-packed" else 0)]
             for row in forty_m],
        ))
    if profiles:
        content.extend(["", "## 🔧 Eligibility root-cause profile", "",
            "1M items, batch 16, probes=24. Both exact-verification implementations "
            "were checked to return identical masks; these stage timings exclude unrelated work.", ""])
        content.append(table(
            ["Stage", "Mean batch ms"],
            [[name, f"{values['latency_mean_ms']:.4f}"] for name, values in profiles[0]["stages"].items()],
        ))
    content.extend(["", "## 🎯 Frozen learned ANN-to-rerank quality", "",
        "MovieLens, 500 candidates, all 64 clusters probed. "
        "This isolates representation/filtering changes from cluster approximation; metrics average the three frozen training seeds.", ""])
    content.append(table(
        ["ANN backend", "Frozen reranker", "Geometric candidate recall", "Target in candidates", "NDCG@10"],
        [[row["backend"], row["ranker"], f"{row['candidate_geometric_recall']:.4f}",
          f"{row['candidate_hit_rate']:.4f}", f"{row['quality']['ndcg@10']:.5f}"]
         for row in summary["trained_ann"] if row["candidates"] == 500 and row["probes"] == 64],
    ))
    content.extend(["", "## 📦 Numerical artifacts", "",
        "The `results/2026-10-10` directory contains full sweep CSVs, sanitized compressed raw journals, "
        "environment metadata, and source checksums. Unsupported FAISS configurations remain explicit in raw ANN rows.", ""])
    tables_path.parent.mkdir(parents=True, exist_ok=True)
    tables_path.write_text("\n".join(content), encoding="utf-8")
    write_json(output / "files.json", {
        path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file() and path.name != "files.json"
    })
    return summary
