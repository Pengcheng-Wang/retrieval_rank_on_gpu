from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from rrbench.io import sha256, write_json
from rrbench.reporting import aggregate, reject_private_fields, table, write_csv

BACKENDS = (
    "faiss-ivf-cpu", "faiss-ivf-gpu", "silver-fp32", "silver-fp16",
    "silver-int8-global", "silver-int8-rowwise",
)
PROBES = (8, 24, 64, 256)


def validate_rows(rows: list[dict[str, Any]], seed: int) -> None:
    expected = {(backend, probe, repeat) for backend in BACKENDS for probe in PROBES for repeat in range(3)}
    actual = []
    for row in rows:
        reject_private_fields(row)
        key = (row["backend"], row["probes"], row["repeat"])
        if any(row[field] != value for field, value in (
            ("seed", seed), ("items", 1000000), ("dimension", 128),
            ("distribution", "clustered"), ("k", 2048), ("batch_size", 16), ("status", "ok"),
        )):
            raise ValueError("Replication row differs from the frozen protocol.")
        samples = np.asarray(row["samples_ms"], dtype=np.float64)
        recalls = np.asarray(row["per_query_recall"], dtype=np.float64)
        if len(samples) != 50 or len(recalls) != 128:
            raise ValueError("Replication must retain 50 timing samples and all 128 query recalls.")
        if not np.isfinite(samples).all() or (samples <= 0).any():
            raise ValueError("Invalid latency samples.")
        if not np.isfinite(recalls).all() or ((recalls < 0) | (recalls > 1)).any():
            raise ValueError("Invalid geometric recall.")
        if not np.isclose(float(samples.mean()), row["latency_mean_ms"], rtol=1e-10):
            raise ValueError("Latency summary does not match its retained samples.")
        if not np.isclose(float(recalls.mean()), row["recall"], rtol=1e-10):
            raise ValueError("Recall summary does not match its retained query observations.")
        actual.append(key)
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Missing or duplicate configuration/repeat rows in replication.")


def protocol_matches(config: dict[str, Any], seed: int) -> None:
    expected = {
        "experiment": "ann", "items": 1000000, "dimension": 128, "queries": 128,
        "seed": seed, "distribution": "clustered", "clusters": 1024, "ks": [2048],
        "probes": list(PROBES), "batches": [16], "warmup": 20, "iterations": 50,
        "repeats": 3, "threads": 32, "backends": list(BACKENDS),
    }
    if config != expected:
        raise ValueError("Bound replication configuration differs from the planned milestone.")


def publish_replication(
    reference: Path, runs: dict[int, Path], output: Path, report: Path,
    resource_receipt: Path,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError("Replication publication must use a new output directory.")
    if set(runs) != {23, 42}:
        raise ValueError("This milestone requires both independent seeds 23 and 42.")
    receipt = json.loads(resource_receipt.read_text())
    reject_private_fields(receipt)
    if (
        receipt.get("restoration_verified") is not True
        or receipt.get("restored_original_profile") is not True
        or receipt.get("restored_gpu_count") != 8
        or receipt.get("other_nodes_modified") is not False
    ):
        raise ValueError("Replication resource-control and restoration receipt is incomplete.")
    baseline_path = reference / "ann-raw.jsonl.gz"
    with gzip.open(baseline_path, "rt", encoding="utf-8") as stream:
        baseline = [
            json.loads(line) for line in stream if line.strip()
        ]
    rows = [
        row for row in baseline
        if row["experiment"] == "ann_1m_clustered_s17"
        and row["k"] == 2048 and row["batch_size"] == 16 and row["probes"] in PROBES
    ]
    validate_rows(rows, 17)
    sources = [{
        "seed": 17, "kind": "immutable initial-study reference",
        "archive": baseline_path.name, "sha256": sha256(baseline_path),
        "selected_rows": len(rows),
    }]
    environments = {"17": json.loads((reference / "summary.json").read_text())["environments"]["ann_1m_clustered_s17"]}
    for seed, directory in sorted(runs.items()):
        completion = json.loads((directory / "completed.json").read_text())
        if completion.get("complete") is not True or completion.get("experiment") != "synthetic ANN":
            raise ValueError("Cannot publish an incomplete or different experiment.")
        protocol_matches(json.loads((directory / "run-config.json").read_text()), seed)
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["environment"]["tf32"] or manifest["environment"]["cpu_threads"] != 32:
            raise ValueError("Replication environment violates numerical/thread contracts.")
        journal = directory / "ann.jsonl"
        current = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
        validate_rows(current, seed)
        for row in current:
            row["experiment"] = directory.name
        rows.extend(current)
        environments[str(seed)] = manifest["environment"]
        sources.append({
            "seed": seed, "kind": "new independent data and index-build seed",
            "experiment": directory.name, "journal_sha256": sha256(journal),
            "bound_config_sha256": sha256(directory / "run-config.json"),
            "selected_rows": len(current),
        })
    grouped = aggregate(rows, ("seed", "backend", "probes", "batch_size", "k"))
    anchor = [row for row in grouped if row["probes"] == 64]
    across_seeds = aggregate([{**row, "status": "ok"} for row in anchor], ("backend", "probes"))
    ratios = []
    for seed in (17, 23, 42):
        selected = {row["backend"]: row for row in anchor if row["seed"] == seed}
        ratios.append({
            "seed": seed,
            "faiss_gpu_to_silver_fp32_latency_ratio": selected["faiss-ivf-gpu"]["latency_mean_ms"] / selected["silver-fp32"]["latency_mean_ms"],
            "faiss_cpu_to_silver_fp32_latency_ratio": selected["faiss-ivf-cpu"]["latency_mean_ms"] / selected["silver-fp32"]["latency_mean_ms"],
            "faiss_gpu_recall": selected["faiss-ivf-gpu"]["recall"],
            "silver_fp32_recall": selected["silver-fp32"]["recall"],
        })
    frontier = []
    for seed in (17, 23, 42):
        for target in (0.8, 0.95):
            for backend in BACKENDS:
                available = [
                    row for row in grouped
                    if row["seed"] == seed and row["backend"] == backend and row["recall"] >= target
                ]
                best = min(available, key=lambda row: row["latency_mean_ms"]) if available else None
                frontier.append({
                    "seed": seed, "backend": backend, "recall_target": target,
                    "status": "attained" if best is not None else "not_attained",
                    "selected_probes": best["probes"] if best else None,
                    "measured_recall": best["recall"] if best else None,
                    "latency_mean_ms": best["latency_mean_ms"] if best else None,
                })
    constrained_ratios = []
    for seed in (17, 23, 42):
        selected = {
            row["backend"]: row for row in frontier
            if row["seed"] == seed and row["recall_target"] == 0.95
        }
        gpu, silver = selected["faiss-ivf-gpu"], selected["silver-fp32"]
        if gpu["status"] == "attained" and silver["status"] == "attained":
            constrained_ratios.append({
                "seed": seed, "recall_target": 0.95,
                "faiss_probes": gpu["selected_probes"], "silver_probes": silver["selected_probes"],
                "faiss_gpu_to_silver_fp32_latency_ratio": gpu["latency_mean_ms"] / silver["latency_mean_ms"],
            })
    result = {
        "milestone": "independent clustered ANN seed replication",
        "seeds": [17, 23, 42], "reference_seed": 17, "new_data_seeds": [23, 42],
        "sources": sources, "environments": environments, "per_seed": grouped,
        "across_seed_anchor": across_seeds, "paired_latency_ratios": ratios,
        "recall_constrained_frontier": frontier,
        "recall_constrained_fp32_ratios": constrained_ratios,
        "resource_receipt": receipt,
        "resource_receipt_sha256": sha256(resource_receipt),
        "interpretation": "Seed changes jointly perturb synthetic vectors, queries, KMeans training sample and initialization; not recommendation-quality or production-QPS evidence.",
        "timing_scope": "Full local ANN; native-resident queries; no RPC or user tower.",
        "resource_control": "New runs use one GPU with confirmed keepalive CUDA work stopped during measurement; original supervisor policy is restored after the milestone.",
        "uncertainty": "Within-seed repeat SD and across-data-seed variation are separate; three data seeds do not establish generalization.",
    }
    reject_private_fields(result)
    output.mkdir(parents=True)
    content = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
    (output / "ann-raw.jsonl.gz").write_bytes(gzip.compress(content.encode(), mtime=0))
    write_json(output / "summary.json", result)
    write_json(output / "resource-receipt.json", receipt)
    write_csv(output / "per-seed.csv", grouped)
    write_csv(output / "across-seed-anchor.csv", across_seeds)
    write_csv(output / "paired-latency-ratios.csv", ratios)
    write_csv(output / "recall-frontier.csv", frontier)
    source_commit = "525b09b6ddf21c2d0d8beaf604c972a8d14561de"
    write_json(output / "protocol.json", {
        "source_commit": source_commit, "upstream_commit": "22c2007ed5d28ec8304de081cc2393778708ad12",
        "items": 1000000, "dimension": 128, "queries": 128, "clusters": 1024,
        "distribution": "clustered", "k": 2048, "probes": list(PROBES), "batch": 16,
        "warmup": 20, "samples_per_repeat": 50, "timing_repeats": 3,
        "data_and_index_seeds": [17, 23, 42], "backends": list(BACKENDS),
    })
    int8_loss_repeats = all(
        next(row["recall"] for row in anchor if row["seed"] == seed and row["backend"] == backend)
        < next(row["recall"] for row in anchor if row["seed"] == seed and row["backend"] == "silver-fp32") - 1e-5
        for seed in (17, 23, 42) for backend in ("silver-int8-global", "silver-int8-rowwise")
    )
    lines = [
        "# Independent data-seed replication", "",
        "_Bounded follow-up milestone, 2026-10-10. Initial artifacts remain immutable._", "", "---", "",
        "## 📋 Protocol and scope", "",
        "The original seed17 is a read-only reference. Seeds23 and42 regenerate both "
        "the clustered catalog/query vectors and their KMeans partition. Each IVF arm "
        "shares that seed's fixed partition. This measures joint data/index sensitivity, "
        "not held-out recommendation quality or Meta production serving QPS.", "",
        "1M normalized 128D items, 1,024 clusters, 128 queries, top-2048, probes "
        "8/24/64/256, batch16, 20 warmups, 50 timed samples, three repeats, 32 CPU threads; TF32 disabled.", "",
        "## 🎯 Findings", "",
        f"At the fixed 64-probe anchor, the FAISS-GPU/SilverTorch-FP32 latency ratio "
        f"ranges from {min(row['faiss_gpu_to_silver_fp32_latency_ratio'] for row in ratios):.3f}x "
        f"to {max(row['faiss_gpu_to_silver_fp32_latency_ratio'] for row in ratios):.3f}x. "
        f"SilverTorch FP32 geometric recall ranges from "
        f"{min(row['silver_fp32_recall'] for row in ratios):.5f} to "
        f"{max(row['silver_fp32_recall'] for row in ratios):.5f}.", "",
        "The INT8-global 64-probe recalls range from "
        f"{min(row['recall'] for row in anchor if row['backend']=='silver-int8-global'):.4f} to "
        f"{max(row['recall'] for row in anchor if row['backend']=='silver-int8-global'):.4f}; "
        "rowwise INT8 ranges from "
        f"{min(row['recall'] for row in anchor if row['backend']=='silver-int8-rowwise'):.4f} to "
        f"{max(row['recall'] for row in anchor if row['backend']=='silver-int8-rowwise'):.4f}. "
        + ("Both INT8 arms lose geometric recall relative to FP32 in all three data/index seeds."
           if int8_loss_repeats else "The data do not establish a consistent loss for both INT8 arms in all three seeds."),
        "",
        "## 📊 Fixed 64-probe anchor", "",
        table(
            ["Seed", "Backend", "Mean batch ms", "Repeat SD ms", "Geometric recall"],
            [[str(row["seed"]), row["backend"], f"{row['latency_mean_ms']:.4f}",
              f"{row['latency_mean_ms_std']:.4f}", f"{row['recall']:.4f}"] for row in anchor],
        ), "", "## ⚖️ Paired FP32 comparisons", "",
        table(
            ["Seed", "FAISS-GPU / SilverTorch latency", "FAISS-CPU / SilverTorch latency", "FAISS recall", "SilverTorch recall"],
            [[str(row["seed"]), f"{row['faiss_gpu_to_silver_fp32_latency_ratio']:.3f}x",
              f"{row['faiss_cpu_to_silver_fp32_latency_ratio']:.3f}x",
              f"{row['faiss_gpu_recall']:.5f}", f"{row['silver_fp32_recall']:.5f}"] for row in ratios],
        ), "", "## 🔬 Across-seed variation", "",
        "These standard deviations are across three seed-level means, not the "
        "within-run sample confidence intervals or serving-tail uncertainty.", "",
        table(
            ["Backend", "Mean batch ms", "Data-seed SD ms", "Mean recall", "Data-seed recall SD"],
            [[row["backend"], f"{row['latency_mean_ms']:.4f}", f"{row['latency_mean_ms_std']:.4f}",
              f"{row['recall']:.4f}", f"{row['recall_std']:.5f}"] for row in across_seeds],
        ), "", "## 🎯 Recall-constrained probe grid", "",
        "Each row chooses the fastest tested probe count meeting a geometric-recall "
        "target. The coarse grid can overdeliver recall; these are constraints, not an exact continuous iso-recall fit.", "",
        table(
            ["Seed", "Backend", "Target", "Chosen probes", "Measured recall", "Batch ms"],
            [[str(row["seed"]), row["backend"], str(row["recall_target"]),
              str(row["selected_probes"]) if row["selected_probes"] else "-",
              f"{row['measured_recall']:.4f}" if row["measured_recall"] is not None else "not attained",
              f"{row['latency_mean_ms']:.4f}" if row["latency_mean_ms"] is not None else "-"] for row in frontier],
        ), "", "## 🛡️ Interpretation boundaries", "",
        "The new runs require an empty, idle measurement GPU; the original keepalive "
        "supervisor and profile are restored afterward. Different elapsed-time windows "
        "and thermal/allocator conditions remain possible across seeds. The raw journal "
        "retains every timing sample and per-query geometric recall.", "",
        "Three synthetic data seeds test repeatability of this recipe, not general "
        "recommendation-model quality or every embedding distribution. The original "
        "MovieLens checkpoints and test evaluations are unchanged; the TorchRec "
        "precision/compilation milestone is not part of this wake.", "",
    ]
    if constrained_ratios:
        lines.extend([
            "## ⚖️ Efficient high-recall operating points", "",
            "Using each arm's fastest tested setting meeting recall>=0.95, "
            "the paired FP32 GPU latency ratios are:",
            "",
            table(
                ["Seed", "FAISS chosen probes", "SilverTorch chosen probes", "FAISS-GPU / SilverTorch latency"],
                [[str(row["seed"]), str(row["faiss_probes"]), str(row["silver_probes"]),
                  f"{row['faiss_gpu_to_silver_fp32_latency_ratio']:.3f}x"] for row in constrained_ratios],
            ),
            "",
            "This is the more efficient grid-constrained comparison; the larger "
            "fixed-64-probe ratio is not the fastest configuration either system can use.",
            "",
        ])
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
    write_json(output / "files.json", {
        path.name: sha256(path) for path in sorted(output.iterdir()) if path.name != "files.json"
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish complete independent-data ANN seed replications.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--seed23", required=True)
    parser.add_argument("--seed42", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--resource-receipt", required=True)
    args = parser.parse_args()
    result = publish_replication(
        Path(args.reference), {23: Path(args.seed23), 42: Path(args.seed42)},
        Path(args.output), Path(args.report), Path(args.resource_receipt),
    )
    print(f"Published {len(result['per_seed'])} configuration means across three data/index seeds.")


if __name__ == "__main__":
    main()
