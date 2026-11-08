from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


MODEL_ORDER = ("sasrec", "hstu", "hstu-large")
METRICS = (
    ("HR@10", "hr@10"),
    ("NDCG@10", "ndcg@10"),
    ("HR@50", "hr@50"),
    ("NDCG@50", "ndcg@50"),
    ("HR@200", "hr@200"),
    ("NDCG@200", "ndcg@200"),
)
EXCLUDED_PUBLIC_KEYS = {
    "checkpoint",
    "cold_heldout_item_ids",
    "destination",
    "event_file",
    "event_root",
    "log",
    "path",
    "selected_user_ids",
    "source",
}


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _publicize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _publicize(item)
            for key, item in value.items()
            if key not in EXCLUDED_PUBLIC_KEYS
        }
    if isinstance(value, list):
        return [_publicize(item) for item in value]
    return value


def _duration_hours(memory_path: Path) -> float | None:
    with memory_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        return None
    start = datetime.fromisoformat(rows[0]["utc"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(rows[-1]["utc"].replace("Z", "+00:00"))
    return (end - start).total_seconds() / 3600.0


def aggregate_quality(
    protocol: dict[str, Any],
    cpu_smoke: list[dict[str, Any]],
    full_root: Path,
) -> list[dict[str, Any]]:
    parameters = {row["model"]: row["parameters"] for row in cpu_smoke}
    targets = protocol["evaluation"]["primary_targets"]
    tolerances = protocol["evaluation"]["descriptive_reproduction_tolerance"]
    rows = []
    for model in MODEL_ORDER:
        summary = _read(full_root / f"{model}-summary.json")
        resource = _read(full_root / f"{model}-resource.json")
        run = summary["runs"][-1]
        final = run["final"]
        best_ndcg10 = max(
            run["epoch_values"]["ndcg@10"],
            key=lambda event: event["value"],
        )
        best_epoch = best_ndcg10["step"]
        best_hr10 = next(
            event["value"]
            for event in run["epoch_values"]["hr@10"]
            if event["step"] == best_epoch
        )
        row: dict[str, Any] = {
            "model": model,
            "final_epoch": run["final_epoch"],
            "best_observed_ndcg10_epoch_diagnostic": best_epoch,
            "best_observed_ndcg10_diagnostic": best_ndcg10["value"],
            "hr10_at_best_ndcg10_epoch_diagnostic": best_hr10,
            "parameters": parameters[model],
            "training_wall_hours": _duration_hours(
                full_root / f"{model}-memory.csv"
            ),
            "maximum_sampled_memory_mib": resource["maximum_memory_used_mib"],
            "minimum_sampled_free_mib": resource["minimum_free_mib"],
        }
        for target_name, observed_name in METRICS:
            target = targets[model][target_name]
            observed = final[observed_name]
            slug = observed_name.replace("@", "")
            row[f"target_{slug}"] = target
            row[f"observed_{slug}"] = observed
            row[f"difference_{slug}"] = observed - target
        row["hr10_within_tolerance"] = (
            abs(row["difference_hr10"]) <= tolerances["HR@10_absolute"]
        )
        row["ndcg10_within_tolerance"] = (
            abs(row["difference_ndcg10"]) <= tolerances["NDCG@10_absolute"]
        )
        rows.append(row)
    return rows


def aggregate_efficiency(benchmark_root: Path) -> list[dict[str, Any]]:
    rows = []
    for model in MODEL_ORDER:
        path = benchmark_root / f"{model}.json"
        if not path.is_file():
            continue
        benchmark = _read(path)
        for precision, precision_result in benchmark["inference"].items():
            for batch_size, batch_result in precision_result["batches"].items():
                for phase in ("encode", "exact_search", "end_to_end"):
                    phase_result = batch_result[phase]
                    rows.append(
                        {
                            "model": model,
                            "operation": phase,
                            "precision": precision,
                            "batch_size": int(batch_size),
                            "sequence_length": batch_result["sequence_length"],
                            "catalog_items": batch_result["catalog_items"],
                            "returned_items": batch_result["returned_items"],
                            "mean_ms": phase_result["mean_ms"],
                            "p50_ms": phase_result["p50_ms"],
                            "p95_ms": phase_result["p95_ms"],
                            "peak_allocated_mib": phase_result["memory"][
                                "peak_allocated_mib"
                            ],
                            "peak_reserved_mib": phase_result["memory"][
                                "peak_reserved_mib"
                            ],
                        }
                    )
        training = benchmark["training_step"]
        rows.append(
            {
                "model": model,
                "operation": "training_step",
                "precision": "fp32-tf32",
                "batch_size": training["batch_size"],
                "sequence_length": training["sequence_length"],
                "catalog_items": None,
                "returned_items": None,
                "mean_ms": training["mean_ms"],
                "p50_ms": training["p50_ms"],
                "p95_ms": training["p95_ms"],
                "peak_allocated_mib": training["memory"]["peak_allocated_mib"],
                "peak_reserved_mib": training["memory"]["peak_reserved_mib"],
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _checksums(output: Path) -> dict[str, Any]:
    records = []
    for path in sorted(candidate for candidate in output.iterdir() if candidate.is_file()):
        if path.name == "checksums.json":
            continue
        records.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return {"files": records}


def _scale_probe(root: Path | None, models: tuple[str, ...]) -> dict[str, Any] | None:
    if root is None or not root.is_dir():
        return None
    result: dict[str, Any] = {
        "status": (root / "status").read_text().strip()
        if (root / "status").is_file()
        else "incomplete",
    }
    for name in ("data-audit.json", "exit-codes.json", "smoke-100-receipt.json"):
        path = root / name
        if path.is_file():
            result[name.removesuffix(".json")] = _publicize(_read(path))
    result["models"] = {}
    for model in models:
        path = root / f"{model}-summary.json"
        if path.is_file():
            result["models"][model] = _publicize(_read(path))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data-audit", type=Path, required=True)
    parser.add_argument("--cpu-smoke", type=Path, required=True)
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--later-attention", type=Path)
    parser.add_argument("--ml20m-probe", type=Path)
    parser.add_argument("--amazon-probe", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    protocol = _read(args.protocol)
    data_audit = _publicize(_read(args.data_audit))
    cpu_smoke = _publicize(_read(args.cpu_smoke))
    quality = aggregate_quality(protocol, cpu_smoke, args.full_root)
    efficiency = aggregate_efficiency(args.benchmark_root)
    _write_csv(args.output / "quality.csv", quality)
    _write_csv(args.output / "efficiency.csv", efficiency)

    smoke = {
        model: {
            "summary": _publicize(_read(args.smoke_root / f"{model}-summary.json")),
            "resource": _publicize(_read(args.smoke_root / f"{model}-resource.json")),
        }
        for model in ("hstu", "sasrec")
    }
    scale_probes = {
        "ml20m": _scale_probe(args.ml20m_probe, ("hstu", "sasrec")),
        "amazon_books": _scale_probe(args.amazon_probe, ("hstu", "sasrec")),
    }
    (args.output / "scale-probes.json").write_text(
        json.dumps(scale_probes, indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "study": protocol["study"],
        "source_commit": protocol["sources"]["paper_era_code"]["commit"],
        "data": data_audit,
        "cpu_smoke": cpu_smoke,
        "gpu_smoke": smoke,
        "quality": quality,
        "efficiency_rows": len(efficiency),
        "scale_probes": scale_probes,
        "later_attention": (
            _publicize(_read(args.later_attention))
            if args.later_attention and args.later_attention.is_file()
            else None
        ),
        "astra_review": {
            "status": "unavailable",
            "reason": (
                "The runtime did not expose the exact gpt-6-astra model or an "
                "independently verifiable one-million-token context."
            ),
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    )
    (args.output / "checksums.json").write_text(
        json.dumps(_checksums(args.output), indent=2, sort_keys=True) + "\n"
    )
    print(args.output)


if __name__ == "__main__":
    main()
