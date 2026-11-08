from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from rrbench.industry_data import validate_run_matrix


MODEL_LABELS = {
    "torchrec-dlrm": "TorchRec DLRM",
    "torchrec-dcn": "TorchRec DLRM_DCN",
    "din": "DIN-style normalized-attention adaptation",
    "twin": "TWIN-style structural adaptation",
    "transact": "Official TransAct module plus ranking head",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _rank_metrics(candidates: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    equal = candidates == targets[:, None]
    present = equal.any(axis=1)
    ranks = np.where(present, equal.argmax(axis=1) + 1, np.inf)
    result: dict[str, Any] = {
        "queries": int(len(targets)),
        "target_present": int(present.sum()),
        "target_coverage": float(present.mean()),
    }
    for k in (10, 50, 200):
        hits = present & (ranks <= k)
        ndcg = np.where(hits, 1.0 / np.log2(ranks + 1), 0.0)
        result[f"unconditional_hr@{k}"] = float(hits.mean())
        result[f"unconditional_ndcg@{k}"] = float(ndcg.mean())
        result[f"conditional_hr@{k}"] = float(hits[present].mean())
        result[f"conditional_ndcg@{k}"] = float(ndcg[present].mean())
    return result


def _coverage_at_caps(
    candidates: np.ndarray, targets: np.ndarray
) -> dict[str, float]:
    return {
        str(cap): float(
            np.any(candidates[:, :cap] == targets[:, None], axis=1).mean()
        )
        for cap in (50, 100, 200, 500)
    }


def _observation_metrics(observations: dict[str, Any]) -> dict[str, Any]:
    present = np.asarray(observations["target_present"], dtype=bool)
    encoded_ranks = np.asarray(observations["target_rank"], dtype=np.uint16)
    ranks = np.where(present, encoded_ranks, np.inf)
    return {
        **_rank_metrics_from_ranks(ranks, present),
        "query_indices": np.asarray(
            observations["query_indices"], dtype=np.int32
        ),
        "encoded_ranks": encoded_ranks,
        "present": present,
    }


def _rank_metrics_from_ranks(
    ranks: np.ndarray,
    present: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "queries": int(len(ranks)),
        "target_present": int(present.sum()),
        "target_coverage": float(present.mean()),
    }
    for k in (10, 50, 200):
        hits = present & (ranks <= k)
        ndcg = np.where(hits, 1.0 / np.log2(ranks + 1), 0.0)
        result[f"unconditional_hr@{k}"] = float(hits.mean())
        result[f"unconditional_ndcg@{k}"] = float(ndcg.mean())
        result[f"conditional_hr@{k}"] = float(hits[present].mean())
        result[f"conditional_ndcg@{k}"] = float(ndcg[present].mean())
    return result


def _validate_reported_metrics(
    reported: dict[str, Any],
    recomputed: dict[str, Any],
    cell: tuple[str, int],
    split: str,
) -> None:
    for key, value in recomputed.items():
        if key in {"query_indices", "encoded_ranks", "present"}:
            continue
        if not math.isclose(
            float(reported[key]),
            float(value),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"{cell} {split} metric {key} differs from observations: "
                f"{reported[key]} != {value}"
            )


def _quality_row(
    *,
    model: str,
    seed: int | str,
    split: str,
    metrics: dict[str, Any],
    parameters: int,
    best_epoch: int | str,
    implementation: str,
) -> dict[str, Any]:
    return {
        "role": "ranking" if model != "hstu-large-retrieval-order" else "retrieval",
        "model": model,
        "implementation": implementation,
        "seed": seed,
        "split": split,
        "fixed_candidate_source": "HSTU-large epoch100 exact top500",
        "scored_candidates": 200,
        "queries": metrics["queries"],
        "target_coverage_at_200": metrics["target_coverage"],
        "unconditional_hr_at_10": metrics["unconditional_hr@10"],
        "unconditional_ndcg_at_10": metrics["unconditional_ndcg@10"],
        "conditional_hr_at_10": metrics["conditional_hr@10"],
        "conditional_ndcg_at_10": metrics["conditional_ndcg@10"],
        "unconditional_hr_at_50": metrics["unconditional_hr@50"],
        "unconditional_ndcg_at_50": metrics["unconditional_ndcg@50"],
        "conditional_hr_at_50": metrics["conditional_hr@50"],
        "conditional_ndcg_at_50": metrics["conditional_ndcg@50"],
        "unconditional_hr_at_200": metrics["unconditional_hr@200"],
        "unconditional_ndcg_at_200": metrics["unconditional_ndcg@200"],
        "conditional_hr_at_200": metrics["conditional_hr@200"],
        "conditional_ndcg_at_200": metrics["conditional_ndcg@200"],
        "parameters": parameters,
        "best_epoch": best_epoch,
    }


def _means(
    rows: Iterable[dict[str, Any]],
    keys: Iterable[str],
) -> dict[str, float]:
    rows = list(rows)
    return {
        key: statistics.fmean(float(row[key]) for row in rows)
        for key in keys
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--logs-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--training-negatives", type=Path, required=True)
    parser.add_argument("--candidate-receipt", type=Path, required=True)
    parser.add_argument("--data-receipt", type=Path, required=True)
    parser.add_argument("--training-negative-receipt", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run_entries = [
        (path, json.loads(path.read_text()))
        for path in sorted(args.runs_dir.glob("*/result.json"))
    ]
    runs = [run for _, run in run_entries]
    expected = {(model, seed) for model in MODEL_LABELS for seed in (17, 42)}
    expected_hashes = {
        "data_sha256": _sha256(args.data),
        "candidates_sha256": _sha256(args.candidates),
        "training_negatives_sha256": _sha256(args.training_negatives),
        "protocol_sha256": _sha256(args.protocol),
    }
    validate_run_matrix(
        runs,
        expected_cells=expected,
        expected_hashes=expected_hashes,
        validation_queries=1000,
        test_queries=5040,
    )

    with np.load(args.candidates, allow_pickle=False) as archive:
        candidates = {key: archive[key] for key in archive.files}
    with np.load(args.data, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    candidate_receipt = json.loads(args.candidate_receipt.read_text())
    data_receipt = json.loads(args.data_receipt.read_text())
    training_negative_receipt = json.loads(
        args.training_negative_receipt.read_text()
    )
    protocol = json.loads(args.protocol.read_text())
    receipt_checks = (
        ("data", data_receipt, expected_hashes["data_sha256"]),
        ("candidates", candidate_receipt, expected_hashes["candidates_sha256"]),
        (
            "training negatives",
            training_negative_receipt,
            expected_hashes["training_negatives_sha256"],
        ),
    )
    for label, receipt, expected_sha in receipt_checks:
        if receipt["output_sha256"] != expected_sha:
            raise RuntimeError(
                f"{label} receipt SHA differs from the aggregated artifact"
            )
    args.output.mkdir(parents=True, exist_ok=True)

    quality_rows: list[dict[str, Any]] = []
    validation_indices = np.asarray(
        data["test_user_partition_validation_indices"]
    ).astype(np.int64)
    test_indices = np.setdiff1d(
        np.arange(len(data["test_targets"]), dtype=np.int64),
        validation_indices,
        assume_unique=False,
    )
    split_indices = {
        "validation": validation_indices,
        "test": test_indices,
    }
    environments = {
        json.dumps(run["environment"], sort_keys=True)
        for run in runs
    }
    if len(environments) != 1:
        raise RuntimeError("Run environments differ across the final matrix.")
    environment = runs[0]["environment"]
    query_rank_arrays: dict[str, np.ndarray] = {}
    run_provenance = []
    for path, run in run_entries:
        cell = (str(run["model"]), int(run["seed"]))
        checkpoint = Path(run["checkpoint"])
        if not checkpoint.is_file():
            raise RuntimeError(f"Missing checkpoint for {cell}: {checkpoint}")
        if _sha256(checkpoint) != run["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint SHA mismatch for {cell}")
        if int(run["best_epoch"]) == 0 and not math.isclose(
            float(run["selected_residual_weight"]),
            0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"Epoch-0 residual is not zero for {cell}")
        if int(run["training_maximum_gap_seconds"]) != int(
            data_receipt["training_maximum_request_gap_seconds"]
        ):
            raise RuntimeError(f"Training normalizer differs for {cell}")
        prefix = f"{run['model'].replace('-', '_')}_seed{run['seed']}"
        for split, expected_indices in split_indices.items():
            recomputed = _observation_metrics(run[split]["observations"])
            if not np.array_equal(
                recomputed["query_indices"],
                expected_indices.astype(np.int32),
            ):
                raise RuntimeError(f"{cell} {split} query partition differs.")
            _validate_reported_metrics(run[split], recomputed, cell, split)
            query_rank_arrays[f"{prefix}_{split}_indices"] = recomputed[
                "query_indices"
            ]
            query_rank_arrays[f"{prefix}_{split}_present"] = recomputed[
                "present"
            ].astype(np.uint8)
            query_rank_arrays[f"{prefix}_{split}_rank"] = recomputed[
                "encoded_ranks"
            ]
        if not math.isclose(
            float(run["best_selection_metric"]),
            float(run["validation"]["unconditional_ndcg@10"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"Selected validation metric differs for {cell}")
        run_provenance.append(
            {
                "model": run["model"],
                "seed": int(run["seed"]),
                "result_sha256": _sha256(path),
                "checkpoint_sha256": run["checkpoint_sha256"],
                "best_epoch": int(run["best_epoch"]),
                "selected_residual_weight": run["selected_residual_weight"],
                "model_class": run["model_class"],
                "base_model_class": run["base_model_class"],
                "upstream_class": run["upstream_class"],
                "parameter_count": int(run["parameter_count"]),
                "training_maximum_gap_seconds": int(
                    run["training_maximum_gap_seconds"]
                ),
                "inputs": {
                    key: run["inputs"][key]
                    for key in expected_hashes
                },
                "environment": run["environment"],
            }
        )
    np.savez_compressed(args.output / "query-ranks.npz", **query_rank_arrays)
    (args.output / "run-provenance.json").write_text(
        json.dumps(run_provenance, indent=2, sort_keys=True) + "\n"
    )
    rejected_protocols = protocol.get("rejected_protocols", [])
    (args.output / "rejected-protocols.json").write_text(
        json.dumps(rejected_protocols, indent=2, sort_keys=True) + "\n"
    )
    for split, indices in split_indices.items():
        retrieval_metrics = _rank_metrics(
            np.asarray(candidates["test_candidates"])[indices, :200],
            np.asarray(candidates["test_targets"])[indices],
        )
        quality_rows.append(
            _quality_row(
                model="hstu-large-retrieval-order",
                seed="frozen",
                split=split,
                metrics=retrieval_metrics,
                parameters=313_000,
                best_epoch=100,
                implementation="Pinned official paper-era Meta HSTU",
            )
        )
    for run in runs:
        for split in ("validation", "test"):
            quality_rows.append(
                _quality_row(
                    model=run["model"],
                    seed=int(run["seed"]),
                    split=split,
                    metrics=run[split],
                    parameters=int(run["parameter_count"]),
                    best_epoch=int(run["best_epoch"]),
                    implementation=MODEL_LABELS[run["model"]],
                )
            )
    _write_csv(args.output / "quality.csv", quality_rows)

    aggregate_quality = []
    metric_keys = [
        key
        for key in quality_rows[0]
        if key.startswith(("unconditional_", "conditional_"))
    ]
    for model in MODEL_LABELS:
        for split in ("validation", "test"):
            selected = [
                row
                for row in quality_rows
                if row["model"] == model and row["split"] == split
            ]
            means = _means(selected, metric_keys)
            aggregate_quality.append(
                {
                    "model": model,
                    "implementation": MODEL_LABELS[model],
                    "split": split,
                    "seeds": 2,
                    "target_coverage_at_200": selected[0][
                        "target_coverage_at_200"
                    ],
                    **{f"mean_{key}": value for key, value in means.items()},
                    "std_unconditional_ndcg_at_10": statistics.pstdev(
                        float(row["unconditional_ndcg_at_10"])
                        for row in selected
                    ),
                    "mean_parameters": statistics.fmean(
                        int(row["parameters"]) for row in selected
                    ),
                    "mean_best_epoch": statistics.fmean(
                        int(row["best_epoch"]) for row in selected
                    ),
                }
            )
    _write_csv(args.output / "quality-aggregate.csv", aggregate_quality)

    efficiency_rows = []
    timing_samples: dict[str, Any] = {}
    for run in runs:
        run_key = f"{run['model']}-seed{run['seed']}"
        timing_samples[run_key] = {"inference": {}, "training_step": {}}
        for precision, measurement in run["benchmark"]["inference"].items():
            if measurement.get("status") == "unsupported":
                efficiency_rows.append(
                    {
                        "model": run["model"],
                        "seed": run["seed"],
                        "kind": "inference",
                        "precision": precision,
                        "status": "unsupported",
                        "batch_size": "",
                        "candidates_per_query": "",
                        "mean_ms": "",
                        "p50_ms": "",
                        "p95_ms": "",
                        "scores_per_second": "",
                        "peak_allocated_mib": "",
                        "peak_reserved_mib": "",
                        "top10_overlap_with_fp32": "",
                    }
                )
                timing_samples[run_key]["inference"][precision] = measurement
                continue
            efficiency_rows.append(
                {
                    "model": run["model"],
                    "seed": run["seed"],
                    "kind": "inference",
                    "precision": precision,
                    "status": "measured",
                    "batch_size": measurement["batch_size"],
                    "candidates_per_query": measurement["candidates_per_query"],
                    "mean_ms": measurement["mean_ms"],
                    "p50_ms": measurement["p50_ms"],
                    "p95_ms": measurement["p95_ms"],
                    "scores_per_second": measurement[
                        "candidate_scores_per_second"
                    ],
                    "peak_allocated_mib": measurement.get(
                        "peak_allocated_mib", ""
                    ),
                    "peak_reserved_mib": measurement.get("peak_reserved_mib", ""),
                    "top10_overlap_with_fp32": measurement.get(
                        "top10_set_overlap_with_fp32", ""
                    ),
                }
            )
            timing_samples[run_key]["inference"][precision] = {
                "raw_ms": measurement["raw_ms"],
                "shape": [
                    measurement["batch_size"],
                    measurement["candidates_per_query"],
                    measurement["history_length"],
                ],
            }
        training = run["benchmark"]["training_step"]
        efficiency_rows.append(
            {
                "model": run["model"],
                "seed": run["seed"],
                "kind": "training_step",
                "precision": "fp32-tf32",
                "status": "measured",
                "batch_size": training["batch_size"],
                "candidates_per_query": training["candidates_per_context"],
                "mean_ms": training["mean_ms"],
                "p50_ms": training["p50_ms"],
                "p95_ms": training["p95_ms"],
                "scores_per_second": training["candidate_scores_per_second"],
                "peak_allocated_mib": training.get("peak_allocated_mib", ""),
                "peak_reserved_mib": training.get("peak_reserved_mib", ""),
                "top10_overlap_with_fp32": "",
            }
        )
        timing_samples[run_key]["training_step"] = {
            "raw_ms": training["raw_ms"],
            "shape": [
                training["batch_size"],
                training["candidates_per_context"],
                200,
            ],
        }
    _write_csv(args.output / "efficiency.csv", efficiency_rows)
    efficiency_aggregate = []
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in efficiency_rows:
        groups[(row["model"], row["kind"], row["precision"])].append(row)
    for (model, kind, precision), rows in sorted(groups.items()):
        measured = [row for row in rows if row["status"] == "measured"]
        if not measured:
            efficiency_aggregate.append(
                {
                    "model": model,
                    "kind": kind,
                    "precision": precision,
                    "status": "unsupported",
                    "seeds": len(rows),
                    "mean_ms": "",
                    "mean_p95_ms": "",
                    "mean_scores_per_second": "",
                    "mean_peak_allocated_mib": "",
                    "mean_top10_overlap_with_fp32": "",
                }
            )
            continue
        overlaps = [
            float(row["top10_overlap_with_fp32"])
            for row in measured
            if row["top10_overlap_with_fp32"] != ""
        ]
        efficiency_aggregate.append(
            {
                "model": model,
                "kind": kind,
                "precision": precision,
                "status": "measured",
                "seeds": len(measured),
                "mean_ms": statistics.fmean(float(row["mean_ms"]) for row in measured),
                "mean_p95_ms": statistics.fmean(
                    float(row["p95_ms"]) for row in measured
                ),
                "mean_scores_per_second": statistics.fmean(
                    float(row["scores_per_second"]) for row in measured
                ),
                "mean_peak_allocated_mib": statistics.fmean(
                    float(row["peak_allocated_mib"]) for row in measured
                ),
                "mean_top10_overlap_with_fp32": (
                    statistics.fmean(overlaps) if overlaps else ""
                ),
            }
        )
    _write_csv(args.output / "efficiency-aggregate.csv", efficiency_aggregate)
    (args.output / "timing-samples.json").write_text(
        json.dumps(timing_samples, indent=2, sort_keys=True) + "\n"
    )

    curves = []
    learning_gates = []
    for run in runs:
        for epoch in run["epochs"]:
            curves.append(
                {
                    "model": run["model"],
                    "seed": run["seed"],
                    "epoch": epoch["epoch"],
                    "training_loss": epoch["training_loss"],
                    "selection_unconditional_ndcg_at_10": epoch[
                        "selection_validation"
                    ]["unconditional_ndcg@10"],
                }
            )
        epoch_zero = next(
            epoch
            for epoch in run["epochs"]
            if int(epoch["epoch"]) == 0
        )
        learned = [
            epoch for epoch in run["epochs"] if int(epoch["epoch"]) > 0
        ]
        best_learned = max(
            learned,
            key=lambda epoch: epoch["selection_validation"][
                "unconditional_ndcg@10"
            ],
        )
        epoch_zero_metric = epoch_zero["selection_validation"][
            "unconditional_ndcg@10"
        ]
        best_learned_metric = best_learned["selection_validation"][
            "unconditional_ndcg@10"
        ]
        learning_gates.append(
            {
                "model": run["model"],
                "seed": run["seed"],
                "epoch_zero_validation_ndcg_at_10": epoch_zero_metric,
                "best_learned_epoch": best_learned["epoch"],
                "best_learned_validation_ndcg_at_10": best_learned_metric,
                "difference_from_epoch_zero": (
                    best_learned_metric - epoch_zero_metric
                ),
                "selected_epoch": run["best_epoch"],
                "gate": "PASS" if int(run["best_epoch"]) > 0 else "FAIL",
            }
        )
    _write_csv(args.output / "training-curves.csv", curves)
    _write_csv(args.output / "learning-gates.csv", learning_gates)
    learning_gate_aggregate = []
    for model in MODEL_LABELS:
        selected = [row for row in learning_gates if row["model"] == model]
        learning_gate_aggregate.append(
            {
                "model": model,
                "seeds": len(selected),
                "passed_seeds": sum(row["gate"] == "PASS" for row in selected),
                "mean_epoch_zero_validation_ndcg_at_10": statistics.fmean(
                    float(row["epoch_zero_validation_ndcg_at_10"])
                    for row in selected
                ),
                "mean_best_learned_validation_ndcg_at_10": statistics.fmean(
                    float(row["best_learned_validation_ndcg_at_10"])
                    for row in selected
                ),
                "mean_difference_from_epoch_zero": statistics.fmean(
                    float(row["difference_from_epoch_zero"])
                    for row in selected
                ),
            }
        )
    _write_csv(
        args.output / "learning-gates-aggregate.csv",
        learning_gate_aggregate,
    )

    wall_rows = []
    log_paths = sorted(args.logs_dir.glob("*.log"))
    if len(log_paths) != len(expected):
        raise RuntimeError(
            f"Expected {len(expected)} run logs, found {len(log_paths)}"
        )
    log_cells: list[tuple[str, int]] = []
    for path in log_paths:
        text = path.read_text(errors="replace")
        start_match = re.search(
            r"START utc=([^ ]+) gpu=(\d+) model=([^ ]+) seed=(\d+)", text
        )
        done_match = re.search(
            r"DONE utc=([^ ]+) model=([^ ]+) seed=(\d+)", text
        )
        if not start_match or not done_match:
            raise RuntimeError(f"Missing bounded start/done receipt in {path}")
        start = datetime.fromisoformat(start_match.group(1).replace("Z", "+00:00"))
        done = datetime.fromisoformat(done_match.group(1).replace("Z", "+00:00"))
        if (
            start_match.group(3) != done_match.group(2)
            or start_match.group(4) != done_match.group(3)
        ):
            raise RuntimeError(f"Run identity changed in {path}")
        cell = (start_match.group(3), int(start_match.group(4)))
        log_cells.append(cell)
        wall_rows.append(
            {
                "model": start_match.group(3),
                "seed": int(start_match.group(4)),
                "gpu_index_within_node": int(start_match.group(2)),
                "started_utc": start_match.group(1),
                "completed_utc": done_match.group(1),
                "complete_run_wall_seconds": (done - start).total_seconds(),
                "scope": "load, 10 training epochs, per-epoch selection validation, final validation/test, efficiency measurements, checkpoint, and serialized durable publish",
            }
        )
    if len(set(log_cells)) != len(log_cells) or set(log_cells) != expected:
        raise RuntimeError(f"Run log matrix mismatch: {sorted(log_cells)}")
    _write_csv(args.output / "run-wall-time.csv", wall_rows)
    wall_aggregate = []
    for model in MODEL_LABELS:
        selected = [row for row in wall_rows if row["model"] == model]
        wall_aggregate.append(
            {
                "model": model,
                "seeds": len(selected),
                "mean_complete_run_wall_seconds": statistics.fmean(
                    float(row["complete_run_wall_seconds"]) for row in selected
                ),
                "minimum_complete_run_wall_seconds": min(
                    float(row["complete_run_wall_seconds"]) for row in selected
                ),
                "maximum_complete_run_wall_seconds": max(
                    float(row["complete_run_wall_seconds"]) for row in selected
                ),
            }
        )
    _write_csv(args.output / "run-wall-time-aggregate.csv", wall_aggregate)

    summary = {
        "study": protocol["study"],
        "data": {
            **{
                key: data_receipt[key]
                for key in (
                    "users",
                    "catalog_rows",
                    "num_item_embeddings",
                    "training_examples",
                    "maximum_history",
                    "training_maximum_request_gap_seconds",
                    "training_target_cold_count",
                )
            },
            "validation_queries": len(validation_indices),
            "test_queries": len(test_indices),
            "validation_target": "last event",
            "test_target": "last event",
            "partition_disjoint": True,
        },
        "candidate_pool": {
            "source": "HSTU-large epoch100 exact full-catalog retrieval",
            "target_injection": False,
            "history_exclusion": True,
            "coverage": {
                split: {
                    "fixed_pool_hstu_large": _coverage_at_caps(
                        np.asarray(candidates["test_candidates"])[indices],
                        np.asarray(candidates["test_targets"])[indices],
                    ),
                    "hstu": _coverage_at_caps(
                        np.asarray(candidates["test_hstu"])[indices],
                        np.asarray(candidates["test_targets"])[indices],
                    ),
                    "sasrec": _coverage_at_caps(
                        np.asarray(candidates["test_sasrec"])[indices],
                        np.asarray(candidates["test_targets"])[indices],
                    ),
                    "rank_interleaved_union_diagnostic": _coverage_at_caps(
                        np.asarray(
                            candidates[
                                "test_rank_interleaved_union_diagnostic"
                            ]
                        )[indices],
                        np.asarray(candidates["test_targets"])[indices],
                    ),
                }
                for split, indices in split_indices.items()
            },
            "full_last_event_diagnostic": candidate_receipt["coverage"]["test"],
            "artifact_sha256": candidate_receipt["output_sha256"],
        },
        "training_negatives": training_negative_receipt,
        "quality_aggregate": aggregate_quality,
        "models": {
            run["model"]: {
                "implementation": MODEL_LABELS[run["model"]],
                "parameter_count": run["parameter_count"],
                "model_class": run["model_class"],
                "base_model_class": run["base_model_class"],
                "upstream_class": run["upstream_class"],
            }
            for run in runs
            if int(run["seed"]) == 17
        },
        "environment": environment,
        "provenance": {
            "run_provenance": "run-provenance.json",
            "query_ranks": "query-ranks.npz",
            "rejected_protocols": "rejected-protocols.json",
        },
        "complete_run_wall_time": wall_aggregate,
        "learning_gates": learning_gate_aggregate,
        "interpretation": protocol["interpretation"],
        "limitations": [
            "MovieLens-1M is a small public sequential recommendation adaptation, not a production traffic reproduction.",
            "The DIN-style arm uses softmax-normalized attention, PReLU instead of Dice, and no mini-batch-aware regularization; it is not a canonical DIN reproduction.",
            "TWIN is a structural adaptation without the production feature cache, 10^4-10^5 event histories, or Kuaishou data.",
            "TransAct preserves Pinterest's public module but uses MovieLens ratings as action types and a learned user embedding instead of Pinterest's production batch embedding.",
            "Exact GPT-6-Astra/MAX was unavailable; no substitute is labeled Astra and no numeric one-million-token context claim is made.",
        ],
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    )
    (args.output / "README.md").write_text(
        """# Industrial ranking aggregate

This package contains only measurements generated by this repository's own
MovieLens-1M experiments. Paper and production numbers are cited in the
research report, not copied into these result tables.

- `quality.csv`: per-seed validation/test metrics plus the frozen HSTU-large
  retrieval-order baseline.
- `quality-aggregate.csv`: two-seed ranker means and NDCG@10 dispersion.
- `efficiency.csv`: per-seed synchronized A100 measurements.
- `efficiency-aggregate.csv`: two-seed efficiency means.
- `timing-samples.json`: all independent synchronized timing samples and exact
  batch/candidate/history shapes.
- `training-curves.csv`: prospective checkpoint-selection evidence.
- `learning-gates.csv`: per-seed epoch-0 versus best learned validation gate.
- `learning-gates-aggregate.csv`: two-seed gate summary.
- `query-ranks.npz`: compact per-query partition membership, target presence,
  and selected target rank for every model/seed.
- `run-provenance.json`: sanitized input/result/checkpoint hashes, package
  versions, model identities, selected epochs, and normalization receipts.
- `rejected-protocols.json`: stable identifiers and reasons for superseded
  experiment protocols whose operational artifacts remain outside Git.
- `run-wall-time.csv`: bounded end-to-end run receipts, including training,
  evaluation, timing, checkpointing, and durable publication.
- `run-wall-time-aggregate.csv`: two-seed complete-run wall-time means.
- `protocol.json`: the frozen experiment contract.
- `summary.json`: data, candidate coverage, model identity, and limitations.
- `checksums.json`: SHA-256 for every other file in this directory.

The rankers score the same first 200 candidates from frozen epoch-100
HSTU-large exact full-catalog retrieval. Targets are never injected, and
retrieval misses remain zero in unconditional metrics.

Training uses the same retrieval distribution: 32 exact HSTU-large hard
negatives per context, with the observed target removed from the negative list
and supplied only as the supervised positive.
""",
        encoding="utf-8",
    )
    checksums = {
        path.name: _sha256(path)
        for path in sorted(args.output.iterdir())
        if path.is_file() and path.name != "checksums.json"
    }
    (args.output / "checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n"
    )
    print(args.output)


if __name__ == "__main__":
    main()
