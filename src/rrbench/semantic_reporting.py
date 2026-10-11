from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from rrbench.io import sha256, write_json
from rrbench.reporting import reject_private_fields, write_csv
from rrbench.semantic_ann import validate_complete_ann, validate_journal
from rrbench.semantic_ir import panel_metrics


def checked_receipt(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("complete") is not True:
        raise ValueError("Cannot export an unfinished research artifact.")
    pairs = (
        ("scores_sha256", "scores.npy"), ("query_rows_sha256", "query_rows.npy"),
        ("ids_sha256", "ids.npy"), ("vectors_sha256", "vectors.npy"),
        ("checkpoint_sha256", "model.pt"), ("model_sha256", "model.pt"),
        ("mf_prediction_sha256", "mf-ranked.npy"),
    )
    for key, filename in pairs:
        if key in value and sha256(path.parent / filename) != value[key]:
            raise ValueError("Raw completed research payload checksum changed before export.")
    return value


def public_quality(quality: dict[str, Any]) -> dict[str, Any]:
    if "by_group" in quality:
        return {
            "by_group": {name: public_quality(values) for name, values in quality["by_group"].items()},
            "queries": quality["queries"], "policy": quality["policy"],
        }
    rows = quality["per_query"]
    if len(rows) != quality["queries"] or not rows:
        raise ValueError("Public metric population count differs from its raw query observations.")
    output = {key: value for key, value in quality.items() if key != "per_query"}
    identifiers = [row["query_id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Public metric population contains duplicated query identities.")
    for name in rows[0]:
        if name == "query_id":
            continue
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        if not np.isfinite(values).all() or not np.isclose(values.mean(), quality[name], atol=1e-12, rtol=1e-10):
            raise ValueError("Published quality mean differs from its checked raw observations.")
        if name.startswith("recall@"):
            cutoff = name.removeprefix("recall@")
            output["hit_queries@" + cutoff] = int((values > 0).sum())
    reject_private_fields(output)
    return output


def public_timing(timing: dict[str, Any]) -> dict[str, Any]:
    samples = np.asarray(timing["samples_ms"], dtype=np.float64)
    if (
        samples.ndim != 1 or len(samples) != timing["sample_count"] or not len(samples)
        or not np.isfinite(samples).all() or (samples <= 0).any()
        or not np.isclose(samples.mean(), timing["latency_mean_ms"], rtol=1e-10, atol=1e-10)
    ):
        raise ValueError("Public timing samples/count/mean are inconsistent.")
    allowed = {
        "samples_ms", "sample_count", "latency_mean_ms", "latency_p50_ms", "observed_min_ms", "observed_max_ms",
        "offline_queries_per_second", "scope", "throughput_kind", "p99_warning",
    }
    if len(samples) >= 50:
        allowed |= {"latency_p95_ms", "latency_p99_ms", "latency_mean_ci95_ms"}
    result = {key: value for key, value in timing.items() if key in allowed}
    reject_private_fields(result)
    return result


def verify_online_parity(value: dict[str, Any]) -> None:
    expected = {(batch, cap) for batch in (1, 16) for cap in (10, 50, 100, 500)}
    if len(value["parity"]) != len(expected):
        raise ValueError("Online parity verification lacks declared batch/candidate conditions.")
    observed = set()
    for row in value["parity"]:
        condition = row["query_batch"], row["candidate_cap"]
        if condition not in expected or condition in observed or not row["same_candidates_scores_final_ids"]:
            raise ValueError("Online same-scorer parity is false, duplicated or outside the declared grid.")
        observed.add(condition)
    if observed != expected:
        raise ValueError("Online parity grid is incomplete.")


def verify_online_protocol(value: dict[str, Any], config: dict[str, Any]) -> None:
    if (
        config.get("primary_warmup_batches") != 1 or config.get("primary_measured_batches") != 3
        or config.get("timing_query_policy") != "Fourdisjointhash-selectedbatches, shared acrossmodels/caps/boundaries."
    ):
        raise ValueError("Online timing source does not use the exact approved matched sampling protocol.")
    expected = {
        (batch, cap, boundary) for batch in (1, 16) for cap in (10, 50, 100, 500)
        for boundary in ("integrated", "local-cascade")
    }
    seen = set()
    for row in value["conditions"]:
        key = row["query_batch"], row["candidate_cap"], row["boundary"]
        timing = row["full_online_latency"]
        batches = [row["warmup_query_rows"], *row["measured_query_row_batches"]]
        if (
            key not in expected or key in seen or timing["sample_count"] != 3 or len(batches) != 4
            or any(len(batch) != row["query_batch"] for batch in batches)
            or len({query for batch in batches for query in batch}) != row["query_batch"] * 4
        ):
            raise ValueError("Online primary measurement protocol has wrong/duplicated conditions or query sampling.")
        public_timing(timing)
        seen.add(key)
    if seen != expected:
        raise ValueError("Online primary measurement protocol has missing expected conditions.")


def public_online_cell(
    value: dict[str, Any], config: dict[str, Any], context: dict[str, Any],
) -> dict[str, Any]:
    if context["corpus"] == "kuairand" and context["ranker"] in (
        "bge-m3-rerank", "qwen06-rerank", "qwen8-rerank",
    ):
        if config.get("history_bound_preprocessing") != "Unique raw queries per scorerbatch; identical bounded tokens reused for repeated candidatepairs.":
            raise ValueError("Final recommendation text cells must all use the same optimized history preprocessing.")
    verify_online_parity(value)
    verify_online_protocol(value, config)
    schedules = {}
    for condition in value["conditions"]:
        schedule = [condition["warmup_query_rows"], *condition["measured_query_row_batches"]]
        batch = str(condition["query_batch"])
        digest = hashlib.sha256(json.dumps(schedule, separators=(",", ":")).encode()).hexdigest()
        if batch in schedules and schedules[batch] != digest:
            raise ValueError("Online candidate caps and boundaries use different measured query batches.")
        schedules[batch] = digest
    samples = value["device_memory_samples"]
    maximum = value["maximum_observed_device_memory_used_bytes"]
    if (
        not samples or any(
            not 0 <= sample["memory_used_bytes"] <= sample["device_memory_bytes"]
            or not 0 <= sample["gpu_utilization_percent"] <= 100
            for sample in samples
        )
        or max(sample["memory_used_bytes"] for sample in samples) != maximum
    ):
        raise ValueError("Online observed device memory samples and maximum are inconsistent.")
    configuration_fields = {
        "backend", "encoder", "ranker", "probes", "same_candidates", "same_precision",
        "same_scorer", "same_weights", "primary_measured_batches", "primary_warmup_batches",
        "timing_estimand", "timing_query_policy", "history_bound_preprocessing",
        "panel_sha256", "items_sha256", "partition_sha256", "model_registry_sha256",
        "checkpoint_sha256", "source_sha256", "offsets_sha256", "token_lengths_sha256",
        "smoke_sha256",
    }
    result = {
        **context, "configuration": {key: value for key, value in config.items() if key in configuration_fields},
        "query_schedule_sha256_by_batch": schedules,
        "parity": value["parity"], "device_memory_samples": samples,
        "device_memory_sampling": value["device_memory_sampling"],
        "maximum_observed_device_memory_used_bytes": maximum,
        "peak_cuda_allocated_bytes": value["peak_cuda_allocated_bytes"],
        "memory_caveat": value["memory_caveat"], "timing_scope": value["timing_scope"],
        "network": value["network"], "text_integrated_boundary": value["text_integrated_boundary"],
    }
    reject_private_fields(result)
    return result


def flatten_quality(context: dict[str, Any], quality: dict[str, Any]) -> list[dict[str, Any]]:
    if "by_group" in quality:
        return [
            row for name, values in quality["by_group"].items()
            for row in flatten_quality({**context, "query_panel": name}, values)
        ]
    sanitized = public_quality(quality)
    return [{**context, **{key: value for key, value in sanitized.items() if not isinstance(value, (dict, list))}}]


def export_results(root: Path, output: Path) -> None:
    plans = json.loads((root / "encoding-plan-v1.json").read_text(encoding="utf-8"))
    merged = json.loads((root / "merged-matrix-v2.json").read_text(encoding="utf-8"))
    required = {
        f"{task['corpus']}/{task['model']}/{dimension}"
        for task in plans["tasks"] for dimension in plans["registry"]["models"][task["model"]]["dimensions"]
    }
    if not merged["complete"] or set(merged["models"]) != required:
        raise ValueError("Cannot publish a final study without every complete full-catalog representation.")
    annex, ann_rows, quality_rows, online_rows, training_rows, sources = [], [], [], [], [], []
    unsupported_ann, online_cells, query_schedules = [], [], {}
    encoding_rows, query_encoding_rows = [], []
    for corpus, model in sorted({(task["corpus"], task["model"]) for task in plans["tasks"]}):
        tasks = [task for task in plans["tasks"] if task["corpus"] == corpus and task["model"] == model]
        receipts = [
            json.loads((Path(task["output"]) / "completed.json").read_text(encoding="utf-8")) for task in tasks
        ]
        expected = plans["rows_per_corpus"][corpus]
        if (
            not all(receipt["complete"] for receipt in receipts)
            or sum(receipt["rows"] for receipt in receipts) != expected
            or any(receipt["source_sha256"] != task["source_sha256"] for receipt, task in zip(receipts, tasks))
        ):
            raise ValueError("Offline encoding measurements omitted source-bound full-catalog rows.")
        elapsed = sum(receipt["elapsed_seconds"] for receipt in receipts)
        real_tokens = sum(receipt["real_tokens"] for receipt in receipts)
        padded_tokens = sum(receipt["padded_tokens"] for receipt in receipts)
        encoding_rows.append({
            "corpus": corpus, "encoder": model, "items": expected,
            "model_parameters": receipts[0]["parameters"], "native_dimension": receipts[0]["native_dimension"],
            "sum_worker_seconds": elapsed, "sum_worker_hours": elapsed / 3600,
            "real_tokens": real_tokens, "padded_tokens": padded_tokens,
            "at_token_cap_rows": sum(receipt["at_token_cap_rows"] for receipt in receipts),
            "real_tokens_per_worker_second": real_tokens / elapsed,
            "item_rows_per_worker_second": expected / elapsed,
            "padding_ratio": padded_tokens / real_tokens,
            "runtime_scope": "Summedmeasuredworkerwalltime,notparallelwalltimeorhardwareMFU;includesCPUtokenization/load/I/O.",
            "concurrency": "SomeexplicitcorrectnesssmokesbrieflysharedownedencoderGPU; notpure uncontendedofflinekernelbenchmark.",
        })
    for receipt in sorted((root / "query-embeddings-v2").glob("*/*/*/completed.json")):
        corpus, model, panel = receipt.parent.relative_to(root / "query-embeddings-v2").parts
        measured = checked_receipt(receipt)
        query_encoding_rows.append({
            "corpus": corpus, "encoder": model, "panel": panel, "queries": measured["rows"],
            "native_dimension": measured["native_dimension"], "elapsed_seconds": measured["elapsed_seconds"],
            "raw_query_overflow_rows": measured["length_audit"]["raw_rows_exceeding_cap"],
            "query_instruction": measured["query_instruction"],
            "latency": {
                batch: public_timing(values) for batch, values in measured["online_query_encoding_latency"].items()
            },
            "latency_population": "Repeatedfirst1/16panelqueries,incontrasttohash-sampledend-to-endprimarytiming.",
        })
    for name in sorted(required):
        corpus, model, dimension = name.split("/")
        directory = root / "ann-full-v1" / corpus / model / f"dim{dimension}"
        completed = json.loads((directory / "completed.json").read_text(encoding="utf-8"))
        records = [json.loads(line) for line in (directory / "ann.jsonl").read_text(encoding="utf-8").splitlines()]
        validate_complete_ann(
            directory, completed["catalog_items"], completed["dimension"], completed["panel_queries"],
            set(completed["conditions"]),
        )
        validate_journal(records, directory, set(completed["conditions"]), completed["panel_queries"], completed["catalog_items"])
        sources.append({"artifact": "ann/" + name, "receipt_sha256": sha256(directory / "completed.json"), "journal_sha256": sha256(directory / "ann.jsonl")})
        oracle = root / "ann-oracles-v1" / corpus / model / f"dim{dimension}" / "panel"
        oracle_receipt = json.loads((oracle / "completed.json").read_text(encoding="utf-8"))
        if sha256(oracle / "ids.npy") != oracle_receipt["ids.npy_sha256"]:
            raise ValueError("Exact semantic model quality oracle payload changed.")
        exact_ids = np.load(oracle / "ids.npy", mmap_mode="r")
        panel_file = root / ("ir-panels-v2/all-final.json" if corpus == "msmarco" else "recommendation-panels-v4/test.json")
        official_panel = json.loads(panel_file.read_text(encoding="utf-8"))
        quality_rows.extend(flatten_quality({
            "corpus": corpus, "encoder": model, "dimension": int(dimension), "stage": "exact_full_catalog_retrieval",
            "catalog_items": completed["catalog_items"],
        }, panel_metrics(official_panel, exact_ids)))
        for record in records:
            context = {"corpus": corpus, "encoder": model, "dimension": int(dimension), "backend": record["backend"], "status": record["status"]}
            if record["status"] != "ok":
                unsupported = {**context, "reason": record["reason"], "catalog_items": completed["catalog_items"]}
                ann_rows.append(unsupported)
                unsupported_ann.append(unsupported)
                continue
            safe = {
                **context, "catalog_items": record["catalog_items"], "k": record["k"], "probes": record["probes"],
                "geometric_recall_at_k": record["geometric_recall_at_k"],
                "relevance": public_quality(record["relevance"]), "environment": record["environment"],
                "prediction_ids_sha256": record["prediction_ids_sha256"],
                "latency": {batch: public_timing(values) for batch, values in record["latency"].items()},
            }
            annex.append(safe)
            for batch, values in safe["latency"].items():
                ann_rows.append({
                    **context, "catalog_items": record["catalog_items"], "k": record["k"], "probes": record["probes"],
                    "query_batch": int(batch), "geometric_recall_at_k": record["geometric_recall_at_k"],
                    **{key: value for key, value in values.items() if not isinstance(value, (dict, list))},
                })
            quality_rows.extend(flatten_quality({
                **context, "k": record["k"], "probes": record["probes"], "stage": "actual_retrieval",
            }, record["relevance"]))
    for corpus in ("msmarco", "kuairand"):
        for architecture in ("semantic-mlp", "torchrec-dlrm", "torchrec-dcn"):
            for seed in (17, 23, 42):
                directory = root / "cached-trained-v1" / corpus / architecture / f"seed{seed}"
                train = checked_receipt(directory / "completed.json")
                sources.append({
                    "artifact": f"cached-training/{corpus}/{architecture}/seed{seed}",
                    "receipt_sha256": sha256(directory / "completed.json"),
                    "checkpoint_sha256": train["checkpoint_sha256"],
                })
                if not train["complete"] or sha256(directory / "model.pt") != train["checkpoint_sha256"]:
                    raise ValueError("Full cached training model/checkpoint is not complete.")
                training_rows.append({
                    "corpus": corpus, "architecture": architecture, "seed": seed,
                    "parameters": train["parameters"], "model_class": train["model_class"],
                    "training_pairs": train["training_pairs"], "epochs": train["epochs"], "training_seconds": train["training_seconds"],
                })
                for phase in ("validation", "final"):
                    quality = checked_receipt(root / "cached-quality-v1" / corpus / architecture / f"seed{seed}" / phase / "completed.json")
                    sources.append({
                        "artifact": f"cached-quality/{corpus}/{architecture}/seed{seed}/{phase}",
                        "receipt_sha256": sha256(root / "cached-quality-v1" / corpus / architecture / f"seed{seed}" / phase / "completed.json"),
                        "scores_sha256": quality["scores_sha256"],
                    })
                    if not quality["complete"]:
                        raise ValueError("Cached full-panel evaluation is incomplete.")
                    for cap, metrics in quality["relevance_by_candidate_cap"].items():
                        quality_rows.extend(flatten_quality({
                            "corpus": corpus, "architecture": architecture, "seed": seed, "phase": phase,
                            "candidate_cap": int(cap), "stage": "actual_retrieval_cached_ranker",
                        }, metrics))
                quality = checked_receipt(root / "cached-common-quality-v1" / corpus / architecture / f"seed{seed}" / "completed.json")
                sources.append({
                    "artifact": f"cached-common-quality/{corpus}/{architecture}/seed{seed}",
                    "receipt_sha256": sha256(root / "cached-common-quality-v1" / corpus / architecture / f"seed{seed}" / "completed.json"),
                    "scores_sha256": quality["scores_sha256"],
                })
                if not quality["complete"]:
                    raise ValueError("Cached common-pool comparison is incomplete.")
                for cap, metrics in quality["relevance_by_candidate_cap"].items():
                    quality_rows.extend(flatten_quality({
                        "corpus": corpus, "architecture": architecture, "seed": seed,
                        "candidate_cap": int(cap), "stage": "common_pool_cached_ranker",
                    }, metrics))
        for ranker in ("bge-m3-rerank", "qwen06-rerank", "qwen8-rerank"):
            path = root / "strong-text-parallel-v2" / corpus / ranker / "collected" / "completed.json"
            if not path.exists():
                path = root / "strong-text-common-quality-v1" / corpus / ranker / "collected" / "completed.json"
            quality = checked_receipt(path)
            sources.append({
                "artifact": f"text-quality/{corpus}/{ranker}", "receipt_sha256": sha256(path),
                "scores_sha256": quality["scores_sha256"],
            })
            if not quality["complete"]:
                raise ValueError("Pretrained full fixed-common-pool text ranking is incomplete.")
            for cap, metrics in quality["relevance_by_candidate_cap"].items():
                quality_rows.extend(flatten_quality({
                    "corpus": corpus, "architecture": ranker, "candidate_cap": int(cap), "stage": "common_pool_text_ranker",
                }, metrics))
        for ranker in ("bge-m3-rerank", "qwen06-rerank", "qwen8-rerank", "semantic-mlp", "torchrec-dlrm", "torchrec-dcn"):
            for backend in ("faiss-gpu-fp32", "silver-fp32"):
                name = (
                    "online-text-unique-history-v2"
                    if corpus == "kuairand" and ranker in ("bge-m3-rerank", "qwen06-rerank", "qwen8-rerank")
                    else "online-full-catalog-v1"
                )
                directory = root / name / corpus / ranker / backend
                complete = json.loads((directory / "completed.json").read_text(encoding="utf-8"))
                if not complete["complete"]:
                    raise ValueError("One-GPU complete-catalog online cell is unfinished.")
                sources.append({"artifact": f"online/{corpus}/{ranker}/{backend}", "receipt_sha256": sha256(directory / "completed.json")})
                context = {"corpus": corpus, "encoder": "qwen8", "ranker": ranker, "backend": backend, "status": complete["status"]}
                if complete["status"] != "ok":
                    online_rows.append({**context, "reason": complete["reason"]})
                    continue
                config = json.loads((directory / "run-config.json").read_text(encoding="utf-8"))
                cell = public_online_cell(complete, config, context)
                for batch, digest in cell["query_schedule_sha256_by_batch"].items():
                    key = corpus, batch
                    if key in query_schedules and query_schedules[key] != digest:
                        raise ValueError("Online models and ANN backends use different measured query batches.")
                    query_schedules[key] = digest
                online_cells.append(cell)
                for condition in complete["conditions"]:
                    timing = public_timing(condition["full_online_latency"])
                    online_rows.append({
                        **context, "boundary": condition["boundary"], "query_batch": condition["query_batch"],
                        "candidate_cap": condition["candidate_cap"],
                        **{key: value for key, value in timing.items() if not isinstance(value, (dict, list))},
                        "samples_ms": timing["samples_ms"],
                        "maximum_observed_device_memory_used_bytes": complete["maximum_observed_device_memory_used_bytes"],
                    })
    for seed in (17, 23, 42):
        train = checked_receipt(root / "mf-full-logged-v1" / f"seed{seed}" / "completed.json")
        sources.append({
            "artifact": f"logged-mf-training/seed{seed}",
            "receipt_sha256": sha256(root / "mf-full-logged-v1" / f"seed{seed}" / "completed.json"),
            "model_sha256": train["model_sha256"],
        })
        training_rows.append({
            "corpus": "kuairand", "architecture": "logged-mf", "seed": seed,
            "parameters": train["model_parameters"], "training_pairs": train["training_pairs"],
            "epochs": train["completed_epochs"], "training_seconds": train["training_seconds"],
        })
        for phase in ("validation", "test"):
            complete = checked_receipt(root / "mf-quality-v1" / f"seed{seed}" / phase / "completed.json")
            if not complete["complete"]:
                raise ValueError("Actual full logged MF evaluation is incomplete.")
            quality_rows.extend(flatten_quality({
                "corpus": "kuairand", "architecture": "logged-mf", "seed": seed, "phase": phase, "stage": "full_catalog_id_baseline",
            }, complete["mf_relevance"]))
            if seed == 17:
                quality_rows.extend(flatten_quality({
                    "corpus": "kuairand", "architecture": "training-positive-popularity", "phase": phase,
                    "stage": "full_catalog_nonpersonalized_baseline",
                }, complete["training_popularity_relevance"]))
    for name in ("trec19", "trec20"):
        complete = checked_receipt(root / "bm25-quality-v2" / name / "completed.json")
        if not complete["complete"]:
            raise ValueError("Full-catalog lexical quality reference is incomplete.")
        quality_rows.extend(flatten_quality({
            "corpus": "msmarco", "architecture": "sqlite-fts5-bm25", "query_panel": name,
            "stage": "full_catalog_lexical_baseline",
        }, complete["relevance"]))
    diagnostic = json.loads((root / "static-profile-results-v1" / "completed.json").read_text(encoding="utf-8"))
    if not diagnostic["complete"] or not diagnostic["diagnostics"]["query_rows_unchanged"]:
        raise ValueError("Exploratory static profile comparison is incomplete or uses different populations.")
    quality_rows.extend(flatten_quality({
        "corpus": "kuairand", "architecture": "training-positive-static-semantic-profile",
        "stage": "exploratory_exact_full_catalog_retrieval",
    }, diagnostic["diagnostics"]["static_profile_exact"]))
    warning_log = root / "logs" / "full-online-text-timing-v1.log"
    runtime_warnings = [{
        "corpus": "msmarco", "ranker": "qwen8-rerank", "backend": "faiss-gpu-fp32",
        "message": line, "source_log_sha256": sha256(warning_log),
        "scope": "Recorded allocator warning; final cell completed all conditions and parity, not a no-OOM robustness guarantee.",
    } for line in warning_log.read_text(encoding="utf-8").splitlines()
        if "memory allocation failed with OOM" in line]
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "complete": True, "ann": annex, "ann_unsupported": unsupported_ann, "quality_rows": quality_rows,
        "online_rows": online_rows, "online_cells": online_cells,
        "training_rows": training_rows, "source_receipts": sources,
        "runtime_warnings": runtime_warnings,
        "models": plans["registry"]["models"],
        "encoding_rows": encoding_rows, "query_encoding_rows": query_encoding_rows,
        "privacy": "Aggregated metrics, measured timings and hashes only; no user/query/item identity, raw text or access details.",
        "coverage": "Recommendation results conditional on frozen catalog and history; full-positive-event coverage14.78%test/25.72%validation.",
        "native_memory_controls": {
            name: json.loads((root / filename).read_text(encoding="utf-8"))
            for name, filename in (
                ("ivf_half_clean_construction", "native-half-clean-gpu-memory-probe.json"),
                ("flat_half_feasibility", "native-full-flat-layout-feasibility-probe.json"),
            )
        },
        "namespace_duration_control": json.loads((root / "caption-id-exact-duration-permutation-audit.json").read_text(encoding="utf-8")),
    }
    reject_private_fields(summary)
    write_json(output / "summary.json", summary)
    write_csv(output / "ann.csv", ann_rows)
    write_csv(output / "quality.csv", quality_rows)
    write_csv(output / "training.csv", training_rows)
    write_csv(output / "offline-encoding.csv", encoding_rows)
    write_json(output / "query-encoding.json", query_encoding_rows)
    write_csv(output / "online.csv", [{key: value for key, value in row.items() if key != "samples_ms"} for row in online_rows])
    write_json(output / "online-raw.json", online_rows)
    write_json(output / "online-cells.json", online_cells)
    write_json(output / "runtime-warnings.json", runtime_warnings)
    files = {path.name: sha256(path) for path in output.iterdir() if path.is_file() and path.name != "checksums.json"}
    write_json(output / "checksums.json", files)
    print("FULL_SEMANTIC_STUDY_PUBLIC_ARTIFACTS_VERIFIED", len(annex), len(quality_rows), len(online_rows), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish fully measured semantic research artifacts without private identifiers.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export_results(Path(args.root), Path(args.output))


if __name__ == "__main__":
    main()
