from __future__ import annotations

import argparse
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
                ann_rows.append({**context, "reason": record["reason"], "catalog_items": completed["catalog_items"]})
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
                if not train["complete"] or sha256(directory / "model.pt") != train["checkpoint_sha256"]:
                    raise ValueError("Full cached training model/checkpoint is not complete.")
                training_rows.append({
                    "corpus": corpus, "architecture": architecture, "seed": seed,
                    "parameters": train["parameters"], "model_class": train["model_class"],
                    "training_pairs": train["training_pairs"], "epochs": train["epochs"], "training_seconds": train["training_seconds"],
                })
                for phase in ("validation", "final"):
                    quality = checked_receipt(root / "cached-quality-v1" / corpus / architecture / f"seed{seed}" / phase / "completed.json")
                    if not quality["complete"]:
                        raise ValueError("Cached full-panel evaluation is incomplete.")
                    for cap, metrics in quality["relevance_by_candidate_cap"].items():
                        quality_rows.extend(flatten_quality({
                            "corpus": corpus, "architecture": architecture, "seed": seed, "phase": phase,
                            "candidate_cap": int(cap), "stage": "actual_retrieval_cached_ranker",
                        }, metrics))
                quality = checked_receipt(root / "cached-common-quality-v1" / corpus / architecture / f"seed{seed}" / "completed.json")
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
            if not quality["complete"]:
                raise ValueError("Pretrained full fixed-common-pool text ranking is incomplete.")
            for cap, metrics in quality["relevance_by_candidate_cap"].items():
                quality_rows.extend(flatten_quality({
                    "corpus": corpus, "architecture": ranker, "candidate_cap": int(cap), "stage": "common_pool_text_ranker",
                }, metrics))
        for ranker in ("bge-m3-rerank", "qwen06-rerank", "qwen8-rerank", "semantic-mlp", "torchrec-dlrm", "torchrec-dcn"):
            for backend in ("faiss-gpu-fp32", "silver-fp32"):
                directory = root / "online-full-catalog-v1" / corpus / ranker / backend
                complete = json.loads((directory / "completed.json").read_text(encoding="utf-8"))
                if not complete["complete"]:
                    raise ValueError("One-GPU complete-catalog online cell is unfinished.")
                sources.append({"artifact": f"online/{corpus}/{ranker}/{backend}", "receipt_sha256": sha256(directory / "completed.json")})
                context = {"corpus": corpus, "encoder": "qwen8", "ranker": ranker, "backend": backend, "status": complete["status"]}
                if complete["status"] != "ok":
                    online_rows.append({**context, "reason": complete["reason"]})
                    continue
                if len(complete["conditions"]) != 16 or len(complete["parity"]) != 8:
                    raise ValueError("Online cell does not cover every declared candidate/batch/boundary condition.")
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
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "complete": True, "ann": annex, "quality_rows": quality_rows,
        "online_rows": online_rows, "training_rows": training_rows, "source_receipts": sources,
        "models": plans["registry"]["models"],
        "privacy": "Aggregated metrics, measured timings and hashes only; no user/query/item identity, raw text or access details.",
        "coverage": "Recommendation results conditional on frozen catalog and history; full-positive-event coverage14.78%test/25.72%validation.",
    }
    reject_private_fields(summary)
    write_json(output / "summary.json", summary)
    write_csv(output / "ann.csv", ann_rows)
    write_csv(output / "quality.csv", quality_rows)
    write_csv(output / "training.csv", training_rows)
    write_csv(output / "online.csv", [{key: value for key, value in row.items() if key != "samples_ms"} for row in online_rows])
    write_json(output / "online-raw.json", online_rows)
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
