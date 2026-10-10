from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from rrbench.inference import BATCHES, CANDIDATE_COUNTS, MODELS, MODES, SEEDS
from rrbench.io import sha256, write_json
from rrbench.reporting import reject_private_fields, table, write_csv


def cell_key(row: dict[str, Any]) -> tuple[str, int, str, int, int]:
    return row["model"], row["seed"], row["mode"], row["batch_size"], row["candidates"]


def validate_experiment(manifest: dict[str, Any], completion: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    expected_protocol = {
        "models": list(MODELS), "modes": list(MODES), "seeds": list(SEEDS),
        "batches": list(BATCHES), "candidate_counts": list(CANDIDATE_COUNTS),
        "panel_users": 128, "warmup": 20, "samples": 50, "repeats": 3,
        "cpu_threads": 4, "smoke": False,
    }
    if any(manifest.get(name) != value for name, value in expected_protocol.items()):
        raise ValueError("Inference experiment differs from the prespecified full protocol.")
    expected = {
        (model, seed, mode, batch, count)
        for model in MODELS for seed in SEEDS for mode in MODES
        for batch in BATCHES for count in CANDIDATE_COUNTS
    }
    keys = [cell_key(row) for row in rows]
    if set(keys) != expected or len(keys) != len(set(keys)):
        raise ValueError("Inference cells contain missing, duplicate, or unexpected conditions.")
    if (
        completion.get("complete") is not True
        or completion.get("expected_cells") != len(expected)
        or completion.get("frozen_checkpoints_unchanged") is not True
    ):
        raise ValueError("Frozen inference completion and checkpoint-preservation evidence is required.")
    counts: dict[str, int] = {}
    for row in rows:
        reject_private_fields(row)
        status = row.get("status")
        if status not in ("ok", "fidelity_failed", "unsupported", "error"):
            raise ValueError("Inference status is not an explicit outcome.")
        counts[status] = counts.get(status, 0) + 1
        if status in ("error", "unsupported"):
            if not row.get("failure_stage"):
                raise ValueError("Failed cells need an explicit failure stage.")
            continue
        panel = manifest["panels"][str(row["seed"])]
        if row["checkpoint_sha256"] != panel["checkpoint_sha256"][row["model"]]:
            raise ValueError("Inference checkpoint differs from the frozen manifest.")
        if row["panel_sha256"] != panel["panel_sha256"] or panel.get("heldout_fields_loaded") is not False:
            raise ValueError("Inference panel identity or holdout-exclusion contract failed.")
        if row.get("eligibility_pass") is not True or row.get("weights_dtype") != "float32":
            raise ValueError("Exact eligibility and original FP32 weights must be preserved.")
        if row.get("eager_sparse_output_dtypes") != ["torch.float32"]:
            raise ValueError("Sparse outputs did not retain the declared FP32 eager representation.")
        if row["fidelity_vs_fp32"]["eligible_scores"] != 128 * row["candidates"]:
            raise ValueError("Fidelity must cover every eligible pair in the frozen panel.")
        if row["changed_input_aba"]["a_repeat_bitwise_identical"] is not True:
            raise ValueError("Changed-input graph-reuse correctness was not verified.")
        if row["changed_input_aba"]["a_and_b_reference_differ"] is not True:
            raise ValueError("Changed-input fixture is not discriminating.")
        if row["mode"].startswith("compiled"):
            evidence = row["compilation"]
            if (
                set(evidence["blocks_wrapped"]) != {"dense_arch", "inter_arch", "over_arch"}
                or not all(evidence["blocks_wrapped"].values())
                or evidence["profile_compiled_region_calls"] < 3
                or evidence["unique_graphs"] < 3 or evidence["graph_breaks"]
                or evidence.get("timing_counters_unchanged") is not True
            ):
                raise ValueError("Compiled-cell execution/fallback evidence is incomplete.")
            if row["mode"] == "compiled-fp32" and status == "ok" and row["fp32_compiler_fidelity_pass"] is not True:
                raise ValueError("Compiled FP32 success must pass its fixed numerical tolerance.")
        repeats = row["timing_repeats"]
        if len(repeats) != 3 or {repeat["repeat"] for repeat in repeats} != {0, 1, 2}:
            raise ValueError("Exactly three timing repeats are required.")
        for repeat in repeats:
            samples = np.asarray(repeat["samples_ms"], dtype=np.float64)
            if len(samples) != 50 or not np.isfinite(samples).all() or (samples <= 0).any():
                raise ValueError("Each repeat must preserve50 finite positive batch timings.")
            if not np.isclose(samples.mean(), repeat["latency_mean_ms"], rtol=1e-10):
                raise ValueError("Timing mean differs from retained samples.")
    if counts != completion["counts"]:
        raise ValueError("Completion status counts differ from cell outcomes.")


def summarize_cells(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        compact = {
            name: row[name] for name in ("model", "seed", "mode", "batch_size", "candidates", "status")
        }
        if row["status"] in ("ok", "fidelity_failed"):
            means = np.asarray([repeat["latency_mean_ms"] for repeat in row["timing_repeats"]])
            fidelity = row["fidelity_vs_fp32"]
            compact.update({
                "latency_mean_ms": float(means.mean()),
                "latency_repeat_sd_ms": float(means.std(ddof=1)),
                "first_call_ms": row["first_call_ms"],
                "initial_warmup_ms": row["initial_warmup_ms"],
                "mae_vs_fp32": fidelity["mae"],
                "rmse_vs_fp32": fidelity["rmse"],
                "normalized_rmse_vs_fp32": fidelity["normalized_rmse"],
                "max_absolute_error_vs_fp32": fidelity["max_absolute_error"],
                "top10_overlap_vs_fp32": row["ranking_vs_fp32"]["top10_set_overlap"],
                "top100_overlap_vs_fp32": row["ranking_vs_fp32"]["top100_set_overlap"],
                "top100_is_full_candidate_set": row["ranking_vs_fp32"]["top100_is_full_candidate_set"],
                "fp32_compiler_fidelity_pass": row["fp32_compiler_fidelity_pass"],
                "execution_label": row["compilation"]["execution_label"],
                "profile_cuda_graph_launches": row["compilation"]["profile_cuda_graph_launches"],
                "same_precision_normalized_rmse": row["fidelity_vs_same_precision_eager"]["normalized_rmse"],
                "same_precision_top10_overlap": row["ranking_vs_same_precision_eager"]["top10_set_overlap"],
            })
        else:
            compact["failure_stage"] = row["failure_stage"]
            compact["error_type"] = row.get("error_type", "worker-process-failure")
        result.append(compact)
    reference = {
        (row["model"], row["seed"], row["batch_size"], row["candidates"]): row
        for row in result if row["mode"] == "eager-fp32" and row["status"] == "ok"
    }
    for row in result:
        baseline = reference.get((row["model"], row["seed"], row["batch_size"], row["candidates"]))
        if baseline is not None and row["status"] == "ok":
            row["runtime_speedup_vs_eager_fp32"] = baseline["latency_mean_ms"] / row["latency_mean_ms"]
    return result


def publish_inference(directory: Path, receipt_path: Path, output: Path, report: Path) -> dict[str, Any]:
    if output.exists():
        raise ValueError("Use a new inference publication directory.")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    completion = json.loads((directory / "completed.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (directory / "cells.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    reject_private_fields(manifest)
    reject_private_fields(receipt)
    if any(receipt.get(name) is not True for name in ("restoration_verified", "restored_original_profile", "measurement_gpu_isolated")):
        raise ValueError("Verified resource isolation and restored keepalive policy are required.")
    validate_experiment(manifest, completion, rows)
    summaries = summarize_cells(rows)
    successful_compiler_fp32 = [
        row for row in summaries if row["mode"] == "compiled-fp32" and row["status"] == "ok"
    ]
    representative = [
        row for row in summaries
        if row["batch_size"] == 16 and row["candidates"] == 1000
    ]
    findings: dict[str, Any] = {
        "compiled_fp32_cells_passing_fixed_gate": len(successful_compiler_fp32),
        "expected_compiled_fp32_cells": 36,
        "compiled_cells_with_observed_graph_replay": sum(
            row.get("profile_cuda_graph_launches", 0) > 0
            for row in summaries if row["mode"].startswith("compiled") and row["status"] == "ok"
        ),
        "representative_shape": {"batch": 16, "candidates": 1000},
        "representative_ranges": {},
        "bf16_equivalence_gate_applied": False,
    }
    for mode in MODES:
        selected = [row for row in representative if row["mode"] == mode and row["status"] == "ok"]
        if selected:
            findings["representative_ranges"][mode] = {
                "cells": len(selected),
                "runtime_ratio_vs_eager_fp32_min": min(row["runtime_speedup_vs_eager_fp32"] for row in selected),
                "runtime_ratio_vs_eager_fp32_max": max(row["runtime_speedup_vs_eager_fp32"] for row in selected),
                "top10_overlap_min": min(row["top10_overlap_vs_fp32"] for row in selected),
                "top10_overlap_max": max(row["top10_overlap_vs_fp32"] for row in selected),
                "normalized_rmse_min": min(row["normalized_rmse_vs_fp32"] for row in selected if row["normalized_rmse_vs_fp32"] is not None),
                "normalized_rmse_max": max(row["normalized_rmse_vs_fp32"] for row in selected if row["normalized_rmse_vs_fp32"] is not None),
            }
    result = {
        "milestone": manifest["milestone"], "protocol": manifest,
        "counts": completion["counts"], "resource_receipt": receipt,
        "source_journal_sha256": sha256(directory / "cells.jsonl"),
        "source_manifest_sha256": sha256(directory / "manifest.json"),
        "cell_summaries": summaries,
        "findings": findings,
        "interpretation": "Frozen training-only candidate-panel inference, not new held-out recommendation quality or serving QPS.",
        "bf16_equivalence_claim": False,
    }
    faster_alternatives = [
        row for row in summaries
        if row["status"] == "ok" and row["mode"] != "eager-fp32"
        and row.get("runtime_speedup_vs_eager_fp32", 0) > 1
    ]
    findings["alternative_cells_faster_than_eager_fp32"] = len(faster_alternatives)
    reject_private_fields(result)
    output.mkdir(parents=True)
    content = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
    (output / "cells-raw.jsonl.gz").write_bytes(gzip.compress(content.encode(), mtime=0))
    write_json(output / "summary.json", result)
    write_json(output / "protocol.json", manifest)
    write_json(output / "resource-receipt.json", receipt)
    write_csv(output / "inference.csv", summaries)
    status_rows = [
        [row["model"], str(row["seed"]), row["mode"], str(row["batch_size"]),
         str(row["candidates"]), row["status"], row.get("failure_stage", "-")]
        for row in summaries if row["status"] != "ok"
    ]
    lines = [
        "# Frozen TorchRec inference comparison", "",
        "_Final bounded milestone, 2026-10-10. Earlier results, checkpoints, and held-out evaluations remain immutable._",
        "", "---", "",
        "## 📋 Scope and execution contract", "",
        "This compares actual TorchRec DLRM and DLRM_DCN checkpoints for seeds17/23/42 "
        "under eager FP32, eager BF16 autocast, and the supported dense-path compilation "
        "in FP32/BF16. Weights and sparse embedding outputs remain FP32. Only "
        "`dense_arch`, `inter_arch`, and `over_arch` are compiled; sparse input/lookup and final selection stay eager.[^1]",
        "",
        "The panel is128 deterministic training-active users and frozen MF candidatecaps "
        "100/500/1000; batches1/16; three50-sample repeats after20 warmups. No training labels, "
        "validation/test targets, or validation/test histories were loaded. Exact training-seen "
        "eligibility is fixed before comparing execution modes.",
        "",
        "```mermaid",
        "flowchart LR",
        "    accTitle: Frozen Inference Comparison Contract",
        "    accDescr: Identical frozen weights and training-only candidate inputs feed eager and compiled precision modes, with numerical fidelity and timing evaluated separately.",
        '    frozen["Frozen checkpoints"] --> modes["Four inference modes"]',
        '    panel["Training-only candidate panel"] --> modes',
        '    modes --> fidelity["Untimed score and order fidelity"]',
        '    modes --> runtime["Warm wrapper batch timing"]',
        "```",
        "",
        "The timer includes step marking, KeyedJaggedTensor creation, forward inference, "
        "masking and stable top100 selection. It excludes candidate/feature preparation, "
        "transfers, reference/fidelity collection, first-call compilation and profiling. "
        "It is warm local reranker-wrapper batch latency, not serving QPS.",
        "",
        "## 🎯 Measured findings", "",
        f"{findings['compiled_fp32_cells_passing_fixed_gate']} of36 compiled-FP32 cells "
        "passed the fixed score-fidelity gate and completed their measurements. "
        f"{findings['compiled_cells_with_observed_graph_replay']} successful compiled cells "
        "also had profiler-observed CUDA-graph replay. This is not inferred merely from a compiled wrapper.",
        "",
        table(
            ["Mode", "Representative cells", "Runtime ratio vs eager FP32 range", "Top10 overlap range", "NRMSE range"],
            [[mode, str(values["cells"]),
              f"{values['runtime_ratio_vs_eager_fp32_min']:.3f}-{values['runtime_ratio_vs_eager_fp32_max']:.3f}x",
              f"{values['top10_overlap_min']:.4f}-{values['top10_overlap_max']:.4f}",
              f"{values['normalized_rmse_min']:.6f}-{values['normalized_rmse_max']:.6f}"]
             for mode, values in findings["representative_ranges"].items()],
        ),
        "",
        "A runtime ratio below1 means the mode was slower than eager FP32 at the "
        "same frozen checkpoint and shape. BF16 score/order agreement is descriptive "
        "and cannot be promoted to held-out recommendation-quality non-inferiority.",
        "",
        f"Across all declared conditions, {len(faster_alternatives)} alternative-mode cells "
        "were faster than their paired eager-FP32 reference. This does not imply that "
        "larger/richer DLRM workloads or different compilation scopes behave identically.",
        "",
        "## ✅ Coverage and correctness outcomes", "",
        f"All144 declared cells were attempted. Outcomes: `{json.dumps(completion['counts'], sort_keys=True)}`. "
        "Errors, unsupported paths and fixed FP32 fidelity failures are retained explicitly rather than replaced by eager fallback.",
        "",
        "Compiled FP32 has a prespecified per-score `atol=1e-5, rtol=1e-4` gate. "
        "BF16 is intentional approximation: its numerical and order agreement is empirical, "
        "not a claim of unchanged recommendation quality. Even passing FP32 score tolerance "
        "does not establish identical ordering. Top100 overlap at cap100 is marked trivial.[^2]",
        "",
        "Graph output snapshots are copied outside timing before the next marked invocation. "
        "An A/B/A same-shape fixture checks changed inputs and repeat determinism. Compiled "
        "cells require three-block Dynamo/profiler evidence, no graph breaks, fail-on-recompile "
        "measurement stance, and unchanged timing counters. `cudaGraphLaunch` observation is "
        "separate from successful Inductor compilation.[^3]",
        "",
        "## 📊 Representative warm runtime", "",
        "Batch16, candidatecap1000. Every row remains paired by frozen checkpoint seed and input shape.",
        "",
        table(
            ["Model", "Seed", "Mode", "Batch ms", "Repeat SD", "Runtime ratio vs eager FP32", "Execution"],
            [[row["model"], str(row["seed"]), row["mode"], f"{row['latency_mean_ms']:.4f}",
              f"{row['latency_repeat_sd_ms']:.4f}", f"{row.get('runtime_speedup_vs_eager_fp32', 0):.3f}x",
              row["execution_label"]]
             for row in summaries if row["status"] == "ok" and row["batch_size"] == 16 and row["candidates"] == 1000],
        ),
        "",
        "## 🔍 Compiler-only fidelity", "",
        "Compiled modes are also compared with their corresponding eager precision, "
        "separating compilation changes from the BF16-versus-FP32 approximation. "
        "Only compiled FP32 has the fixed score-tolerance acceptance gate.",
        "",
        table(
            ["Model", "Seed", "Compiled mode", "NRMSE vs same-precision eager", "Top10 overlap", "FP32 gate"],
            [[row["model"], str(row["seed"]), row["mode"],
              f"{row['same_precision_normalized_rmse']:.6f}" if row["same_precision_normalized_rmse"] is not None else "undefined",
              f"{row['same_precision_top10_overlap']:.4f}",
              str(row["fp32_compiler_fidelity_pass"]) if row["mode"] == "compiled-fp32" else "not an equivalence gate"]
             for row in summaries if row["status"] in ("ok", "fidelity_failed")
             and row["batch_size"] == 16 and row["candidates"] == 1000 and row["mode"].startswith("compiled")],
        ),
        "",
        "## 🔬 Score and ordering fidelity", "",
        "Same representative shape; FP32 is the reference. Normalized RMSE divides by "
        "reference RMS over every eligible panel score, and an all-zero reference makes it undefined.",
        "",
        table(
            ["Model", "Seed", "Mode", "NRMSE vs FP32", "Max score error", "Top10 set overlap", "Top100 set overlap"],
            [[row["model"], str(row["seed"]), row["mode"],
              f"{row['normalized_rmse_vs_fp32']:.6f}" if row["normalized_rmse_vs_fp32"] is not None else "undefined",
              f"{row['max_absolute_error_vs_fp32']:.6f}", f"{row['top10_overlap_vs_fp32']:.4f}",
              f"{row['top100_overlap_vs_fp32']:.4f}"]
             for row in summaries if row["status"] in ("ok", "fidelity_failed")
             and row["batch_size"] == 16 and row["candidates"] == 1000 and row["mode"] != "eager-fp32"],
        ),
        "",
        "## ⏱️ First-call costs", "",
        "First-call wall time can include compilation, compiler-cache lookup and initial graph "
        "preparation. It is not deducted from a deployment cold start. Compilation caches can "
        "be warm across processes; these are observed first calls, not guaranteed clean-cache build costs.",
        "",
        table(
            ["Model", "Seed", "Compiled mode", "First-call ms", "Initial warmup ms", "Observed graph launches"],
            [[row["model"], str(row["seed"]), row["mode"], f"{row['first_call_ms']:.1f}",
              f"{row['initial_warmup_ms']:.1f}", str(row["profile_cuda_graph_launches"])]
             for row in summaries if row["status"] == "ok" and row["mode"].startswith("compiled")
             and row["batch_size"] == 16 and row["candidates"] == 1000],
        ),
        "",
        "## ⚠️ Explicit unsuccessful cells", "",
        table(["Model", "Seed", "Mode", "Batch", "Candidates", "Status", "Stage"], status_rows)
        if status_rows else "No declared cell required an unsupported or error fallback.",
        "",
        "## 🛡️ Limits and final handoff", "",
        "This is an inference/fidelity extension conditional on one training-only candidate "
        "panel and small two-table rankers, not another test-set evaluation, quality non-inferiority "
        "test, distributed embedding benchmark, or full SilverTorch compiled serving graph. "
        "No checkpoints were reselected or retrained. Earlier studies are unchanged.",
        "",
        "The original keepalive supervisor/profile is restored, and only node0 writes the "
        "durable experiment bank. Access and operational details remain local/ignored. This "
        "completes the two bounded hourly follow-ups; no additional unbounded compute is scheduled.",
        "",
        "[^1]: Meta-PyTorch. TorchRec DLRM `compile_dense_path`. https://github.com/meta-pytorch/torchrec/blob/v1.8.0-rc1/torchrec/models/dlrm.py#L573-L609",
        "[^2]: PyTorch2.13. Numerical accuracy and automatic mixed precision. https://docs.pytorch.org/docs/2.13/notes/numerical_accuracy.html ; https://docs.pytorch.org/docs/2.13/amp.html",
        "[^3]: PyTorch2.13. CUDAGraph Trees and step boundaries. https://docs.pytorch.org/docs/2.13/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html ; https://docs.pytorch.org/docs/2.13/generated/torch.compiler.cudagraph_mark_step_begin.html",
        "",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
    write_json(output / "files.json", {
        path.name: sha256(path) for path in sorted(output.iterdir()) if path.name != "files.json"
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish verified frozen TorchRec inference cells.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--resource-receipt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    result = publish_inference(
        Path(args.input), Path(args.resource_receipt), Path(args.output), Path(args.report)
    )
    print(f"Published frozen inference outcomes: {result['counts']}")


if __name__ == "__main__":
    main()
