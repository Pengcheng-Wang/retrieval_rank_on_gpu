import copy
import gzip
import hashlib
import json

import pytest

from rrbench.inference import BATCHES, CANDIDATE_COUNTS, MODELS, MODES, SEEDS
from rrbench.inference_reporting import publish_inference, summarize_cells, validate_experiment


def complete_fixture():
    manifest = {
        "models": list(MODELS), "modes": list(MODES), "seeds": list(SEEDS),
        "batches": list(BATCHES), "candidate_counts": list(CANDIDATE_COUNTS),
        "panel_users": 128, "warmup": 20, "samples": 50, "repeats": 3,
        "cpu_threads": 4, "smoke": False,
        "panels": {
            str(seed): {
                "checkpoint_sha256": {model: "checkpoint" for model in MODELS},
                "panel_sha256": "panel", "heldout_fields_loaded": False,
            } for seed in SEEDS
        },
    }
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            for mode in MODES:
                for batch in BATCHES:
                    for count in CANDIDATE_COUNTS:
                        rows.append({
                            "model": model, "seed": seed, "mode": mode, "batch_size": batch,
                            "candidates": count, "status": "ok",
                            "checkpoint_sha256": "checkpoint", "panel_sha256": "panel",
                            "eligibility_pass": True, "weights_dtype": "float32",
                            "eager_sparse_output_dtypes": ["torch.float32"],
                            "fidelity_vs_fp32": {
                                "eligible_scores": 128 * count, "mae": 0, "rmse": 0,
                                "normalized_rmse": 0, "max_absolute_error": 0,
                            },
                            "fidelity_vs_same_precision_eager": {"normalized_rmse": 0},
                            "ranking_vs_fp32": {
                                "top10_set_overlap": 1, "top100_set_overlap": 1,
                                "top100_is_full_candidate_set": count == 100,
                            },
                            "ranking_vs_same_precision_eager": {"top10_set_overlap": 1},
                            "changed_input_aba": {
                                "a_repeat_bitwise_identical": True,
                                "a_and_b_reference_differ": True,
                            },
                            "fp32_compiler_fidelity_pass": True if mode == "compiled-fp32" else None,
                            "first_call_ms": 2, "initial_warmup_ms": 3,
                            "compilation": {
                                "blocks_wrapped": {"dense_arch": True, "inter_arch": True, "over_arch": True},
                                "profile_compiled_region_calls": 3, "unique_graphs": 3, "graph_breaks": 0,
                                "timing_counters_unchanged": True, "profile_cuda_graph_launches": 0,
                                "execution_label": "inductor_without_observed_graph_replay",
                            },
                            "timing_repeats": [
                                {"repeat": repeat, "samples_ms": [1.0] * 50, "latency_mean_ms": 1.0}
                                for repeat in range(3)
                            ],
                        })
    completion = {
        "complete": True, "expected_cells": 144, "frozen_checkpoints_unchanged": True,
        "counts": {"ok": 144},
    }
    return manifest, completion, rows


def test_complete_shape_and_mode_coverage_has_paired_runtime_ratios():
    manifest, completion, rows = complete_fixture()
    validate_experiment(manifest, completion, rows)
    summaries = summarize_cells(rows)
    assert len(summaries) == 144
    assert all(row["runtime_speedup_vs_eager_fp32"] == 1 for row in summaries)


@pytest.mark.parametrize("mutation", [
    "missing", "duplicate", "holdout", "fallback", "fidelity", "samples", "checkpoint",
])
def test_inference_publication_rejects_incomplete_or_invalid_evidence(mutation):
    manifest, completion, rows = copy.deepcopy(complete_fixture())
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = rows[0]
    elif mutation == "holdout":
        manifest["panels"]["17"]["heldout_fields_loaded"] = True
    elif mutation == "fallback":
        next(row for row in rows if row["mode"] == "compiled-fp32")["compilation"]["unique_graphs"] = 0
    elif mutation == "fidelity":
        next(row for row in rows if row["mode"] == "compiled-fp32")["fp32_compiler_fidelity_pass"] = False
    elif mutation == "samples":
        rows[0]["timing_repeats"][0]["samples_ms"].pop()
    else:
        rows[0]["checkpoint_sha256"] = "different"
    with pytest.raises(ValueError):
        validate_experiment(manifest, completion, rows)


def test_explicit_error_cell_is_not_success_shaped_or_given_a_speedup():
    manifest, completion, rows = complete_fixture()
    failed = rows[-1]
    failed.clear()
    failed.update({
        "model": MODELS[-1], "seed": SEEDS[-1], "mode": MODES[-1],
        "batch_size": BATCHES[-1], "candidates": CANDIDATE_COUNTS[-1],
        "status": "unsupported", "failure_stage": "compile-setup",
    })
    completion["counts"] = {"ok": 143, "unsupported": 1}
    validate_experiment(manifest, completion, rows)
    summary = summarize_cells(rows)[-1]
    assert summary["status"] == "unsupported"
    assert "runtime_speedup_vs_eager_fp32" not in summary


def test_full_inference_publication_has_exact_hashes_and_no_overwrite(tmp_path):
    manifest, completion, rows = complete_fixture()
    manifest["milestone"] = "frozen TorchRec precision and dense-path compilation"
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps(manifest))
    (source / "completed.json").write_text(json.dumps(completion))
    (source / "cells.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    receipt = tmp_path / "resource.json"
    receipt.write_text(json.dumps({
        "restoration_verified": True, "restored_original_profile": True,
        "measurement_gpu_isolated": True,
    }))
    output, report = tmp_path / "publication", tmp_path / "report.md"
    result = publish_inference(source, receipt, output, report)
    assert result["bf16_equivalence_claim"] is False
    assert result["findings"]["alternative_cells_faster_than_eager_fp32"] == 0
    assert len(result["cell_summaries"]) == 144
    with gzip.open(output / "cells-raw.jsonl.gz", "rt", encoding="utf-8") as stream:
        assert len([line for line in stream if line.strip()]) == 144
    for name, expected in json.loads((output / "files.json").read_text()).items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    assert "not an equivalence gate" in report.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="new inference publication"):
        publish_inference(source, receipt, output, report)
