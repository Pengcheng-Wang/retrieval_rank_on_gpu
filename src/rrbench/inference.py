from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch

from rrbench.io import append_jsonl, bind_run_config, sha256, write_json
from rrbench.metrics import latency_summary, recall_at_k
from rrbench.models import TorchRecRanker, make_model
from rrbench.runtime import configure, environment_manifest, require_cuda

MODELS = ("torchrec-dlrm", "torchrec-dcn")
MODES = ("eager-fp32", "eager-bf16", "compiled-fp32", "compiled-bf16")
SEEDS = (17, 23, 42)
CANDIDATE_COUNTS = (100, 500, 1000)
BATCHES = (1, 16)
FP32_ATOL = 1e-5
FP32_RTOL = 1e-4


def array_digest(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def score_fidelity(reference: np.ndarray, actual: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    if reference.shape != actual.shape or valid.shape != reference.shape or valid.dtype != np.bool_:
        raise ValueError("Score arrays and Boolean eligibility must have identical shapes.")
    if not valid.any():
        raise ValueError("At least one eligible score is required.")
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError("Raw scores must be finite, before applying the eligibility mask.")
    baseline = reference[valid].astype(np.float64)
    observed = actual[valid].astype(np.float64)
    difference = observed - baseline
    rms = float(np.sqrt(np.mean(baseline * baseline)))
    rmse = float(np.sqrt(np.mean(difference * difference)))
    tolerance = FP32_ATOL + FP32_RTOL * np.abs(baseline)
    violations = int(np.count_nonzero(np.abs(difference) > tolerance))
    return {
        "eligible_scores": len(baseline), "mae": float(np.abs(difference).mean()),
        "rmse": rmse, "max_absolute_error": float(np.abs(difference).max()),
        "reference_rms": rms, "normalized_rmse": rmse / rms if rms else None,
        "fp32_tolerance_violation_count": violations, "fp32_tolerance_pass": violations == 0,
        "fp32_atol": FP32_ATOL, "fp32_rtol": FP32_RTOL,
    }


def stable_rankings(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if scores.ndim != 2 or scores.shape != valid.shape or valid.dtype != np.bool_:
        raise ValueError("Ranking requires aligned score and Boolean eligibility matrices.")
    masked = np.where(valid, scores, -np.inf)
    positions = np.argsort(-masked, axis=1, kind="stable")
    return np.where(np.take_along_axis(valid, positions, axis=1), positions, -1)


def ranking_fidelity(reference: np.ndarray, actual: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    expected, observed = stable_rankings(reference, valid), stable_rankings(actual, valid)
    result: dict[str, Any] = {
        "tie_policy": "descending stable sort; candidate order breaks exact score ties",
        "identical_full_order_fraction": float((expected == observed).all(axis=1).mean()),
    }
    for cutoff in (10, 100):
        overlaps = recall_at_k(observed, expected, cutoff)
        result[f"top{cutoff}_set_overlap"] = float(overlaps.mean())
        result[f"top{cutoff}_minimum_query_overlap"] = float(overlaps.min())
        result[f"top{cutoff}_is_full_candidate_set"] = bool((valid.sum(axis=1) <= cutoff).all())
    return result


def training_arrays(data_directory: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    data_manifest = json.loads((data_directory / "manifest.json").read_text(encoding="utf-8"))
    prepared = data_directory / "prepared.npz"
    digest = sha256(prepared)
    if digest != data_manifest["prepared_sha256"]:
        raise ValueError("Training data checksum differs from the original prepared manifest.")
    with np.load(prepared) as data:
        return data["user_counts"], data["item_counts"], data["catalog"], data["train_seen"], digest


def prepare_panel(
    data_directory: Path, checkpoint_directory: Path, seed: int, panel_users: int,
    max_candidates: int, destination: Path,
) -> dict[str, Any]:
    device = require_cuda()
    counts_user, counts_item, catalog, history, data_digest = training_arrays(data_directory)
    active = np.flatnonzero(counts_user > 0)
    if panel_users > len(active) or max_candidates > len(catalog):
        raise ValueError("Panel or candidate count exceeds the training-only inventory.")
    rng = np.random.default_rng(20261010)
    selected = np.sort(rng.choice(active, panel_users, replace=False))
    state = torch.load(checkpoint_directory / f"mf_s{seed}.pt", map_location=device, weights_only=True)
    if state["model"] != "mf" or state["seed"] != seed:
        raise ValueError("Frozen MF checkpoint identity differs from the protocol.")
    retriever = make_model("mf", len(counts_user), len(counts_item), device, seed).eval()
    retriever.load_state_dict(state["state_dict"])
    with torch.inference_mode():
        query = retriever.user(torch.tensor(selected, device=device))
        item = retriever.item(torch.tensor(catalog, device=device))
        scores = query @ item.T
        eligible = torch.tensor(~history[selected][:, catalog], device=device)
        scores.masked_fill_(~eligible, -torch.inf)
        positions = scores.argsort(dim=1, descending=True, stable=True)[:, :max_candidates]
        candidates = torch.tensor(catalog, device=device)[positions]
        valid = eligible.gather(1, positions)
        candidates.masked_fill_(~valid, -1)
    candidates_numpy, valid_numpy = candidates.cpu().numpy(), valid.cpu().numpy()
    if not valid_numpy.all():
        raise ValueError("The frozen panel contains fewer eligible items than its candidate cap.")
    if history[selected[:, None], candidates_numpy].any():
        raise RuntimeError("Training-seen items survived candidate preparation.")
    count_user = np.log1p(counts_user).astype(np.float32)
    count_item = np.log1p(counts_item).astype(np.float32)
    count_user /= max(float(count_user.max()), 1.0)
    count_item /= max(float(count_item.max()), 1.0)
    dense = np.stack((
        np.broadcast_to(count_user[selected, None], candidates_numpy.shape),
        count_item[candidates_numpy],
    ), axis=-1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError("Frozen panel destination already exists; do not overwrite it.")
    np.savez_compressed(
        destination, users=selected, candidates=candidates_numpy, valid=valid_numpy, dense=dense,
        user_count=np.asarray(len(counts_user)), item_count=np.asarray(len(counts_item)),
    )
    return {
        "seed": seed, "panel_users": panel_users, "max_candidates": max_candidates,
        "prepared_data_sha256": data_digest,
        "panel_sha256": sha256(destination),
        "user_selection_sha256": array_digest(selected),
        "candidate_ids_sha256": array_digest(candidates_numpy),
        "eligibility_sha256": array_digest(valid_numpy),
        "dense_features_sha256": array_digest(dense),
        "history_source": "training-only observed interactions; no validation/test histories",
        "selection": "fixed RNG20261010 sample from training-active users",
        "retriever": "already validation-selected frozen MF; stable full warm-catalog ordering",
        "checkpoint_sha256": {
            name: sha256(checkpoint_directory / f"{name}_s{seed}.pt")
            for name in ("mf", *MODELS)
        },
        "heldout_fields_loaded": False,
    }


@dataclass
class StaticInputs:
    users: torch.Tensor
    items: torch.Tensor
    dense: torch.Tensor
    valid: torch.Tensor
    all_users: torch.Tensor
    all_items: torch.Tensor
    all_dense: torch.Tensor
    all_valid: torch.Tensor
    batch: int
    candidates: int

    @classmethod
    def from_panel(cls, panel: Path, batch: int, candidates: int, device: torch.device) -> StaticInputs:
        with np.load(panel) as data:
            users = data["users"]
            items = data["candidates"][:, :candidates].copy()
            dense = data["dense"][:, :candidates].copy()
            valid = data["valid"][:, :candidates].copy()
        if len(users) % batch or items.shape != (len(users), candidates):
            raise ValueError("Frozen panel must cover full fixed-shape batches.")
        all_users = torch.tensor(np.repeat(users[:, None], candidates, axis=1), device=device)
        all_items = torch.tensor(items, device=device)
        all_dense = torch.tensor(dense, device=device)
        all_valid = torch.tensor(valid, device=device)
        result = cls(
            torch.empty(batch * candidates, dtype=torch.long, device=device),
            torch.empty(batch * candidates, dtype=torch.long, device=device),
            torch.empty(batch * candidates, 2, dtype=torch.float32, device=device),
            torch.empty(batch, candidates, dtype=torch.bool, device=device),
            all_users, all_items, all_dense, all_valid, batch, candidates,
        )
        result.load(0)
        return result

    def load(self, start: int) -> None:
        end = start + self.batch
        self.users.copy_(self.all_users[start:end].reshape(-1))
        self.items.copy_(self.all_items[start:end].clamp_min(0).reshape(-1))
        self.dense.copy_(self.all_dense[start:end].reshape(-1, 2))
        self.valid.copy_(self.all_valid[start:end])


def forward(model: TorchRecRanker, inputs: StaticInputs, bf16: bool) -> torch.Tensor:
    torch.compiler.cudagraph_mark_step_begin()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        return model(inputs.users, inputs.items, inputs.dense).reshape(inputs.batch, inputs.candidates)


def inference_step(model: TorchRecRanker, inputs: StaticInputs, bf16: bool) -> torch.Tensor:
    logits = forward(model, inputs, bf16).float()
    logits = logits.masked_fill(~inputs.valid, -torch.inf)
    positions = logits.argsort(dim=1, descending=True, stable=True)[:, :min(100, inputs.candidates)]
    ids = inputs.items.reshape(inputs.batch, inputs.candidates).gather(1, positions)
    return ids.masked_fill(~inputs.valid.gather(1, positions), -1)


def collect_logits(model: TorchRecRanker, inputs: StaticInputs, bf16: bool) -> np.ndarray:
    pieces = []
    for start in range(0, len(inputs.all_users), inputs.batch):
        inputs.load(start)
        # Graph outputs are borrowed: snapshot before the next marked invocation.
        logits = forward(model, inputs, bf16).detach().clone().float().cpu().numpy()
        pieces.append(logits)
    return np.concatenate(pieces)


def compiler_counters() -> dict[str, int]:
    from torch._dynamo.utils import counters

    return {
        "unique_graphs": int(counters["stats"]["unique_graphs"]),
        "calls_captured": int(counters["stats"]["calls_captured"]),
        "graph_breaks": int(sum(counters["graph_break"].values())),
    }


def profile_compilation(model: TorchRecRanker, inputs: StaticInputs, bf16: bool) -> dict[str, Any]:
    from torch._dynamo.eval_frame import OptimizedModule

    blocks = {
        name: isinstance(getattr(model.model, name), OptimizedModule)
        for name in ("dense_arch", "inter_arch", "over_arch")
    }
    if not all(blocks.values()):
        raise RuntimeError("One or more requested dense blocks are not compiled wrappers.")
    with torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
    ]) as profile:
        inference_step(model, inputs, bf16)
        torch.cuda.synchronize()
    events = [
        {"name": event.key, "count": event.count}
        for event in profile.key_averages()
        if "compiled" in event.key.lower() or "cudagraph" in event.key.lower()
    ]
    regions = sum(event["count"] for event in events if "compiled region" in event["name"].lower())
    launches = sum(event["count"] for event in events if "cudagraphlaunch" in event["name"].lower())
    counters = compiler_counters()
    if counters["unique_graphs"] < 3 or counters["graph_breaks"] or regions < 3:
        raise RuntimeError("Requested three-block compilation lacks complete execution evidence.")
    return {
        "blocks_wrapped": blocks, "profile_events": events, **counters,
        "profile_compiled_region_calls": regions, "profile_cuda_graph_launches": launches,
        "execution_label": "inductor_with_cuda_graph_replay" if launches else "inductor_without_observed_graph_replay",
    }


@torch.inference_mode()
def run_cell(args: Any) -> None:
    configure(args.seed, args.threads)
    device = require_cuda()
    output = Path(args.output)
    panel_path, checkpoint = Path(args.panel), Path(args.checkpoint)
    row: dict[str, Any] = {
        "model": args.model, "seed": args.seed, "mode": args.mode,
        "batch_size": args.batch, "candidates": args.candidates,
        "panel_sha256": sha256(panel_path), "checkpoint_sha256": sha256(checkpoint),
        "environment": environment_manifest(), "status": "pending",
    }
    stage = "loading-frozen-model"
    try:
        if min(args.batch, args.candidates, args.samples, args.repeats) <= 0 or args.warmup < 0:
            raise ValueError("Positive shapes/sample counts and nonnegative warmup are required.")
        bf16 = args.mode.endswith("bf16")
        compiled = args.mode.startswith("compiled")
        if bf16 and not torch.cuda.is_bf16_supported():
            raise NotImplementedError("BF16 is unavailable on this GPU.")
        with np.load(panel_path) as panel:
            user_count, item_count = int(panel["user_count"]), int(panel["item_count"])
        inputs = StaticInputs.from_panel(panel_path, args.batch, args.candidates, device)
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if state["model"] != args.model or state["seed"] != args.seed:
            raise ValueError("Frozen checkpoint model/seed does not match the cell.")
        model = make_model(args.model, user_count, item_count, device, args.seed).eval()
        if not isinstance(model, TorchRecRanker):
            raise TypeError("This milestone requires actual TorchRec DLRM/DCN.")
        model.load_state_dict(state["state_dict"])
        if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise RuntimeError("Frozen weights must remain FP32 in every mode.")
        from torchrec.models.dlrm import DLRM

        row["torchrec_dlrm_source_sha256"] = sha256(Path(inspect.getsourcefile(DLRM)))
        row["model_class"] = model.model_class
        row["parameters"] = sum(parameter.numel() for parameter in model.parameters())
        row["bf16_reduced_precision_reduction"] = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        row["weights_dtype"] = "float32"
        stage = "eager-references"
        observed_sparse_dtypes = []
        hook = model.model.sparse_arch.register_forward_hook(
            lambda module, inputs, output: observed_sparse_dtypes.append(str(output.dtype))
        )
        reference = collect_logits(model, inputs, False)
        precision_reference = collect_logits(model, inputs, True) if bf16 else reference
        hook.remove()
        row["eager_sparse_output_dtypes"] = sorted(set(observed_sparse_dtypes))
        row["reference_scores_sha256"] = array_digest(reference)
        row["precision_reference_sha256"] = array_digest(precision_reference)
        inputs.load(0)
        compilation = {"execution_label": "eager", "profile_cuda_graph_launches": 0}
        started = time.perf_counter()
        if compiled:
            stage = "compile-setup"
            import torch._dynamo.config as dynamo_config

            dynamo_config.suppress_errors = False
            dynamo_config.fail_on_recompile_limit_hit = True
            row["compiler_recompile_limit"] = dynamo_config.recompile_limit
            model.model.compile_dense_path(compile_mode="reduce-overhead")
        row["compile_setup_ms"] = (time.perf_counter() - started) * 1000
        stage = "first-call"
        torch.cuda.synchronize()
        started = time.perf_counter()
        first = inference_step(model, inputs, bf16)
        torch.cuda.synchronize()
        row["first_call_ms"] = (time.perf_counter() - started) * 1000
        del first
        stage = "warmup"
        started = time.perf_counter()
        for _ in range(args.warmup):
            inference_step(model, inputs, bf16)
        torch.cuda.synchronize()
        row["initial_warmup_ms"] = (time.perf_counter() - started) * 1000
        row["logits_dtype"] = str(forward(model, inputs, bf16).dtype)
        stance = torch.compiler.set_stance("fail_on_recompile") if compiled else nullcontext()
        with stance:
            stage = "fidelity-panel"
            actual = collect_logits(model, inputs, bf16)
            valid = inputs.all_valid.cpu().numpy()
            row["fidelity_vs_fp32"] = score_fidelity(reference, actual, valid)
            row["fidelity_vs_same_precision_eager"] = score_fidelity(precision_reference, actual, valid)
            row["ranking_vs_fp32"] = ranking_fidelity(reference, actual, valid)
            row["ranking_vs_same_precision_eager"] = ranking_fidelity(precision_reference, actual, valid)
            stage = "changed-input-graph-check"
            snapshots = []
            for start in (0, args.batch, 0):
                inputs.load(start)
                snapshots.append(forward(model, inputs, bf16).detach().clone().float().cpu().numpy())
            np.testing.assert_array_equal(snapshots[0], snapshots[2])
            aba_reference = precision_reference[args.batch:2 * args.batch]
            row["changed_input_aba"] = {
                "a_repeat_bitwise_identical": True,
                "a_and_b_reference_differ": bool(not np.array_equal(precision_reference[:args.batch], aba_reference)),
                "b_fidelity": score_fidelity(
                    aba_reference, snapshots[1], valid[args.batch:2 * args.batch]
                ),
            }
            if not row["changed_input_aba"]["a_and_b_reference_differ"]:
                raise RuntimeError("Changed-input fixture is not discriminating.")
            inputs.load(0)
            stage = "compilation-evidence"
            if compiled:
                compilation = profile_compilation(model, inputs, bf16)
            row["compilation"] = compilation
            row["timing_repeats"] = []
            counters_before = compiler_counters() if compiled else {}
            stage = "timing"
            for repeat in range(args.repeats):
                inputs.load(repeat * args.batch % (len(inputs.all_users) - args.batch + 1))
                for _ in range(args.warmup):
                    inference_step(model, inputs, bf16)
                torch.cuda.synchronize()
                samples = []
                for _ in range(args.samples):
                    torch.cuda.synchronize()
                    started = time.perf_counter_ns()
                    result = inference_step(model, inputs, bf16)
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter_ns() - started) / 1e6)
                if (result < 0).any():
                    raise RuntimeError("Invalid IDs survived the exact eligibility mask.")
                allowed = inputs.all_items[repeat * args.batch:repeat * args.batch + args.batch]
                if not (result[:, :, None] == allowed[:, None, :]).any(dim=-1).all():
                    raise RuntimeError("A returned item is not in the frozen eligible candidate set.")
                row["timing_repeats"].append({"repeat": repeat, **latency_summary(samples, args.batch)})
            if compiled:
                counters_after = compiler_counters()
                if counters_after != counters_before:
                    raise RuntimeError("Compiler work or a graph break occurred during measurement.")
                row["compilation"]["timing_counters_unchanged"] = True
            row["fp32_compiler_fidelity_pass"] = (
                row["fidelity_vs_fp32"]["fp32_tolerance_pass"] if compiled and not bf16 else None
            )
            row["precision_interpretation"] = (
                "BF16 is an intentional approximation; score/order agreement is empirical, not equivalence."
                if bf16 else "Fixed FP32 numerical tolerance; order agreement is separately reported."
            )
            row["eligibility_pass"] = True
            row["status"] = (
                "fidelity_failed" if row["fp32_compiler_fidelity_pass"] is False else "ok"
            )
        write_json(output, row)
        print(f"CELL_COMPLETE {args.model} s{args.seed} {args.mode} b{args.batch} k{args.candidates} {row['status']}", flush=True)
    except (RuntimeError, ValueError, TypeError, AssertionError, AttributeError, NotImplementedError) as error:
        traceback.print_exc()
        row.update(
            status="unsupported" if isinstance(error, (NotImplementedError, AttributeError)) else "error",
            failure_stage=stage, error_type=f"{type(error).__module__}.{type(error).__name__}",
            failure_notice="Full traceback retained in the node-local cell log; no success or speedup claim.",
        )
        write_json(output, row)
        raise SystemExit(2)


def run_experiment(args: Any) -> None:
    configure(17, args.threads)
    device = require_cuda()
    output, checkpoint_directory = Path(args.output), Path(args.checkpoints)
    if output.exists():
        raise ValueError("Use a new immutable inference experiment directory.")
    seeds = (17,) if args.smoke else SEEDS
    batches = (1,) if args.smoke else BATCHES
    candidates = (100,) if args.smoke else CANDIDATE_COUNTS
    panel_users = 2 if args.smoke else 128
    warmup, samples, repeats = (3, 2, 1) if args.smoke else (20, 50, 3)
    manifest = {
        "milestone": "frozen TorchRec precision and dense-path compilation",
        "models": list(MODELS), "modes": list(MODES), "seeds": list(seeds),
        "batches": list(batches), "candidate_counts": list(candidates), "panel_users": panel_users,
        "warmup": warmup, "samples": samples, "repeats": repeats, "cpu_threads": args.threads,
        "worker_timeout_seconds": args.timeout, "smoke": args.smoke,
        "environment": environment_manifest(),
        "heldout_policy": "No train labels, validation/test targets, or validation/test histories loaded.",
        "compile_scope": "TorchRec compile_dense_path(compile_mode=reduce-overhead); sparse path eager",
        "weight_precision": "FP32 frozen weights in every mode",
        "fp32_gate": {"atol": FP32_ATOL, "rtol": FP32_RTOL},
        "bf16_gate": "No arbitrary quality/equivalence cutoff; report score and ordering differences.",
        "latency_scope": "Resident pair inputs to masked stable top100; sparse-input creation and marker included",
        "excluded": "Retrieval, features preparation, data transfer, reference/fidelity collection, compilation/first-call/profiling",
        "worker_isolation": "Fresh process per model, checkpoint seed, precision/compile mode and exact shape; sequential GPU",
    }
    bind_run_config(output, manifest)
    panels = {}
    for seed in seeds:
        panels[str(seed)] = prepare_panel(
            Path(args.data), checkpoint_directory, seed, panel_users, max(candidates),
            output / "private-inputs" / f"panel_s{seed}.npz",
        )
    write_json(output / "manifest.json", {**manifest, "panels": panels})
    journal = output / "cells.jsonl"
    counts: dict[str, int] = {}
    for model in MODELS:
        for seed in seeds:
            for batch in batches:
                for count in candidates:
                    for mode in MODES:
                        name = f"{model}_s{seed}_{mode}_b{batch}_k{count}"
                        receipt = output / "cells" / f"{name}.json"
                        log = output / "private-logs" / f"{name}.log"
                        log.parent.mkdir(parents=True, exist_ok=True)
                        command = [
                            sys.executable, "-m", "rrbench.inference", "--worker",
                            "--model", model, "--seed", str(seed), "--mode", mode,
                            "--batch", str(batch), "--candidates", str(count),
                            "--panel", str(output / "private-inputs" / f"panel_s{seed}.npz"),
                            "--checkpoint", str(checkpoint_directory / f"{model}_s{seed}.pt"),
                            "--output", str(receipt), "--threads", str(args.threads),
                            "--warmup", str(warmup), "--samples", str(samples), "--repeats", str(repeats),
                        ]
                        with log.open("wb") as stream:
                            try:
                                result = subprocess.run(
                                    command, stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=args.timeout, check=False,
                                )
                                exit_code = result.returncode
                            except subprocess.TimeoutExpired:
                                exit_code = -1
                        if receipt.exists():
                            row = json.loads(receipt.read_text())
                            if row["status"] == "ok" and exit_code:
                                raise RuntimeError("A worker claims success despite a nonzero exit.")
                        else:
                            row = {
                                "model": model, "seed": seed, "mode": mode,
                                "batch_size": batch, "candidates": count, "status": "error",
                                "failure_stage": "worker-timeout" if exit_code == -1 else "worker-exited-without-receipt",
                                "worker_exit_code": exit_code,
                                "failure_notice": "Node-local log retains the failure; no fallback was executed.",
                            }
                        row["cell_log_sha256"] = sha256(log)
                        append_jsonl(journal, row)
                        counts[row["status"]] = counts.get(row["status"], 0) + 1
                        print(f"{name}: {row['status']}", flush=True)
    for seed in seeds:
        for name, expected in panels[str(seed)]["checkpoint_sha256"].items():
            if sha256(checkpoint_directory / f"{name}_s{seed}.pt") != expected:
                raise RuntimeError("A frozen checkpoint changed during the experiment.")
    write_json(output / "completed.json", {
        "complete": True, "milestone": manifest["milestone"], "counts": counts,
        "expected_cells": len(MODELS) * len(seeds) * len(batches) * len(candidates) * len(MODES),
        "completion_means": "All declared cells attempted; errors and fidelity failures remain explicit.",
        "frozen_checkpoints_unchanged": True,
    })
    if args.smoke and counts.get("ok", 0) != 8:
        raise RuntimeError("Mandatory 100-pair smoke did not pass all declared modes.")
    print("FROZEN_INFERENCE_EXPERIMENT_COMPLETE", counts, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen TorchRec inference precision/compilation milestone.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--candidates", type=int)
    parser.add_argument("--panel")
    parser.add_argument("--checkpoint")
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--data")
    parser.add_argument("--checkpoints")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.threads <= 0 or args.timeout <= 0:
        parser.error("Thread count and worker timeout must be positive.")
    if args.worker:
        required = (
            "model", "seed", "mode", "batch", "candidates", "panel", "checkpoint",
            "warmup", "samples", "repeats",
        )
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            parser.error("Worker arguments are required: " + ", ".join(missing))
        run_cell(args)
    else:
        if args.data is None or args.checkpoints is None:
            parser.error("Frozen inference requires --data and --checkpoints.")
        run_experiment(args)


if __name__ == "__main__":
    main()
