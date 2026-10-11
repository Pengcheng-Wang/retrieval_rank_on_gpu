from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from rrbench import semantic_personalization as study
from rrbench import semantic_ir as ir_metrics
from rrbench import semantic_statistics as statistics
from rrbench.exact import exact_topk
from rrbench.io import bind_run_config, sha256, write_json
from rrbench.runtime import configure, environment_manifest, require_cuda
from rrbench.semantic_cascade import device_telemetry
from rrbench.semantic_ir import panel_metrics
from rrbench.semantic_recommendation_baselines import publish_mf_epoch, validate_mf_resume
from rrbench.semantic_sampling import stable_rank
from rrbench.semantic_statistics import paired_cluster_effect


def checked_raw_queries(
    vectors: Path, receipt_file: Path, panel_file: Path, item_receipt: dict[str, Any],
) -> np.ndarray:
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    panel = json.loads(panel_file.read_text(encoding="utf-8"))
    values = np.load(vectors, mmap_mode="r")
    representation = item_receipt["representation"]
    if (
        not receipt["complete"] or sha256(vectors) != receipt["vectors_sha256"]
        or sha256(panel_file) != receipt["panel_sha256"] or receipt["query_ids"] != panel["query_ids"]
        or receipt["model"] != representation["model"] or receipt["revision"] != representation["revision"]
        or values.dtype != np.float16 or values.shape != (len(panel["query_ids"]), item_receipt["stored_dimension"])
        or not np.isfinite(values).all()
    ):
        raise ValueError("Raw query bytes, panel identities or frozen encoder representation changed.")
    return values


def train_one_epoch(
    model: study.HistoryAttentionHead, items: torch.Tensor, histories: np.ndarray, candidates: np.ndarray,
    positives: np.ndarray, optimizer: torch.optim.Optimizer, batch_size: int, seed: int, epoch: int,
    temperature: float = 0.05,
) -> tuple[float, int]:
    if (
        items.requires_grad or items.ndim != 2 or histories.ndim != 2 or candidates.ndim != 2
        or len(histories) != len(candidates) or positives.shape != candidates.shape
        or min(batch_size, len(histories)) <= 0 or epoch < 0
        or ((histories < -1) | (histories >= len(items))).any()
        or ((candidates < -1) | (candidates >= len(items))).any()
    ):
        raise ValueError("Fitting requires a frozen bank and valid aligned history/candidate inputs.")
    model.train()
    order = np.random.default_rng(seed + epoch).permutation(len(histories))
    total, count = 0.0, 0
    for start in range(0, len(order), batch_size):
        rows = order[start:start + batch_size]
        history = torch.as_tensor(histories[rows], device=items.device, dtype=torch.long)
        candidate = torch.as_tensor(candidates[rows], device=items.device, dtype=torch.long)
        mask = history >= 0
        query = model(items[history.clamp_min(0)], mask)
        logits = (query[:, None] * items[candidate.clamp_min(0)].float()).sum(dim=-1)
        positive = torch.as_tensor(positives[rows], device=items.device, dtype=torch.bool)
        optimizer.zero_grad(set_to_none=True)
        loss = study.sampled_softmax_loss(logits, positive, candidate >= 0, temperature)
        if not torch.isfinite(loss):
            raise RuntimeError("User-head fitting loss is nonfinite.")
        loss.backward()
        if any(value.grad is not None and not torch.isfinite(value.grad).all() for value in model.parameters()):
            raise RuntimeError("User-head fitting gradient is nonfinite.")
        optimizer.step()
        total += float(loss.item()) * len(rows)
        count += len(rows)
    return total / count, count


def pilot_gate(
    aligned: dict[str, Any], wrong: dict[str, Any], popularity: dict[str, Any], users: np.ndarray,
    draws: int = 2000, minimum_hit_users: int = 10, minimum_separation: float = 0.005,
) -> dict[str, Any]:
    identities = [row["query_id"] for row in aligned["per_query"]]
    if any([row["query_id"] for row in quality["per_query"]] != identities for quality in (wrong, popularity)):
        raise ValueError("Pilot controls changed the evaluation population.")
    values = [np.asarray([row["recall@1000"] for row in quality["per_query"]]) for quality in (aligned, wrong, popularity)]
    summary = study.equal_user_metrics(aligned, users)
    separation = paired_cluster_effect(values[0], values[1], users, seed=20261011, draws=draws)
    versus_popularity = paired_cluster_effect(values[0], values[2], users, seed=20261011, draws=draws)
    return {
        "go": summary["hit_users@1000"] >= minimum_hit_users and separation["mean_user_effect"] >= minimum_separation,
        "evaluated_users": summary["users"], "paired_rows": summary["queries"],
        "hit_users": summary["hit_users@1000"], "aligned_minus_wrong_user": separation,
        "aligned_minus_popularity": versus_popularity,
        "interpretation": "Information-value pilot gate; stopping is not evidence that population personalization signal is zero.",
    }


def validate_smoke(receipt: dict[str, Any], items_hash: str, fitting_hash: str) -> None:
    if (
        not receipt["complete"] or receipt["study_role"] != "correctness-only" or receipt["rows"] != 100
        or receipt["item_vectors_sha256"] != items_hash or receipt["fitting_panel_sha256"] != fitting_hash
    ):
        raise ValueError("Actual 100-row user-head smoke is missing or bound to different inputs.")


def checked_model(directory: Path, binding: dict[str, Any], queries: int) -> dict[str, Any]:
    receipt = json.loads((directory / "completed.json").read_text())
    if (
        not receipt["complete"] or receipt["epochs"] != 3 or receipt["fitting_queries"] != queries
        or receipt["binding"] != binding or sha256(directory / "model.pt") != receipt["model_sha256"]
    ):
        raise ValueError("Completed user-head source, fitting scope or model bytes changed.")
    state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    if state["binding"] != binding or any(not torch.isfinite(value).all() for value in state["model"].values()):
        raise ValueError("Completed user-head state binding or finite weights changed.")
    return receipt


def checked_evaluation(directory: Path, binding: dict[str, Any]) -> dict[str, Any]:
    receipt = json.loads((directory / "completed.json").read_text())
    if not receipt["complete"] or receipt["binding"] != binding:
        raise ValueError("Completed evaluation source binding changed.")
    if not receipt["files"] or any(sha256(directory / name) != digest for name, digest in receipt["files"].items()):
        raise ValueError("Completed personalization evaluation payload changed.")
    return receipt


def evaluation_implementation(
    core: Path | None = None, runtime: Path | None = None,
    metrics: Path | None = None, uncertainty: Path | None = None,
) -> dict[str, str]:
    return {
        "implementation_sha256": sha256(core if core is not None else Path(study.__file__)),
        "runtime_sha256": sha256(runtime if runtime is not None else Path(__file__)),
        "metric_sha256": sha256(metrics if metrics is not None else Path(ir_metrics.__file__)),
        "statistics_sha256": sha256(uncertainty if uncertainty is not None else Path(statistics.__file__)),
    }


def _utc(value: str) -> datetime.datetime:
    parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Admission time must include an explicit timezone.")
    return parsed.astimezone(datetime.timezone.utc)


def make_admission_decision(
    validation: dict[str, Any], decided_at: str, cutoff: str,
) -> dict[str, Any]:
    if (
        not validation["complete"] or validation["phase"] != "validation"
        or validation["model_seed"] != 17 or validation["model_epochs"] != 3
        or not isinstance(validation["pilot_gate"]["go"], bool)
    ):
        raise ValueError("Admission decision requires the fixed complete validation pilot.")
    on_time = _utc(decided_at) <= _utc(cutoff)
    go = on_time and validation["pilot_gate"]["go"]
    return {
        "complete": True, "phase": "validation", "model_seed": 17, "model_epochs": 3,
        "binding": validation["binding"], "pilot_gate": validation["pilot_gate"],
        "go": go, "decision_status": "admitted" if go else ("not_admitted" if on_time else "time_budget_incomplete"),
        "decision_time_utc": _utc(decided_at).isoformat(), "pilot_cutoff_utc": cutoff,
        "interpretation": "A late pilot is time-budget incomplete, not evidence of absent personalized signal. No late expansion is admitted.",
    }


def persist_admission(validation_path: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    validation = json.loads(validation_path.read_text())
    checked_evaluation(validation_path.parent, {**validation["binding"], **evaluation_implementation()})
    if validation["binding"]["protocol_sha256"] != sha256(protocol_path):
        raise ValueError("Admission protocol differs from the evaluated fixed pilot.")
    cutoff = json.loads(protocol_path.read_text())["authorization"]["first_pilot_gate_utc"]
    digest = sha256(validation_path)
    if output.exists():
        prior = json.loads(output.read_text())
        rebuilt = make_admission_decision(validation, prior["decision_time_utc"], cutoff)
        if (
            prior["validation_receipt_sha256"] != digest
            or any(prior.get(key) != value for key, value in rebuilt.items())
        ):
            raise ValueError("Persisted admission decision or fixed pilot source changed.")
        return prior
    decision = make_admission_decision(
        validation, datetime.datetime.now(datetime.timezone.utc).isoformat(), cutoff,
    )
    decision.update({
        "validation_receipt_path": str(validation_path.resolve()),
        "validation_receipt_sha256": digest,
    })
    write_json(output, decision)
    return decision


def validate_admission(
    receipt: dict[str, Any], items_hash: str, protocol_hash: str,
    cutoff: str = "2026-10-11T15:03:01.929Z",
) -> None:
    if (
        not receipt["complete"] or receipt["phase"] != "validation"
        or receipt["model_seed"] != 17 or receipt["model_epochs"] != 3
        or receipt["binding"]["items_sha256"] != items_hash
        or receipt["binding"]["protocol_sha256"] != protocol_hash or receipt["pilot_gate"]["go"] is not True
        or receipt.get("go") is not True or receipt.get("decision_status") != "admitted"
        or receipt.get("pilot_cutoff_utc") != cutoff or "decision_time_utc" not in receipt
        or _utc(receipt["decision_time_utc"]) > _utc(cutoff)
    ):
        raise ValueError("Seed23 admission requires the passing fixed seed17 epoch3 validation pilot.")


def select_second_seed(
    admission: dict[str, Any], second: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if admission.get("complete") is not True or not isinstance(admission.get("go"), bool):
        raise ValueError("Qualification requires a complete persisted admission decision.")
    if not admission["go"]:
        return None
    if second is None or second.get("complete") is not True:
        raise ValueError("The admitted second-seed result is missing or incomplete.")
    return second


def validate_shared_evaluation_bindings(first: dict[str, Any], second: dict[str, Any]) -> None:
    shared = (
        "protocol_sha256", "items_sha256", "panel_sha256", "raw_queries_sha256",
        "popularity_sha256", "history_metadata_sha256", "coverage_receipt_sha256",
        "implementation_sha256", "runtime_sha256", "metric_sha256", "statistics_sha256", "phase",
    )
    for key in shared:
        if key not in first["binding"] or key not in second["binding"] or first["binding"][key] != second["binding"][key]:
            raise ValueError("Two-seed evaluation shared source binding differs.")


def validate_evaluation_phase(panel: dict[str, Any], phase: str) -> None:
    if phase not in ("validation", "test") or panel["name"] != "kuairand-" + phase:
        raise ValueError("Evaluation phase and the frozen panel identity disagree.")


def phase_admission(phase: str, values: dict[str, Any]) -> dict[str, Any]:
    if phase == "test":
        return {"status": "not_applicable", "go": None, "reason": "Final test metrics cannot authorize new fitting or select checkpoints."}
    if phase != "validation":
        raise ValueError("Unknown pilot evaluation phase.")
    return {"status": "validation_only", **values}


def qualification_summary(
    first: dict[str, Any], second: dict[str, Any] | None, admission: dict[str, Any],
    admission_hash: str | None = None, draws: int = 2000,
) -> dict[str, Any]:
    second = select_second_seed(admission, second)
    for key in ("items_sha256", "protocol_sha256", "checkpoint_sha256"):
        if first["binding"].get(key) != admission["binding"].get(key):
            raise ValueError("Final seed17 evaluation and fixed admission source bindings differ.")
    if second is not None:
        validate_admission(admission, first["binding"]["items_sha256"], first["binding"]["protocol_sha256"])
        validate_shared_evaluation_bindings(first, second)
        if admission_hash is None or second["binding"].get("admission_sha256") != admission_hash:
            raise ValueError("Second-seed evaluation is bound to a different admission decision.")
    records = [first] + ([second] if second is not None else [])
    if any(
        not record["complete"] or record["phase"] != "test" or record["model_epochs"] != 3
        for record in records
    ) or first["model_seed"] != 17 or (second is not None and second["model_seed"] != 23):
        raise ValueError("Qualification requires the fixed final test checkpoint/seed set.")
    latency = max(record["incremental_user_encoding_timing"]["16"]["latency_mean_ms"] for record in records)
    memory = max(sample["memory_used_bytes"] for record in records for sample in record["device_memory_samples"])
    runtime = {
        "batch16_mean_ms": latency, "batch16_within_5ms": latency <= 5,
        "maximum_observed_device_bytes": memory, "within_24GiB_observed_budget": memory <= 24 * (1 << 30),
        "scope": "Incremental user encoding and finite sampled memory, not a full-serving or unseen-peak guarantee.",
    }
    if second is None:
        return {
            "two_seed_status": "not_assessed_by_design", "research_qualification": False,
            "reason": "Persisted validation decision did not admit expansion; no stale second-seed file is included.",
            "admission_status": admission["decision_status"],
            "runtime_budget": runtime,
        }
    ids = [row["query_id"] for row in first["quality"]["learned"]["per_query"]]
    users = np.asarray(first["query_users"])
    learned = []
    population = [row["query_id"] for row in first["quality"]["training-popularity"]["per_query"]]
    if population != ids:
        raise ValueError("Qualification popularity and learned population differ.")
    popularity = np.asarray([row["recall@1000"] for row in first["quality"]["training-popularity"]["per_query"]])
    directions = []
    for record in records:
        if (
            record["query_users"] != first["query_users"]
            or [row["query_id"] for row in record["quality"]["learned"]["per_query"]] != ids
            or [row["query_id"] for row in record["quality"]["training-popularity"]["per_query"]] != ids
        ):
            raise ValueError("Two-seed qualification changes the users or query population.")
        control = np.asarray([row["recall@1000"] for row in record["quality"]["training-popularity"]["per_query"]])
        if not np.array_equal(control, popularity):
            raise ValueError("Two-seed popularity control changed.")
        values = np.asarray([row["recall@1000"] for row in record["quality"]["learned"]["per_query"]])
        learned.append(values)
        directions.append(paired_cluster_effect(values, popularity, users, seed=20261011, draws=draws)["mean_user_effect"])
    effect = paired_cluster_effect(np.mean(learned, axis=0), popularity, users, seed=20261011, draws=draws)
    qualifies = (
        effect["mean_user_effect"] >= 0.01 and effect["ci95_mean_user_effect"][0] > 0
        and all(value > 0 for value in directions) and runtime["batch16_within_5ms"]
        and runtime["within_24GiB_observed_budget"]
    )
    return {
        "two_seed_status": "assessed", "effect": effect, "seed_user_effects": directions,
        "research_qualification": qualifies, "runtime_budget": runtime,
        "aggregation": "Mean of fixed-seed per-query metrics, not prediction ensemble or test seed selection.",
        "interpretation": "Scoped post-test exploratory qualification only; no causal engagement or production adoption claim.",
    }


def load_item_bank(vectors: Path, receipt_file: Path, ids_file: Path) -> tuple[np.ndarray, dict[str, Any]]:
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    values = np.load(vectors, mmap_mode="r")
    identifiers = np.load(ids_file, mmap_mode="r")
    if (
        not receipt["complete"] or sha256(vectors) != receipt["vectors_sha256"]
        or sha256(ids_file) != receipt["ids_sha256"]
        or values.dtype != np.float16 or values.shape != (1798875, 1024)
        or not np.array_equal(identifiers, np.arange(len(values)))
        or receipt["representation"]["model"] != "Qwen/Qwen3-Embedding-0.6B"
        or receipt["representation"]["revision"] != "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    ):
        raise ValueError("Personalization must use the complete original source-bound Qwen0.6 item bank.")
    return values, receipt


def _bank_tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.array(values, copy=True), device=device)


def run_smoke(
    vectors: Path, item_receipt: Path, ids: Path, fitting: Path, output: Path,
) -> None:
    configure(17, 4)
    device = require_cuda()
    device_telemetry(device, require_exclusive=True)
    values, _ = load_item_bank(vectors, item_receipt, ids)
    prepared = study.checked_training_panel(fitting)
    panel = json.loads((fitting / "training-panel.json").read_text())
    study.validate_history_panel(panel, len(values), require_clock_bounds=True)
    if prepared["queries"] < 100:
        raise ValueError("Actual smoke requires 100 real admissible training queries.")
    smoke_binding = {
        "items_sha256": sha256(vectors), "fitting_sha256": prepared["panel_sha256"],
        "core_sha256": sha256(Path(study.__file__)), "runtime_sha256": sha256(Path(__file__)),
        "rows": 100, "seed": 17, "role": "correctness-only",
    }
    bind_run_config(output, smoke_binding)
    if (output / "completed.json").exists():
        prior = json.loads((output / "completed.json").read_text())
        validate_smoke(prior, smoke_binding["items_sha256"], smoke_binding["fitting_sha256"])
        if sha256(output / "smoke-state.pt") != prior["state_sha256"]:
            raise ValueError("Actual smoke state checksum changed.")
        return
    histories = study.padded_histories(panel["history_item_rows"][:100], len(values))
    selected = {
        **panel, "query_ids": panel["query_ids"][:100],
        "qrels": {query: panel["qrels"][query] for query in panel["query_ids"][:100]},
    }
    candidates, positive, _, _ = study.training_candidates(selected, len(values), seed=17)
    item_gpu = _bank_tensor(values, device)
    model = study.HistoryAttentionHead(1024).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss, count = train_one_epoch(model, item_gpu, histories, candidates, positive, optimizer, 100, 17, 0)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, output / "smoke-state.pt")
    state = torch.load(output / "smoke-state.pt", map_location=device, weights_only=True)
    restored = study.HistoryAttentionHead(1024).to(device)
    restored.load_state_dict(state["model"])
    row_tensor = torch.as_tensor(histories, device=device)
    with torch.inference_mode():
        torch.testing.assert_close(
            model(item_gpu[row_tensor.clamp_min(0)], row_tensor >= 0),
            restored(item_gpu[row_tensor.clamp_min(0)], row_tensor >= 0), rtol=0, atol=0,
        )
    write_json(output / "completed.json", {
        "complete": True, "study_role": "correctness-only", "rows": count, "loss": loss,
        "model_parameters": sum(value.numel() for value in model.parameters()),
        "item_vectors_sha256": sha256(vectors), "fitting_panel_sha256": prepared["panel_sha256"],
        "state_sha256": sha256(output / "smoke-state.pt"), "device": device_telemetry(device),
        "scope": "Actual forward/backward and immutable state reload; no quality or efficiency claim.",
    })
    print("ACTUAL_100_QUERY_FROZEN_BANK_USER_HEAD_SMOKE_COMPLETE", count, loss, flush=True)


def train(
    vectors: Path, item_receipt: Path, ids: Path, fitting: Path, smoke: Path,
    protocol_file: Path, output: Path, seed: int, admission: Path | None = None,
) -> None:
    configure(seed, 4)
    device = require_cuda()
    device_telemetry(device, require_exclusive=True)
    values, _ = load_item_bank(vectors, item_receipt, ids)
    prepared = study.checked_training_panel(fitting)
    panel = json.loads((fitting / "training-panel.json").read_text())
    study.validate_history_panel(panel, len(values), require_clock_bounds=True)
    protocol = json.loads(protocol_file.read_text())["personalization"]
    if seed not in protocol["seeds"] or protocol["maximum_epochs"] != 3:
        raise ValueError("Fitting seed or epoch budget differs from the frozen extension.")
    if seed == 23:
        if admission is None:
            raise ValueError("Seed23 fitting requires the fixed passing validation admission receipt.")
        decision = json.loads(admission.read_text())
        validate_admission(
            decision, sha256(vectors), sha256(protocol_file),
            json.loads(protocol_file.read_text())["authorization"]["first_pilot_gate_utc"],
        )
        validation_path = Path(decision["validation_receipt_path"])
        if sha256(validation_path) != decision["validation_receipt_sha256"]:
            raise ValueError("Persisted admission's validation receipt bytes changed.")
        checked_evaluation(validation_path.parent, {**decision["binding"], **evaluation_implementation()})
    elif admission is not None:
        raise ValueError("Seed17 fixed pilot does not use an admission receipt.")
    smoke_record = json.loads(smoke.read_text())
    validate_smoke(smoke_record, sha256(vectors), prepared["panel_sha256"])
    if sha256(smoke.parent / "smoke-state.pt") != smoke_record["state_sha256"]:
        raise ValueError("Successful model smoke payload checksum changed.")
    binding = {
        "protocol_sha256": sha256(protocol_file), "item_vectors_sha256": sha256(vectors),
        "fitting_panel_sha256": prepared["panel_sha256"], "smoke_sha256": sha256(smoke),
        "implementation_sha256": sha256(Path(study.__file__)), "runtime_sha256": sha256(Path(__file__)),
        "seed": seed, "epochs": 3, "dimension": 1024, "maximum_history": 12,
        "loss": protocol["loss"], "temperature": protocol["temperature"],
        "admission_sha256": sha256(admission) if admission is not None else None,
    }
    bind_run_config(output, binding)
    if (output / "completed.json").exists():
        checked_model(output, binding, len(panel["query_ids"]))
        return
    candidates, positive, _, audit = study.training_candidates(
        panel, len(values), seed, protocol["maximum_logged_negatives_per_query"],
        protocol["maximum_sampled_catalog_negatives_per_query"],
    )
    histories = study.padded_histories(panel["history_item_rows"], len(values))
    item_gpu = _bank_tensor(values, device)
    model = study.HistoryAttentionHead(1024).to(device)
    if sum(value.numel() for value in model.parameters()) >= 2_000_000:
        raise ValueError("User-head trainable parameter budget exceeded.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["learning_rate"], weight_decay=protocol["weight_decay"])
    losses, first_epoch, previous_seconds = [], 0, 0.0
    if (output / "training-state.json").exists():
        pointer = json.loads((output / "training-state.json").read_text())
        checkpoint = output / pointer["checkpoint"]
        if checkpoint.parent != output or sha256(checkpoint) != pointer["checkpoint_sha256"]:
            raise ValueError("User-head optimizer checkpoint pointer changed.")
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if state["binding"] != binding:
            raise ValueError("Resumed user-head protocol/source binding changed.")
        validate_mf_resume(state, 3)
        if pointer["completed_epochs"] != state["epochs"]:
            raise ValueError("User-head resume epoch pointer and state disagree.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        losses, first_epoch, previous_seconds = state["losses"], state["epochs"], state["elapsed_seconds"]
    started = time.perf_counter()
    for epoch in range(first_epoch, 3):
        loss, count = train_one_epoch(
            model, item_gpu, histories, candidates, positive, optimizer,
            protocol["batch_size"], seed, epoch, protocol["temperature"],
        )
        if count != len(panel["query_ids"]):
            raise RuntimeError("User-head fitting omitted prepared queries.")
        losses.append(loss)
        publish_mf_epoch(output, {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "binding": binding,
            "epochs": epoch + 1, "losses": losses,
            "elapsed_seconds": previous_seconds + time.perf_counter() - started,
        })
        print("CONTROLLED_USER_HEAD_EPOCH", seed, epoch + 1, count, loss, flush=True)
    model.eval()
    torch.save({"model": model.state_dict(), "binding": binding}, output / "model.partial.pt")
    (output / "model.partial.pt").replace(output / "model.pt")
    write_json(output / "completed.json", {
        "complete": True, "model_sha256": sha256(output / "model.pt"), "binding": binding,
        "epochs": 3, "losses": losses, "fitting_queries": len(histories), "negative_sources": audit,
        "parameters": sum(value.numel() for value in model.parameters()),
        "training_seconds": previous_seconds + time.perf_counter() - started,
        "training_scope": "Head fitting segments including batch feature fetch; frozen bank load and negative preparation excluded.",
        "environment": environment_manifest(), "device": device_telemetry(device),
    })


@torch.inference_mode()
def encode_histories(
    item_gpu: torch.Tensor, histories: list[list[int]], model: study.HistoryAttentionHead | None,
) -> torch.Tensor:
    rows = study.padded_histories(histories, len(item_gpu))
    outputs = []
    for start in range(0, len(rows), 256):
        selected = torch.as_tensor(rows[start:start + 256], device=item_gpu.device)
        features, mask = item_gpu[selected.clamp_min(0)], selected >= 0
        values = study.mean_history(features, mask) if model is None else model(features, mask)
        outputs.append(values.to(torch.float16).float())
    return torch.cat(outputs)


def _timing(item_gpu: torch.Tensor, panel: dict[str, Any], model: study.HistoryAttentionHead) -> dict[str, Any]:
    result = {}
    for batch in (1, 16):
        order = sorted(
            range(len(panel["query_ids"])),
            key=lambda row: stable_rank("extension-user-latency-" + str(batch), 20261011, panel["query_ids"][row]),
        )[:60 * batch]
        if len(order) != 60 * batch:
            raise ValueError("User-head timing panel lacks enough distinct query rows.")
        samples = []
        for iteration in range(60):
            selected = [panel["history_item_rows"][row] for row in order[iteration * batch:(iteration + 1) * batch]]
            torch.cuda.synchronize()
            started = time.perf_counter()
            encode_histories(item_gpu, selected, model)
            torch.cuda.synchronize()
            if iteration >= 10:
                samples.append((time.perf_counter() - started) * 1000)
        result[str(batch)] = {
            "samples_ms": samples, "sample_count": 50, "latency_mean_ms": float(np.mean(samples)),
            "scope": "History ID transfer, frozen-bank gathering, head forward, normalization and canonical query rounding; not full retrieval.",
        }
    return result


def evaluate(
    vectors: Path, item_receipt: Path, ids: Path, panel_file: Path, raw_queries: Path,
    raw_receipt: Path, popularity_file: Path, popularity_receipt: Path, checkpoint: Path,
    protocol_file: Path, output: Path, phase: str, history_metadata_file: Path,
    coverage_receipt_file: Path,
) -> None:
    configure(17, 4)
    device = require_cuda()
    device_telemetry(device, require_exclusive=True)
    values, bank_receipt = load_item_bank(vectors, item_receipt, ids)
    panel = json.loads(panel_file.read_text())
    validate_evaluation_phase(panel, phase)
    study.validate_history_panel(panel, len(values))
    history_diagnostic = json.loads(history_metadata_file.read_text())
    if (
        not history_diagnostic["complete"] or history_diagnostic["panel_sha256"] != sha256(panel_file)
        or history_diagnostic["query_ids"] != panel["query_ids"]
    ):
        raise ValueError("Held-out frozen-history diagnostic identities or source binding differ.")
    study.validate_history_panel({
        **panel, "history_upper_bounds": history_diagnostic["history_upper_bounds"],
    }, len(values), require_clock_bounds=True)
    raw = checked_raw_queries(raw_queries, raw_receipt, panel_file, bank_receipt)
    pop_receipt = json.loads(popularity_receipt.read_text())
    popularity = np.load(popularity_file, mmap_mode="r")
    if (
        sha256(popularity_file) != pop_receipt["positive_popularity_sha256"]
        or popularity.shape != (len(values),) or (popularity < 0).any()
        or panel["catalog_sha256"] != pop_receipt["input_contract"]["catalog_sha256"]
        or panel["catalog_ids_sha256"] != pop_receipt["input_contract"]["catalog_ids_sha256"]
    ):
        raise ValueError("Popularity/control vectors use a different frozen catalog.")
    coverage_receipt = json.loads(coverage_receipt_file.read_text())
    if (
        coverage_receipt["catalog_sha256"] != panel["catalog_sha256"]
        or coverage_receipt["ids_sha256"] != panel["catalog_ids_sha256"]
        or coverage_receipt["rows"] != len(values) or "coverage_counts" not in coverage_receipt
    ):
        raise ValueError("Numeric coverage receipt and the frozen candidate catalog disagree.")
    completed_model = json.loads((checkpoint.parent / "completed.json").read_text())
    if not completed_model["complete"] or sha256(checkpoint) != completed_model["model_sha256"]:
        raise ValueError("Evaluation requires a complete immutable user-head checkpoint.")
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if state["binding"]["item_vectors_sha256"] != sha256(vectors) or state["binding"]["protocol_sha256"] != sha256(protocol_file):
        raise ValueError("Evaluation head and item/protocol bindings differ.")
    checked_model(checkpoint.parent, state["binding"], completed_model["fitting_queries"])
    binding = {
        "panel_sha256": sha256(panel_file), "raw_queries_sha256": sha256(raw_queries),
        "checkpoint_sha256": sha256(checkpoint), "items_sha256": sha256(vectors),
        "popularity_sha256": sha256(popularity_file), "protocol_sha256": sha256(protocol_file), "phase": phase,
        "history_metadata_sha256": sha256(history_metadata_file),
        "coverage_receipt_sha256": sha256(coverage_receipt_file),
        "admission_sha256": state["binding"].get("admission_sha256"),
        **evaluation_implementation(),
        "evaluation_status": "Post-test exploratory, unchanged conditional population; no positive injection.",
    }
    bind_run_config(output, binding)
    if (output / "completed.json").exists():
        checked_evaluation(output, binding)
        return
    item_gpu = _bank_tensor(values, device)
    model = study.HistoryAttentionHead(1024).to(device).eval()
    model.load_state_dict(state["model"])
    users = np.asarray(panel["users"])
    histories = panel["history_item_rows"]
    query_vectors = {
        "raw-qwen06": torch.as_tensor(np.array(raw), device=device).float(),
        "same-history-mean": encode_histories(item_gpu, histories, None),
        "learned": encode_histories(item_gpu, histories, model),
        "learned-wrong-user": encode_histories(item_gpu, study.wrong_user_histories(histories, users, 20261011), model),
        "learned-order-permuted": encode_histories(item_gpu, study.permute_history_order(histories, 20261011), model),
    }
    incremental_timing = _timing(item_gpu, panel, model)
    memory_samples = [device_telemetry(device)]
    full_item_float = item_gpu.float()
    memory_samples.append(device_telemetry(device))
    rankings, files = {}, {}
    for name, query in query_vectors.items():
        batches = []
        for start in range(0, len(query), 128):
            _, found = exact_topk(full_item_float, query[start:start + 128], 1000, chunk_size=65536)
            batches.append(found.cpu().numpy())
        rankings[name] = np.concatenate(batches)
        np.save(output / (name + "-queries.npy"), query.half().cpu().numpy())
        files[name + "-queries.npy"] = sha256(output / (name + "-queries.npy"))
    del full_item_float
    popularity_order = np.lexsort((np.arange(len(values)), -popularity))
    rankings["training-popularity"] = np.tile(popularity_order[:1000], (len(users), 1))
    rankings["recurrence-popularity"] = study.recurrence_popularity(histories, popularity_order, 1000)
    uniform = {
        user: np.random.default_rng(int.from_bytes(stable_rank("extension-uniform", 20261011, str(user))[0][:8], "little")).choice(len(values), 1000, replace=False)
        for user in np.unique(users)
    }
    rankings["uniform"] = np.stack([uniform[user] for user in users])
    quality, summaries, strata = {}, {}, {}
    for name, ranked in rankings.items():
        np.save(output / (name + "-ids.npy"), ranked.astype(np.int64))
        files[name + "-ids.npy"] = sha256(output / (name + "-ids.npy"))
        quality[name] = panel_metrics(panel, ranked)
        summaries[name] = study.equal_user_metrics(quality[name], users)
        strata[name] = study.target_strata(panel, ranked, popularity)
    protocol = json.loads(protocol_file.read_text())["personalization"]
    gate_values = {}
    if phase == "validation":
        gate_values = pilot_gate(
            quality["learned"], quality["learned-wrong-user"], quality["training-popularity"], users,
            minimum_hit_users=protocol["pilot_gate"]["minimum_hit_users"],
            minimum_separation=protocol["pilot_gate"]["minimum_equal_user_aligned_minus_wrong_user_recall1000"],
        )
    gate = phase_admission(phase, gate_values)
    observed_memory = device_telemetry(device)
    memory_samples.append(observed_memory)
    write_json(output / "completed.json", {
        "complete": True, "phase": phase, "binding": binding, "files": files, "quality": quality,
        "model_seed": state["binding"]["seed"], "model_epochs": completed_model["epochs"],
        "query_users": panel["users"],
        "user_summaries": summaries, "pilot_gate": gate, "incremental_user_encoding_timing": incremental_timing,
        "target_support_strata": strata,
        "history_age_ms_quantiles": np.quantile(history_diagnostic["history_age_ms"], [0, 0.25, 0.5, 0.75, 1]).tolist(),
        "memory": observed_memory, "coverage": {
            "counts": coverage_receipt["coverage_counts"],
            "catalog_items": len(values), "source_manifest_sha256": sha256(coverage_receipt_file),
            "clock_policy": coverage_receipt["clock_contract"],
            "conditional_panel_sha256": sha256(panel_file),
            "scope": "All-positive-event catalog audit alongside unchanged conditional queries; no multiplication of incompatible means.",
        },
        "device_memory_samples": memory_samples,
        "population": {"queries": len(users), "users": len(set(users)), "catalog_items": len(values)},
        "denominator": "All original conditional warm/history queries; no successful-user selection.",
        "memory_scope": "Observed post-evaluation device sample, not continuous peak or robustness guarantee.",
    })
    print("FULL_CATALOG_CONTROLLED_PERSONALIZATION_EVALUATED", phase, summaries["learned"], gate, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled past-only personalization on frozen semantic item vectors.")
    commands = parser.add_subparsers(dest="operation", required=True)
    for operation in ("smoke", "train", "evaluate"):
        command = commands.add_parser(operation)
        for name in ("items", "items-receipt", "ids", "output"):
            command.add_argument("--" + name, required=True)
        if operation in ("smoke", "train"):
            command.add_argument("--fitting", required=True)
        if operation == "train":
            command.add_argument("--smoke", required=True)
            command.add_argument("--protocol", required=True)
            command.add_argument("--seed", type=int, choices=[17, 23], required=True)
            command.add_argument("--admission")
        if operation == "evaluate":
            for name in ("panel", "raw-queries", "raw-receipt", "popularity", "popularity-receipt", "checkpoint", "protocol", "history-metadata", "coverage-receipt"):
                command.add_argument("--" + name, required=True)
            command.add_argument("--phase", choices=["validation", "test"], required=True)
    args = parser.parse_args()
    common = Path(args.items), Path(args.items_receipt), Path(args.ids)
    if args.operation == "smoke":
        run_smoke(*common, Path(args.fitting), Path(args.output))
    elif args.operation == "train":
        train(
            *common, Path(args.fitting), Path(args.smoke), Path(args.protocol), Path(args.output), args.seed,
            Path(args.admission) if args.admission else None,
        )
    else:
        evaluate(
            *common, Path(args.panel), Path(args.raw_queries), Path(args.raw_receipt),
            Path(args.popularity), Path(args.popularity_receipt), Path(args.checkpoint),
            Path(args.protocol), Path(args.output), args.phase, Path(args.history_metadata), Path(args.coverage_receipt),
        )


if __name__ == "__main__":
    main()
