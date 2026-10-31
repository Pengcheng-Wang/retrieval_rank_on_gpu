"""Separate, frozen Figure 8 temperature queue over the baseline full tokenizer."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from fresh_ids import file_sha256, read_json, write_json
from fresh_pipeline import Runtime, cell_is_banked


TEMPERATURES = (("0p6", 0.6), ("0p8", 0.8), ("0p9", 0.9), ("1p1", 1.1), ("1p2", 1.2))
SWEEP_TEMPERATURES = (
    ("0p6", 0.6), ("0p7", 0.7), ("0p8", 0.8), ("0p9", 0.9),
    ("1p0", 1.0), ("1p1", 1.1), ("1p2", 1.2),
)
TOKENIZER_ARMS = (
    "alpha-0p001", "alpha-0p01", "alpha-0p1",
    "beta-0p00001", "beta-0p001", "beta-0p01",
    "clusters-5", "clusters-15", "clusters-20",
    "levels-2", "levels-6", "levels-8",
    "codes-64", "codes-128", "codes-512",
)
TEMPERATURE_CONTROL_FILES = (
    "author.py",
    "bank.py",
    "bank_watch.py",
    "compare.py",
    "evaluate.py",
    "figure8_temperature.py",
    "figure8_temperature_admission.py",
    "figure8_temperature_compare.py",
    "figure8_temperature_preflight.py",
    "figure8_temperature_probe.py",
    "fresh_ids.py",
    "fresh_pipeline.py",
    "fresh_rq_admission.py",
    "prepare.py",
    "protocol.py",
    "run.py",
    "runtime_patch.py",
)


def control_source_hashes(scripts: Path) -> dict:
    return {
        name: file_sha256(scripts / name)
        for name in TEMPERATURE_CONTROL_FILES
    }


def verify_durable_tree(path: Path) -> dict:
    marker = path / "bank-receipt.json"
    if path.is_symlink() or not marker.is_file() or marker.is_symlink():
        raise ValueError("Durable artifact receipt is missing or redirected")
    receipt = read_json(marker)
    files = receipt.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Durable artifact receipt has no file inventory")
    actual = set()
    for candidate in path.rglob("*"):
        if candidate.is_symlink():
            raise ValueError("Durable artifact contains a redirected path")
        if candidate.is_file() and candidate != marker:
            actual.add(candidate.relative_to(path).as_posix())
    if actual != set(files):
        raise ValueError("Durable artifact file inventory differs from its receipt")
    for name, metadata in files.items():
        target = path / name
        if (not target.resolve().is_relative_to(path.resolve())
                or target.stat().st_size != metadata.get("bytes")
                or file_sha256(target) != metadata.get("sha256")):
            raise ValueError(f"Durable artifact checksum differs: {name}")
    return {
        "bank_receipt_sha256": file_sha256(marker),
        "files": len(files),
    }


def verify_tokenizer_barrier(
    audits: dict[str, Path], durable_root: Path, spec_root: Path,
) -> dict:
    if set(audits) != set(TOKENIZER_ARMS):
        raise ValueError("Figure8 tokenizer barrier requires all exact one-factor arms")
    verified = {}
    for arm in TOKENIZER_ARMS:
        audit_path = audits[arm]
        if audit_path.is_symlink():
            raise ValueError("Figure8 tokenizer audit may not be redirected")
        audit = read_json(audit_path)
        bundle_name = audit.get("bundle", "")
        if (audit.get("status") != "pass" or audit.get("arm") != arm
                or audit.get("final_epoch") != 20000
                or type(audit.get("selected_epoch")) is not int
                or not 1 <= audit["selected_epoch"] <= 20000
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", bundle_name)
                or bundle_name in (".", "..")):
            raise ValueError(f"Figure8 tokenizer final audit differs: {arm}")
        target = durable_root / bundle_name
        durable = verify_durable_tree(target)
        spec_path = spec_root / f"{arm}.json"
        if spec_path.is_symlink():
            raise ValueError("Figure8 tokenizer specification may not be redirected")
        spec = read_json(spec_path)
        config = read_json(target / "payload" / "run-config.json")
        spec_fields = (
            "dataset", "codes", "epsilons", "dimension", "layers", "clusters",
            "alpha", "beta", "seed", "epochs", "batch_size", "eval_step",
        )
        if any(config.get(field) != spec.get(field) for field in spec_fields):
            raise ValueError(f"Figure8 tokenizer one-factor specification differs: {arm}")
        required = {
            "bundle_completed_sha256": target / "completed.json",
            "run_completed_sha256": target / "payload" / "completed.json",
            "index_sha256": target / "payload" / "export" / "index.json",
            "codes_sha256": target / "payload" / "export" / "codes.npy",
        }
        if (durable["bank_receipt_sha256"] != audit.get("durable_bank_receipt_sha256")
                or any(file_sha256(path) != audit.get(field)
                       for field, path in required.items())):
            raise ValueError(f"Figure8 tokenizer final audit checksum differs: {arm}")
        verified[arm] = {
            "status": "verified_durable_final",
            "bundle": bundle_name,
            "selected_epoch": audit["selected_epoch"],
            "final_epoch": 20000,
            **{field: audit[field] for field in required},
            "bank_receipt_sha256": durable["bank_receipt_sha256"],
            "audit_sha256": file_sha256(audit_path),
            "spec_sha256": file_sha256(spec_path),
            "run_config_sha256": file_sha256(target / "payload" / "run-config.json"),
            "durable_files": durable["files"],
        }
    return verified


def verify_fresh_queue_artifacts(
    work: Path, reference_binding: Path, expected_queue: list[dict], durable_root: Path,
) -> dict:
    if (len(expected_queue) != 39
            or len({cell.get("run_id") for cell in expected_queue}) != 39):
        raise ValueError("Fresh queue barrier requires the exact 39 unique cells")
    if (reference_binding.is_symlink() or reference_binding.parent.resolve() != work.resolve()):
        raise ValueError("Fresh queue binding must be an immutable work-local file")
    binding = read_json(reference_binding)
    driver = read_json(work / "driver.json")
    status = read_json(work / "status.json")
    if (binding.get("queue") != expected_queue
            or Path(binding.get("work", "")).resolve() != work.resolve()
            or Path(binding.get("durable_root", "")).resolve() != durable_root.resolve()
            or driver.get("queue_binding_file") != reference_binding.name
            or driver.get("queue_binding_sha256") != file_sha256(reference_binding)
            or status.get("status") != "all_39_cells_durably_complete"):
        raise ValueError("Fresh queue binding or completion status differs")
    acknowledgments = {}
    for cell in expected_queue:
        if not cell_is_banked(work, cell):
            raise ValueError("Fresh queue is not durably complete")
        run_id = cell["run_id"]
        ack_path = work / "acks" / f"{run_id}.json"
        result = work / "results" / run_id / "result.json"
        durable_result = durable_root / "results" / run_id
        verified = verify_durable_tree(durable_result)
        if (file_sha256(durable_result / "result.json") != file_sha256(result)
                or verified["bank_receipt_sha256"]
                != file_sha256(durable_result / "bank-receipt.json")):
            raise ValueError("Fresh queue durable result checksum differs")
        acknowledgments[run_id] = file_sha256(ack_path)
    return acknowledgments


def verify_baseline(
    plan: dict, bundle: Path, rq: dict, durable: dict,
    baseline_results: dict[str, Path],
) -> dict:
    expected = plan["baseline"]
    rq_expected = {
        "rq_run_id": plan["baseline_rq_run_id"],
        "bundle_completed_sha256": expected["bundle_completed_sha256"],
        "run_completed_sha256": expected["run_completed_sha256"],
        "final_epoch": 20000,
        "final_state_sha256": expected["final_state_sha256"],
        "selected_epoch": expected["selected_epoch"],
        "selected_state_sha256": expected["selected_state_sha256"],
        "index_sha256": expected["index_sha256"],
        "codes_sha256": expected["codes_sha256"],
    }
    durable_expected = {
        "bank_receipt_sha256": expected["durable_bank_receipt_sha256"],
        "bundle_completed_sha256": expected["bundle_completed_sha256"],
        "run_completed_sha256": expected["run_completed_sha256"],
    }
    if (bundle.name != expected["bundle_name"] or bundle.is_symlink()
            or file_sha256(bundle / "completed.json") != expected["bundle_completed_sha256"]
            or any(rq.get(key) != value for key, value in rq_expected.items())
            or any(durable.get(key) != value for key, value in durable_expected.items())
            or set(baseline_results) != set(plan["baseline_results"])
            or any(
                path.is_symlink() or file_sha256(path) != plan["baseline_results"][field]
                for field, path in baseline_results.items()
            )):
        raise ValueError("Figure8 baseline RQ, durability or result identity differs")
    return expected


def verify_admitted_preflight(work: Path, admission: dict) -> None:
    preflight = admission["temperature_preflight"]
    stage = work / "stages" / "Instruments-full-ids-v1" / "stage.json"
    audit = work / "audits" / "Instruments-full-ids-v1" / "audit.json"
    if (file_sha256(stage) != preflight["stage_sha256"]
            or file_sha256(audit) != preflight["audit_sha256"]):
        raise ValueError("Figure8 admitted preflight stage/audit bytes differ")


def verify_admission_sources(scripts: Path, admission: dict) -> None:
    if admission.get("source_sha256") != control_source_hashes(scripts):
        raise ValueError("Figure8 admission control source bytes differ")


def validate_preflight(receipt: dict) -> dict:
    labels = [label for label, _ in SWEEP_TEMPERATURES]
    temperatures = [temperature for _, temperature in SWEEP_TEMPERATURES]
    if (receipt.get("status") != "pass"
            or receipt.get("kind") != "figure8_temperature_preflight_v1"
            or receipt.get("temperatures") != temperatures
            or receipt.get("world_size") != 2
            or type(receipt.get("batches_per_rank")) is not int
            or receipt["batches_per_rank"] < 1
            or not re.fullmatch(r"[a-f0-9]{64}", receipt.get("stage_sha256", ""))
            or not re.fullmatch(r"[a-f0-9]{64}", receipt.get("audit_sha256", ""))
            or set(receipt.get("probes", {})) != set(labels)):
        raise ValueError("Figure8 temperature preflight identity differs")
    reference = None
    for label, temperature in SWEEP_TEMPERATURES:
        probe = receipt["probes"][label]
        ranks = probe.get("ranks", {})
        if probe.get("temperature") != temperature or set(ranks) != {"0", "1"}:
            raise ValueError("Figure8 temperature preflight rank coverage differs")
        normalized = {}
        for rank in ("0", "1"):
            model = ranks[rank].get("initial_model_sha256", "")
            batches = ranks[rank].get("ordered_batch_sha256")
            if (not re.fullmatch(r"[a-f0-9]{64}", model)
                    or not isinstance(batches, list)
                    or len(batches) != receipt["batches_per_rank"]
                    or any(not re.fullmatch(r"[a-f0-9]{64}", value) for value in batches)):
                raise ValueError("Figure8 temperature preflight hashes differ")
            normalized[rank] = {
                "initial_model_sha256": model,
                "ordered_batch_sha256": batches,
            }
        if (normalized["0"]["initial_model_sha256"]
                != normalized["1"]["initial_model_sha256"]):
            raise ValueError("Figure8 DDP ranks do not share the same initial model")
        if reference is None:
            reference = normalized
        else:
            for rank in ("0", "1"):
                if (normalized[rank]["initial_model_sha256"]
                        != reference[rank]["initial_model_sha256"]):
                    raise ValueError("Figure8 temperatures do not share the same initial model")
                if (normalized[rank]["ordered_batch_sha256"]
                        != reference[rank]["ordered_batch_sha256"]):
                    raise ValueError("Figure8 temperatures do not share the same ordered batch order")
    return receipt


def assemble_preflight(
    probe_root: Path, stage_receipt: Path, audit_receipt: Path, batches_per_rank: int,
) -> dict:
    stage_sha256 = file_sha256(stage_receipt)
    audit_sha256 = file_sha256(audit_receipt)
    probes = {}
    for label, temperature in SWEEP_TEMPERATURES:
        ranks = {}
        for rank in range(2):
            path = probe_root / label / f"rank-{rank}.json"
            if path.is_symlink():
                raise ValueError("Figure8 temperature probe may not be redirected")
            probe = read_json(path)
            if (probe.get("status") != "pass"
                    or probe.get("kind") != "figure8_temperature_probe_v1"
                    or probe.get("temperature") != temperature
                    or probe.get("rank") != rank
                    or probe.get("world_size") != 2
                    or probe.get("stage_sha256") != stage_sha256
                    or probe.get("audit_sha256") != audit_sha256
                    or len(probe.get("ordered_batch_sha256", [])) != batches_per_rank):
                raise ValueError("Figure8 temperature probe identity differs")
            ranks[str(rank)] = {
                "initial_model_sha256": probe["initial_model_sha256"],
                "ordered_batch_sha256": probe["ordered_batch_sha256"],
                "probe_sha256": file_sha256(path),
            }
        probes[label] = {"temperature": temperature, "ranks": ranks}
    return validate_preflight({
        "status": "pass",
        "kind": "figure8_temperature_preflight_v1",
        "temperatures": [temperature for _, temperature in SWEEP_TEMPERATURES],
        "world_size": 2,
        "batches_per_rank": batches_per_rank,
        "stage_sha256": stage_sha256,
        "audit_sha256": audit_sha256,
        "probes": probes,
    })


def aggregate_sweep(arms: dict[str, Path], seed: int, replicates: int) -> dict:
    from compare import compare_arms, load_arm

    labels = [label for label, _ in SWEEP_TEMPERATURES]
    if set(arms) != set(labels):
        raise ValueError("Figure8 aggregation requires the exact seven-point temperature grid")
    loaded = {}
    points = []
    for label, temperature in SWEEP_TEMPERATURES:
        completed, config, rows = load_arm(arms[label])
        if config["source_run"]["temperature"] != temperature:
            raise ValueError("Figure8 arm path is bound to a different temperature")
        loaded[label] = (completed, config, rows)
        points.append({
            "label": label,
            "temperature": temperature,
            "users": completed["users"],
            "predictions_sha256": completed["predictions_sha256"],
            "code": completed["metrics"]["code"],
            "item": completed["metrics"]["item"],
        })
    contrasts = {
        label: compare_arms(arms[label], arms["1p0"], seed, replicates)
        for label, _ in SWEEP_TEMPERATURES if label != "1p0"
    }
    reference_run = loaded["1p0"][1]["source_run"]
    return {
        "status": "pass",
        "kind": "figure8_temperature_sweep_v1",
        "dataset": "Instruments",
        "points": points,
        "reference_temperature": 1.0,
        "contrasts": contrasts,
        "experiment_lineage": reference_run["experiment_lineage"],
        "bootstrap_seed": seed,
        "bootstrap_replicates": replicates,
        "intervals": "paired-user percentile bootstrap; marginal and unadjusted",
        "selection": "descriptive only; no test-based winner",
    }


def temperature_cells() -> list[dict]:
    return [
        {
            "dataset": "Instruments", "variant": "full", "seed": 42,
            "temperature": temperature,
            "rq_run_id": "openllama-v2-rq-Instruments-full-s42",
            "stage_id": "Instruments-full-ids-v1",
            "run_id": f"openllama-v2-figure8-t5-Instruments-full-s42-t{label}-v1",
        }
        for label, temperature in TEMPERATURES
    ]


def parse_gpu_pair(value: str) -> tuple[int, int]:
    if not re.fullmatch(r"\d+,\d+", value):
        raise ValueError("GPU pair must contain two comma-separated physical indexes")
    indexes = tuple(map(int, value.split(",")))
    if len(set(indexes)) != 2:
        raise ValueError("GPU pair must contain two distinct physical indexes")
    return indexes


def bind_temperature_queue(work: Path, binding: dict) -> Path:
    marker = work / "figure8-temperature-binding.json"
    if marker.exists() and read_json(marker) != binding:
        raise ValueError("Figure8 temperature queue is frozen to different bytes")
    if not marker.exists():
        write_json(marker, binding)
    return marker


def execute_next(runtime: Runtime, bundle: Path, cells: list[dict]) -> dict | None:
    pending = [cell for cell in cells if not cell_is_banked(runtime.work, cell)]
    if not pending:
        return None
    cell = pending[0]
    runtime.execute_cell(cell, bundle)
    return cell


def load_plan(path: Path) -> dict:
    plan = read_json(path)
    expected = {
        "kind": "letter_figure8_temperature_v1",
        "dataset": "Instruments",
        "variant": "full",
        "seed": 42,
        "temperatures": [temperature for _, temperature in TEMPERATURES],
        "baseline_temperatures": [0.7, 1.0],
        "learning_rate": 0.0005,
        "epochs": 200,
        "world_size": 2,
        "effective_batch_size": 1024,
        "beams": 20,
        "bootstrap_seed": 20261027,
        "bootstrap_replicates": 2000,
        "seen_item_exclusion": False,
        "selection": (
            "within-arm minimum validation loss with patience20; "
            "never compare validation loss across temperatures"
        ),
        "comparison_reference": 1.0,
        "comparison_metrics": ["recall@5", "recall@10", "ndcg@5", "ndcg@10"],
        "metric_identities": ["code", "item"],
        "test_selection": "descriptive only; no best-temperature field or recipe selection",
        "preflight_temperatures": [
            temperature for _, temperature in SWEEP_TEMPERATURES
        ],
        "preflight_batches_per_rank": 3,
        "require_completed_inactive_fresh_queue": True,
        "require_distinct_gpu_lease": True,
        "tokenizer_barrier": list(TOKENIZER_ARMS),
        "baseline_rq_run_id": "openllama-v2-rq-Instruments-full-s42",
        "baseline": {
            "bundle_name":
                "openllama-v2-rq-Instruments-full-s42-completed-8a999c191f8ec7fd",
            "bundle_completed_sha256":
                "621fff5a7e7f42e425b3ca45b4f61fcd23c808d439a1054002a842422cd6579e",
            "run_completed_sha256":
                "c88a19484d073ab5e119c0e6d4062014c75c8755055c0b9bb885e1ba40d03af9",
            "final_state_sha256":
                "796d484fab903b239e02e2c90a2323073408e6f875ba9178c76cb78329ded771",
            "selected_epoch": 18000,
            "selected_state_sha256":
                "59a2dcc4f61c6c9fd15d96d946db33fa590a9689af55e7b08400d48eb2d001f9",
            "index_sha256":
                "44048e87910d0367ebe1733e1fcec6d833f2d51b02241fc3edc5800194e82a74",
            "codes_sha256":
                "00444111176fd11997ab4c6450dffd417360dec501506b0a9133fb104a2c2688",
            "durable_bank_receipt_sha256":
                "8a6e3ebbc42bd602ab7c114b54598f10f35d946280cedb01e9bf7b3cd064f08f",
        },
        "baseline_results": {
            "temperature_0p7_result_sha256":
                "2c7cfc6ba34a28f00cd8f154b3de6686124e60277841dc0736886afd78ba61a8",
            "temperature_1p0_result_sha256":
                "b76344c10082fbe6bdd3a0ac11390942bee387e4edf38e4ed9e36478de1abafe",
        },
        "reference_fresh_queue_cells": 39,
        "work_basename": "fresh-t5-figure8-temperature-v1",
        "durability":
            "separate node0-only work/durable namespace under the common bank lock",
    }
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError("Figure8 temperature plan differs from the frozen scientific grid")
    return plan


def validate_admission(path: Path, expected_sha256: str, plan: dict, bundle: Path) -> dict:
    if not re.fullmatch(r"[a-f0-9]{64}", expected_sha256):
        raise ValueError("Figure8 admission requires an external digest")
    if file_sha256(path) != expected_sha256:
        raise ValueError("Figure8 admission is not externally authorized")
    admission = read_json(path)
    tokenizer_arms = admission.get("tokenizer_arms")
    result_acks = admission.get("fresh_queue_result_acks")
    preflight = admission.get("temperature_preflight")
    sources = admission.get("source_sha256")
    barriers_valid = (
        isinstance(tokenizer_arms, dict)
        and set(tokenizer_arms) == set(TOKENIZER_ARMS)
        and all(
            value.get("status") == "verified_durable_final"
            and value.get("final_epoch") == 20000
            and type(value.get("selected_epoch")) is int
            and 1 <= value["selected_epoch"] <= 20000
            and re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]*", value.get("bundle", ""),
            )
            and value.get("bundle") not in (".", "..")
            and all(
                re.fullmatch(r"[a-f0-9]{64}", value.get(field, ""))
                for field in (
                    "bundle_completed_sha256", "bank_receipt_sha256",
                    "run_completed_sha256", "index_sha256", "codes_sha256",
                    "audit_sha256", "spec_sha256", "run_config_sha256",
                )
            )
            for value in tokenizer_arms.values()
        )
    )
    queue_complete = (
        admission.get("fresh_queue_durably_complete") is True
        and admission.get("fresh_queue_driver_inactive") is True
        and isinstance(result_acks, dict) and len(result_acks) == 39
        and all(
            isinstance(name, str) and re.fullmatch(r"[a-f0-9]{64}", digest)
            for name, digest in result_acks.items()
        )
    )
    preflight_valid = (
        isinstance(preflight, dict)
        and preflight.get("temperatures")
        == [temperature for _, temperature in SWEEP_TEMPERATURES]
        and preflight.get("batches_per_rank") == plan["preflight_batches_per_rank"]
        and preflight.get("initial_model_identical") is True
        and preflight.get("ordered_batches_identical") is True
        and all(
            re.fullmatch(r"[a-f0-9]{64}", preflight.get(name, ""))
            for name in (
                "receipt_sha256", "durable_bank_receipt_sha256",
                "stage_sha256", "audit_sha256",
            )
        )
    )
    sources_valid = (
        isinstance(sources, dict)
        and set(sources) == set(TEMPERATURE_CONTROL_FILES)
        and all(re.fullmatch(r"[a-f0-9]{64}", value) for value in sources.values())
    )
    if (admission.get("status") != "pass"
            or admission.get("kind") != "figure8_temperature_admission_v1"
            or admission.get("temperatures") != plan["temperatures"]
            or admission.get("baseline_bundle") != bundle.name
            or admission.get("baseline_bundle_completed_sha256")
            != file_sha256(bundle / "completed.json")
            or admission.get("baseline") != plan["baseline"]
            or admission.get("baseline_results") != plan["baseline_results"]
            or admission.get("all_tokenizer_one_factor_arms_durable") is not True
            or not barriers_valid
            or admission.get("existing_fresh_queue_unchanged") is not True
            or admission.get("existing_fresh_queue_cells") != 39
            or not re.fullmatch(
                r"[a-f0-9]{64}", admission.get("existing_fresh_queue_binding_sha256", ""),
            )
            or not queue_complete
            or not preflight_valid
            or not sources_valid):
        raise ValueError("Figure8 temperature prerequisites are incomplete or changed")
    return admission


def queue_binding(
    runtime: Runtime, plan_path: Path, admission_path: Path,
    bundle: Path, reference_binding: Path,
) -> dict:
    return {
        "kind": "figure8_temperature_queue_v1",
        "plan_sha256": file_sha256(plan_path),
        "admission_sha256": file_sha256(admission_path),
        "baseline_bundle": bundle.name,
        "baseline_bundle_completed_sha256": file_sha256(bundle / "completed.json"),
        "reference_fresh_queue_binding_sha256": file_sha256(reference_binding),
        "work": str(runtime.work),
        "durable_root": str(runtime.bank_root),
        "control": str(runtime.control),
        "source_sha256": control_source_hashes(runtime.scripts),
        "gpu_indexes": list(runtime.gpu_indexes),
        "cells": temperature_cells(),
        "selection": "within-arm minimum validation loss only; no cross-temperature selection",
        "comparison": (
            "all six temperatures versus predeclared temperature1 reference; "
            "test descriptive only"
        ),
    }


def driver(
    runtime: Runtime, bundle: Path, plan_path: Path, admission_path: Path,
    admission_sha256: str, reference_binding: Path, checks: int,
) -> None:
    import fcntl

    plan = load_plan(plan_path)
    admission = validate_admission(admission_path, admission_sha256, plan, bundle)
    if runtime.work.name != plan["work_basename"]:
        raise ValueError("Figure8 temperature work namespace differs")
    verify_admission_sources(runtime.scripts, admission)
    verify_admitted_preflight(runtime.work, admission)
    if (admission.get("gpu_indexes") != list(runtime.gpu_indexes)
            or admission.get("fresh_queue_gpu_indexes") != [2, 3]
            or set(admission["gpu_indexes"]) & set(admission["fresh_queue_gpu_indexes"])
            or admission.get("same_allocation_distinct_gpu_lease") is not True):
        raise ValueError("Figure8 and fresh queues do not have distinct declared GPU leases")
    binding = queue_binding(runtime, plan_path, admission_path, bundle, reference_binding)
    with (runtime.work / "driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = bind_temperature_queue(runtime.work, binding)
        write_json(runtime.work / "driver.json", {
            "pid": os.getpid(), "started_unix": time.time(),
            "binding_sha256": file_sha256(marker), "checks": checks,
        })
        for check in range(checks):
            cell = execute_next(runtime, bundle, binding["cells"])
            if cell is None:
                write_json(runtime.work / "status.json", {
                    "status": "all_temperature_arms_durably_complete",
                })
                return
            write_json(runtime.work / "status.json", {
                "status": "completed_one_arm", "check": check + 1, "cell": cell,
            })


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "binding", "project", "control", "work", "bundle", "plan",
        "admission", "reference-binding",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--admission-sha256", required=True)
    parser.add_argument("--gpu-pair", required=True)
    parser.add_argument("--checks", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.checks <= len(TEMPERATURES):
        parser.error(f"Use 1..{len(TEMPERATURES)} checks")
    gpu_indexes = parse_gpu_pair(args.gpu_pair)
    runtime = Runtime(
        args.binding, args.project, args.control, args.work,
        gpu_indexes=gpu_indexes,
    )
    driver(
        runtime, args.bundle.resolve(), args.plan.resolve(), args.admission.resolve(),
        args.admission_sha256, args.reference_binding.resolve(), args.checks,
    )


if __name__ == "__main__":
    main()
