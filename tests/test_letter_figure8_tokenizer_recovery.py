"""Fail-closed recovery gates for the two Figure 8 tokenizer endpoint faults."""

from __future__ import annotations

import importlib
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "letter"))


def recovery():
    assert importlib.util.find_spec("figure8_tokenizer_recovery") is not None, (
        "Figure 8 tokenizer recovery must use a dedicated sidecar controller"
    )
    return importlib.import_module("figure8_tokenizer_recovery")


def runtime():
    assert importlib.util.find_spec("figure8_tokenizer_recovery_runtime") is not None, (
        "Figure 8 tokenizer recovery must use an independently audited runtime overlay"
    )
    return importlib.import_module("figure8_tokenizer_recovery_runtime")


@pytest.mark.parametrize(
    "arm,gpu,base_epoch,failure_epoch",
    [
        ("beta-0p001", 2, 16000, 16401),
        ("clusters-20", 7, 11000, 11431),
    ],
)
def test_recovery_scope_is_limited_to_the_two_observed_node1_faults(
    arm, gpu, base_epoch, failure_epoch,
):
    spec = {
        "host": "node-1",
        "gpu": gpu,
        "output": f"/scratch/letter_20261028_a77dfb13/figure8-sweeps-v1/runs/{arm}",
    }
    identity = recovery().validate_recovery_identity(spec, arm, "node-1", gpu)
    assert identity == {
        "arm": arm,
        "host": "node-1",
        "gpu": gpu,
        "base_epoch": base_epoch,
        "first_failed_epoch": failure_epoch,
    }
    for changed in (
        {**spec, "host": "node-2"},
        {**spec, "gpu": (gpu + 1) % 8},
        {**spec, "output": spec["output"] + "-other"},
    ):
        with pytest.raises(ValueError):
            recovery().validate_recovery_identity(changed, arm, "node-1", gpu)
    with pytest.raises(ValueError):
        recovery().validate_recovery_identity(spec, "alpha-0p001", "node-1", gpu)


def native_failure_log(arm: str, completed_epoch: int) -> str:
    block = (
        f'{{"epoch": {completed_epoch}, "training_seconds": 4.0}}\n'
        'joblib.externals.loky.process_executor._RemoteTraceback:\n'
        'IndexError: index 256 is out of bounds for axis 0 with size 256\n'
        'The above exception was the direct cause of the following exception:\n'
        'IndexError: index 256 is out of bounds for axis 0 with size 256\n'
    )
    return f"arm={arm}\n" + block * 5


@pytest.mark.parametrize(
    "arm,completed_epoch,failed_epoch",
    [
        ("beta-0p001", 16400, 16401),
        ("clusters-20", 11430, 11431),
    ],
)
def test_native_failure_admission_requires_five_identical_exact_endpoint_faults(
    arm, completed_epoch, failed_epoch,
):
    result = recovery().parse_native_failures(native_failure_log(arm, completed_epoch), arm)
    assert result == {
        "attempts": 5,
        "completed_epoch_before_fault": completed_epoch,
        "first_failed_epoch": failed_epoch,
        "candidate_index": 256,
        "sample_count": 256,
    }
    with pytest.raises(ValueError):
        recovery().parse_native_failures(
            native_failure_log(arm, completed_epoch).replace(
                "joblib.externals.loky.process_executor._RemoteTraceback:\n", "", 1,
            ),
            arm,
        )
    with pytest.raises(ValueError):
        recovery().parse_native_failures(
            native_failure_log(arm, completed_epoch).replace("index 256", "index 257"),
            arm,
        )


def operational_spec(arm: str = "beta-0p001") -> dict:
    root = "/scratch/letter_20261028_a77dfb13"
    gpu = 2 if arm == "beta-0p001" else 7
    return {
        "kind": "figure8-one-factor-tokenizer-v1",
        "root": root,
        "control": f"{root}/figure8-sweeps-v1/control",
        "output": f"{root}/figure8-sweeps-v1/runs/{arm}",
        "host": "node-1",
        "gpu": gpu,
        "dataset": "Instruments",
        "semantic": f"{root}/embeddings-openllama-v2-v1/Instruments/embeddings.npy",
        "cf": f"{root}/runs/Instruments-cf-bce-s42/item-embeddings.pt",
        "embedding_receipt":
            f"{root}/embeddings-openllama-v2-v1/Instruments/embedding-receipt.json",
        "manifest": f"{root}/figure8-sweeps-v1/control/configs/letter-reproduction-v1.json",
        "method_spec":
            f"{root}/figure8-sweeps-v1/control/configs/letter-openllama-v2-method-v1.json",
        "codes": [256, 256, 256, 256],
        "epsilons": [0.0, 0.0, 0.0, 0.003],
        "dimension": 32,
        "layers": [2048, 1024, 512, 256, 128, 64],
        "clusters": 20 if arm == "clusters-20" else 10,
        "alpha": 0.02,
        "beta": 0.001 if arm == "beta-0p001" else 0.0001,
        "seed": 42,
        "epochs": 20000,
        "batch_size": 1024,
        "eval_step": 2000,
        "checkpoint_every": 500,
    }


def test_sidecar_replays_the_exact_frozen_native_command_without_recipe_changes():
    spec = operational_spec()
    output = Path("/scratch/letter_20261028_a77dfb13/recovery-probes/beta/run")
    resume = Path(
        "/scratch/letter_20261028_a77dfb13/figure8-sweeps-v1/"
        "runs/beta-0p001/checkpoints/epoch-16000"
    )
    argv = recovery().native_arguments(spec, output, resume, 16401)
    assert argv[0].as_posix().endswith(
        "/figure8-sweeps-v1/control/scripts/letter/tokenizer_run.py"
    )
    value = lambda flag: argv[argv.index(flag) + 1]
    assert value("--output") == str(output)
    assert value("--resume") == str(resume)
    assert value("--alpha") == "0.02"
    assert value("--beta") == "0.001"
    assert value("--clusters") == "10"
    assert value("--seed") == "42"
    assert value("--epochs") == "20000"
    assert value("--stop-after-epoch") == "16401"
    assert argv[argv.index("--codes") + 1:argv.index("--epsilons")] == [
        "256", "256", "256", "256",
    ]


def test_probe_admission_binds_the_exact_native_state_and_control_hashes():
    digest = "a" * 64
    admission = recovery().probe_admission(
        identity={
            "arm": "beta-0p001", "host": "node-1", "gpu": 2,
            "base_epoch": 16000, "first_failed_epoch": 16401,
        },
        spec_sha256=digest,
        run_config_sha256="b" * 64,
        resume_completed_sha256="c" * 64,
        resume_state_sha256="d" * 64,
        overlay_receipt_sha256="e" * 64,
        native_module_sha256="f" * 64,
        failure_log_sha256="0" * 64,
        failure={"attempts": 5, "completed_epoch_before_fault": 16400,
                 "first_failed_epoch": 16401, "candidate_index": 256, "sample_count": 256},
        control_sha256={"figure8_tokenizer_recovery.py": "1" * 64},
    )
    assert admission["kind"] == "figure8_tokenizer_kmeans_probe_admission_v1"
    assert admission["resume"] == {
        "epoch": 16000,
        "completed_sha256": "c" * 64,
        "state_sha256": "d" * 64,
    }
    assert admission["policy"] == "rq-kmeans-cdf-endpoint-v1"
    broken = dict(admission)
    broken["resume"] = {**admission["resume"], "state_sha256": "not-a-hash"}
    with pytest.raises(ValueError, match="digest"):
        recovery().validate_probe_admission(broken)


def test_production_admission_requires_a_fault_crossing_probe_and_durable_receipt():
    probe = {
        "status": "pass",
        "kind": "figure8_tokenizer_kmeans_probe_result_v1",
        "arm": "beta-0p001",
        "admission_sha256": "a" * 64,
        "checkpoint": "/scratch/probe/checkpoints/epoch-16401",
        "checkpoint_state_sha256": "b" * 64,
        "attestation_sha256": "c" * 64,
        "runtime_summary": {
            "tasks": 16040,
            "failed_tasks": 0,
            "corrections": 1,
            "phases": {"training": 1},
            "worker_pids": [1, 2],
            "native_defined": 16039,
            "native_faults": 1,
        },
        "corrections_by_epoch_level_phase": {"16401": {"training": {"0": 1}}},
    }
    admission = recovery().production_admission(
        recovery().probe_admission(
            identity={
                "arm": "beta-0p001", "host": "node-1", "gpu": 2,
                "base_epoch": 16000, "first_failed_epoch": 16401,
            },
            spec_sha256="d" * 64,
            run_config_sha256="e" * 64,
            resume_completed_sha256="f" * 64,
            resume_state_sha256="1" * 64,
            overlay_receipt_sha256="2" * 64,
            native_module_sha256="3" * 64,
            failure_log_sha256="4" * 64,
            failure={"attempts": 5, "completed_epoch_before_fault": 16400,
                     "first_failed_epoch": 16401, "candidate_index": 256, "sample_count": 256},
            control_sha256={"figure8_tokenizer_recovery.py": "5" * 64},
        ),
        probe,
        probe_bank_receipt_sha256="6" * 64,
        probe_result_sha256="7" * 64,
        production_control_sha256={"figure8_tokenizer_recovery.py": "8" * 64},
    )
    assert admission["kind"] == "figure8_tokenizer_kmeans_production_admission_v1"
    assert admission["probe"]["native_faults"] == 1
    assert admission["probe"]["native_defined"] == 16039
    broken = json.loads(json.dumps(admission))
    broken["probe"]["native_faults"] = 0
    with pytest.raises(ValueError, match="fault"):
        recovery().validate_production_admission(broken)


def rng_state(seed: int):
    return np.random.RandomState(seed).get_state()


def test_defined_native_initialization_must_match_centers_and_rng_bitwise():
    centers = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert runtime().validate_differential(
        native_error=None,
        native_centers=centers,
        native_rng=rng_state(7),
        corrected_centers=centers.copy(),
        corrected_rng=rng_state(7),
        corrections=[],
        sample_count=256,
    ) == "native-defined-bitwise-equal"
    with pytest.raises(ValueError, match="centers"):
        runtime().validate_differential(
            native_error=None,
            native_centers=centers,
            native_rng=rng_state(7),
            corrected_centers=centers + 1,
            corrected_rng=rng_state(7),
            corrections=[],
            sample_count=256,
        )
    with pytest.raises(ValueError, match="RNG"):
        runtime().validate_differential(
            native_error=None,
            native_centers=centers,
            native_rng=rng_state(7),
            corrected_centers=centers,
            corrected_rng=rng_state(8),
            corrections=[],
            sample_count=256,
        )


def test_only_the_exact_native_endpoint_fault_can_be_compatibility_defined():
    correction = {
        "native_candidates": [10, 256],
        "corrected_candidates": [10, 255],
        "affected_trials": [1],
    }
    assert runtime().validate_differential(
        native_error=IndexError("index 256 is out of bounds for axis 0 with size 256"),
        native_centers=None,
        native_rng=rng_state(11),
        corrected_centers=np.ones((10, 32), dtype=np.float32),
        corrected_rng=rng_state(12),
        corrections=[correction],
        sample_count=256,
    ) == "native-endpoint-fault-corrected"
    for error in (
        RuntimeError("solver failed"),
        IndexError("index 257 is out of bounds for axis 0 with size 256"),
        IndexError("index 256 is out of bounds for axis 1 with size 256"),
    ):
        with pytest.raises((IndexError, RuntimeError, ValueError)):
            runtime().validate_differential(
                native_error=error,
                native_centers=None,
                native_rng=rng_state(11),
                corrected_centers=np.ones((10, 32), dtype=np.float32),
                corrected_rng=rng_state(12),
                corrections=[correction],
                sample_count=256,
            )
    with pytest.raises(ValueError, match="correction"):
        runtime().validate_differential(
            native_error=IndexError("index 256 is out of bounds for axis 0 with size 256"),
            native_centers=None,
            native_rng=rng_state(11),
            corrected_centers=np.ones((10, 32), dtype=np.float32),
            corrected_rng=rng_state(12),
            corrections=[],
            sample_count=256,
        )


def test_actual_differential_execution_uses_a_cloned_native_rng():
    values = np.arange(24, dtype=np.float32).reshape(6, 4)

    def implementation(X, _clusters, _norms, random_state, _trials):
        first = random_state.randint(len(X))
        random_state.random_sample(4)
        return X[[first, (first + 1) % len(X)]]

    actual = np.random.RandomState(19)
    expected = np.random.RandomState(19)
    result, evidence = runtime().execute_differential(
        implementation,
        implementation,
        values,
        2,
        np.square(values).sum(axis=1),
        actual,
        None,
        [],
    )
    np.testing.assert_array_equal(result, implementation(
        values, 2, np.square(values).sum(axis=1), expected, None,
    ))
    assert evidence["classification"] == "native-defined-bitwise-equal"
    assert runtime().same_rng(actual.get_state(), expected.get_state())


def test_actual_differential_execution_allows_only_a_recorded_endpoint_repair():
    values = np.arange(24, dtype=np.float32).reshape(6, 4)
    corrections = []

    def native(_X, _clusters, _norms, random_state, _trials):
        random_state.random_sample(4)
        raise IndexError("index 6 is out of bounds for axis 0 with size 6")

    def corrected(X, _clusters, _norms, random_state, _trials):
        random_state.random_sample(4)
        corrections.append({
            "native_candidates": [6],
            "corrected_candidates": [5],
            "affected_trials": [0],
        })
        return X[[5]]

    result, evidence = runtime().execute_differential(
        corrected,
        native,
        values,
        1,
        np.square(values).sum(axis=1),
        np.random.RandomState(23),
        None,
        corrections,
    )
    np.testing.assert_array_equal(result, values[[5]])
    assert evidence["classification"] == "native-endpoint-fault-corrected"
    assert evidence["native_error"] == "IndexError: index 6 is out of bounds for axis 0 with size 6"


def test_nonendpoint_native_failure_stops_before_the_compatibility_function_runs():
    called = []

    def native(*_args):
        raise RuntimeError("solver failed")

    def corrected(*_args):
        called.append(True)
        return np.zeros((1, 1), dtype=np.float32)

    with pytest.raises(RuntimeError, match="solver failed"):
        runtime().execute_differential(
            corrected,
            native,
            np.zeros((6, 4), dtype=np.float32),
            1,
            np.zeros(6, dtype=np.float32),
            np.random.RandomState(29),
            None,
            [],
        )
    assert called == []


def write_defined_event_archive(
    path: Path, invocation: str = "run-1", *, include_differential: bool = True,
) -> None:
    context = {
        "call_id": "call-1",
        "invocation": invocation,
        "expected_input_sha256": "input",
        "initialization_seeds": [31],
        "epoch": 16001,
        "phase": "training",
        "level": 0,
    }
    context_bytes = (json.dumps(context, indent=2) + "\n").encode()
    event = {
        "task_id": "task-1",
        "pid": 123,
        "policy": "rq-kmeans-cdf-endpoint-v1",
        "status": "success",
        "input_sha256": "input",
        "initial_seed": 31,
        "actual_init_centroids_binding": True,
        "context": context,
        "context_sha256": hashlib.sha256(context_bytes).hexdigest(),
        "corrections": [],
    }
    if include_differential:
        event["differential"] = {
            "classification": "native-defined-bitwise-equal",
            "centers_equal": True,
            "rng_equal": True,
            "native_error": None,
        }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("contexts/call-1.json", context_bytes)
        archive.writestr("tasks-123.jsonl", json.dumps(event) + "\n")


def test_event_audit_rejects_cross_invocation_or_unaudited_native_tasks(tmp_path):
    archive = tmp_path / "events.zip"
    write_defined_event_archive(archive)
    result = runtime().audit_event_archive(archive, "run-1")
    assert result["summary"]["tasks"] == 1
    assert result["summary"]["native_defined"] == 1
    assert result["summary"]["native_faults"] == 0
    with pytest.raises(ValueError, match="invocation"):
        runtime().audit_event_archive(archive, "run-2")

    broken = tmp_path / "broken.zip"
    write_defined_event_archive(broken, include_differential=False)
    with pytest.raises(ValueError, match="differential"):
        runtime().audit_event_archive(broken, "run-1")


def test_compatibility_attestation_parent_must_match_the_exact_resume_state():
    common = {
        "base_epoch": 16000,
        "base_state_sha256": "a" * 64,
        "parent_epoch": 16500,
        "parent_state_sha256": "b" * 64,
    }
    parent = {
        "artifact": "checkpoint",
        "epoch": 16500,
        "state_sha256": "b" * 64,
    }
    assert recovery().validate_parent_binding(common, parent) == "compatible-parent"
    with pytest.raises(ValueError, match="parent"):
        recovery().validate_parent_binding(
            common, {**parent, "state_sha256": "c" * 64},
        )
    base = {
        **common,
        "parent_epoch": 16000,
        "parent_state_sha256": "a" * 64,
    }
    assert recovery().validate_parent_binding(base, None) == "native-base"
    with pytest.raises(ValueError, match="base"):
        recovery().validate_parent_binding(common, None)


def test_repaired_arms_cannot_be_admitted_without_semantic_compatibility_evidence(tmp_path):
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "completed.json").write_text(
        json.dumps({"status": "complete"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="compatibility"):
        recovery().verify_completed_compatibility(
            payload, "beta-0p001", "a" * 64,
        )
    assert recovery().verify_completed_compatibility(
        payload, "alpha-0p001", None,
    ) is None


def test_figure8_arm_admission_requires_the_trusted_repair_digest(tmp_path):
    import figure8_tokenizer

    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "completed.json").write_text(
        json.dumps({"status": "complete"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="trusted"):
        figure8_tokenizer.verify_arm_compatibility(
            "beta-0p001", {}, payload,
        )
    assert figure8_tokenizer.verify_arm_compatibility(
        "alpha-0p001", {}, payload,
    ) is None


def test_overlay_builder_refuses_any_unpinned_native_module(tmp_path):
    source = tmp_path / "source"
    module = source / "sklearn_import" / "cluster" / "k_means_.py"
    module.parent.mkdir(parents=True)
    module.write_text("def _k_init(): return None\n", encoding="utf-8")
    destination = tmp_path / "overlay"
    with pytest.raises(ValueError, match="pinned"):
        runtime().make_overlay(source, destination)
    assert not destination.exists()
