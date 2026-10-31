import copy
import importlib
import json
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "letter"))


def recovery():
    assert importlib.util.find_spec("lc_recovery"), "Narrow LC timeout recovery is missing"
    return importlib.import_module("lc_recovery")


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def fixture(tmp_path):
    from lc_checkpoint import seal, snapshot
    from protocol import file_sha256
    run, source = tmp_path / "run", tmp_path / "source"
    source.mkdir()
    (source / "lc_run.py").write_text("frozen learner")
    (source / "lc_checkpoint.py").write_text("frozen checkpoint")
    launcher = tmp_path / "fit.sh"
    launcher.write_text("unchanged launcher")
    admission = tmp_path / "admission.json"
    dump(admission, {"status": "pass"})
    config = {
        "kind": "full-fit", "inputs": {
            "dataset": "Instruments",
            "harness_sha256": {p.name: file_sha256(p) for p in source.iterdir()},
            "verified_files_sha256": {},
        },
        "gate_report_sha256": file_sha256(admission),
    }
    dump(run / "run-config.json", config)
    path = run / "checkpoint-10"
    path.mkdir()
    for name in ("adapter_model.bin", "optimizer.pt", "scheduler.pt"):
        (path / name).write_bytes(b"fixture complete state")
    dump(path / "adapter_config.json", {})
    (path / "run-config.json").write_bytes((run / "run-config.json").read_bytes())
    dump(path / "trainer_state.json", {"global_step": 10, "epoch": 0.1})
    dump(path / "callback-state.json", {"validation_calls": []})
    for rank in range(4):
        (path / f"rng_state_{rank}.pth").write_bytes(b"rank rng")
        (path / f"rank-state-{rank}.pt").write_bytes(b"rank sampler")
    receipt = seal(path, 4)
    snap = run / "snapshots" / path.name
    snapshot(path, snap)
    bank = tmp_path / "durable"
    ack = {
        "status": "verified", "writer_host": "writer",
        "global_step": 10, "run_config_sha256": file_sha256(run / "run-config.json"),
        "adapter_sha256": receipt["files"]["adapter_model.bin"],
        "complete_sha256": file_sha256(snap / "complete.json"),
        "bank_directory": str(bank / path.name), "verified_unix": time.time(),
        "bank_helper_sha256": file_sha256(source / "lc_checkpoint.py"),
    }
    dump(run / "acks" / "checkpoint-10.json", ack)
    dump(run / "latest-state.json", {
        "global_step": 10, "checkpoint": "checkpoint-10",
        "run_config_sha256": ack["run_config_sha256"],
        "complete_sha256": ack["complete_sha256"], "adapter_sha256": ack["adapter_sha256"],
    })
    dump(run / "live.json", {"global_step": 10, "pid": 999})
    dump(run / "runtime.json", {"pid": 999, "started_unix": 50})
    (run / "supervisor.pid").write_text("997\n")
    (run / "torchrun.pid").write_text("998\n")
    error = {
        "status": "error", "error_type": "TimeoutError",
        "message": "No hash-bound durable acknowledgment for checkpoint-10", "unix": 100.0,
        "traceback": f'File "{source / "lc_run.py"}", line 633, in on_save\n'
                     '    wait_for_ack(sealed, args.output / "acks")\n'
                     f'File "{source / "lc_checkpoint.py"}", line 122, in wait_for_ack\n'
                     "TimeoutError: No hash-bound durable acknowledgment for checkpoint-10\n",
    }
    failure = run / "failed-rank-0-999.json"
    dump(failure, error)
    settings = {
        "run": str(run), "project": str(tmp_path), "frozen_source": str(source),
        "launcher": str(launcher), "admission": str(admission),
        "run_config_sha256": ack["run_config_sha256"],
        "admission_sha256": file_sha256(admission), "launcher_sha256": file_sha256(launcher),
        "writer_host": "writer", "worker_host": "worker", "bank_directory": str(bank),
        "bank_helper_sha256": ack["bank_helper_sha256"], "allowed_gpus": [0, 1, 2, 3],
        "worker_ledger": str(tmp_path / "ledger"), "broker_pid": 77,
    }
    return settings, failure, error, ack


def test_only_exact_rank_zero_ack_timeout_is_recoverable(tmp_path):
    lib = recovery()
    settings, path, error, _ = fixture(tmp_path)
    candidate = lib.classify_failure(path, settings)
    assert candidate["step"] == 10
    for field, value in (("error_type", "RuntimeError"), ("message", "NCCL operation timed out"),
                         ("traceback", "unrelated timeout"), ("status", "complete")):
        changed = copy.deepcopy(error)
        changed[field] = value
        dump(path, changed)
        with pytest.raises(ValueError):
            lib.classify_failure(path, settings)
    dump(path, error)
    wrong_rank = path.with_name("failed-rank-1-999.json")
    wrong_rank.write_bytes(path.read_bytes())
    with pytest.raises(ValueError):
        lib.classify_failure(wrong_rank, settings)


def test_complete_checkpoint_snapshot_and_exact_ack_are_required(tmp_path):
    lib = recovery()
    settings, failure, _, ack = fixture(tmp_path)
    candidate = lib.classify_failure(failure, settings)
    proof = lib.verify_local_candidate(settings, candidate)
    assert proof["checkpoint_identity"]["global_step"] == 10
    ack["adapter_sha256"] = "0" * 64
    dump(Path(settings["run"]) / "acks" / "checkpoint-10.json", ack)
    with pytest.raises(ValueError):
        lib.verify_local_candidate(settings, candidate)


@pytest.mark.parametrize("filename", ["complete.json", "optimizer.pt", "rng_state_3.pth", "rank-state-2.pt"])
def test_incomplete_state_never_recovers(tmp_path, filename):
    lib = recovery()
    settings, failure, _, _ = fixture(tmp_path)
    candidate = lib.classify_failure(failure, settings)
    (Path(settings["run"]) / "checkpoint-10" / filename).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        lib.verify_local_candidate(settings, candidate)


def test_stale_failure_cannot_resume_latest_or_changed_source(tmp_path):
    lib = recovery()
    settings, failure, _, _ = fixture(tmp_path)
    candidate = lib.classify_failure(failure, settings)
    dump(Path(settings["run"]) / "live.json", {"global_step": 11, "pid": 1001})
    with pytest.raises(ValueError, match="stale|step"):
        lib.verify_local_candidate(settings, candidate)
    dump(Path(settings["run"]) / "live.json", {"global_step": 10, "pid": 999})
    (Path(settings["frozen_source"]) / "lc_run.py").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        lib.verify_local_candidate(settings, candidate)


def test_owned_rank_detection_ignores_pid_notes_and_bigrec_gpu4(tmp_path):
    lib = recovery()
    settings, _, _, _ = fixture(tmp_path)
    run = settings["run"]
    assert lib.owns_process({"argv": ["python", "-u", "scripts/letter/lc_run.py", "fit",
                                     "--output", run], "cwd": settings["project"]}, settings)
    assert lib.owns_process({"argv": ["bash", settings["launcher"]], "cwd": "/"}, settings)
    assert lib.owns_process({"argv": ["torchrun", "scripts/letter/lc_run.py", "fit", "--output=" + run],
                            "cwd": settings["project"]}, settings)
    assert not lib.owns_process({"argv": ["python", "bigrec_train.py", "--output", run + "-bigrec"],
                                "cwd": settings["project"]}, settings)
    assert not lib.owns_process({"argv": ["python", "other.py"], "cwd": settings["project"]}, settings)


def test_gpu4_does_not_authorize_or_block_assigned_gpu_guard():
    lib = recovery()
    devices = "\n".join(f"{i}, uuid-{i}" for i in range(8))
    assert lib.gpus_free(devices, "999, uuid-4", [0, 1, 2, 3])
    assert not lib.gpus_free(devices, "999, uuid-4\n1000, uuid-0", [0, 1, 2, 3])
    with pytest.raises(ValueError):
        lib.gpus_free("0, uuid-0", "", [0, 1, 2, 3])


def test_parent_handled_failure_is_permanently_consumed(tmp_path):
    lib = recovery()
    settings, failure, _, _ = fixture(tmp_path)
    candidate = lib.classify_failure(failure, settings)
    ledger = Path(settings["worker_ledger"])
    lib.consume_external(ledger, candidate, {"reason": "explicit parent resume"})
    assert lib.consumed(ledger, candidate["failure_id"])
    with pytest.raises(FileExistsError):
        lib.reserve_attempt(ledger, candidate, {"permit": "new"})


def test_atomic_reservation_allows_one_attempt_even_after_crash(tmp_path):
    lib = recovery()
    settings, failure, _, _ = fixture(tmp_path)
    candidate = lib.classify_failure(failure, settings)
    ledger = Path(settings["worker_ledger"])

    def reserve():
        try:
            lib.reserve_attempt(ledger, candidate, {"permit": "same"})
            return True
        except FileExistsError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(lambda _: reserve(), range(2))) == [False, True]
    with pytest.raises(FileExistsError):
        lib.reserve_attempt(ledger, candidate, {"permit": "after-controller-restart"})


@pytest.mark.parametrize("race", ["owned_process", "gpu_busy", "generation_changed", "permit_expired"])
def test_final_launch_recheck_blocks_races_and_never_retries(tmp_path, race):
    lib = recovery()
    settings, failure, _, _ = fixture(tmp_path)
    candidate = lib.classify_failure(failure, settings)
    expected = {"generation_sha256": "a", "candidate": candidate}
    permit = {"expires_unix": time.time() + 30}
    directory = lib.reserve_attempt(Path(settings["worker_ledger"]), candidate, permit)
    current = {"owned_processes": [], "gpus_free": True, "generation_sha256": "a", "candidate": candidate}
    if race == "owned_process":
        current["owned_processes"] = [{"pid": 123}]
    elif race == "gpu_busy":
        current["gpus_free"] = False
    elif race == "generation_changed":
        current["generation_sha256"] = "b"
    else:
        permit["expires_unix"] = 0
    launched = []
    with pytest.raises(ValueError):
        lib.commit_reserved_launch(directory, expected, permit, lambda: current, lambda: launched.append(True))
    assert not launched
    assert (directory / "aborted.json").is_file()
    with pytest.raises(FileExistsError):
        lib.reserve_attempt(Path(settings["worker_ledger"]), candidate, permit)


def test_fuse_wait_or_missing_exporter_never_means_ready():
    lib = recovery()
    assert not lib.storage_processes_ready([{"pid": 77, "state": "D", "wchan": "request_wait_answer"}], 77)
    assert not lib.storage_processes_ready([{"pid": 77, "state": "S", "wchan": "poll"},
                                           {"pid": 78, "state": "D", "wchan": "io_schedule"}], 77)
    assert not lib.storage_processes_ready([], 77)
    assert lib.storage_processes_ready([{"pid": 77, "state": "S", "wchan": "poll"}], 77)


@pytest.mark.skipif(os.name != "posix", reason="Uses the production nonblocking flock")
def test_busy_common_lock_does_not_probe_storage_or_launch(tmp_path):
    import fcntl
    lib = recovery()
    path = tmp_path / "bank.lock"
    path.touch()
    with path.open("r") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with lib.try_bank_lock(path) as lease:
            assert lease is None


def test_queue_passthrough_is_only_after_instruments_completion():
    lib = recovery()
    assert lib.route_probe({"complete": False, "owned_processes": [{"pid": 1}],
                            "unhandled": [], "classification_error": None}) == "observe-current"
    assert lib.route_probe({"complete": True, "owned_processes": [],
                            "unhandled": [], "classification_error": None}) == "frozen-queue"
    assert lib.route_probe({"complete": False, "owned_processes": [],
                            "unhandled": [], "classification_error": "CUDA error"}) == "blocked"


def test_current_parent_resume_never_touches_storage_or_launches(monkeypatch, tmp_path):
    lib = recovery()
    observed = {"complete": False, "owned_processes": [{"pid": 455147}],
                "unhandled": [], "classification_error": None}
    monkeypatch.setattr(lib, "worker_rpc", lambda *args: observed)
    monkeypatch.setattr(lib, "attempt_recovery", lambda *args: pytest.fail("Touched active parent resume"))
    queue = SimpleNamespace(
        rpc=lambda *args: {"original_queue": "unchanged"},
        tick=lambda *args: pytest.fail("Advanced queue while learner active"),
    )
    phase, facts, result = lib.tick_recovery({"queue_plan": str(tmp_path / "plan.json")}, queue, {})
    assert phase == "waiting-instruments-learner"
    assert facts == {"original_queue": "unchanged"}
    assert result["automatic_recovery_of_parent_handled_failure"] is False


def test_missing_ack_waits_without_storage_probe(monkeypatch, tmp_path):
    lib = recovery()
    observed = {"complete": False, "owned_processes": [], "unhandled": [{"step": 10}],
                "classification_error": None, "candidate_ack_present": False}
    monkeypatch.setattr(lib, "worker_rpc", lambda *args: observed)
    monkeypatch.setattr(lib, "attempt_recovery", lambda *args: pytest.fail("Probed storage before ack"))
    queue = SimpleNamespace(rpc=lambda *args: {})
    phase, _, _ = lib.tick_recovery({"queue_plan": str(tmp_path / "plan.json")}, queue, {})
    assert phase == "recovery-waiting-verified-bank-ack"


def test_fuse_wait_never_reads_bank_or_restarts(monkeypatch, tmp_path):
    lib = recovery()
    monkeypatch.setattr(lib, "exporter_records", lambda _: [
        {"pid": 77, "state": "D", "wchan": "request_wait_answer"}])
    monkeypatch.setattr(lib, "worker_rpc", lambda *args: pytest.fail("Recovery advanced while I/O blocked"))
    monkeypatch.setattr(lib, "durable_ack_proof", lambda *args: pytest.fail("Read blocked bank"))
    result = lib.attempt_recovery({"writer_ledger": str(tmp_path / "writer"), "broker_pid": 77}, {})
    assert result["status"] == "waiting-storage"


def test_reserved_launch_is_exactly_once_and_lease_failure_is_preserved(tmp_path):
    lib = recovery()
    settings, failure, _, _ = fixture(tmp_path)
    candidate = lib.classify_failure(failure, settings)
    permit = {"expires_unix": time.time() + 30}
    directory = lib.reserve_attempt(Path(settings["worker_ledger"]), candidate, permit)
    expected = {"generation_sha256": "same", "candidate": candidate}
    current = {**expected, "owned_processes": [], "gpus_free": True}
    calls = []
    assert lib.commit_reserved_launch(
        directory, expected, permit, lambda: current, lambda: calls.append(1) or {"pid": 7}) == {"pid": 7}
    with pytest.raises(FileExistsError):
        lib.commit_reserved_launch(directory, expected, permit, lambda: current, lambda: calls.append(2))
    assert calls == [1]


@pytest.mark.skipif(os.name != "posix", reason="Verifies Linux /proc/locks without GPU use")
def test_storage_lease_is_bound_to_live_owner_and_common_lock(tmp_path):
    lib = recovery()
    lock = tmp_path / "bank.lock"
    lock.touch()
    permit_id = "a" * 32
    process = lib.process_record(os.getpid())
    settings = {"bank_lock": str(lock), "writer_ledger": str(tmp_path / "writer")}
    permit = {"writer_pid": os.getpid(), "writer_start_ticks": process["start_ticks"],
              "expires_unix": time.time() + 30}
    dump(tmp_path / "writer" / "permits" / f"{permit_id}.json", permit)
    with lib.try_bank_lock(lock) as owned:
        assert owned is not None
        assert lib.lease_status(settings, permit_id)["status"] == "held"
    assert lib.lease_status(settings, permit_id)["status"] == "released"
