"""GPU ownership and quiescence tests; no test launches a CUDA workload."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "letter"
sys.path.insert(0, str(SCRIPTS))
import fresh_pipeline as pipeline
from protocol import write_json


def snapshot(memory=(0, 0), clients=(), timestamp=0.0):
    return {
        "timestamp": timestamp,
        "gpus": {
            "2": {"uuid": "GPU-2", "memory_used_mib": memory[0]},
            "3": {"uuid": "GPU-3", "memory_used_mib": memory[1]},
        },
        "clients": list(clients),
    }


def client(pid, gpu="GPU-3", name="python", memory=123):
    return {
        "gpu_uuid": gpu, "pid": pid, "process_name": name,
        "used_memory_mib": memory,
    }


def process(pid, start, *, ppid=1, pgid=None, session=None):
    return {
        "pid": pid, "ppid": ppid, "pgid": pid if pgid is None else pgid,
        "session_id": pid if session is None else session,
        "start_time_ticks": start,
    }


def baseline(tmp_path):
    value = {
        "gpu_indexes": [2, 3],
        "gpu_uuids": {"2": "GPU-2", "3": "GPU-3"},
        "memory_used_mib": {"2": 0, "3": 0},
    }
    path = tmp_path / "gpu-idle-baseline.json"
    write_json(path, value)
    return path, value


def owner():
    return {
        "pid": 100, "ppid": 1, "pgid": 100, "session_id": 100,
        "start_time_ticks": 10, "label": "fixture-fit",
        "gpu_uuids": ["GPU-2", "GPU-3"], "members": {"100": 10},
    }


def test_snapshot_classifier_is_fail_closed_for_foreign_unknown_and_reused_pids(tmp_path):
    _, idle = baseline(tmp_path)
    owned = owner()
    processes = {
        100: process(100, 10),
        200: process(200, 20, ppid=100, pgid=100, session=100),
        900: process(900, 90),
    }
    reader = processes.get

    live = pipeline.classify_gpu_snapshot(
        snapshot(clients=[client(200)]), idle, [owned], process_reader=reader,
    )
    assert live["status"] == "owned_live"
    assert owned["members"]["200"] == 20

    foreign = pipeline.classify_gpu_snapshot(
        snapshot(clients=[client(900)]), idle, [owned], process_reader=reader,
    )
    assert foreign["status"] == "foreign"
    assert foreign["foreign_clients"][0]["pid"] == 900

    unknown = pipeline.classify_gpu_snapshot(
        snapshot(clients=[client(None, name="nvidia-cuda-mps-server")]),
        idle, [owned], process_reader=reader,
    )
    assert unknown["status"] == "foreign"

    reused = pipeline.classify_gpu_snapshot(
        snapshot(clients=[client(100)]), idle, [owned],
        process_reader=lambda pid: process(pid, 11),
    )
    assert reused["status"] == "foreign"

    residual = pipeline.classify_gpu_snapshot(snapshot(memory=(0, 7)), idle, [])
    assert residual["status"] == "residual_memory"
    assert pipeline.classify_gpu_snapshot(snapshot(), idle, [])["status"] == "clean"


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


def sequenced(values):
    calls = []

    def take():
        value = values[min(len(calls), len(values) - 1)]
        calls.append(value)
        return value

    return take, calls


def test_quiescence_waits_for_three_baseline_samples_spanning_two_seconds(tmp_path):
    baseline_path, _ = baseline(tmp_path)
    evidence = tmp_path / "prelaunch.json"
    take, calls = sequenced([
        snapshot(memory=(0, 9)),
        snapshot(), snapshot(), snapshot(),
    ])
    clock = FakeClock()

    result = pipeline.wait_for_gpu_quiescence(
        take, baseline_path, evidence, owners=(), drain_owned=lambda _: None,
        quiet_samples=3, quiet_seconds=2.0, timeout_seconds=10.0,
        monotonic=clock.monotonic, sleep=clock.sleep,
    )

    assert result == snapshot()
    assert len(calls) == 4
    receipt = json.loads(evidence.read_text())
    assert receipt["status"] == "launchable"
    assert receipt["clean_samples"] == 3
    assert receipt["quiet_seconds"] >= 2.0
    assert [entry["classification"]["status"] for entry in receipt["samples"]] == [
        "residual_memory", "clean", "clean", "clean",
    ]


def test_first_idle_baseline_is_captured_without_an_invented_memory_threshold(tmp_path):
    baseline_path = tmp_path / "gpu-idle-baseline.json"
    evidence = tmp_path / "initial-baseline.json"
    stable = snapshot(memory=(4, 6))
    take, calls = sequenced([stable, stable, stable])
    clock = FakeClock()

    pipeline.wait_for_gpu_quiescence(
        take, baseline_path, evidence, owners=(), drain_owned=lambda _: None,
        quiet_samples=3, quiet_seconds=2.0, timeout_seconds=10.0,
        monotonic=clock.monotonic, sleep=clock.sleep,
    )

    captured = json.loads(baseline_path.read_text())
    assert len(calls) == 3
    assert captured["memory_used_mib"] == {"2": 4, "3": 6}
    assert captured["gpu_uuids"] == {"2": "GPU-2", "3": "GPU-3"}


def test_initial_baseline_quiet_window_restarts_when_memory_changes(tmp_path):
    baseline_path = tmp_path / "gpu-idle-baseline.json"
    evidence = tmp_path / "changing-baseline.json"
    first = snapshot(memory=(4, 6))
    stable = snapshot(memory=(5, 6))
    take, calls = sequenced([first, stable, stable, stable])
    clock = FakeClock()

    pipeline.wait_for_gpu_quiescence(
        take, baseline_path, evidence, owners=(), drain_owned=lambda _: None,
        quiet_samples=3, quiet_seconds=2.0, timeout_seconds=10.0,
        monotonic=clock.monotonic, sleep=clock.sleep,
    )

    assert len(calls) == 4
    assert json.loads(baseline_path.read_text())["memory_used_mib"] == {
        "2": 5, "3": 6,
    }


def test_owned_live_context_is_drained_once_before_quiet_window(tmp_path):
    baseline_path, _ = baseline(tmp_path)
    evidence = tmp_path / "owned-live.json"
    owned = owner()
    processes = {
        100: process(100, 10),
        200: process(200, 20, ppid=100, pgid=100, session=100),
    }
    take, _ = sequenced([
        snapshot(clients=[client(200)]),
        snapshot(clients=[client(200)]),
        snapshot(), snapshot(), snapshot(),
    ])
    drained = []
    clock = FakeClock()

    pipeline.wait_for_gpu_quiescence(
        take, baseline_path, evidence, owners=[owned],
        drain_owned=lambda value: drained.append(value["pid"]),
        quiet_samples=3, quiet_seconds=2.0, timeout_seconds=10.0,
        monotonic=clock.monotonic, sleep=clock.sleep,
        process_reader=processes.get,
    )

    assert drained == [100]
    receipt = json.loads(evidence.read_text())
    assert receipt["drained_owned_pids"] == [100]
    assert receipt["status"] == "launchable"


def test_foreign_canary_blocks_launch_and_is_never_signaled(tmp_path):
    baseline_path, _ = baseline(tmp_path)
    evidence = tmp_path / "foreign-canary.json"
    canary = process(900, 90)
    signaled = []

    with pytest.raises(pipeline.GpuOccupancyError, match="foreign"):
        pipeline.wait_for_gpu_quiescence(
            lambda: snapshot(clients=[client(900)]),
            baseline_path, evidence, owners=(),
            drain_owned=lambda value: signaled.append(value["pid"]),
            process_reader=lambda pid: canary if pid == 900 else None,
        )

    assert signaled == []
    receipt = json.loads(evidence.read_text())
    assert receipt["status"] == "blocked_foreign"
    assert receipt["samples"][0]["classification"]["foreign_clients"][0]["pid"] == 900


class FakeProcess:
    def __init__(self, pid, polls):
        self.pid = pid
        self.polls = list(polls)
        self.returncode = None

    def poll(self):
        if self.polls:
            self.returncode = self.polls.pop(0)
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0


def test_foreign_client_race_after_spawn_drains_only_the_owned_group(tmp_path):
    _, idle = baseline(tmp_path)
    owned = owner()
    processes = {
        100: process(100, 10),
        200: process(200, 20, ppid=100, pgid=100, session=100),
        900: process(900, 90),
    }
    take, _ = sequenced([
        snapshot(clients=[client(200)]),
        snapshot(clients=[client(200), client(900)]),
    ])
    drained = []
    evidence = tmp_path / "post-spawn.json"

    with pytest.raises(pipeline.GpuOccupancyError, match="after spawn"):
        pipeline.monitor_gpu_process(
            FakeProcess(100, [None, None]), owned, idle, evidence,
            snapshotter=take, drain_owned=lambda value: drained.append(value["pid"]),
            process_reader=processes.get, sleep=lambda _: None, poll_seconds=0,
        )

    assert drained == [100]
    receipt = json.loads(evidence.read_text())
    assert receipt["status"] == "foreign_after_spawn"
    assert receipt["samples"][-1]["classification"]["foreign_clients"][0]["pid"] == 900


def test_banker_cuda_context_after_spawn_drains_banker_and_training(tmp_path):
    _, idle = baseline(tmp_path)
    training = owner()
    banker = {
        **owner(), "pid": 300, "pgid": 300, "session_id": 300,
        "start_time_ticks": 30, "label": "checkpoint-watcher",
        "members": {"300": 30},
    }
    processes = {
        100: process(100, 10),
        200: process(200, 20, ppid=100, pgid=100, session=100),
        300: process(300, 30),
    }
    drained = []
    evidence = tmp_path / "banker-post-spawn.json"

    with pytest.raises(pipeline.GpuOccupancyError, match="lifecycle"):
        pipeline.monitor_gpu_process(
            FakeProcess(100, [None]), training, idle, evidence,
            snapshotter=lambda: snapshot(clients=[client(200), client(300)]),
            drain_owned=lambda value: drained.append(value["pid"]),
            process_reader=processes.get, other_owners=[banker],
            sleep=lambda _: None, poll_seconds=0,
        )

    assert drained == [300, 100]
    receipt = json.loads(evidence.read_text())
    assert receipt["status"] == "owned_lifecycle_violation"
    assert receipt["unexpected_owned_pids"] == [300]


def test_gpu_snapshot_failure_after_spawn_drains_owned_group(tmp_path):
    _, idle = baseline(tmp_path)
    owned = owner()
    drained = []
    evidence = tmp_path / "snapshot-failed.json"

    def fail():
        raise subprocess.CalledProcessError(9, ["nvidia-smi"])

    with pytest.raises(pipeline.GpuOccupancyError, match="snapshot"):
        pipeline.monitor_gpu_process(
            FakeProcess(100, [None]), owned, idle, evidence,
            snapshotter=fail, drain_owned=lambda value: drained.append(value["pid"]),
            sleep=lambda _: None,
        )

    assert drained == [100]
    receipt = json.loads(evidence.read_text())
    assert receipt["status"] == "snapshot_failed_after_spawn"


def test_cpu_helpers_hide_cuda_while_gpu_children_keep_the_lease(tmp_path):
    runtime = object.__new__(pipeline.Runtime)
    runtime.env = tmp_path / "env.sh"

    cpu_command, cpu_env = runtime.command(["python", "bank_watch.py"], gpu=False)
    gpu_command, gpu_env = runtime.command(["python", "run.py"], seed=17, gpu=True)

    assert "CUDA_VISIBLE_DEVICES=" in cpu_command
    assert "CUDA_VISIBLE_DEVICES=2,3" in gpu_command
    assert cpu_env["CUDA_VISIBLE_DEVICES"] == gpu_env["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert gpu_env["LETTER_SEED"] == "17"


def test_banker_cuda_regression_drains_banker_and_prevents_torchrun(tmp_path, monkeypatch):
    runtime = object.__new__(pipeline.Runtime)
    runtime.work, runtime.root, runtime.env = tmp_path / "work", tmp_path, tmp_path / "env.sh"
    runtime.gpu_baseline = tmp_path / "baseline.json"
    runtime.work.mkdir()
    banker = FakeProcess(100, [])
    owned = owner()
    runtime._owned_gpu_processes = {100: (banker, owned)}
    drained, launches = [], []

    def quiescence(*args, **kwargs):
        kwargs["drain_owned"](owned)
        return snapshot()

    def drain(value):
        drained.append(value["pid"])
        banker.returncode = 0
        return "completed"

    monkeypatch.setattr(pipeline, "wait_for_gpu_quiescence", quiescence)
    monkeypatch.setattr(runtime, "_drain_gpu_owner", drain)
    monkeypatch.setattr(
        runtime, "command",
        lambda *args, **kwargs: ([sys.executable, "-c", "raise SystemExit(99)"], os.environ.copy()),
    )
    monkeypatch.setattr(
        pipeline.subprocess, "Popen",
        lambda *args, **kwargs: launches.append(args) or pytest.fail("torchrun must not start"),
    )

    with pytest.raises(pipeline.ExporterFailure):
        runtime.invoke(
            ["torchrun"], "banker-regression", gpu=True,
            exporter=banker, completed=lambda: False,
        )

    assert drained == [100]
    assert launches == []


def test_runtime_registers_cpu_banker_as_owned_for_cuda_regression_detection(monkeypatch):
    runtime = object.__new__(pipeline.Runtime)
    runtime._owned_gpu_processes = {}
    banker = FakeProcess(100, [])
    recorded = owner()
    recorded["label"] = "checkpoint-watcher"
    monkeypatch.setattr(pipeline, "capture_process_identity", lambda *args: recorded)

    result = runtime._record_owned_process(
        banker, "checkpoint-watcher", ["GPU-2", "GPU-3"],
    )

    assert result is recorded
    assert runtime._owned_gpu_processes == {100: (banker, recorded)}


def test_identity_capture_failure_drains_the_new_process_session(monkeypatch):
    runtime = object.__new__(pipeline.Runtime)
    runtime._owned_gpu_processes = {}
    process_value = FakeProcess(100, [])
    drained = []
    monkeypatch.setattr(
        pipeline, "capture_process_identity",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            pipeline.GpuOccupancyError("identity unavailable")
        ),
    )
    monkeypatch.setattr(
        pipeline, "_drain_owned_training",
        lambda process, identity=None: drained.append(process.pid) or "completed",
    )

    with pytest.raises(pipeline.GpuOccupancyError, match="identity"):
        runtime._record_owned_process(
            process_value, "new-gpu-child", ["GPU-2", "GPU-3"],
        )

    assert drained == [100]
    assert runtime._owned_gpu_processes == {}


def test_lingering_recorded_member_is_drained_after_launcher_exit(monkeypatch):
    owned = owner()
    owned["members"]["200"] = 20
    training = FakeProcess(100, [0])
    live = {200: process(200, 20, ppid=1, pgid=100, session=100)}
    signals = []

    def killpg(pgid, sig):
        signals.append((pgid, sig))
        if sig == pipeline.signal.SIGTERM:
            live.clear()
        else:
            raise ProcessLookupError

    monkeypatch.setattr(
        pipeline, "os",
        SimpleNamespace(name="posix", killpg=killpg),
    )
    monkeypatch.setattr(pipeline, "read_process_identity", lambda pid: live.get(pid))

    assert pipeline._drain_owned_training(training, owned) == "completed"
    assert signals == [(100, pipeline.signal.SIGTERM), (100, 0)]


def test_drain_waits_for_the_entire_process_group(monkeypatch):
    owned = owner()
    owned["members"]["200"] = 20
    training = FakeProcess(100, [0])
    live = {200: process(200, 20, ppid=1, pgid=100, session=100)}
    signals, group_checks = [], [True, False]

    def killpg(pgid, sig):
        signals.append((pgid, sig))
        if sig == pipeline.signal.SIGTERM:
            live.clear()
            return
        if not group_checks.pop(0):
            raise ProcessLookupError

    monkeypatch.setattr(
        pipeline, "os", SimpleNamespace(name="posix", killpg=killpg),
    )
    monkeypatch.setattr(pipeline, "read_process_identity", lambda pid: live.get(pid))

    assert pipeline._drain_owned_training(training, owned) == "completed"
    assert signals == [
        (100, pipeline.signal.SIGTERM),
        (100, 0),
        (100, 0),
    ]


def test_reused_lingering_member_is_never_signaled(monkeypatch):
    owned = owner()
    owned["members"]["200"] = 20
    training = FakeProcess(100, [0])
    signals = []

    monkeypatch.setattr(
        pipeline, "os",
        SimpleNamespace(name="posix", killpg=lambda *args: signals.append(args)),
    )
    monkeypatch.setattr(
        pipeline, "read_process_identity",
        lambda pid: process(200, 21, ppid=1, pgid=100, session=100) if pid == 200 else None,
    )

    assert pipeline._drain_owned_training(training, owned) == "already_exited"
    assert signals == []


def test_training_never_retries_a_foreign_occupancy_failure(tmp_path, monkeypatch):
    runtime = object.__new__(pipeline.Runtime)
    runtime.work, runtime.root, runtime.bank_root = (
        tmp_path / "work", tmp_path, tmp_path / "durable",
    )
    runtime.scripts = tmp_path / "control"
    runtime.gpu_baseline = runtime.work / "gpu-idle-baseline.json"
    runtime._owned_gpu_processes = {}
    runtime.spec = {
        "queue": {"max_invocation_failures": 3, "poll_seconds": 0},
        "datasets": {"Beauty": {"micro_batch_size": 512}},
    }
    runtime.work.mkdir()
    (tmp_path / "bank.lock").touch()
    stage, audit, output = tmp_path / "stage", tmp_path / "audit", tmp_path / "output"
    cell = {
        "run_id": "fixture", "dataset": "Beauty", "seed": 42, "temperature": 1.0,
    }

    class Banker:
        pid = 300
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

    banker = Banker()
    identity = {
        "pid": 300, "ppid": 1, "pgid": 300, "session_id": 300,
        "start_time_ticks": 30, "label": "checkpoint-watcher",
        "gpu_uuids": ["GPU-2", "GPU-3"], "members": {"300": 30},
    }
    calls = []
    monkeypatch.setitem(
        sys.modules, "fcntl",
        SimpleNamespace(LOCK_EX=1, flock=lambda *args: None),
    )
    monkeypatch.setattr(pipeline, "verify_stage", lambda _: {})
    monkeypatch.setattr(pipeline, "require_real_stage", lambda _: None)
    monkeypatch.setattr(pipeline, "wait_for_gpu_quiescence", lambda *args, **kwargs: snapshot())
    monkeypatch.setattr(pipeline, "wait_for_exporter", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "capture_process_identity", lambda *args, **kwargs: identity)
    monkeypatch.setattr(
        pipeline, "read_process_identity",
        lambda pid: identity if pid == 300 and banker.returncode is None else None,
    )
    monkeypatch.setattr(pipeline.subprocess, "Popen", lambda *args, **kwargs: banker)
    monkeypatch.setattr(
        runtime, "command",
        lambda *args, **kwargs: ([sys.executable, "-c", "pass"], os.environ.copy()),
    )

    def launch(*args, **kwargs):
        calls.append("launch")
        raise pipeline.GpuOccupancyError("foreign canary")

    monkeypatch.setattr(runtime, "invoke", launch)
    with pytest.raises(pipeline.GpuOccupancyError, match="foreign"):
        runtime.train(cell, stage, audit, output)
    assert calls == ["launch"]


def test_gate_proof_generation_probe_receives_the_gpu_lease(tmp_path, monkeypatch):
    runtime = object.__new__(pipeline.Runtime)
    runtime.scripts = tmp_path / "control"
    stage, audit, gate = tmp_path / "stage", tmp_path / "audit", tmp_path / "gate"
    calls = []
    monkeypatch.setattr(runtime, "audit", lambda *args: None)
    monkeypatch.setattr(pipeline, "verify_stage", lambda _: {"dataset": "Beauty"})
    monkeypatch.setattr(pipeline, "validate_training", lambda *args, **kwargs: {})
    monkeypatch.setattr(pipeline, "select_resume", lambda _: "checkpoint-3")

    def invoke(argv, label, *args, gpu=False, **kwargs):
        calls.append((label, gpu))
        if label == "gate-proof":
            write_json(gate / "gate.json", {"status": "pass"})

    monkeypatch.setattr(runtime, "invoke", invoke)

    assert runtime.gate(stage, audit, gate) == {"status": "pass"}
    assert calls == [
        ("gate-continuous", True),
        ("gate-stopped", True),
        ("gate-resumed", True),
        ("gate-proof", True),
    ]


@pytest.mark.skipif(os.name != "posix", reason="Linux process-session identity is required")
def test_gpu_invoke_uses_a_process_session_and_records_start_time(tmp_path, monkeypatch):
    runtime = object.__new__(pipeline.Runtime)
    runtime.work, runtime.root, runtime.env = tmp_path / "work", tmp_path, tmp_path / "env.sh"
    runtime.gpu_baseline = tmp_path / "baseline.json"
    runtime._owned_gpu_processes = {}
    runtime.work.mkdir()
    write_json(runtime.gpu_baseline, {
        "gpu_indexes": [2, 3],
        "gpu_uuids": {"2": "GPU-2", "3": "GPU-3"},
        "memory_used_mib": {"2": 0, "3": 0},
    })
    script = tmp_path / "identity.py"
    output = tmp_path / "identity.json"
    script.write_text(
        "import json,os,time\n"
        f"open({str(output)!r},'w').write(json.dumps("
        "{'pid':os.getpid(),'pgid':os.getpgid(0),'sid':os.getsid(0)}))\n"
        "time.sleep(.2)\n"
    )
    idle = snapshot()
    monkeypatch.setattr(
        pipeline, "wait_for_gpu_quiescence",
        lambda *args, **kwargs: idle,
    )
    monkeypatch.setattr(
        pipeline, "monitor_gpu_process",
        lambda process, *args, **kwargs: process.wait(timeout=10),
    )
    monkeypatch.setattr(
        runtime, "command",
        lambda *args, **kwargs: ([sys.executable, str(script)], os.environ.copy()),
    )

    runtime.invoke(["ignored"], "gpu-session-fixture", gpu=True)

    identity = json.loads(output.read_text())
    assert identity["pid"] == identity["pgid"] == identity["sid"]
    command_file = next((runtime.work / "commands").glob("*/command.json"))
    receipt = json.loads(command_file.read_text())
    assert receipt["process_identity"]["pid"] == identity["pid"]
    assert receipt["process_identity"]["pgid"] == identity["pgid"]
    assert receipt["process_identity"]["session_id"] == identity["sid"]
    assert receipt["process_identity"]["start_time_ticks"] > 0
