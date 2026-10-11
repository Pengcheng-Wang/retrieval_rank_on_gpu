import shutil
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from rrbench.io import write_json
from rrbench.process_lease import register_worker, terminate_owned_workers, wait_for_start_gate
from rrbench import process_lease


def fake_process(root, pid, parent=40, start=100):
    directory = root / str(pid)
    directory.mkdir()
    fields = ["S", str(parent)] + ["0"] * 17 + [str(start)]
    (directory / "stat").write_text(f"{pid} (python) " + " ".join(fields), encoding="utf-8")
    command = ["python", "-m", "rrbench.semantic_encoding", "worker"]
    (directory / "cmdline").write_bytes(b"\0".join(value.encode() for value in command) + b"\0")
    return {"pid": pid, "parent": parent, "start_ticks": start, "command": command}


def test_disappearing_worker_does_not_abort_cleanup_of_remaining_workers(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    identities = [fake_process(proc, 1), fake_process(proc, 2)]
    registry = tmp_path / "workers.json"
    write_json(registry, identities)
    attempted = []

    def signal(pid, _):
        attempted.append(pid)
        shutil.rmtree(proc / str(pid))
        if pid == 1:
            raise ProcessLookupError("worker exited between inspection and signal")

    terminate_owned_workers(registry, 40, proc_root=proc, send_signal=signal, timeout=0)
    assert attempted == [1, 2]


def test_pid_reuse_is_not_signalled_but_other_owned_workers_are_still_drained(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    first = fake_process(proc, 1)
    second = fake_process(proc, 2)
    registry = tmp_path / "workers.json"
    write_json(registry, [first, second])
    fields = ["S", "40"] + ["0"] * 17 + ["999"]
    (proc / "1" / "stat").write_text("1 (python) " + " ".join(fields), encoding="utf-8")
    attempted = []

    def signal(pid, _):
        attempted.append(pid)
        shutil.rmtree(proc / str(pid))

    with pytest.raises(RuntimeError, match="identity"):
        terminate_owned_workers(registry, 40, proc_root=proc, send_signal=signal, timeout=0)
    assert attempted == [2]


def test_unregistered_worker_never_leaves_closed_start_gate(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    fake_process(proc, 40, parent=30)
    with pytest.raises(TimeoutError):
        wait_for_start_gate(tmp_path / "gate.json", "owner", 0, 40, 100, proc_root=proc, timeout=0)


def test_foreign_gate_or_dead_parent_cannot_authorize_gpu_work(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    fake_process(proc, 40, parent=30)
    gate = tmp_path / "gate.json"
    write_json(gate, {"lease_owner": "foreign", "worker": 0})
    with pytest.raises(ValueError, match="owner"):
        wait_for_start_gate(gate, "owner", 0, 40, 100, proc_root=proc)
    shutil.rmtree(proc / "40")
    write_json(gate, {"lease_owner": "owner", "worker": 0})
    with pytest.raises(RuntimeError, match="parent"):
        wait_for_start_gate(gate, "owner", 0, 40, 100, proc_root=proc)


def test_model_smoke_is_registered_before_its_gpu_start_gate_opens(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    fake_process(proc, 1)
    (proc / "1" / "cmdline").write_bytes(b"python\0-m\0rrbench.semantic_models\0smoke\0")
    registry = tmp_path / "registry.json"
    register_worker(1, 40, registry, proc_root=proc, expected_module="rrbench.semantic_models")
    assert registry.exists()


def test_absolute_deadline_drains_only_the_registered_worker(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    fake_process(proc, 40, parent=30)
    registered = fake_process(proc, 1)
    fake_process(proc, 2)
    registry = tmp_path / "registry.json"
    write_json(registry, [registered])
    signalled = []

    def signal(pid, _):
        signalled.append(pid)
        shutil.rmtree(proc / str(pid))

    with pytest.raises(TimeoutError, match="deadline"):
        process_lease.wait_owned_worker_until(
            registry, 40, 1, "2026-01-01T00:00:00Z",
            proc_root=proc, send_signal=signal,
        )
    assert signalled == [1]
    assert (proc / "2").exists()


def test_expired_deadline_does_not_signal_a_reused_pid(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    fake_process(proc, 40, parent=30)
    original = fake_process(proc, 1)
    registry = tmp_path / "registry.json"
    write_json(registry, [original])
    fields = ["S", "40"] + ["0"] * 17 + ["999"]
    (proc / "1" / "stat").write_text("1 (python) " + " ".join(fields))
    signalled = []
    with pytest.raises(RuntimeError, match="identity"):
        process_lease.wait_owned_worker_until(
            registry, 40, 1, "2026-01-01T00:00:00Z",
            proc_root=proc, send_signal=lambda pid, requested: signalled.append(pid),
        )
    assert signalled == []


@pytest.mark.parametrize("operation", ["monitor", "drain"])
def test_exiting_worker_empty_command_is_rechecked_before_identity_failure(tmp_path, monkeypatch, operation):
    proc = tmp_path / "proc"
    proc.mkdir()
    fake_process(proc, 40, parent=30)
    original = fake_process(proc, 1)
    registry = tmp_path / "registry.json"
    write_json(registry, [original])
    (proc / "1" / "cmdline").write_bytes(b"")
    real_read = process_lease.read_process
    seen = False

    def read_then_exit(pid, root):
        nonlocal seen
        value = real_read(pid, root)
        if pid == 1 and not seen:
            seen = True
            fields = ["Z", "40"] + ["0"] * 17 + ["100"]
            (proc / "1" / "stat").write_text("1 (python) " + " ".join(fields))
        return value

    monkeypatch.setattr(process_lease, "read_process", read_then_exit)
    attempted = []
    signal = lambda pid, requested: attempted.append(pid)
    if operation == "monitor":
        process_lease.wait_owned_worker_until(
            registry, 40, 1, "2026-01-01T00:00:00Z", proc_root=proc, send_signal=signal,
        )
    else:
        process_lease.terminate_owned_workers(registry, 40, proc_root=proc, send_signal=signal, timeout=0)
    assert attempted == []


def test_extended_empty_command_teardown_waits_for_confirmed_terminal_state(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    proc.mkdir()
    fake_process(proc, 40, parent=30)
    original = fake_process(proc, 1)
    registry = tmp_path / "registry.json"
    write_json(registry, [original])
    (proc / "1" / "cmdline").write_bytes(b"")
    real_read = process_lease.read_process
    observations = 0

    def read_with_delayed_exit(pid, root):
        nonlocal observations
        if pid == 1:
            observations += 1
            if observations >= 5:
                fields = ["Z", "40"] + ["0"] * 17 + ["100"]
                (proc / "1" / "stat").write_text("1 (python) " + " ".join(fields))
        return real_read(pid, root)

    monkeypatch.setattr(process_lease, "read_process", read_with_delayed_exit)
    attempted = []
    process_lease.wait_owned_worker_until(
        registry, 40, 1, "2099-01-01T00:00:00Z", proc_root=proc,
        send_signal=lambda pid, requested: attempted.append(pid), poll_seconds=0.001,
    )
    assert observations >= 5
    assert attempted == []


def test_persistently_empty_command_is_not_certified_as_drained_or_signalled(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    original = fake_process(proc, 1)
    registry = tmp_path / "registry.json"
    write_json(registry, [original])
    (proc / "1" / "cmdline").write_bytes(b"")
    attempted = []
    with pytest.raises(RuntimeError):
        terminate_owned_workers(
            registry, 40, proc_root=proc, send_signal=lambda pid, requested: attempted.append(pid), timeout=0,
        )
    assert attempted == []


def test_pending_exit_does_not_prevent_draining_another_confirmed_worker(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    pending = fake_process(proc, 1)
    live = fake_process(proc, 2)
    registry = tmp_path / "registry.json"
    write_json(registry, [pending, live])
    (proc / "1" / "cmdline").write_bytes(b"")
    attempted = []

    def signal(pid, _):
        attempted.append(pid)
        shutil.rmtree(proc / str(pid))

    with pytest.raises(RuntimeError, match="drain"):
        terminate_owned_workers(registry, 40, proc_root=proc, send_signal=signal, timeout=0)
    assert attempted == [2]
    assert (proc / "1").exists()
    assert not (proc / "2").exists()


def test_identity_change_while_pending_is_never_signalled(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    proc.mkdir()
    fake_process(proc, 40, parent=30)
    original = fake_process(proc, 1)
    registry = tmp_path / "registry.json"
    write_json(registry, [original])
    (proc / "1" / "cmdline").write_bytes(b"")
    real_read = process_lease.read_process
    observations = 0

    def read_with_reuse(pid, root):
        nonlocal observations
        if pid == 1:
            observations += 1
            if observations >= 3:
                fields = ["S", "40"] + ["0"] * 17 + ["999"]
                (proc / "1" / "stat").write_text("1 (python) " + " ".join(fields))
        return real_read(pid, root)

    monkeypatch.setattr(process_lease, "read_process", read_with_reuse)
    attempted = []
    with pytest.raises(RuntimeError, match="identity"):
        process_lease.wait_owned_worker_until(
            registry, 40, 1, "2099-01-01T00:00:00Z", proc_root=proc,
            send_signal=lambda pid, requested: attempted.append(pid), poll_seconds=0.001,
        )
    assert attempted == []
@pytest.mark.skipif(sys.platform != "linux" or shutil.which("bash") is None, reason="Linux process ownership")
def test_deadline_interrupts_a_waiting_worker_and_reaches_restoration_cleanup(tmp_path):
    registry = tmp_path / "registry.json"
    restored = tmp_path / "restored.txt"
    script = r"""
set -euo pipefail
P="$1"; REGISTRY="$2"; RESTORED="$3"
cleanup() { printf 'restoration cleanup reached\n' > "$RESTORED"; }
trap cleanup EXIT
"$P" -c 'import time; time.sleep(30)' &
pid=$!
"$P" - "$REGISTRY" "$$" "$pid" <<'PY'
from pathlib import Path
import sys
from rrbench.io import write_json
from rrbench.process_lease import read_process
identity,state=read_process(int(sys.argv[3]),Path('/proc'))
assert identity['parent']==int(sys.argv[2])
write_json(Path(sys.argv[1]),[identity])
PY
end=$("$P" -c 'import datetime; print((datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=1)).isoformat())')
"$P" -m rrbench.process_lease wait --registry "$REGISTRY" --parent "$$" --pid "$pid" --deadline "$end"
wait "$pid"
"""
    result = subprocess.run(
        ["bash", "-c", script, "deadline-check", sys.executable, str(registry), str(restored)],
        capture_output=True, text=True, timeout=15,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    assert result.returncode != 0
    assert "deadline" in result.stderr.lower()
    assert restored.read_text() == "restoration cleanup reached\n"
    worker = json.loads(registry.read_text())[0]
    proc = Path("/proc") / str(worker["pid"])
    if proc.exists():
        fields = (proc / "stat").read_text().rsplit(") ", 1)[1].split()
        assert fields[0] in ("Z", "X")
