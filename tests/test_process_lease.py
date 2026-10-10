import shutil

import pytest

from rrbench.io import write_json
from rrbench.process_lease import register_worker, terminate_owned_workers, wait_for_start_gate


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
