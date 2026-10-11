import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

from rrbench import leased_worker


def test_module_is_not_imported_before_parent_guard_and_registered_gate(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(leased_worker, "guard_worker_parent", lambda *args: events.append("parent"))
    monkeypatch.setattr(leased_worker, "wait_for_start_gate", lambda *args: events.append("gate"))
    before = sys.argv

    def execute(module, **kwargs):
        events.append((module, sys.argv[:], kwargs))

    monkeypatch.setattr(leased_worker.runpy, "run_module", execute)
    leased_worker.run_registered_module(40, 100, tmp_path / "gate.json", "owner", 0, "rrbench.example", ["--rows", "20"])
    assert events == ["parent", "gate", ("rrbench.example", ["rrbench.example", "--rows", "20"], {"run_name": "__main__"})]
    assert sys.argv is before


def test_closed_gate_cannot_execute_the_gpu_module(tmp_path, monkeypatch):
    monkeypatch.setattr(leased_worker, "guard_worker_parent", lambda *args: None)

    def closed(*args):
        raise TimeoutError("closed")

    monkeypatch.setattr(leased_worker, "wait_for_start_gate", closed)
    monkeypatch.setattr(leased_worker.runpy, "run_module", lambda *args, **kwargs: pytest.fail("Unregistered module ran."))
    with pytest.raises(TimeoutError, match="closed"):
        leased_worker.run_registered_module(40, 100, tmp_path / "gate.json", "owner", 0, "rrbench.example", [])


@pytest.mark.skipif(sys.platform != "linux" or shutil.which("bash") is None, reason="Linux parent-death signaling")
@pytest.mark.parametrize("open_gate", [False, True])
def test_real_parent_loss_removes_registered_child_before_or_after_module_start(tmp_path, open_gate):
    module = tmp_path / "owned_lifecycle_probe.py"
    marker = tmp_path / "module-started.txt"
    registry = tmp_path / "registry.json"
    gate = tmp_path / "gate.json"
    module.write_text("import pathlib,sys,time\npathlib.Path(sys.argv[1]).write_text('started')\ntime.sleep(60)\n")
    script = r"""
set -euo pipefail
P="$1"; ROOT="$2"; REGISTRY="$3"; GATE="$4"; MARKER="$5"
start=$("$P" - "$$" <<'PY'
import sys
from pathlib import Path
from rrbench.process_lease import read_process
print(read_process(int(sys.argv[1]),Path('/proc'))[0]['start_ticks'])
PY
)
"$P" -m rrbench.leased_worker --parent "$$" --parent-start "$start" --gate "$GATE" --owner probe --worker 0 owned_lifecycle_probe "$MARKER" &
pid=$!
"$P" -m rrbench.process_lease register --registry "$REGISTRY" --parent "$$" --pid "$pid" --module rrbench.leased_worker
wait "$pid"
"""
    parent = subprocess.Popen(
        ["bash", "-c", script, "owned-parent", sys.executable, str(tmp_path), str(registry), str(gate), str(marker)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), *sys.path]), "CUDA_VISIBLE_DEVICES": ""},
    )
    try:
        end = time.monotonic() + 5
        while not registry.exists() and time.monotonic() < end:
            time.sleep(0.02)
        assert registry.exists()
        if open_gate:
            gate.write_text(json.dumps({"lease_owner": "probe", "worker": 0}))
            while not marker.exists() and time.monotonic() < end:
                time.sleep(0.02)
            assert marker.exists()
        else:
            time.sleep(0.15)
            assert not marker.exists()
        entry = json.loads(registry.read_text())[0]
        parent.kill()
        parent.wait(timeout=5)
        directory = Path("/proc") / str(entry["pid"])
        end = time.monotonic() + 5
        while directory.exists() and time.monotonic() < end:
            state = (directory / "stat").read_text().rsplit(") ", 1)[1].split()[0]
            if state in ("Z", "X"):
                break
            time.sleep(0.02)
        assert not directory.exists() or state in ("Z", "X")
        assert marker.exists() == open_gate
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
