import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "letter"
sys.path.insert(0, str(SCRIPTS))


def module():
    path = SCRIPTS / "p5_xnode_adopt.py"
    assert path.exists(), "An adopted child needs a non-restarting one-cell controller"
    spec = importlib.util.spec_from_file_location("adoption_test", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_missing_adopted_pid_is_ambiguous_not_permission_to_restart():
    adopted = module()
    expected = {"pid": 42, "start_ticks": 123, "boot_id": "boot", "command_sha256": "a"}
    assert adopted.adoption_state(expected, None, False) == "blocked-exited-without-completion"
    assert adopted.adoption_state(expected, None, True) == "compute-complete"
    assert adopted.adoption_state(expected, expected, False) == "running"


def test_reused_pid_cannot_be_adopted():
    adopted = module()
    expected = {"pid": 42, "start_ticks": 123, "boot_id": "boot", "command_sha256": "a"}
    actual = dict(expected, start_ticks=124)
    with pytest.raises(ValueError, match="identity"):
        adopted.adoption_state(expected, actual, False)
