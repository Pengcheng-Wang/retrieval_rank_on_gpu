import ast
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

from test_letter_p5_xnode import workspace


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "letter"
sys.path.insert(0, str(SCRIPTS))


def control(workspace=None):
    path = SCRIPTS / "p5_xnode_control.py"
    assert path.exists(), "Node-specific operational guards must be versioned separately"
    spec = importlib.util.spec_from_file_location("p5_xnode_control_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if workspace is not None:
        from test_letter_p5_xnode_configuration import configure
        configure(module.state, workspace)
    return module


def test_only_the_two_operational_guards_are_recompiled(workspace):
    module = control(workspace)
    text = (SCRIPTS / "p5_run.py").read_text(encoding="utf-8")
    before = hashlib.sha256(text.replace("\r\n", "\n").encode()).hexdigest()
    tree, changes = module.operational_ast(text)
    assert before == module.state.configuration()["learner_sha256"]
    assert {node.name for node in tree.body} == {"setup", "train"}
    assert changes == {"device_guard": 1, "transport_liveness": 1, "transport_host_guard": 1}
    rendered = ast.unparse(tree)
    assert "_xnode_require_device(args)" in rendered
    assert "_xnode_require_transport(args.output, watcher)" in rendered
    assert "900" not in rendered or "checkpoint_every_seconds" in rendered
    assert "CUDA_VISIBLE_DEVICES" not in ast.unparse(tree.body[0])


def test_modified_learner_cannot_enter_operational_adapter(workspace):
    module = control(workspace)
    text = (SCRIPTS / "p5_run.py").read_text(encoding="utf-8") + "\n# unknown edit\n"
    with pytest.raises(ValueError, match="unchanged"):
        module.operational_ast(text)


def test_worker_runtime_denies_every_direct_durable_writer():
    module = control()
    with pytest.raises(RuntimeError, match="writer"):
        module.forbid_durable_write()
