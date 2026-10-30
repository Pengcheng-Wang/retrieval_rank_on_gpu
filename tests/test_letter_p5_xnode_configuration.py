import ast
import copy
import hashlib
import json
import re
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from test_letter_p5_xnode import protocol, workspace
from test_letter_p5_xnode_broker import broker


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "letter"


def binding_document(workspace):
    learner = (SCRIPTS / "p5_run.py").read_text(encoding="utf-8").replace("\r\n", "\n")
    conditions = [ast.unparse(node.test) for node in ast.walk(ast.parse(learner))
                  if isinstance(node, ast.If)]
    device = next(text for text in conditions
                  if text.startswith("os.environ.get(") and "CUDA_VISIBLE_DEVICES" in text)
    transport = next(text for text in conditions
                     if "watcher[" in text and "status" in text and "host" in text)
    assignment = {
        "role": "worker", "gpu": 0, "gpu_uuid": "fixture-device-zero",
        "microbatch": 16, "accumulation": 8, "root_step": 100, "ownership_epoch": 2,
    }
    return {
        "kind": "p5-xnode-bindings-v1",
        "root": str(workspace), "p5_root": str(workspace / "experiment"),
        "control_root": str(workspace / "control"), "durable_root": str(workspace / "bank"),
        "python": str(Path(sys.executable).resolve()),
        "writer_host": socket.gethostname(), "worker_host": "worker.invalid",
        "cells": {
            "example-cell": assignment,
            "retained-cell": {**assignment, "role": "writer", "gpu": 1,
                              "gpu_uuid": "fixture-device-one", "microbatch": 64, "accumulation": 2},
        },
        "adopted_cell": "retained-cell", "learner_sha256": hashlib.sha256(learner.encode()).hexdigest(),
        "legacy_device_guard": device, "legacy_transport_guard": transport,
    }


def configure(module, workspace, changes=None):
    value = binding_document(workspace)
    if changes:
        value.update(changes)
    path = workspace / "private-bindings.json"
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    assert hasattr(module, "configure_bindings"), "An explicit private bootstrap is required"
    module.configure_bindings(path)
    return path, value


@pytest.mark.parametrize("operation", ["write", "locked"])
def test_unconfigured_helpers_fail_before_creating_any_file(workspace, operation):
    module = protocol()
    target = workspace / "must-not-exist" / "artifact.json"
    with pytest.raises((ValueError, RuntimeError), match="bindings|configured"):
        if operation == "write":
            module.write(target, {"not": "authorized"})
        else:
            with module.locked(target):
                pass
    assert not target.parent.exists()


def test_import_has_no_usable_allocation_defaults():
    module = protocol()
    for name in ("ROOT", "P5", "XROOT"):
        assert getattr(module, name) is None, "Import must not choose a real or placeholder output path"
    assert not module.DESTINATIONS


def test_unconfigured_broker_fetch_cannot_create_staging_or_invoke_transport(workspace, monkeypatch):
    monkeypatch.delitem(sys.modules, "p5_xnode_state", raising=False)
    module = broker()
    instance = module.Broker.__new__(module.Broker)
    instance.directory = workspace / "must-not-exist"

    def forbidden(*args, **kwargs):
        pytest.fail("Unconfigured transport was invoked")

    monkeypatch.setattr(module.subprocess, "run", forbidden)
    with pytest.raises((ValueError, RuntimeError), match="bindings|configured"):
        instance.fetch_package(
            {"cell": "example-cell", "lease_id": "fixture-lease"},
            {"name": "g00000001-fixture"},
        )
    assert not instance.directory.exists()


def test_explicit_configuration_allows_only_bound_paths(workspace):
    module = protocol()
    path, value = configure(module, workspace)
    assert module.ROOT == workspace
    assert module.P5 == Path(value["p5_root"])
    assert module.XROOT == Path(value["control_root"])
    assert module.DESTINATIONS == {"example-cell": 0}
    target = workspace / "allowed" / "artifact.json"
    module.write(target, {"ok": True})
    assert module.read(target) == {"ok": True}
    module.configure_bindings(path)


@pytest.mark.parametrize("damage", [
    "empty-root", "relative-root", "traversal-root", "foreign-experiment",
    "foreign-control", "relative-durable", "relative-python", "same-host",
    "host-option", "unknown-adoption", "negative-gpu", "invalid-batch",
])
def test_invalid_bindings_leave_bootstrap_unconfigured_and_create_nothing(workspace, damage):
    module = protocol()
    value = binding_document(workspace)
    if damage == "empty-root":
        value["root"] = ""
    elif damage == "relative-root":
        value["root"] = "unbound-output"
    elif damage == "traversal-root":
        value["root"] = str(workspace / ".." / "escape")
    elif damage == "foreign-experiment":
        value["p5_root"] = str(workspace.parent / "foreign-experiment")
    elif damage == "foreign-control":
        value["control_root"] = str(workspace.parent / "foreign-control")
    elif damage == "relative-durable":
        value["durable_root"] = "unbound-bank"
    elif damage == "relative-python":
        value["python"] = "python"
    elif damage == "same-host":
        value["worker_host"] = value["writer_host"]
    elif damage == "host-option":
        value["worker_host"] = "-unsafe-target"
    elif damage == "unknown-adoption":
        value["adopted_cell"] = "not-authorized"
    elif damage == "negative-gpu":
        value["cells"]["example-cell"]["gpu"] = -1
    else:
        value["cells"]["example-cell"]["accumulation"] = 7
    path = workspace / "invalid-bindings.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    assert hasattr(module, "configure_bindings"), "Bindings must be validated before setting any paths"
    with pytest.raises((ValueError, RuntimeError)):
        module.configure_bindings(path)
    assert module.ROOT is None and not (workspace / "experiment").exists()
    with pytest.raises((ValueError, RuntimeError)):
        module.write(workspace / "not-authorized.json", {})
    assert not (workspace / "not-authorized.json").exists()


@pytest.mark.parametrize("damage", ["foreign", "traversal", "relative"])
def test_configured_write_rejects_foreign_or_ambiguous_paths(workspace, damage):
    module = protocol()
    configure(module, workspace)
    target = {
        "foreign": workspace.parent / "foreign-output" / "artifact.json",
        "traversal": workspace / "ordinary" / ".." / "artifact.json",
        "relative": Path("unbound-output") / "artifact.json",
    }[damage]
    with pytest.raises((ValueError, RuntimeError), match="path|root|absolute|traversal"):
        module.write(target, {})
    assert not (workspace / "ordinary").exists()


def test_binding_content_cannot_change_after_configuration(workspace):
    module = protocol()
    path, value = configure(module, workspace)
    value["worker_host"] = "different.invalid"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError), match="bindings|changed|configured"):
        module.write(workspace / "not-authorized.json", {})
    assert not (workspace / "not-authorized.json").exists()


def test_binding_snapshot_is_recursively_immutable(workspace):
    module = protocol()
    configure(module, workspace)
    bindings = module.configuration()
    with pytest.raises(TypeError):
        bindings["root"] = "different"
    with pytest.raises(TypeError):
        bindings["cells"]["example-cell"]["gpu"] = 1


def test_read_only_protocol_validation_uses_the_pinned_snapshot_without_file_io(workspace, monkeypatch):
    from test_letter_p5_xnode import lease
    module = protocol()
    configure(module, workspace)

    def no_read(*args, **kwargs):
        pytest.fail("Read-only protocol validation repeatedly reopened its binding file")

    monkeypatch.setattr(module, "sha", no_read)
    assert module.require_assignment(lease(), module.WORKER_HOST, 0) == lease()


def test_child_bootstrap_rejects_a_different_parent_binding_digest(workspace):
    module = protocol()
    document = binding_document(workspace)
    path = workspace / "private-bindings.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="digest|checksum|pin"):
        module.configure_bindings(path, expected_sha256="0" * 64)
    assert module.ROOT is None
    module.configure_bindings(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert module.BINDINGS_SHA256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_foreign_host_and_worker_durable_write_are_rejected(workspace, monkeypatch):
    module = protocol()
    _, value = configure(module, workspace)
    monkeypatch.setattr(socket, "gethostname", lambda: "foreign.invalid")
    with pytest.raises((ValueError, RuntimeError), match="host|writer"):
        module.write(workspace / "foreign.json", {})
    monkeypatch.setattr(socket, "gethostname", lambda: value["worker_host"])
    module.write(workspace / "local-worker.json", {})
    with pytest.raises((ValueError, RuntimeError), match="writer|durable"):
        module.write(Path(value["durable_root"]) / "not-authorized.json", {})


def test_runtime_paths_cannot_silently_select_another_allocation(workspace):
    module = protocol()
    configure(module, workspace)
    assert hasattr(module, "validate_runtime_paths"), "Runtime fields must agree with explicit private bindings"
    with pytest.raises((ValueError, RuntimeError), match="path|root|binding"):
        module.validate_runtime_paths({
            "base": str(workspace.parent / "foreign-runtime"), "root": str(workspace.parent),
        })


def test_configured_symlink_escape_is_rejected_before_write(workspace):
    module = protocol()
    configure(module, workspace)
    link = workspace / "escape"
    target = workspace.parent / (workspace.name + "-unowned")
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("This platform does not permit test directory symlinks")
    with pytest.raises(ValueError, match="root|path"):
        module.write(link / "must-not-exist.json", {})
    assert not target.exists()


def test_runtime_source_member_cannot_escape_through_a_symlink(workspace):
    from test_letter_p5_xnode_control import control
    module = control(workspace)
    runtime = workspace / "runtime"
    runtime.mkdir()
    source = runtime / "p5_xnode_control.py"
    try:
        source.symlink_to(SCRIPTS / source.name)
    except OSError:
        pytest.skip("This platform does not permit test file symlinks")
    (runtime / "p5_xnode_state.py").write_text("# fixture\n", encoding="utf-8")
    inventory = workspace / "input.json"
    inventory.write_text("{}", encoding="utf-8")
    manifest = runtime / "runtime.json"
    manifest.write_text(json.dumps({
        "kind": "p5-xnode-runtime-v1",
        "files_sha256": {name: module.state.sha(runtime / name) for name in
                         ("p5_xnode_control.py", "p5_xnode_state.py")},
        "input_manifest": str(inventory), "input_manifest_sha256": module.state.sha(inventory),
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="root|path"):
        module.verify_runtime(manifest)


def test_rpc_carries_the_explicit_binding_file_and_parent_pin(workspace, monkeypatch):
    module = broker(workspace)
    instance = module.Broker.__new__(module.Broker)
    calls = []

    def response(command, **kwargs):
        calls.append(command)
        return type("Reply", (), {"stdout": 'P5_XNODE_RPC:{"status":"ok"}\n'})()

    monkeypatch.setattr(module.subprocess, "run", response)
    assert instance.rpc(["inventory"]) == {"status": "ok"}
    assert "--bindings" in calls[0][-1]
    assert module.state.BINDINGS_SHA256 in calls[0][-1]
    assert calls[0][1] == "-n" and module.state.WORKER_HOST in calls[0]


def test_send_does_not_write_remotely_before_remote_bootstrap(workspace, monkeypatch):
    from test_letter_p5_xnode import lease
    module = broker(workspace)
    instance = module.Broker.__new__(module.Broker)
    instance.directory = module.state.XROOT
    remote_calls = []

    def unavailable(arguments):
        remote_calls.append(arguments)
        raise ValueError("Remote bindings are unavailable")

    def no_write_transport(*args, **kwargs):
        pytest.fail("A remote directory or upload was attempted before remote bootstrap")

    instance.rpc = unavailable
    monkeypatch.setattr(module.subprocess, "run", no_write_transport)
    with pytest.raises(ValueError, match="bindings"):
        instance.send(lease(), {"kind": "test"})
    assert remote_calls[0][0] == "prepare-message"


@pytest.mark.parametrize("script", ["p5_xnode_broker.py", "p5_xnode_control.py", "p5_xnode_adopt.py"])
def test_entrypoints_require_private_bindings_before_other_io(workspace, script):
    result = subprocess.run([sys.executable, "-B", str(SCRIPTS / script)],
                            cwd=workspace, capture_output=True, text=True)
    assert result.returncode != 0 and "--bindings" in result.stderr
    assert list(workspace.iterdir()) == []


def test_owned_public_surface_contains_no_allocation_literals():
    files = [*SCRIPTS.glob("p5_xnode*.py"), *(ROOT / "tests").glob("test_letter_p5_xnode*.py"),
             ROOT / "docs" / "letter-p5-cross-node.md"]
    patterns = [
        r"/(?:scratch|data)/[^\s\"')]+",
        r"letter_\d{8}_[0-9a-f]+",
        r"\bnode-\d+\b",
        r"\b(?:Instruments|Beauty|Yelp)-(?:TID|CID|SemID)-s\d+\b",
        r"\b[0-9a-f]{64}\b",
        r"[A-Za-z]:\\(?:Users|repos)\\",
        r"(?:blob|dfs|vault)\.core\.windows\.net",
        r"\b[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\b",
    ]
    findings = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        findings.extend((str(path.relative_to(ROOT)), match.group(0))
                        for pattern in patterns for match in re.finditer(pattern, text))
        if path.parent == SCRIPTS:
            for node in ast.walk(ast.parse(text)):
                if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                        and node.value.startswith("/") and len(node.value) > 1
                        and not node.value.startswith("/proc")):
                    findings.append((str(path.relative_to(ROOT)), node.value))
    assert findings == [], "Actual access/binding/evidence literals belong only in ignored private inputs"
