"""Privately bound worker lease/transport controls around the unchanged scientific learner."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import p5_xnode_state as state


STOP = False


def operational_ast(text):
    bindings = state.configuration()
    text = text.replace("\r\n", "\n")
    if hashlib.sha256(text.encode()).hexdigest() != bindings["learner_sha256"]:
        raise ValueError("Only the unchanged frozen learner may receive operational adapters")
    tree = ast.parse(text)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("setup", "train")]
    changes = {"device_guard": 0, "transport_liveness": 0, "transport_host_guard": 0}
    device = ast.parse(bindings["legacy_device_guard"], mode="eval").body
    transport = ast.parse(bindings["legacy_transport_guard"], mode="eval").body
    liveness = ast.parse('os.kill(watcher["pid"], 0)').body[0]

    class OperationalChecks(ast.NodeTransformer):
        def visit_If(self, node):
            if ast.dump(node.test) == ast.dump(device):
                changes["device_guard"] += 1
                return ast.copy_location(ast.parse("_xnode_require_device(args)").body[0], node)
            if ast.dump(node.test) == ast.dump(transport):
                changes["transport_host_guard"] += 1
                return ast.copy_location(ast.parse("_xnode_require_transport(args.output, watcher)").body[0], node)
            return self.generic_visit(node)

        def visit_Expr(self, node):
            if ast.dump(node) == ast.dump(liveness):
                changes["transport_liveness"] += 1
                return None
            return self.generic_visit(node)

    result = OperationalChecks().visit(ast.Module(body=functions, type_ignores=[]))
    ast.fix_missing_locations(result)
    if changes != {"device_guard": 1, "transport_liveness": 1, "transport_host_guard": 1}:
        raise ValueError("The exact operational-only patch boundary changed")
    return result, changes


def forbid_durable_write(*args, **kwargs):
    raise RuntimeError("Only the configured writer may write durable artifacts; workers use local outboxes")


def load_module(name, path):
    state.configuration()
    state.require_path(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_runtime(manifest):
    state.configuration()
    state.require_path(manifest)
    manifest = Path(manifest)
    config = state.read(manifest)
    state.validate_runtime_paths(config)
    if (config.get("kind") != "p5-xnode-runtime-v1"
            or set(config["files_sha256"]) != {"p5_xnode_control.py", "p5_xnode_state.py"}):
        raise ValueError("Unknown cross-node runtime")
    for name, digest in config["files_sha256"].items():
        state.require_path(manifest.parent / name)
        if state.sha(manifest.parent / name) != digest:
            raise ValueError("Cross-node runtime source changed")
    if state.sha(Path(__file__)) != config["files_sha256"]["p5_xnode_control.py"]:
        raise ValueError("A different control runtime was invoked")
    if state.sha(config["input_manifest"]) != config["input_manifest_sha256"]:
        raise ValueError("The copied input inventory changed")
    return config


def ownership_certificate(path, lease):
    state.configuration()
    state.require_path(path)
    certificate = state.read(path)
    event = certificate["event"]
    encoded = (__import__("json").dumps(event, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    event_sha = hashlib.sha256(encoded).hexdigest()
    receipt = certificate["bank_receipt"]
    if (certificate.get("writer_host") != state.WRITER_HOST
            or certificate.get("event_sha256") != event_sha
            or state.digest(receipt) != certificate.get("bank_receipt_sha256")
            or receipt["files"]["ownership.json"]["sha256"] != event_sha
            or state.require_owner(event, lease["cell"], state.WORKER_HOST, lease["lease_id"]) != lease):
        raise ValueError("No matching durable checkpoint-bound ownership certificate")
    return certificate


def registration(run, lease, config):
    state.configuration()
    state.require_path(run)
    state.validate_runtime_paths(config)
    certificate = ownership_certificate(config["ownership_certificate"], lease)
    proof = state.read(config["crossnode_proof"])
    cell_proof = proof.get("cells", {}).get(lease["cell"], {})
    if (proof.get("status") != "pass" or proof.get("source_node") != state.WRITER_HOST
            or proof.get("destination_node") != state.WORKER_HOST
            or proof.get("runtime_manifest_sha256") != lease["control_manifest_sha256"]
            or certificate["event"].get("crossnode_proof_sha256") != state.sha(config["crossnode_proof"])
            or cell_proof.get("source_checkpoint_seal_sha256") != lease["root"]["seal_sha256"]
            or not all(cell_proof.get(key) is True for key in
                       ("bitwise_state_match", "sampled_rows_losses_match", "cold_save_reload",
                        "populated_optimizer_memory_pass", "complete_20beam_geometry_pass"))):
        raise ValueError("A matching actual cross-node continuation/memory proof is mandatory")
    value = state.read(Path(run) / ".xnode" / "registration.json")
    if (value.get("kind") != "p5-xnode-registration-v1" or value.get("writer_host") != state.WRITER_HOST
            or value.get("lease_sha256") != state.digest(lease)
            or value.get("event_sha256") != certificate["event_sha256"]
            or value.get("input_manifest_sha256") != config["input_manifest_sha256"]):
        raise ValueError("Node0 has not registered this destination owner")
    return value


def make_attestation(config, manifest, lease, purpose, run):
    return {
        "kind": "p5-xnode-execution-attestation-v1", "purpose": purpose,
        "lease_sha256": state.digest(lease), "runtime_manifest_sha256": state.sha(manifest),
        "input_manifest_sha256": config["input_manifest_sha256"],
        "host": socket.gethostname(), "gpu": lease["gpu"], "gpu_uuid": lease["gpu_uuid"],
        "root_checkpoint_seal_sha256": lease["root"]["seal_sha256"],
        "scientific_contract_file_sha256": state.sha(Path(run) / "p5-run-config.json"),
        "scientific_contract_rewritten": False,
        "operational_patches": ["setup.device_guard", "train.transport_guard",
                                "checkpoint.owner_ancestry_outbox", "wait_for_bank.local_request_reply"],
    }


def install_runtime(manifest, lease, run, purpose):
    state.configuration()
    state.require_path(run, writable=True)
    config = verify_runtime(manifest)
    run = Path(run)
    if state.sha(manifest) != lease["control_manifest_sha256"]:
        raise ValueError("Lease is bound to another control release")
    state.require_assignment(lease, socket.gethostname(), lease["gpu"])
    if purpose not in ("probe", "production"):
        raise ValueError("Invalid execution purpose")
    if purpose == "production":
        if run != state.P5 / "runs" / lease["cell"]:
            raise ValueError("Production may only use this lease's canonical local run")
        registration(run, lease, config)
    elif state.XROOT / "gates" not in run.parents:
        raise ValueError("Probes must use separate owned gate runs, never production copies")
    if state.sha(run / "p5-run-config.json") != lease["contract_file_sha256"]:
        raise ValueError("The original scientific contract changed")
    maintenance_path = Path(config["maintenance_manifest"]).parent / "p5_maintenance.py"
    maintenance = load_module("p5_xnode_existing_maintenance", maintenance_path)
    maintenance.verify_compatibility(config["maintenance_manifest"], config["maintenance_proof"])
    context = maintenance.install(config["base"], config["maintenance_manifest"])
    maintenance.attest_run(context, run, config["maintenance_manifest"], config["maintenance_proof"])
    attestation = make_attestation(config, manifest, lease, purpose, run)
    attestation_path = run / "crossnode-attestation.json"
    if attestation_path.exists() and state.read(attestation_path) != attestation:
        raise ValueError("Cross-node attestations are immutable within a lease")
    state.write(attestation_path, attestation)

    def device_check(args):
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible != str(lease["gpu"]) or args.microbatch_size != lease["microbatch"]:
            raise ValueError("Physical GPU lease or frozen microbatch differs")
        gpu_uuid = subprocess.check_output([
            "nvidia-smi", "--id=" + visible, "--query-gpu=uuid", "--format=csv,noheader",
        ], text=True).strip()
        state.require_assignment(lease, socket.gethostname(), int(visible), gpu_uuid)
        if args.gate_subset != 0:
            raise ValueError("Cross-node continuation cannot substitute a subset training contract")

    def transport_check(output, watcher):
        if purpose != "production":
            raise ValueError("A probe cannot enter the production training loop")
        current = registration(output, lease, config)
        if (watcher.get("kind") != "p5-xnode-local-transport-v1" or watcher.get("status") != "ready"
                or watcher.get("host") != state.WORKER_HOST or watcher.get("writer_host") != state.WRITER_HOST
                or watcher.get("lease_sha256") != state.digest(lease)
                or watcher.get("registration_sha256") != state.digest(current)):
            raise ValueError("The local transport guardian is not ready for this fenced owner")
        os.kill(watcher["pid"], 0)
        proc = Path("/proc") / str(watcher["pid"])
        start = int(proc.joinpath("stat").read_text().rsplit(")", 1)[1].split()[19])
        if start != watcher["start_ticks"] or b"guardian" not in proc.joinpath("cmdline").read_bytes().split(b"\0"):
            raise ValueError("The local guardian PID identity changed")

    context.learner.__dict__["_xnode_require_device"] = device_check
    context.learner.__dict__["_xnode_require_transport"] = transport_check
    text = Path(context.learner.__file__).read_text()
    adapted, changes = operational_ast(text)
    exec(compile(adapted, context.learner.__file__, "exec"), context.learner.__dict__)
    original_save, original_restore = context.checkpoint.save_checkpoint, context.checkpoint.restore_checkpoint

    def selected_header(output, saved_state):
        selected = saved_state.get("selected_artifact")
        return (state.checkpoint_header(Path(output) / "states" / state.plain_name(selected),
                                        lease["root"]["contract_sha256"]) if selected else None)

    def save(output, model, optimizer, scheduler, sampler, saved_state, contract, selected=False):
        before = context.checkpoint.rng_state()
        result = original_save(output, model, optimizer, scheduler, sampler, saved_state, contract, selected)
        state.publish_checkpoint(output, result[0], lease, purpose, selected_header(output, saved_state))
        maintenance._assert_rng_equal(before, context.checkpoint.rng_state())
        return result

    def restore(path, model, optimizer, scheduler, sampler, contract):
        saved = original_restore(path, model, optimizer, scheduler, sampler, contract)
        before = context.checkpoint.rng_state()
        state.publish_checkpoint(Path(path).parent.parent, path, lease, purpose,
                                 selected_header(Path(path).parent.parent, saved))
        maintenance._assert_rng_equal(before, context.checkpoint.rng_state())
        return saved

    def wait_for_bank(output, checkpoint, timeout_seconds):
        if purpose != "production":
            raise ValueError("Probes have no production bank authority")
        registration(output, lease, config)
        header = state.checkpoint_header(checkpoint, lease["root"]["contract_sha256"])
        nonce = uuid.uuid4().hex
        directory = Path(output) / ".xnode"
        state.write(directory / "requests" / (nonce + ".json"), {
            "kind": "p5-xnode-checkpoint-request-v1", "nonce": nonce,
            "cell": lease["cell"], "lease_id": lease["lease_id"], "lease_sha256": state.digest(lease),
            "checkpoint_seal_sha256": header["seal_sha256"],
        })
        deadline = time.monotonic() + timeout_seconds
        reply_path = directory / "responses" / (nonce + ".json")
        while not reply_path.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("No fresh owner-checked writer acknowledgement; final test blocked")
            time.sleep(5)
        reply = state.read(reply_path)
        if reply.get("nonce") != nonce or reply.get("lease_sha256") != state.digest(lease):
            raise ValueError("A stale transport reply cannot authorize testing")
        state.validate_ack(reply["ack"], lease, header)
        return reply["ack"]

    context.checkpoint.save_checkpoint = context.learner.save_checkpoint = save
    context.checkpoint.restore_checkpoint = context.learner.restore_checkpoint = restore
    old_bank = sys.modules["p5_bank"]
    old_bank.wait_for_bank = wait_for_bank
    for module in (old_bank, context.bank):
        module.bank_checkpoint = forbid_durable_write
        module.bank_results = forbid_durable_write
        module.watch = forbid_durable_write
    return SimpleNamespace(context=context, maintenance=maintenance, config=config, changes=changes)


def worker(manifest, lease_path, arguments):
    state.configuration()
    state.require_path(lease_path)
    lease = state.read(lease_path)
    run = state.P5 / "runs" / lease["cell"]
    with state.locked(run / ".xnode" / "fit.lock", nonblocking=True):
        installed = install_runtime(manifest, lease, run, "production")
        parsed = installed.context.learner.parser().parse_args(arguments)
        if parsed.output != run or parsed.stop_steps is not None:
            raise ValueError("Production continuation must use the original run and full budget")
        installed.context.learner.train(parsed)


def _request_stop(*_):
    global STOP
    STOP = True


def guardian(manifest, lease_path, spec_path):
    state.configuration()
    state.require_path(lease_path)
    state.require_path(spec_path)
    lease = state.read(lease_path)
    config = verify_runtime(manifest)
    state.require_assignment(lease, socket.gethostname(), lease["gpu"])
    run = state.P5 / "runs" / lease["cell"]
    with state.locked(run / ".xnode" / "guardian.lock", nonblocking=True):
        ownership_certificate(config["ownership_certificate"], lease)
        deadline = time.monotonic() + 600
        while not (run / ".xnode" / "registration.json").exists():
            if time.monotonic() > deadline:
                raise TimeoutError("Node0 has not admitted the destination transport")
            time.sleep(2)
        admitted = registration(run, lease, config)
        state.write(run / "bank-watcher-ready.json", {
            "kind": "p5-xnode-local-transport-v1", "status": "ready", "host": state.WORKER_HOST,
            "writer_host": state.WRITER_HOST, "pid": os.getpid(), "lease_sha256": state.digest(lease),
            "registration_sha256": state.digest(admitted),
            "start_ticks": int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]),
        })
        specification = state.read(spec_path)
        cell = next(cell for cell in specification["cells"] if cell["name"] == lease["cell"])
        base_arguments = []
        for key in ("native", "upstream", "manifest", "recipe", "model"):
            state.require_path(specification["common"][key])
            base_arguments += ["--" + key, specification["common"][key]]
        state.require_path(cell["corpus"])
        state.require_path(lease["native_gate"])
        base_arguments += [
            "--corpus", cell["corpus"], "--output", str(run),
            "--microbatch-size", str(lease["microbatch"]), "--gate", lease["native_gate"],
        ]
        environment = dict(
            os.environ, CUDA_VISIBLE_DEVICES=str(lease["gpu"]), CUBLAS_WORKSPACE_CONFIG=":4096:8",
            PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", RAYON_NUM_THREADS="2",
            TOKENIZERS_PARALLELISM="true", TMPDIR=str(state.P5 / "cache"),
        )
        signal.signal(signal.SIGTERM, _request_stop)
        signal.signal(signal.SIGINT, _request_stop)
        attempts = 0
        while not STOP:
            if (run / "completed.json").exists():
                if (run / "results-banked.json").exists():
                    completed = state.read(run / "results-banked.json")
                    ready = state.read(run / ".xnode" / "results-ready.json")
                    state.validate_result_ack(completed, lease, run / ready["snapshot"])
                    state.write(run / ".xnode" / "guardian-status.json", {
                        "status": "verified-complete", "pid": os.getpid(),
                        "lease_sha256": state.digest(lease),
                    })
                    return
                if not (run / ".xnode" / "results-ready.json").exists():
                    maintenance = load_module(
                        "p5_xnode_result_maintenance",
                        Path(config["maintenance_manifest"]).parent / "p5_maintenance.py",
                    )
                    context = maintenance.install(config["base"], config["maintenance_manifest"])
                    with context.fixed_checkpoint.pointer_lock(run):
                        context.fixed_checkpoint._recover_selection_locked(run)
                        snapshot = context.bank._build_result_snapshot_locked(run)
                    state.write(run / ".xnode" / "results-ready.json", {
                        "kind": "p5-xnode-results-ready-v1", "cell": lease["cell"],
                        "lease_sha256": state.digest(lease),
                        "snapshot": str(snapshot.relative_to(run)),
                        "result_seal_sha256": state.sha(snapshot / "p5-result-seal.json"),
                        "head": state.read(run / ".xnode" / "head.json"),
                    })
                state.write(run / ".xnode" / "guardian-status.json", {
                    "status": "compute-complete-awaiting-writer-result-export", "pid": os.getpid(),
                    "lease_sha256": state.digest(lease),
                })
                time.sleep(10)
                continue
            command = [
                sys.executable, "-B", str(Path(__file__).resolve()),
                "--bindings", str(state.BINDINGS_PATH), "--bindings-sha256", state.BINDINGS_SHA256,
                "--runtime-manifest", str(manifest),
                "--lease", str(lease_path), "worker", "--",
            ] + base_arguments
            with (run / "train.log").open("a") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
                state.write(run / ".xnode" / "guardian-status.json", {
                    "status": "training", "pid": os.getpid(), "fit_pid": child.pid,
                    "command": command, "lease_sha256": state.digest(lease),
                })
                signaled = False
                while child.poll() is None:
                    if STOP and not signaled:
                        child.send_signal(signal.SIGTERM)
                        signaled = True
                    time.sleep(2)
                if child.returncode != 0:
                    state.write(run / ".xnode" / "guardian-status.json", {
                        "status": "failed", "pid": os.getpid(), "fit_pid": child.pid,
                        "returncode": child.returncode, "lease_sha256": state.digest(lease),
                    })
                    raise RuntimeError("Destination learner failed; no recipe fallback is allowed")
            attempts += 1
            if attempts > 3 and not (run / "completed.json").exists():
                raise RuntimeError("Repeated drained continuation requires inspection")
        state.write(run / ".xnode" / "guardian-status.json", {
            "status": "drained", "pid": os.getpid(), "lease_sha256": state.digest(lease),
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--bindings-sha256")
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--lease", type=Path, required=True)
    parser.add_argument("mode", choices=("worker", "guardian"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    state.configure_bindings(args.bindings, args.bindings_sha256)
    rest = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    if args.mode == "worker":
        worker(args.runtime_manifest, args.lease, rest)
    else:
        child = argparse.ArgumentParser()
        child.add_argument("--spec", type=Path, required=True)
        guardian(args.runtime_manifest, args.lease, child.parse_args(rest).spec)


if __name__ == "__main__":
    main()
