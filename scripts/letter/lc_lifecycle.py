"""CPU-only LC final-run/evaluation banking and a durable, gate-ordered continuation queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path


POLICIES = ("author-position-sets", "full-prefix-trie-correction")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + f".writing-{os.getpid()}")
    staging.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    staging.replace(path)


def require_hash(path, expected):
    if sha(path) != expected:
        raise ValueError(f"LC lifecycle checksum mismatch: {path}")


def _bundle_inventory(directory):
    result = {}
    for path in sorted(Path(directory).rglob("*")):
        if path.is_symlink():
            raise ValueError("Bundle symlinks are forbidden")
        if path.is_file() and path.name not in {"bundle.json", "bank-receipt.json"}:
            result[path.relative_to(directory).as_posix()] = sha(path)
    return result


def publish_bundle(output, kind, identity, files):
    output = Path(output)
    expected = {}
    for name, source in files.items():
        if Path(name).is_absolute() or ".." in Path(name).parts or Path(source).is_symlink():
            raise ValueError("Unsafe immutable bundle member")
        expected[name] = sha(source)
    if output.exists():
        existing = verify_bundle(output, kind)
        if existing["identity"] != identity or existing["files"] != expected:
            raise ValueError("Existing bundle is immutable and differs from its source")
        return existing
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".building-" + uuid.uuid4().hex)
    staging.mkdir()
    for name, source in files.items():
        target = staging / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        require_hash(target, expected[name])
        require_hash(source, expected[name])
    manifest = {"schema_version": 1, "kind": kind, "status": "sealed",
                "identity": identity, "files": expected}
    write(staging / "bundle.json", manifest)
    verify_bundle(staging, kind)
    staging.rename(output)
    return manifest


def verify_bundle(directory, kind=None):
    directory = Path(directory)
    manifest = read(directory / "bundle.json")
    if (manifest.get("schema_version") != 1 or manifest.get("status") != "sealed"
            or (kind is not None and manifest.get("kind") != kind)):
        raise ValueError("Wrong or incomplete LC bundle kind")
    if _bundle_inventory(directory) != manifest["files"]:
        raise ValueError("LC bundle checksum or file inventory mismatch")
    return manifest


def verify_bundle_ack(directory, acknowledgment, kind):
    manifest = verify_bundle(directory, kind)
    if (acknowledgment.get("kind") != "lc-bundle-bank"
            or acknowledgment.get("bundle_kind") != kind
            or acknowledgment.get("status") != "verified"
            or acknowledgment.get("bundle_sha256") != sha(Path(directory) / "bundle.json")
            or acknowledgment.get("identity") != manifest["identity"]):
        raise ValueError("A checkpoint acknowledgment is not a verified final-run/evaluation bundle receipt")


def final_run_identity(run):
    from lc_checkpoint import snapshot_identity, verify, verify_ack

    run = Path(run)
    config, completed = read(run / "run-config.json"), read(run / "completed.json")
    epochs = config["recipe"]["epochs"]
    updates = config["geometry"]["updates_per_epoch"]
    total = epochs * updates
    if (config.get("kind") != "full-fit" or completed.get("kind") != "full-fit"
            or completed.get("status") != "complete" or completed.get("global_step") != total
            or completed.get("epoch") != epochs):
        raise ValueError("Final-run banking requires the entire declared training budget")
    config_sha = sha(run / "run-config.json")
    if completed.get("run_config_sha256") != config_sha:
        raise ValueError("Root completion/config checksum mismatch")
    calls = completed["validation_calls"]
    if (len(calls) != epochs or [call["global_step"] for call in calls] != [
            updates * epoch for epoch in range(1, epochs + 1)]
            or any(not math.isfinite(call["eval_loss"]) for call in calls)):
        raise ValueError("Final-run validation schedule is incomplete")
    selected_step = min(calls, key=lambda call: (call["eval_loss"], call["global_step"]))["global_step"]
    selected_name = f"checkpoint-{selected_step}"
    if completed["selected_checkpoint"] != selected_name:
        raise ValueError("Root completion did not freeze the validation-only checkpoint selection")
    identities = {}
    for label, step in (("selected", selected_step), ("final", total)):
        directory = run / "snapshots" / f"checkpoint-{step}"
        receipt = verify(directory)
        if receipt["global_step"] != step or receipt["files"]["run-config.json"] != config_sha:
            raise ValueError("Selected/final snapshot differs from the completed run")
        acknowledgment = read(run / "acks" / f"checkpoint-{step}.json")
        verify_ack(directory, acknowledgment)
        identities[label] = snapshot_identity(directory)
    if completed["selected_complete_sha256"] != identities["selected"]["complete_sha256"]:
        raise ValueError("Selected snapshot changed after root completion")
    return {
        "run_config_sha256": config_sha, "run_completed_sha256": sha(run / "completed.json"),
        "dataset": config["inputs"]["dataset"], "global_step": total, "epochs": epochs,
        "expected_test_users": config["inputs"]["validation_examples"]
                               // config["inputs"]["format"]["validation_prompts"],
        **identities,
    }


def freeze_final_run(run, output):
    run = Path(run)
    identity = final_run_identity(run)
    files = {
        "run/completed.json": run / "completed.json", "run/run-config.json": run / "run-config.json",
        "run/runtime.json": run / "runtime.json", "run/loader.json": run / "loader" / "run-config.json",
    }
    for label in ("selected", "final"):
        name = f'checkpoint-{identity[label]["global_step"]}'
        source = run / "snapshots" / name
        for filename in ("complete.json", "trainer_state.json", "callback-state.json"):
            files[f"{label}/{filename}"] = source / filename
        files[f"{label}/ack.json"] = run / "acks" / f"{name}.json"
    return publish_bundle(output, "final-run", identity, files)


def authorize_evaluation(run, bundle, bank_ack, policy):
    if policy not in POLICIES:
        raise ValueError("Undeclared evaluation policy")
    verify_bundle_ack(bundle, bank_ack, "final-run")
    identity = verify_bundle(bundle, "final-run")["identity"]
    require_hash(Path(run) / "completed.json", identity["run_completed_sha256"])
    require_hash(Path(run) / "run-config.json", identity["run_config_sha256"])
    if final_run_identity(run) != identity:
        raise ValueError("Final-run completion/selection changed before test")
    return identity


def freeze_evaluation(run, evaluation, output, policy, final_ack, job_dir=None, failed=False):
    run, evaluation = Path(run), Path(evaluation)
    final_identity = final_run_identity(run)
    authorization = read(final_ack)
    if (authorization.get("kind") != "lc-bundle-bank" or authorization.get("bundle_kind") != "final-run"
            or authorization.get("status") != "verified" or authorization.get("identity") != final_identity):
        raise ValueError("Evaluation bundle requires its separate verified final-run bank receipt")
    files = {}
    if evaluation.exists():
        files.update({
            "evaluation/" + path.relative_to(evaluation).as_posix(): path
            for path in evaluation.rglob("*") if path.is_file()
        })
    if job_dir is not None:
        job_dir = Path(job_dir)
        files.update({"attempt/" + path.name: path for path in job_dir.iterdir() if path.is_file()})
    files["authorization/final-run-bank.json"] = Path(final_ack)
    if not failed:
        completed = read(evaluation / "completed.json")
        config = read(evaluation / "run-config.json")
        if (completed.get("status") != "complete" or completed.get("decoder_policy") != policy
                or completed.get("users") != final_identity["expected_test_users"]
                or config.get("run_config_sha256") != final_identity["run_config_sha256"]
                or config.get("decoder_policy") != policy or config.get("num_beams") != 20
                or config.get("drop_last") is not False or config.get("seen_item_exclusion") is not False):
            raise ValueError("Evaluation did not cover the frozen run and declared policy")
        require_hash(evaluation / "run-config.json", completed["run_config_sha256"])
        users = []
        for rank in range(config["world_size"]):
            path = evaluation / f"predictions-rank-{rank}.jsonl"
            marker = read(evaluation / f"completed-rank-{rank}.json")
            require_hash(path, marker["sha256"])
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            if marker["status"] != "complete" or len(rows) != marker["users"]:
                raise ValueError("Incomplete per-rank evaluation records")
            users.extend(row["user"] for row in rows)
            if any(len(row["raw_codes"]) != 20 or len(row["raw_scores"]) != 20
                   or any(not math.isfinite(score) for score in row["raw_scores"]) for row in rows):
                raise ValueError("Evaluation lost fixed-beam raw finite code scores")
        if len(users) != final_identity["expected_test_users"] or len(set(users)) != len(users):
            raise ValueError("Evaluation user records are missing or duplicated")
    identity = {
        "run_config_sha256": final_identity["run_config_sha256"],
        "run_completed_sha256": final_identity["run_completed_sha256"],
        "selected": final_identity["selected"], "policy": policy,
        "outcome": "failed-preserved" if failed else "complete",
        "final_run_bank_sha256": sha(final_ack),
    }
    return publish_bundle(output, "failed-evaluation" if failed else "evaluation", identity, files)


def bank_bundle(source, destination, lock_file, writer_host):
    import fcntl

    if socket.gethostname() != writer_host:
        raise RuntimeError("Only the designated single writer may persist LC bundles")
    source, destination, lock_file = Path(source), Path(destination), Path(lock_file)
    manifest = verify_bundle(source)
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with lock_file.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            ack = read(destination / "bank-receipt.json")
            verify_bundle_ack(source, ack, manifest["kind"])
            verify_bundle_ack(destination, ack, manifest["kind"])
            return ack
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.with_name(destination.name + ".building-" + uuid.uuid4().hex)
        shutil.copytree(source, staging)
        if verify_bundle(staging) != manifest or verify_bundle(source) != manifest:
            raise ValueError("LC bundle changed during bank transfer")
        ack = {
            "kind": "lc-bundle-bank", "bundle_kind": manifest["kind"], "status": "verified",
            "bundle_sha256": sha(source / "bundle.json"), "identity": manifest["identity"],
            "writer_host": writer_host, "bank_directory": str(destination),
            "verified_unix": time.time(), "controller_source_sha256": sha(Path(__file__)),
        }
        write(staging / "bank-receipt.json", ack)
        staging.rename(destination)
    return ack


def require_queue_gate(stage, facts):
    instruments = (facts.get("instruments_complete") and facts.get("instruments_exited")
                   and facts.get("instruments_final_banked"))
    evaluated = set(facts.get("instruments_evaluations_banked", [])) == set(POLICIES)
    ready_yelp = instruments and evaluated and facts.get("yelp_cpu_pass") and facts.get("gpus_free")
    allowed = {
        "instruments-evaluate": instruments and facts.get("gpus_free"),
        "yelp-gpu-gate": ready_yelp,
        "yelp-fit": ready_yelp and facts.get("yelp_gpu_pass"),
        "yelp-evaluate": (ready_yelp and facts.get("yelp_gpu_pass") and facts.get("yelp_complete")
                         and facts.get("yelp_exited") and facts.get("yelp_final_banked")),
    }
    if stage not in allowed or not allowed[stage]:
        raise ValueError(f"LC queue prerequisites are not met: {stage}")


def reserve_job(directory, spec):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "reservation.json").open("x", encoding="utf-8") as stream:
        json.dump(spec, stream, sort_keys=True, indent=2)
        stream.write("\n")


def process_alive(pid):
    try:
        stat = Path(f"/proc/{int(pid)}/stat")
        if stat.is_file() and stat.read_text().split(") ", 1)[1].split()[0] == "Z":
            return False
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, FileNotFoundError, ValueError):
        return False


def run_alive(run):
    for filename in ("supervisor.pid", "torchrun.pid"):
        path = Path(run) / filename
        if path.is_file() and process_alive(path.read_text().strip()):
            return True
    return False


def assigned_gpus_free():
    query = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits",
    ], text=True)
    allocated = {line.split(",")[1].strip() for line in query.splitlines()
                 if int(line.split(",")[0]) in (0, 1, 2, 3)}
    if len(allocated) != 4:
        raise RuntimeError("Expected the four explicitly assigned devices")
    applications = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits",
    ], text=True)
    return not any(line.split(",")[-1].strip() in allocated
                   for line in applications.splitlines() if "," in line)


def verify_sources(plan):
    for name, digest in plan["frozen_sources_sha256"].items():
        require_hash(Path(plan["frozen_source"]) / name, digest)


def paths(plan, case):
    cell = plan[case]
    control = Path(plan["control_dir"])
    return {
        "run": Path(cell["run"]), "bundle": control / "bundles" / f"{case}-final",
        "final_ack": control / "receipts" / f"{case}-final-bank.json",
        "evaluations": {p: Path(cell["evaluations"]) / p for p in POLICIES},
        "eval_acks": {p: control / "receipts" / f"{case}-{p}-bank.json" for p in POLICIES},
    }


def job_status(plan, name):
    directory = Path(plan["control_dir"]) / "jobs" / name
    if not directory.exists():
        return {"status": "absent"}
    if (directory / "exit.json").is_file():
        return read(directory / "exit.json")
    if (directory / "running.json").is_file():
        state = read(directory / "running.json")
        if process_alive(state["pid"]) or process_alive(state["child_pid"]):
            return {"status": "running", **state}
    return {"status": "lost-or-reserved", "directory": str(directory)}


def worker_facts(plan, include_gpus=False):
    facts = {}
    for case in ("instruments", "yelp"):
        locations = paths(plan, case)
        run = locations["run"]
        facts[case + "_complete"] = (run / "completed.json").is_file()
        facts[case + "_exited"] = not run_alive(run)
        facts[case + "_live"] = read(run / "live.json") if (run / "live.json").is_file() else None
        final_banked = locations["final_ack"].is_file()
        if final_banked:
            ack = read(locations["final_ack"])
            verify_bundle_ack(locations["bundle"], ack, "final-run")
            if ack["writer_host"] != plan["writer_host"]:
                raise ValueError("Final-run receipt names a different writer")
            require_hash(run / "completed.json", ack["identity"]["run_completed_sha256"])
        facts[case + "_final_banked"] = final_banked
        evaluation_banks = []
        for suffix, policy in zip(("author", "trie"), POLICIES):
            ack_path = locations["eval_acks"][policy]
            if ack_path.is_file():
                ack = read(ack_path)
                directory = Path(plan["control_dir"]) / "bundles" / (case + "-" + suffix + "-evaluation")
                verify_bundle_ack(directory, ack, "evaluation")
                if ack["identity"]["policy"] != policy or ack["writer_host"] != plan["writer_host"]:
                    raise ValueError("Evaluation bank receipt names the wrong policy/writer")
                evaluation_banks.append(policy)
        facts[case + "_evaluations_banked"] = evaluation_banks
    cpu = Path(plan["yelp"]["cpu_receipt"])
    admission = Path(plan["yelp"]["gate"]) / "admission.json"
    facts["yelp_cpu_pass"] = cpu.is_file() and read(cpu).get("status") == "pass"
    facts["yelp_gpu_pass"] = admission.is_file() and read(admission).get("status") == "pass"
    facts["gpus_free"] = assigned_gpus_free() if include_gpus else False
    facts["jobs"] = {name: job_status(plan, name) for name in (
        "instruments-author", "instruments-trie", "yelp-gate", "yelp-fit", "yelp-author", "yelp-trie",
    )}
    return facts


def input_arguments(plan, case):
    cell = plan[case]
    result = []
    for key in ("upstream", "manifest", "method_spec", "model", "model_receipt", "audit", "execution_spec"):
        result.extend(["--" + key.replace("_", "-"), str(cell[key])])
    result.extend(["--dataset", "Instruments" if case == "instruments" else "Yelp"])
    return result


def make_job(plan, name):
    directory = Path(plan["control_dir"]) / "jobs" / name
    source = Path(plan["frozen_source"])
    case = "instruments" if name.startswith("instruments-") else "yelp"
    facts = worker_facts(plan, include_gpus=True)
    cell, locations = plan[case], paths(plan, case)
    if name.endswith(("-author", "-trie")):
        require_queue_gate(case + "-evaluate", facts)
        policy = POLICIES[0] if name.endswith("-author") else POLICIES[1]
        ack = read(locations["final_ack"])
        if ack["writer_host"] != plan["writer_host"]:
            raise ValueError("Final-run bundle was not banked by the designated writer")
        authorize_evaluation(locations["run"], locations["bundle"], ack, policy)
        script = source / "lc_evaluate.py" if case == "instruments" else Path(plan["yelp"]["entry"])
        command = ["torchrun", "--nproc_per_node=4", "--master_port", str(24000 + os.getpid() % 15000),
                   "--log_dir", str(directory / "elastic"), str(script)]
        if case == "yelp":
            command += ["evaluate", "--frozen-source", str(source)]
        command += input_arguments(plan, case) + [
            "--run", str(locations["run"]), "--output", str(locations["evaluations"][policy]),
            "--decoder-policy", policy, "--batch-size", "1",
        ]
        kind = "evaluation"
    elif name == "yelp-gate":
        require_queue_gate("yelp-gpu-gate", facts)
        command = ["python", plan["yelp"]["entry"], "gate-sequence",
                   "--frozen-source", str(source), "--gate", plan["yelp"]["gate"],
                   "--cpu-receipt", plan["yelp"]["cpu_receipt"],
                   "--env-script", str(source / "env.sh")] + input_arguments(plan, "yelp")
        kind = "gate"
    elif name == "yelp-fit":
        require_queue_gate("yelp-fit", facts)
        command = ["python", plan["yelp"]["entry"], "supervise-fit",
                   "--frozen-source", str(source), "--output", cell["run"],
                   "--gate-report", str(Path(cell["gate"]) / "admission.json"),
                   "--env-script", str(source / "env.sh")] + input_arguments(plan, "yelp")
        kind = "fit"
    else:
        raise ValueError("Unknown GPU queue entry")
    return {"kind": kind, "case": case, "name": name, "cuda": "0,1,2,3",
            "cwd": plan["project"], "argv": ["bash", str(source / "env.sh")] + command}


def execute_job(plan, name):
    directory = Path(plan["control_dir"]) / "jobs" / name
    spec = read(directory / "reservation.json")
    if (directory / "running.json").exists() or (directory / "exit.json").exists():
        raise FileExistsError("GPU attempts are single-use")
    verify_sources(plan)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=spec["cuda"])
    with (directory / "stdout.log").open("x", encoding="utf-8") as stream:
        child = subprocess.Popen(spec["argv"], cwd=spec["cwd"], env=env,
                                 stdout=stream, stderr=subprocess.STDOUT)
        write(directory / "running.json", {
            "pid": os.getpid(), "child_pid": child.pid, "started_unix": time.time(),
        })
        code = child.wait()
    write(directory / "exit.json", {
        "status": "succeeded" if code == 0 else "failed", "exit_code": code,
        "finished_unix": time.time(), "pid": os.getpid(), "child_pid": child.pid,
        "log_sha256": sha(directory / "stdout.log"),
    })


def spawn_job(plan, plan_path, name):
    existing = job_status(plan, name)
    if existing["status"] != "absent":
        return existing
    if not assigned_gpus_free():
        return {"status": "waiting-for-assigned-gpus", "name": name}
    spec = make_job(plan, name)
    directory = Path(plan["control_dir"]) / "jobs" / name
    reserve_job(directory, spec)
    command = shlex.join([
        "env", "CUDA_VISIBLE_DEVICES=-1", "bash", str(Path(plan["frozen_source"]) / "env.sh"),
        "python", str(Path(__file__).resolve()), "execute", "--plan", str(plan_path), "--job", name,
    ])
    subprocess.run(["tmux", "new-session", "-d", "-s", plan["session_prefix"] + "-" + name, command],
                   check=True)
    return {"status": "reserved", "name": name}


def worker(plan, plan_path, action, case=None, name=None, failed=False):
    verify_sources(plan)
    if action == "status":
        return worker_facts(plan)
    if action == "spawn":
        return spawn_job(plan, plan_path, name)
    locations = paths(plan, case)
    if action == "finalize":
        if run_alive(locations["run"]):
            raise ValueError("Learner must exit before final-run freezing")
        return freeze_final_run(locations["run"], locations["bundle"])
    if action == "evaluation-bundle":
        policy = POLICIES[0] if name.endswith("-author") else POLICIES[1]
        job_dir = Path(plan["control_dir"]) / "jobs" / name
        if job_status(plan, name)["status"] == "running":
            raise ValueError("Cannot bank an active evaluation attempt")
        output = Path(plan["control_dir"]) / "bundles" / (
            name + ("-failed" if failed else "-evaluation"))
        return freeze_evaluation(locations["run"], locations["evaluations"][policy], output, policy,
                                 locations["final_ack"], job_dir, failed=failed)
    if action == "gate-bundle":
        gate = Path(plan["yelp"]["gate"])
        if read(gate / "admission.json")["status"] != "pass":
            raise ValueError("Yelp real GPU gate has not passed")
        files = {
            path.relative_to(gate).as_posix(): path for path in gate.rglob("*.json")
            if not any(part.startswith(("checkpoint-", "snapshots")) for part in path.relative_to(gate).parts)
        }
        return publish_bundle(Path(plan["control_dir"]) / "bundles" / "yelp-gate", "gate-evidence",
                              {"admission_sha256": sha(gate / "admission.json")}, files)
    if action == "failed-gate-bundle":
        gate = Path(plan["yelp"]["gate"])
        files = {path.relative_to(gate).as_posix(): path for pattern in ("*.json", "*.log")
                 for path in gate.rglob(pattern)
                 if not any(part.startswith(("checkpoint-", "snapshots")) for part in path.relative_to(gate).parts)}
        directory = Path(plan["control_dir"]) / "jobs" / "yelp-gate"
        files.update({"attempt/" + path.name: path for path in directory.iterdir() if path.is_file()})
        return publish_bundle(Path(plan["control_dir"]) / "bundles" / "yelp-gate-failed",
                              "failed-gpu-gate", {"status": "failed-preserved"}, files)
    raise ValueError("Unknown LC worker control action")


def rpc(plan, plan_path, action, **kwargs):
    argv = [
        "env", "CUDA_VISIBLE_DEVICES=-1", "bash", str(Path(plan["frozen_source"]) / "env.sh"),
        "python", plan["worker_entry"], "worker", "--plan", str(plan_path), "--action", action,
    ]
    for key, value in kwargs.items():
        if value is True:
            argv.append("--" + key.replace("_", "-"))
        elif value is not None:
            argv.extend(["--" + key.replace("_", "-"), str(value)])
    result = subprocess.check_output(["ssh", "-n", "-o", "BatchMode=yes", plan["worker_host"],
                                      shlex.join(argv)], text=True)
    return json.loads(result)


def return_ack(plan, path, ack):
    import base64

    encoded = base64.b64encode((json.dumps(ack) + "\n").encode()).decode()
    target = shlex.quote(str(path))
    command = (f"mkdir -p {shlex.quote(str(Path(path).parent))}; "
               f"printf '%s' '{encoded}' | base64 -d > {target}.writing; mv {target}.writing {target}")
    subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", plan["worker_host"], command], check=True)


def pull_bank(plan, bundle_name, destination_name, acknowledgment_path):
    source = Path(plan["control_dir"]) / "bundles" / bundle_name
    inbox = Path(plan["writer_state"]) / "inbox" / bundle_name
    inbox.parent.mkdir(parents=True, exist_ok=True)
    if not inbox.exists():
        staging = inbox.with_name(inbox.name + ".incoming")
        subprocess.run(["rsync", "-a", "--", f'{plan["worker_host"]}:{source}/', str(staging) + "/"],
                       check=True)
        verify_bundle(staging)
        staging.rename(inbox)
    ack = bank_bundle(inbox, Path(plan["bank_root"]) / destination_name,
                      Path(plan["bank_lock"]), plan["writer_host"])
    return_ack(plan, acknowledgment_path, ack)
    write(Path(plan["writer_state"]) / "receipts" / (bundle_name + ".json"), ack)
    return ack


def start_yelp_broker(plan):
    state = Path(plan["writer_state"])
    note = state / "yelp-broker.json"
    if note.exists() and process_alive(read(note)["pid"]):
        return
    source = Path(plan["frozen_source"])
    command = [
        "bash", str(source / "env.sh"), "python", str(source / "lc_checkpoint.py"), "broker",
        "--source-host", plan["worker_host"], "--source", str(Path(plan["yelp"]["run"]) / "snapshots"),
        "--inbox", str(state / "yelp-checkpoints"), "--destination", plan["yelp"]["checkpoint_bank"],
        "--lock-file", plan["bank_lock"], "--writer-host", plan["writer_host"],
    ]
    stream = (state / "yelp-broker.log").open("a", encoding="utf-8")
    child = subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"),
                             stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    stream.close()
    write(note, {"pid": child.pid, "started_unix": time.time(), "argv": command})


def _handle_case(plan, plan_path, facts, case):
    locations = paths(plan, case)
    if not facts[case + "_complete"] or not facts[case + "_exited"]:
        return "waiting-" + case + "-learner"
    if not facts[case + "_final_banked"]:
        rpc(plan, plan_path, "finalize", case=case)
        pull_bank(plan, case + "-final", "final-runs/" + case, locations["final_ack"])
        return "banked-" + case + "-final-run"
    for suffix, policy in zip(("author", "trie"), POLICIES):
        name = case + "-" + suffix
        if policy in facts[case + "_evaluations_banked"]:
            continue
        state = facts["jobs"][name]
        if state["status"] == "absent":
            result = rpc(plan, plan_path, "spawn", name=name)
            return result["status"] + "-" + name
        if state["status"] in ("running", "reserved"):
            return "waiting-" + name
        if state["status"] == "succeeded":
            rpc(plan, plan_path, "evaluation-bundle", case=case, name=name)
            pull_bank(plan, name + "-evaluation", "evaluations/" + name, locations["eval_acks"][policy])
            return "banked-" + name
        if state["status"] in ("failed", "lost-or-reserved"):
            rpc(plan, plan_path, "evaluation-bundle", case=case, name=name, failed=True)
            pull_bank(plan, name + "-failed", "failed-attempts/" + name,
                      Path(plan["control_dir"]) / "receipts" / (name + "-failed-bank.json"))
        raise RuntimeError(f"Single-use evaluation cannot be retried: {name}: {state}")
    return case + "-complete-and-banked"


def tick(plan, plan_path):
    facts = rpc(plan, plan_path, "status")
    instruments = _handle_case(plan, plan_path, facts, "instruments")
    if instruments != "instruments-complete-and-banked":
        return instruments, facts
    if not facts["yelp_cpu_pass"]:
        return "yelp-queued-awaiting-cpu-preparation", facts
    gate_job = facts["jobs"]["yelp-gate"]
    if gate_job["status"] == "absent":
        result = rpc(plan, plan_path, "spawn", name="yelp-gate")
        return result["status"] + "-yelp-gpu-gate", facts
    if gate_job["status"] in ("running", "reserved"):
        return "waiting-yelp-gpu-gate", facts
    if gate_job["status"] != "succeeded" or not facts["yelp_gpu_pass"]:
        rpc(plan, plan_path, "failed-gate-bundle", case="yelp")
        pull_bank(plan, "yelp-gate-failed", "failed-attempts/yelp-gate",
                  Path(plan["control_dir"]) / "receipts" / "yelp-gate-failed-bank.json")
        raise RuntimeError("Yelp GPU gate failed or was lost; full fitting is not admitted")
    gate_ack = Path(plan["writer_state"]) / "receipts" / "yelp-gate.json"
    if not gate_ack.exists():
        rpc(plan, plan_path, "gate-bundle", case="yelp")
        pull_bank(plan, "yelp-gate", "gates/yelp", Path(plan["control_dir"]) / "receipts" / "yelp-gate-bank.json")
        return "banked-yelp-gpu-gate", facts
    fit_job = facts["jobs"]["yelp-fit"]
    if fit_job["status"] == "absent":
        start_yelp_broker(plan)
        time.sleep(20)
        result = rpc(plan, plan_path, "spawn", name="yelp-fit")
        return result["status"] + "-yelp-fit", facts
    if fit_job["status"] in ("failed", "lost-or-reserved"):
        raise RuntimeError("Yelp fit exited unsuccessfully; preserve checkpoint and failure evidence")
    result = _handle_case(plan, plan_path, facts, "yelp")
    return ("all-complete-and-banked" if result == "yelp-complete-and-banked" else result), facts


def drive(plan, plan_path):
    import fcntl
    import traceback

    if socket.gethostname() != plan["writer_host"] or os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("Continuation controller must be CPU-only on the designated writer")
    state = Path(plan["writer_state"])
    state.mkdir(parents=True, exist_ok=True)
    with (state / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(state / "controller.pid.json", {
            "pid": os.getpid(), "plan_sha256": sha(plan_path), "source_sha256": sha(Path(__file__)),
            "started_unix": time.time(), "cuda_visible_devices": "-1",
        })
        identity, last = sha(plan_path), None
        while True:
            require_hash(plan_path, identity)
            try:
                phase, facts = tick(plan, plan_path)
                status = {"status": "done" if phase == "all-complete-and-banked" else "active",
                          "phase": phase, "facts": facts, "pid": os.getpid(), "updated_unix": time.time()}
                write(state / "state.json", status)
                if phase != last:
                    print(json.dumps(status), flush=True)
                    last = phase
                if status["status"] == "done":
                    write(state / "completed.json", status)
                    return
            except Exception as error:
                status = {"status": "blocked", "error": str(error), "traceback": traceback.format_exc(),
                          "pid": os.getpid(), "updated_unix": time.time()}
                write(state / "state.json", status)
                print(json.dumps(status), flush=True)
                return
            time.sleep(plan.get("poll_seconds", 30))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("drive", "worker", "execute"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--action")
    parser.add_argument("--case", choices=("instruments", "yelp"))
    parser.add_argument("--name")
    parser.add_argument("--job")
    parser.add_argument("--failed", action="store_true")
    args = parser.parse_args()
    plan = read(args.plan)
    sys.path.insert(0, plan["frozen_source"])
    if args.mode == "drive":
        drive(plan, args.plan)
    elif args.mode == "execute":
        execute_job(plan, args.job)
    else:
        print(json.dumps(worker(plan, args.plan, args.action, args.case, args.name, args.failed)), flush=True)


if __name__ == "__main__":
    main()
