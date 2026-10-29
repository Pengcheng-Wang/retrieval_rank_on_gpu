import importlib.util
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "letter"
GUARD = SCRIPTS / "p5_maintenance.py"


def module():
    assert GUARD.exists(), "A narrowly attested immutable-base maintenance guard is required"
    spec = importlib.util.spec_from_file_location("p5_guard_under_test", GUARD)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_guard_changes_only_io_boundaries_and_retains_old_learner_pin():
    guard = module()
    assert guard.BASE_HASHES["p5_run.py"] == "6b9572e4cb3405b79c1247339bf78e21bbfb6904abf72ec30113533760c1c9b5"
    assert set(guard.PATCH_TARGETS) == {
        "p5_checkpoint.save_checkpoint", "p5_checkpoint.restore_checkpoint",
        "p5_checkpoint.pointer_path", "p5_checkpoint.write_json",
        "p5_bank.result_snapshot_and_publication", "p5_queue.command",
    }


@pytest.fixture
def workspace():
    path = ROOT / ".local" / "letter" / "p5-maintenance-tests" / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path)


def test_guard_never_falls_back_to_an_unwrapped_training_child():
    guard = module()
    original = ["python", "frozen/p5_run.py", "--output", "owned-run"]
    wrapped = guard.wrap_child_command(original, Path("base"), Path("manifest"), Path("proof"))
    assert "run" in wrapped and wrapped[-2:] == ["--output", "owned-run"]
    assert any(Path(argument).name == "p5_maintenance.py" for argument in wrapped)
    with pytest.raises(ValueError, match="unchanged learner"):
        guard.wrap_child_command(["python", "other.py"], Path("base"), Path("manifest"), Path("proof"))


def test_guard_rejects_modified_base_source(workspace):
    guard = module()
    path = workspace / "base" / "scripts" / "letter"
    path.mkdir(parents=True)
    for name in guard.BASE_HASHES:
        (path / name).write_text("not the frozen code")
    with pytest.raises(ValueError, match="base"):
        guard.verify_base(workspace / "base", "0" * 64)


def test_guard_rejects_an_incomplete_continuation_proof(workspace):
    guard = module()
    manifest = workspace / "release.json"
    manifest.write_text("{}")
    proof = workspace / "proof.json"
    proof.write_text(json.dumps({
        "status": "pass", "release_manifest_sha256": guard._sha(manifest),
        "model_optimizer_scheduler_sampler_rng_equal": True,
        "subsequent_sampled_rows_and_losses_equal": True, "scientific_contracts_unchanged": True,
        "cells": {},
    }))
    with pytest.raises(ValueError, match="continuation proof"):
        guard.verify_compatibility(manifest, proof)


def test_guarded_original_checkpoint_writer_recovers_without_rewriting_base(workspace):
    base = os.environ.get("LETTER_P5_BASE_BUNDLE")
    if not base:
        pytest.skip("Set LETTER_P5_BASE_BUNDLE to the immutable running learner")
    release = workspace / "release"
    release.mkdir()
    hashes = {}
    for name in ("p5_maintenance.py", "p5_checkpoint.py", "p5_bank.py"):
        content = (SCRIPTS / name).read_bytes().replace(b"\r\n", b"\n")
        (release / name).write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    guard = module()
    manifest = release / "maintenance-release.json"
    manifest.write_text(json.dumps({
        "kind": "p5-crash-consistency-maintenance-v1", "patch_targets": guard.PATCH_TARGETS,
        "base_source_seal_sha256": guard._sha(Path(base) / "p5-source-seal.json"),
        "files_sha256": hashes,
    }))
    code = r"""
import importlib.util,sys
from pathlib import Path
import torch
base,manifest,run=map(Path,sys.argv[1:])
spec=importlib.util.spec_from_file_location('checked_guard',manifest.parent/'p5_maintenance.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
before={n:guard._sha(base/'scripts/letter'/n) for n in guard.BASE_HASHES}
c=guard.install(base,manifest)
assert Path(c.learner.__file__).parent==base/'scripts/letter'
model=torch.nn.Linear(2,2);opt=torch.optim.AdamW(model.parameters())
scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
sampler=c.checkpoint.IndexSampler(20,42)
state={'epoch':1,'global_step':1,'phase':'train','best_metric':.1,'selected_artifact':None,
       'validation_history':[{'primary':{'recall@10':.1}}]}
contract={'guard_cpu_test':True}
first,_=c.checkpoint.save_checkpoint(run,model,opt,scheduler,sampler,state,contract,selected=True)
write=c.checkpoint.write_json
def crash(path,value):
 if Path(path).name=='selected.json':raise RuntimeError('injected old writer crash')
 write(path,value)
c.checkpoint.write_json=crash
state.update(epoch=20,global_step=20,phase='frozen',best_metric=.2)
state['validation_history']=[{'primary':{'recall@10':.1}}]*19+[{'primary':{'recall@10':.2}}]
try:c.checkpoint.save_checkpoint(run,model,opt,scheduler,sampler,state,contract,selected=True)
except RuntimeError as error:assert 'injected' in str(error)
else:raise AssertionError('fault did not occur')
c.checkpoint.write_json=write
latest=c.checkpoint.pointer_path(run)
restored=c.checkpoint.restore_checkpoint(latest,model,opt,scheduler,sampler,contract)
assert c.checkpoint.pointer_path(run,'selected').name==restored['selected_artifact']
assert latest!=first
assert before=={n:guard._sha(base/'scripts/letter'/n) for n in guard.BASE_HASHES}
print('GUARDED_ORIGINAL_WRITER_CPU_PASS')
"""
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(Path(base)), str(manifest), str(workspace / "run")],
        capture_output=True, text=True, env=environment,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "GUARDED_ORIGINAL_WRITER_CPU_PASS" in result.stdout
