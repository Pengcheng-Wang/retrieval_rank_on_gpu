"""P5-specific sealed full-state checkpoints and exact serial sampling."""

from __future__ import annotations

import copy
import math
import os
import random
import shutil
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from p5_source import digest_json, read_json, require_hash, sha256, write_json


_METADATA_CACHE = OrderedDict()


@contextmanager
def pointer_lock(run):
    """Keep this lock inode stable across every publisher and recovery reader."""
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    with (run / ".p5-pointer.lock").open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            if lock.seek(0, os.SEEK_END) == 0:
                lock.write(b"\0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


def fsync_directory(path):
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class IndexSampler:
    def __init__(self, size, seed):
        if size <= 0:
            raise ValueError("Empty native training support")
        self.size, self.epoch, self.cursor = size, 0, 0
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(size, generator=self.generator)

    def next(self, count):
        if self.cursor == self.size:
            return []
        indices = self.order[self.cursor:min(self.cursor + count, self.size)].tolist()
        self.cursor += len(indices)
        return indices

    def advance(self):
        if self.cursor != self.size:
            raise ValueError("Cannot omit the unfinished tail of an epoch")
        self.epoch += 1
        self.cursor = 0
        self.order = torch.randperm(self.size, generator=self.generator)

    def state_dict(self):
        return {
            "size": self.size, "epoch": self.epoch, "cursor": self.cursor,
            "generator": self.generator.get_state(), "order": self.order.clone(),
        }

    def load_state_dict(self, state):
        if state["size"] != self.size or not 0 <= state["cursor"] <= self.size:
            raise ValueError("Checkpoint sampler size/cursor differs")
        order = state["order"]
        if order.dtype != torch.int64 or not torch.equal(torch.sort(order).values, torch.arange(self.size)):
            raise ValueError("Checkpoint sampler is not a complete no-replacement permutation")
        self.epoch, self.cursor, self.order = state["epoch"], state["cursor"], order.clone()
        self.generator.set_state(state["generator"])


def rng_state():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Exact P5 resume requires the same visible GPU count")
        torch.cuda.set_rng_state_all(state["cuda"])


def verify_checkpoint(path, contract=None):
    path = Path(path)
    seal = read_json(path / "p5-seal.json")
    if (seal.get("kind") != "p5-full-state-v1" or seal.get("status") != "sealed"
            or set(seal["files_sha256"]) != {"model.safetensors", "training-state.pt"}):
        raise ValueError("Not a sealed full-state P5 artifact")
    if contract is not None and seal["contract_sha256"] != digest_json(contract):
        raise ValueError("P5 checkpoint belongs to different code/data/recipe/runtime")
    for name, digest in seal["files_sha256"].items():
        if (path / name).is_symlink():
            raise ValueError("P5 artifacts may not contain symlinks")
        require_hash(path / name, digest)
    return seal


def _pointer_path(run, pointer="latest"):
    run = Path(run)
    data = read_json(run / (pointer + ".json"))
    name = data["artifact"]
    if Path(name).name != name or not name.startswith("p5-state-"):
        raise ValueError("Unsafe checkpoint pointer")
    path = run / "states" / name
    require_hash(path / "p5-seal.json", data["seal_sha256"])
    return path


def _metadata(path, contract=None):
    path = Path(path)
    marker = path / "p5-seal.json"
    signature = tuple(
        (name, (path / name).stat().st_size, (path / name).stat().st_mtime_ns)
        for name in ("model.safetensors", "training-state.pt")
    )
    key = (str(path.resolve()), sha256(marker), signature)
    if key not in _METADATA_CACHE:
        seal = verify_checkpoint(path, contract)
        payload = torch.load(path / "training-state.pt", map_location="cpu", weights_only=False)
        if (digest_json(payload["contract"]) != seal["contract_sha256"]
                or payload["checkpoint_boundary"] !=
                "after_complete_optimizer_update_no_partial_gradients"):
            raise ValueError("Selection authority has an inconsistent sealed training contract")
        state = payload["state"]
        if any(state[name] != seal[name] for name in ("epoch", "global_step", "phase")):
            raise ValueError("Selection authority state differs from its seal")
        _METADATA_CACHE[key] = (copy.deepcopy(state), payload["contract"])
        while len(_METADATA_CACHE) > 64:
            _METADATA_CACHE.popitem(last=False)
    state, saved_contract = _METADATA_CACHE[key]
    if contract is not None and saved_contract != contract:
        raise ValueError("Selection authority belongs to a different contract")
    return state, saved_contract


def _publish_projection(path, value, authority):
    path = Path(path)
    if path.exists():
        try:
            if read_json(path) == value:
                fsync_directory(path.parent)
                return
        except (ValueError, UnicodeError):
            pass
        if not path.is_file() or path.is_symlink():
            raise ValueError("Selection projection is not a regular owned file")
        backup = path.parent / "selection-recovery" / (uuid.uuid4().hex + "-" + path.name)
        backup.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, backup)
        with backup.open("r+b") as stream:
            os.fsync(stream.fileno())
        write_json(backup.with_suffix(backup.suffix + ".receipt.json"), {
            "authority": authority, "displaced_sha256": sha256(backup), "projection": path.name,
        })
        fsync_directory(backup.parent)
    # The unchanged learner also writes validation-history.json after save returns.
    # A unique recovery staging name cannot collide with that writer's .partial.
    if path.name == "validation-history.json":
        staging = path.with_name(path.name + ".recovery-" + uuid.uuid4().hex)
        write_json(staging, value)
        staging.replace(path)
    else:
        write_json(path, value)
    fsync_directory(path.parent)


def _recover_selection_locked(run, contract=None):
    run = Path(run)
    authority_path = _pointer_path(run)
    state, saved_contract = _metadata(authority_path, contract)
    selected_name = state.get("selected_artifact")
    metric = state.get("best_metric")
    history = state.get("validation_history", [])
    if not isinstance(history, list):
        raise ValueError("Invalid sealed validation history")
    authority = {"artifact": authority_path.name,
                 "seal_sha256": sha256(authority_path / "p5-seal.json")}
    selected_file = run / "selected.json"
    if selected_name is None:
        if metric is not None or history or (run / "frozen.json").exists():
            raise ValueError("Sealed state has validation/freeze state but no selected artifact")
        if selected_file.exists():
            quarantine = run / "selection-recovery" / (uuid.uuid4().hex + "-selected.json")
            quarantine.parent.mkdir(exist_ok=True)
            selected_file.replace(quarantine)
            fsync_directory(quarantine.parent)
            fsync_directory(run)
        return None
    if (not isinstance(selected_name, str) or Path(selected_name).name != selected_name
            or not selected_name.startswith("p5-state-") or "/" in selected_name
            or "\\" in selected_name or metric is None or not math.isfinite(metric)):
        raise ValueError("Invalid sealed selection identity/metric")
    selected = run / "states" / selected_name
    if selected.is_symlink() or not selected.is_dir():
        raise ValueError("Authoritative selected artifact is missing or unsafe")
    selected_state, _ = _metadata(selected, saved_contract)
    if (selected_state.get("selected_artifact") != selected_name
            or selected_state.get("best_metric") != metric):
        raise ValueError("Selected artifact does not carry the authoritative validation decision")
    native_history = history and all(
        isinstance(entry, dict) and "recall@10" in entry.get("primary", {}) for entry in history
    )
    if native_history:
        values = [entry["primary"]["recall@10"] for entry in history]
        if not all(math.isfinite(value) for value in values) or max(values) != metric:
            raise ValueError("Saved best metric conflicts with validation history")
        selected_history = selected_state.get("validation_history")
        if (not isinstance(selected_history, list)
                or selected_history != history[:len(selected_history)]
                or len(selected_history) != values.index(metric) + 1):
            raise ValueError("Saved selection violates the frozen earliest-validation-best policy")
    selected_sha = sha256(selected / "p5-seal.json")
    if (run / "frozen.json").exists():
        frozen = read_json(run / "frozen.json")
        if (frozen["selected"] != selected_name or frozen["selected_seal_sha256"] != selected_sha
                or frozen["epochs"] != state["epoch"]
                or frozen.get("best_validation_recall10", metric) != metric):
            raise ValueError("Frozen generation conflicts with authoritative sealed selection")
    if "validation_history" in state:
        _publish_projection(run / "validation-history.json", history, authority)
    _publish_projection(selected_file, {
        "artifact": selected_name, "seal_sha256": selected_sha,
        "global_step": selected_state["global_step"], "phase": selected_state["phase"],
    }, authority)
    return selected


def recover_selection(run, contract=None):
    with pointer_lock(run):
        return _recover_selection_locked(run, contract)


def pointer_path(run, pointer="latest"):
    if pointer not in ("latest", "selected"):
        raise ValueError("Unknown P5 pointer")
    with pointer_lock(run):
        if pointer == "selected":
            selected = _recover_selection_locked(run)
            if selected is None:
                raise FileNotFoundError("No validation-selected P5 checkpoint exists yet")
            return selected
        return _pointer_path(run, pointer)


def save_checkpoint(run, model, optimizer, scheduler, sampler, state, contract, selected=False):
    with pointer_lock(run):
        return _save_checkpoint_locked(
            run, model, optimizer, scheduler, sampler, state, contract, selected,
        )


def _save_checkpoint_locked(run, model, optimizer, scheduler, sampler, state, contract, selected):
    run = Path(run)
    name = f"p5-state-{state['global_step']:09d}-{state['epoch']:03d}-{uuid.uuid4().hex[:12]}"
    final = run / "states" / name
    partial = run / "states" / (name + ".partial")
    partial.mkdir(parents=True)
    if selected:
        state["selected_artifact"] = name
    captured = rng_state()
    tensors = {key: value.detach().cpu().contiguous().clone()
               for key, value in model.state_dict().items()}
    save_file(tensors, str(partial / "model.safetensors"), metadata={"format": "pt"})
    payload = {
        "schema_version": 1, "contract": contract, "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "sampler": sampler.state_dict(),
        "rng": captured, "state": copy.deepcopy(state),
        "checkpoint_boundary": "after_complete_optimizer_update_no_partial_gradients",
    }
    torch.save(payload, partial / "training-state.pt")
    write_json(partial / "p5-seal.json", {
        "kind": "p5-full-state-v1", "status": "sealed", "contract_sha256": digest_json(contract),
        "global_step": state["global_step"], "epoch": state["epoch"], "phase": state["phase"],
        "files_sha256": {
            file: sha256(partial / file) for file in ("model.safetensors", "training-state.pt")
        },
    })
    for filename in ("model.safetensors", "training-state.pt"):
        with (partial / filename).open("r+b") as stream:
            os.fsync(stream.fileno())
    fsync_directory(partial)
    partial.replace(final)
    fsync_directory(final.parent)
    seal = verify_checkpoint(final, contract)
    pointer = {"artifact": name, "seal_sha256": sha256(final / "p5-seal.json"),
               "global_step": state["global_step"], "phase": state["phase"]}
    write_json(run / "latest.json", pointer)
    fsync_directory(run)
    if selected:
        write_json(run / "selected.json", pointer)
        fsync_directory(run)
    return final, seal


def restore_checkpoint(path, model, optimizer, scheduler, sampler, contract):
    path = Path(path)
    run = path.parent.parent
    with pointer_lock(run):
        if (run / "latest.json").exists() and _pointer_path(run) != path:
            raise ValueError("Refusing to restore a different generation than committed latest")
        state, captured_rng = _restore_checkpoint_locked(
            path, model, optimizer, scheduler, sampler, contract,
        )
        if (run / "latest.json").exists():
            _recover_selection_locked(run, contract)
        restore_rng(captured_rng)
        return state


def _restore_checkpoint_locked(path, model, optimizer, scheduler, sampler, contract):
    verify_checkpoint(path, contract)
    model.load_state_dict(load_file(str(Path(path) / "model.safetensors")), strict=True)
    # Only our own sealed, SHA-verified training state is unpickled; never downloaded model pickle.
    payload = torch.load(Path(path) / "training-state.pt", map_location="cpu", weights_only=False)
    if payload["contract"] != contract or payload["checkpoint_boundary"] != (
        "after_complete_optimizer_update_no_partial_gradients"
    ):
        raise ValueError("Not an exact optimizer-boundary P5 continuation")
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    sampler.load_state_dict(payload["sampler"])
    return payload["state"], payload["rng"]
