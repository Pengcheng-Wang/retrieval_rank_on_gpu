"""Complete BIGRec adapter state, not disguised full-model checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

from protocol import file_sha256, require_hash, write_json


REQUIRED = (
    "adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "scheduler.pt",
    "scaler.pt", "rng_state.pth", "trainer_state.json", "continuation.json",
)


def checkpoint_files(path: Path) -> list[Path]:
    for name in REQUIRED:
        if not (path / name).is_file():
            raise ValueError(f"Incomplete BIGRec checkpoint: missing {name}")
    files = sorted(p for p in path.iterdir() if p.is_file() and p.name != "complete.json")
    if any(p.is_symlink() or p.name.endswith((".tmp", ".part", ".transfer")) for p in files):
        raise ValueError("Incomplete or linked BIGRec checkpoint files")
    if (path / "pytorch_model.bin").exists():
        raise ValueError("BIGRec adapters must not masquerade as full model weights")
    return files


def validate_position(path: Path, state: dict) -> None:
    continuation = json.loads((path / "continuation.json").read_text(encoding="utf-8"))
    if continuation.get("microbatches_per_epoch") != 256 or continuation.get("accumulation") != 32:
        raise ValueError("A complete BIGRec state requires the frozen native loader geometry")
    if continuation.get("evaluations") != state["global_step"] // 8:
        raise ValueError("Epoch-boundary validation is pending or its counter is inconsistent")
    if continuation.get("microbatches") != state["global_step"] * 32:
        raise ValueError("Saved microbatch and optimizer-step positions are inconsistent")


def seal(path: Path) -> dict:
    files = checkpoint_files(path)
    state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
    validate_position(path, state)
    receipt = {
        "kind": "bigrec_complete_adapter_optimizer_scheduler_fp16_rng_sampler_state",
        "global_step": state["global_step"], "epoch": state["epoch"],
        "files": {p.name: file_sha256(p) for p in files},
    }
    write_json(path / "complete.json", receipt)
    return receipt


def verify(path: Path) -> dict:
    files = checkpoint_files(path)
    receipt = json.loads((path / "complete.json").read_text(encoding="utf-8"))
    if set(receipt["files"]) != {p.name for p in files}:
        raise ValueError("BIGRec checkpoint checksum file-set mismatch")
    for name, digest in receipt["files"].items():
        require_hash(path / name, digest)
    state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
    validate_position(path, state)
    if state["global_step"] != receipt["global_step"]:
        raise ValueError("Checkpoint position differs from its seal")
    return receipt


def latest(run: Path) -> Path | None:
    for path in sorted(run.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]), reverse=True):
        if (path / "complete.json").is_file():
            verify(path)
            return path
    return None
