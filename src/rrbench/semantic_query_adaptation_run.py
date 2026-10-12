"""Fixed query-only Qwen0.6 pilot; private explicit inputs, no allocation management.

Run phases with ``python -m rrbench.semantic_query_adaptation_run --phase PHASE
--inputs inputs.json --protocol protocol.json --admission admission.json
--output NEW_RUN_ROOT --deadline ISO_UTC``. Runtime additionally requires
``--session 0|1|2`` in three separately launched processes. The output root is
private: rendered prompts, query identities, raw results and checkpoints belong
there, never in public results. Only completed source-bound artifacts are reusable.

Admission JSON requires complete=true, decision="admitted",
primary_evidence_verified=true, protocol_sha256, sources (every file input except
the cache directory), admitted_at_utc, engineering_ready_deadline_utc,
gpu_end_utc and hard_end_utc. Dates must obey the public protocol; --deadline may
only tighten them. Paths in the input map are absolute or relative to that JSON.
Each phase writes its own immutable run-config.json/completed.json; only train
can resume, from complete adapter/AdamW/RNG step receipts. Interrupted other
phases require a new version, never in-place recertification or partial cohorts.
The output filesystem must support atomic same-filesystem hard links (use a
local execution directory, then let the parent publish verified artifacts).
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import random
import re
import time
from typing import Any
import uuid

import numpy as np
import torch

from rrbench.io import sha256
from rrbench.semantic_models import SemanticEncoder, configuration_digest
from rrbench.semantic_profile_contracts import UnsupportedMemory

QUERY_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
PHASES = ("audit", "mine", "smoke", "train", "evaluate", "runtime")
INPUT_KEYS = (
    "items", "item_receipt", "item_ids", "fit_panel", "fit_queries", "fit_query_receipt",
    "validation_panel", "validation_queries", "validation_query_receipt", "final_panel",
    "partition", "partition_receipt", "registry", "model_smoke", "cache",
)
METRICS = ("mrr@10", "ndcg@10", "recall@10", "recall@100", "recall@1000")
GIB = 1 << 30
STORED_NORMALIZATION = "FP32 unit normalization then canonicalFP16storage; MRLprefix re-normalized beforestorage."
PINNED_SPECIFICATION = "f5e7adb27413d6d36fb69787f719e505410a42fc2663461d3b809d6187df9864"
PINNED_ITEMS = "f3e17c17853b67e6e54179b391a519d81c1df6d14a9bad7835a3c0f88fd9f5da"
PROCESS_CONTEXT_ID = uuid.uuid4().hex


class MemoryGateFailure(UnsupportedMemory):
    def __init__(self, message: str, observation: dict[str, Any]) -> None:
        self.observation = observation
        super().__init__(message)


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON duplicate key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON constant: {value}")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_json_pairs,
        parse_constant=_invalid_constant,
    )
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    return value


def artifact_path(root: Path, name: str) -> Path:
    relative, windows = Path(name), PureWindowsPath(name)
    if (
        not name or not relative.parts or relative.is_absolute() or windows.is_absolute() or windows.drive
        or name.startswith(("/", "\\")) or ".." in relative.parts or ".." in windows.parts
        or "\\" in name or ":" in name
    ):
        raise ValueError("Artifact requires a relative path without traversal or another scope.")
    target = root / relative
    for path in (target, *target.parents):
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise ValueError("Artifact symlink/junction is outside the immutable output scope.")
        if path.absolute() == root.absolute():
            break
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes its output scope.")
    return target


def _publish(root: Path, name: str, writer: Callable[[Any], None]) -> Path:
    target = artifact_path(root, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_name(target.name + ".pending-" + uuid.uuid4().hex)
    try:
        descriptor = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        # A same-filesystem link publishes the complete bytes without replacing a winner.
        os.link(pending, target)
    finally:
        if pending.exists():
            pending.unlink()
    return target


def publish_json(root: Path, name: str, value: Any) -> Path:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return _publish(root, name, lambda stream: stream.write(payload))


def publish_array(root: Path, name: str, values: np.ndarray) -> Path:
    if values.dtype.hasobject:
        raise ValueError("Object arrays cannot be published as numeric pilot evidence.")
    return _publish(root, name, lambda stream: np.save(stream, values, allow_pickle=False))


def _inventory(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise ValueError("Artifact inventory cannot contain symlinks or junctions.")
        if path.is_file() and path != root / "completed.json":
            name = path.relative_to(root).as_posix()
            result[name] = sha256(artifact_path(root, name))
    return result


def checked_phase(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    receipt = read_json(root / "completed.json")
    if (
        receipt.get("complete") is not True or receipt.get("status") != "ok"
        or receipt.get("binding") != binding or read_json(root / "run-config.json") != binding
    ):
        raise ValueError("Phase completion or source binding changed.")
    actual = _inventory(root)
    if set(actual) != set(receipt["artifacts"]):
        raise ValueError("Completed artifact inventory changed.")
    if actual != receipt["artifacts"]:
        raise ValueError("Completed artifact checksum/bytes changed.")
    return receipt


def begin_phase(
    root: Path, binding: dict[str, Any], *, resumable: bool = False,
) -> dict[str, Any] | None:
    artifact_path(root, "run-config.json")
    root.mkdir(parents=True, exist_ok=True)
    config = root / "run-config.json"
    if not config.exists():
        if any(root.iterdir()):
            raise ValueError("Output collision without source binding; use a new output version.")
        publish_json(root, "run-config.json", binding)
        return None
    if read_json(config) != binding:
        raise ValueError("Output source binding differs; use a new output version.")
    if (root / "completed.json").exists():
        return checked_phase(root, binding)
    if not resumable:
        raise ValueError("Incomplete phase is not reusable; use a new output version.")
    allowed = {"run-config.json", "checkpoints"}
    if {path.name for path in root.iterdir()} - allowed:
        raise ValueError("Incomplete training artifacts are not an attested checkpoint-only resume.")
    return None


def finish_phase(root: Path, binding: dict[str, Any], details: dict[str, Any]) -> dict[str, Any]:
    if read_json(root / "run-config.json") != binding:
        raise ValueError("Cannot publish completion under a changed run binding.")
    receipt = {
        "complete": True, "status": "ok", "binding": binding,
        "artifacts": _inventory(root), "details": details,
    }
    if any(".pending-" in name or name == "failure.json" for name in receipt["artifacts"]):
        raise ValueError("Partial or failed artifacts cannot be certified complete.")
    publish_json(root, "completed.json", receipt)
    return checked_phase(root, binding)


def validate_panel(
    panel: dict[str, Any], rows: int, catalog_items: int, *, unique_text: bool = True,
) -> list[list[int]]:
    from rrbench.semantic_ir import canonical_query

    ids, texts, labels = panel["query_ids"], panel["queries"], panel["qrels"]
    if (
        not isinstance(ids, list) or len(ids) != rows or rows <= 0
        or any(not isinstance(value, str) or not value.strip() for value in ids)
        or len(set(ids)) != rows or not isinstance(texts, list) or len(texts) != rows
        or any(not isinstance(value, str) or not canonical_query(value) for value in texts)
        or set(labels) != set(ids)
        or (unique_text and len({canonical_query(value) for value in texts}) != rows)
    ):
        raise ValueError("Panel has empty, duplicate, missing or misaligned query rows.")
    groups = panel.get("evaluation_groups")
    if groups is not None and len(groups) != rows:
        raise ValueError("Evaluation populations do not align with query rows.")
    positives = []
    for row, query in enumerate(ids):
        threshold = (
            panel["binary_threshold"] if groups is None
            else panel["threshold_by_group"][groups[row]]
        )
        if type(threshold) is not int or not 1 <= threshold <= 3:
            raise ValueError("Panel relevance threshold is invalid.")
        selected = []
        for item, grade in labels[query].items():
            if (
                not isinstance(item, str) or not item.isascii() or not item.isdecimal()
                or str(int(item)) != item or not 0 <= int(item) < catalog_items
                or type(grade) is not int or not 0 <= grade <= 3
            ):
                raise ValueError("Panel has noncanonical item IDs or invalid official judgments.")
            if grade >= threshold:
                selected.append(int(item))
        if not selected:
            raise ValueError("Every selected query must retain all covered official positives.")
        positives.append(sorted(selected))
    return positives


def require_disjoint_panels(*panels: dict[str, Any]) -> None:
    from rrbench.semantic_ir import canonical_query

    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    for panel in panels:
        ids = set(panel["query_ids"])
        texts = {canonical_query(text) for text in panel["queries"]}
        if seen_ids & ids or seen_text & texts:
            raise ValueError("Fitting, validation and final cohorts must be canonically disjoint.")
        seen_ids |= ids
        seen_text |= texts


def checked_queries(
    path: Path, receipt: dict[str, Any], panel_path: Path, panel: dict[str, Any],
    spec: dict[str, Any],
) -> np.ndarray:
    from rrbench.semantic_models import validate_encoding_contract

    if (
        receipt.get("complete") is not True or receipt["model"] != spec["model"]
        or receipt["revision"] != spec["revision"] or receipt["rows"] != len(panel["query_ids"])
        or receipt["query_ids"] != panel["query_ids"] or receipt["panel_sha256"] != sha256(panel_path)
        or receipt["vectors_sha256"] != sha256(path)
        or receipt["query_instruction"] != QUERY_INSTRUCTION
    ):
        raise ValueError("Frozen query receipt/panel/model/source order does not match.")
    validate_encoding_contract(spec, receipt)
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    checked_vectors(values, len(panel["query_ids"]), max(spec["dimensions"]))
    return values


def checked_vectors(values: np.ndarray, rows: int, dimension: int) -> None:
    if values.shape != (rows, dimension) or values.dtype != np.float16:
        raise ValueError("Expected complete canonical FP16 vector shape.")
    for start in range(0, rows, 8192):
        block = np.asarray(values[start:start + 8192], dtype=np.float32)
        if not np.isfinite(block).all() or not np.allclose(
            np.linalg.norm(block, axis=1), 1, atol=0.002, rtol=0,
        ):
            raise ValueError("Canonical stored vectors must be finite and unit normalized.")


def verify_file_hashes(
    paths: dict[str, Path], expected: dict[str, str], deadline: dt.datetime | None = None,
) -> None:
    if set(paths) != set(expected):
        raise ValueError("Source checksum inventory does not cover every explicit file.")
    for name, path in paths.items():
        if deadline is not None:
            check_deadline(deadline)
        if sha256(path) != expected[name]:
            raise ValueError(f"Frozen source checksum changed during the phase: {name}")


def execution_groups(query_ids: list[str], batch: int = 16) -> list[list[int]]:
    if (
        not query_ids or len(query_ids) != len(set(query_ids)) or type(batch) is not int
        or batch <= 0 or any(not isinstance(value, str) or not value for value in query_ids)
    ):
        raise ValueError("Execution partition requires unique query IDs and a positive batch.")
    order = sorted(
        range(len(query_ids)),
        key=lambda row: (
            configuration_digest({
                "domain": "query-adaptation-groups-v1", "seed": 20261011, "query_id": query_ids[row],
            }), row,
        ),
    )
    return [order[start:start + batch] for start in range(0, len(order), batch)]


def token_audit(
    tokenizer: Any, texts: list[str], groups: list[list[int]], cap: int = 512,
    deadline: dt.datetime | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    if (
        tokenizer.padding_side != "left" or not texts or cap != 512
        or any(not isinstance(text, str) or not text.strip() for text in texts)
        or sorted(row for group in groups for row in group) != list(range(len(texts)))
        or any(not group for group in groups)
    ):
        raise ValueError("Token audit requires complete nonempty strings and exact left-padded groups.")
    lengths = np.empty(len(texts), dtype=np.int64)
    for start in range(0, len(texts), 1024):
        if deadline is not None:
            check_deadline(deadline)
        formatted = [f"Instruct: {QUERY_INSTRUCTION}\nQuery:{text}" for text in texts[start:start + 1024]]
        encoded = tokenizer(
            formatted, padding=False, truncation=False, return_length=True,
            return_attention_mask=False, return_token_type_ids=False,
        )
        actual = np.asarray(encoded["length"])
        if (
            actual.shape != (len(formatted),) or not np.issubdtype(actual.dtype, np.integer)
            or (actual <= 0).any()
        ):
            raise ValueError("Full actual strings produced invalid tokenizer lengths.")
        lengths[start:start + len(formatted)] = actual
    quantiles = np.quantile(lengths, [0.5, 0.9, 0.99])
    overflow = int((lengths > cap).sum())
    return {
        "count": len(lengths), "p50": float(quantiles[0]), "p90": float(quantiles[1]),
        "p99": float(quantiles[2]), "max": int(lengths.max()), "overflow": overflow,
        "token_cap": cap, "real_tokens": int(lengths.sum()),
        "padded_tokens": sum(
            ((int(lengths[group].max()) + 7) // 8) * 8 * len(group) for group in groups
        ),
        "strict_clean": len(lengths) - overflow, "pass": overflow == 0,
        "generation_metadata": "N/A: static contrastive embedding inputs; no thinking or finish_reason.",
        "policy": "Every full instructed string, original special tokens, no truncation or filtering.",
    }, lengths


def _retrieved_ids(values: np.ndarray, rows: int, catalog_items: int) -> None:
    if (
        values.ndim != 2 or values.shape[0] != rows or values.dtype != np.int64
        or ((values < -1) | (values >= catalog_items)).any()
    ):
        raise ValueError("Retrieved IDs have an invalid source shape/dtype/catalog mapping.")
    for candidates in values:
        valid = candidates[candidates >= 0]
        if len(valid) != len(np.unique(valid)):
            raise ValueError("Retrieved source IDs contain duplicates.")


def checked_retrieval(
    ids: np.ndarray, scores: np.ndarray, rows: int, catalog_items: int, *, allow_missing: bool = False,
) -> None:
    _retrieved_ids(ids, rows, catalog_items)
    if (
        scores.shape != ids.shape or scores.dtype != np.float32
        or np.isnan(scores).any() or np.isposinf(scores).any()
        or not np.isfinite(scores[ids >= 0]).all()
    ):
        raise ValueError("All retrieved score slots must be valid; only absent IDs may have negative infinity.")
    if not allow_missing and (ids < 0).any():
        raise ValueError("Exact retrieval cannot contain missing neighbor slots.")


def sample_negatives(
    query_ids: list[str], positives: list[list[int]], first100: np.ndarray,
    catalog_items: int, seed: int = 17,
) -> dict[str, np.ndarray]:
    if (
        len(query_ids) != len(positives) or not query_ids or len(set(query_ids)) != len(query_ids)
        or catalog_items <= 0 or seed != 17
    ):
        raise ValueError("Negative sampling requires the fixed query/source contract.")
    _retrieved_ids(first100, len(query_ids), catalog_items)
    if first100.shape[1] > 100:
        raise ValueError("Hard negative mining may only consider the frozen first100.")
    width = max(len(positive) for positive in positives) + 12 + 64
    result = {
        "candidate_ids": np.full((len(query_ids), width), -1, dtype=np.int64),
        "positive_mask": np.zeros((len(query_ids), width), dtype=np.bool_),
        "valid_mask": np.zeros((len(query_ids), width), dtype=np.bool_),
        "hard_ids": np.full((len(query_ids), 12), -1, dtype=np.int64),
        "uniform_ids": np.empty((len(query_ids), 64), dtype=np.int64),
    }
    for row, (query, positive) in enumerate(zip(query_ids, positives)):
        if (
            not positive or len(positive) != len(set(positive))
            or any(type(item) is not int or not 0 <= item < catalog_items for item in positive)
        ):
            raise ValueError("All official positives must be distinct valid catalog rows.")
        positive = sorted(positive)
        known = set(positive)
        candidates = [int(item) for item in first100[row] if item >= 0 and item not in known]
        rng = np.random.default_rng(int(configuration_digest({
            "domain": "negative", "seed": seed, "query_id": query,
        })[:16], 16))
        hard = rng.choice(candidates, min(12, len(candidates)), replace=False).tolist()
        excluded = known | set(hard)
        if catalog_items - len(excluded) < 64:
            raise ValueError("Catalog cannot provide 64 distinct uniform alternatives outside P/H.")
        uniform: list[int] = []
        while len(uniform) < 64:
            for item in rng.integers(0, catalog_items, size=128).tolist():
                if item not in excluded:
                    excluded.add(item)
                    uniform.append(item)
                    if len(uniform) == 64:
                        break
        selected = positive + hard + uniform
        result["candidate_ids"][row, :len(selected)] = selected
        result["positive_mask"][row, :len(positive)] = True
        result["valid_mask"][row, :len(selected)] = True
        result["hard_ids"][row, :len(hard)] = hard
        result["uniform_ids"][row] = uniform
    return result


def runtime_schedule(groups: list[list[int]], session: int) -> dict[str, list[int]]:
    if type(session) is not int or session not in (0, 1, 2):
        raise ValueError("Runtime requires one of the three fresh contexts 0,1,2.")
    eligible = [index for index, group in enumerate(groups) if len(group) == 16]
    if len(eligible) < 60:
        raise ValueError("Runtime requires at least60 distinct production-shaped fixed groups.")
    rng = np.random.default_rng(20261011 + session)
    ordered = rng.permutation(eligible).tolist()
    probes = (rng.permutation(eligible).tolist() * math.ceil(100 / len(eligible)))[:100]
    return {"warmup": ordered[:10], "measured": ordered[10:60], "probes": probes}


def paired_order(session: int, index: int) -> tuple[str, str]:
    if type(session) is not int or session not in (0, 1, 2) or index < 0:
        raise ValueError("Pair order requires a declared context and nonnegative sample.")
    return ("base", "adapter") if (session + index) % 2 == 0 else ("adapter", "base")


def optimizer_step(
    parameters: list[torch.nn.Parameter], optimizer: torch.optim.Optimizer, rows: Sequence[int],
    loss_for_rows: Callable[[list[int]], torch.Tensor], microbatch: int = 8,
    deadline: dt.datetime | None = None,
) -> dict[str, Any]:
    if not rows or microbatch <= 0 or not parameters:
        raise ValueError("Optimizer step requires actual query rows and adapter parameters.")
    if deadline is not None:
        check_deadline(deadline)
    optimizer.zero_grad(set_to_none=True)
    loss_sum, micros = 0.0, 0
    with torch.enable_grad():
        for start in range(0, len(rows), microbatch):
            selected = list(rows[start:start + microbatch])
            losses = loss_for_rows(selected)
            if losses.shape != (len(selected),) or not torch.isfinite(losses).all():
                raise ValueError("Loss must be a finite per-query vector for the actual microbatch.")
            (losses.sum() / len(rows)).backward()
            loss_sum += float(losses.detach().double().sum().item())
            micros += 1
    if any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in parameters):
        raise ValueError("Every adapter must have a finite gradient; zero initial A gradients are allowed.")
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    optimizer.step()
    if any(not torch.isfinite(parameter).all() for parameter in parameters):
        raise ValueError("Adapter optimizer produced a nonfinite update.")
    return {
        "loss": loss_sum / len(rows), "queries": len(rows), "microbatches": micros,
        "gradient_norm": float(norm), "query_weight": 1 / len(rows),
    }


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def capture_rng(cuda: bool) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [torch.cuda.get_rng_state(torch.cuda.current_device()).cpu()] if cuda else [],
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
    }


def restore_rng(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if len(state["torch_cuda"]) != 1:
            raise ValueError("Checkpoint may address only the parent-selected benchmark CUDA device.")
        torch.cuda.set_rng_state(state["torch_cuda"][0], torch.cuda.current_device())
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))


def tensor_tree_equal(first: Any, second: Any) -> bool:
    if isinstance(first, torch.Tensor):
        return (
            isinstance(second, torch.Tensor) and first.dtype == second.dtype
            and torch.equal(first.detach().cpu(), second.detach().cpu())
        )
    if isinstance(first, dict):
        return (
            isinstance(second, dict) and first.keys() == second.keys()
            and all(tensor_tree_equal(value, second[key]) for key, value in first.items())
        )
    if isinstance(first, (tuple, list)):
        return (
            type(first) is type(second) and len(first) == len(second)
            and all(tensor_tree_equal(a, b) for a, b in zip(first, second))
        )
    return first == second


def make_checkpoint(
    adapters: dict[str, torch.Tensor], optimizer: torch.optim.Optimizer,
    binding: dict[str, Any], completed_steps: int, queries_seen: int,
    loss_history: list[float], total_queries: int = 50000, effective_batch: int = 64,
) -> dict[str, Any]:
    state = {
        "format_version": 1, "adapters": _cpu_tree(adapters), "optimizer": _cpu_tree(optimizer.state_dict()),
        "rng": capture_rng(any(
            parameter.is_cuda for group in optimizer.param_groups for parameter in group["params"]
        )), "binding": binding,
        "completed_steps": completed_steps, "queries_seen": queries_seen,
        "total_queries": total_queries, "effective_batch": effective_batch,
        "completed_epochs": int(queries_seen == total_queries), "loss_history": list(loss_history),
    }
    validate_checkpoint(state, binding, total_queries, effective_batch)
    return state


def validate_checkpoint(
    state: dict[str, Any], binding: dict[str, Any], total_queries: int = 50000,
    effective_batch: int = 64, require_final: bool = False,
) -> None:
    fields = {
        "format_version", "adapters", "optimizer", "rng", "binding", "completed_steps",
        "queries_seen", "total_queries", "effective_batch", "completed_epochs", "loss_history",
    }
    if set(state) != fields or state["format_version"] != 1 or state["binding"] != binding:
        raise ValueError("Checkpoint schema/source binding changed; full base states are forbidden.")
    steps, seen = state["completed_steps"], state["queries_seen"]
    if (
        type(steps) is not int or type(seen) is not int
        or total_queries <= 0 or effective_batch <= 0
        or not 1 <= steps <= math.ceil(total_queries / effective_batch)
        or seen != min(steps * effective_batch, total_queries)
        or state["total_queries"] != total_queries or state["effective_batch"] != effective_batch
        or state["completed_epochs"] != int(seen == total_queries)
        or len(state["loss_history"]) != steps or not np.isfinite(state["loss_history"]).all()
    ):
        raise ValueError("Checkpoint logical data position/optimizer steps/epoch are inconsistent.")
    if require_final and seen != total_queries:
        raise ValueError("Evaluation requires the completed final epoch, never an intermediate checkpoint.")
    adapters = state["adapters"]
    if not adapters or any(
        not isinstance(name, str) or not name or not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32 or not torch.isfinite(value).all()
        for name, value in adapters.items()
    ):
        raise ValueError("Checkpoint requires only finite FP32 adapter tensors.")
    if set(state["rng"]) != {"torch_cpu", "torch_cuda", "python", "numpy"}:
        raise ValueError("Checkpoint lacks complete optimizer/RNG continuation metadata.")
    if set(state["optimizer"]) != {"state", "param_groups"} or not state["optimizer"]["param_groups"]:
        raise ValueError("Checkpoint lacks a resumable optimizer state.")
    optimizer = state["optimizer"]
    identifiers = [identifier for group in optimizer["param_groups"] for identifier in group["params"]]
    if (
        len(identifiers) != len(adapters) or len(set(identifiers)) != len(identifiers)
        or set(optimizer["state"]) != set(identifiers)
        or any(
            group["lr"] != 2e-5 or tuple(group["betas"]) != (0.9, 0.999)
            or group["eps"] != 1e-8 or group["weight_decay"] != 0 or group["amsgrad"]
            or group["maximize"]
            for group in optimizer["param_groups"]
        )
    ):
        raise ValueError("Checkpoint optimizer inventory or fixed AdamW recipe changed.")
    for identifier, parameter in zip(identifiers, adapters.values()):
        moment = optimizer["state"][identifier]
        if (
            set(moment) != {"step", "exp_avg", "exp_avg_sq"}
            or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in moment.values())
            or moment["step"].numel() != 1 or float(moment["step"].item()) != steps
            or any(
                moment[name].shape != parameter.shape or moment[name].dtype != torch.float32
                for name in ("exp_avg", "exp_avg_sq")
            )
            or (moment["exp_avg_sq"] < 0).any()
        ):
            raise ValueError("Checkpoint optimizer moments/step counter are not a finite completed update.")


def publish_checkpoint(root: Path, state: dict[str, Any]) -> Path:
    validate_checkpoint(state, state["binding"], state["total_queries"], state["effective_batch"])
    name = f"step-{state['completed_steps']:06d}"
    path = _publish(root, name + ".pt", lambda stream: torch.save(state, stream))
    publish_json(root, name + ".json", {
        "complete": True, "checkpoint": path.name, "checkpoint_sha256": sha256(path),
        "binding": state["binding"], "completed_steps": state["completed_steps"],
        "queries_seen": state["queries_seen"], "completed_epochs": state["completed_epochs"],
    })
    return path


def read_checkpoint(
    root: Path, binding: dict[str, Any], total_queries: int = 50000,
    effective_batch: int = 64, require_final: bool = False,
) -> tuple[dict[str, Any], Path]:
    payloads = sorted(root.glob("step-*.pt"))
    metadata = sorted(root.glob("step-*.json"))
    if (
        not payloads or {value.stem for value in payloads} != {value.stem for value in metadata}
        or any(".pending-" in path.name for path in root.iterdir())
    ):
        raise ValueError("Checkpoint is incomplete or has no completed metadata.")
    if (
        set(root.iterdir()) != set(payloads + metadata)
        or any(re.fullmatch(r"step-\d{6}", path.stem) is None for path in payloads + metadata)
    ):
        raise ValueError("Checkpoint inventory contains unattested or misnamed metadata/payloads.")
    for path in metadata:
        receipt = read_json(path)
        payload = artifact_path(root, receipt["checkpoint"])
        if (
            receipt.get("complete") is not True or receipt["binding"] != binding
            or payload != path.with_suffix(".pt") or sha256(payload) != receipt["checkpoint_sha256"]
            or receipt["completed_steps"] != int(path.stem.split("-")[1])
            or receipt["queries_seen"] != min(receipt["completed_steps"] * effective_batch, total_queries)
            or receipt["completed_epochs"] != int(receipt["queries_seen"] == total_queries)
        ):
            raise ValueError("Checkpoint binding, completed metadata position or payload checksum changed.")
    receipt = read_json(metadata[-1])
    payload = artifact_path(root, receipt["checkpoint"])
    state = torch.load(payload, map_location="cpu", weights_only=True)
    validate_checkpoint(state, binding, total_queries, effective_batch, require_final)
    if any(receipt[key] != state[key] for key in ("completed_steps", "queries_seen", "completed_epochs")):
        raise ValueError("Completed checkpoint metadata differs from its logical data position.")
    return state, payload


def utc(value: str) -> dt.datetime:
    result = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() != dt.timedelta(0):
        raise ValueError("Deadline/admission timestamps must explicitly specify UTC.")
    return result


def check_deadline(deadline: dt.datetime, now: dt.datetime | None = None) -> None:
    if (now or dt.datetime.now(dt.timezone.utc)) >= deadline:
        raise TimeoutError("Bounded query adaptation deadline reached; no abbreviated result is admissible.")


def validate_admission(
    admission: dict[str, Any], protocol: dict[str, Any], protocol_sha256: str,
    sources: dict[str, str], phase: str, requested: dt.datetime,
    now: dt.datetime | None = None,
) -> dt.datetime:
    current = now or dt.datetime.now(dt.timezone.utc)
    if (
        admission.get("complete") is not True or admission.get("decision") != "admitted"
        or admission.get("primary_evidence_verified") is not True
        or admission.get("protocol_sha256") != protocol_sha256 or admission.get("sources") != sources
        or phase not in PHASES
    ):
        raise ValueError("Private admission must bind verified primary evidence, protocol and every source.")
    limits = protocol["admission"]
    started = utc(admission["admitted_at_utc"])
    engineering = utc(admission["engineering_ready_deadline_utc"])
    gpu_end = utc(admission["gpu_end_utc"])
    hard_end = utc(admission["hard_end_utc"])
    if (
        not started <= current or started > utc(limits["latest_start_utc"])
        or engineering != started + dt.timedelta(seconds=limits["engineering_ready_seconds"])
        or gpu_end != min(
            started + dt.timedelta(seconds=limits["maximum_gpu_seconds"]),
            utc(limits["global_gpu_cutoff_utc"]),
        )
        or hard_end != min(
            started + dt.timedelta(seconds=limits["maximum_wall_seconds"]),
            utc(limits["global_hard_end_utc"]),
        )
        or hard_end - gpu_end < dt.timedelta(seconds=limits["reserved_closure_seconds"])
        or not started < engineering < gpu_end < hard_end
    ):
        raise ValueError("Private admission dates exceed or alter the fixed bounded protocol.")
    bound = engineering if phase == "audit" else gpu_end
    if requested.tzinfo is None or requested.utcoffset() != dt.timedelta(0) or requested > bound:
        raise ValueError("Requested deadline exceeds the phase's admitted bound.")
    check_deadline(requested, current)
    return requested


def memory_failure(error: BaseException) -> bool:
    from rrbench.semantic_ann import is_faiss_gpu_allocation_failure

    return isinstance(error, (UnsupportedMemory, torch.cuda.OutOfMemoryError)) or (
        isinstance(error, RuntimeError) and is_faiss_gpu_allocation_failure(error)
    )


def validate_memory_sample(value: dict[str, Any]) -> None:
    required = (
        "memory_used_bytes", "device_memory_bytes", "torch_peak_allocated_bytes",
        "torch_allocated_bytes", "cuda_free_bytes",
    )
    if (
        any(type(value[name]) is not int or value[name] < 0 for name in required)
        or value["device_memory_bytes"] <= 0
        or value["memory_used_bytes"] > value["device_memory_bytes"]
        or value["cuda_free_bytes"] > value["device_memory_bytes"]
        or value["torch_peak_allocated_bytes"] < value["torch_allocated_bytes"]
    ):
        raise ValueError("Device memory observations must be actual nonnegative integer byte counts.")
    sensitivity = max(0, value["torch_peak_allocated_bytes"] - value["torch_allocated_bytes"])
    conservative = value["memory_used_bytes"] + sensitivity
    if (
        conservative > 48 * GIB or value["cuda_free_bytes"] < 8 * GIB
        or value["device_memory_bytes"] - conservative < 8 * GIB
    ):
        raise MemoryGateFailure(
            f"Observed usage plus peak sensitivity={conservative} bytes, "
            f"CUDA free={value['cuda_free_bytes']} bytes; fixed48GiB/8GiB gate failed.",
            dict(value),
        )


def bank_tensor(values: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_inference_mode_enabled() or dtype not in (torch.float16, torch.float32):
        raise ValueError("Frozen item bank must be an ordinary tensor constructed outside inference_mode.")
    result = torch.empty(values.shape, device=device, dtype=dtype, requires_grad=False)
    for start in range(0, len(values), 65536):
        block = torch.from_numpy(np.array(values[start:start + 65536], copy=True))
        result[start:start + len(block)].copy_(block)
    return result


def exact_search(
    item_gpu: torch.Tensor, queries: torch.Tensor, k: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        item_gpu.ndim != 2 or item_gpu.dtype != torch.float32 or item_gpu.requires_grad
        or torch.is_inference(item_gpu) or queries.ndim != 2 or queries.dtype != torch.float32
        or queries.shape[1] != item_gpu.shape[1] or queries.device != item_gpu.device
        or not 0 < k <= len(item_gpu)
    ):
        raise ValueError("Exact retrieval needs an already-resident ordinary frozen FP32 whole bank.")
    return torch.mm(queries, item_gpu.T).topk(k, dim=1)


def paired_bootstrap(
    base: list[dict[str, Any]], adapter: list[dict[str, Any]],
    metrics: Sequence[str] = METRICS, draws: int = 20000,
) -> dict[str, Any]:
    if (
        not base or [row["query_id"] for row in base] != [row["query_id"] for row in adapter]
        or len({row["query_id"] for row in base}) != len(base) or draws <= 0
    ):
        raise ValueError("Paired uncertainty requires the exact ordered original query population.")
    first = np.asarray([[row[metric] for metric in metrics] for row in base], dtype=np.float64)
    second = np.asarray([[row[metric] for metric in metrics] for row in adapter], dtype=np.float64)
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Paired metrics must be finite.")
    difference = second - first
    bootstrap = np.empty((draws, len(metrics)), dtype=np.float64)
    rng = np.random.default_rng(20261011)
    for start in range(0, draws, 128):
        indices = rng.integers(0, len(base), (min(128, draws - start), len(base)))
        bootstrap[start:start + len(indices)] = difference[indices].mean(axis=1)
    interval = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    return {
        "queries": len(base), "draws": draws, "seed": 20261011,
        "metrics": {
            metric: {
                "base": float(first[:, index].mean()), "adapter": float(second[:, index].mean()),
                "difference": float(difference[:, index].mean()), "ci95": interval[:, index].tolist(),
            } for index, metric in enumerate(metrics)
        },
        "scope": "Paired original-query pointwise95% bootstrap; one population, no NI margin or qualification.",
    }


def validate_protocol(protocol: dict[str, Any]) -> None:
    expected = {
        "model": {
            "key": "qwen06", "model": "Qwen/Qwen3-Embedding-0.6B",
            "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3", "dimension": 1024,
            "base_dtype": "bfloat16", "attention": "sdpa", "adapter_dtype": "float32",
            "targets": ["q_proj", "v_proj"], "rank": 16, "alpha": 32, "dropout": 0,
            "base_frozen": True, "items_frozen": True, "merge": False, "compile": False,
            "quantize_base": False,
        },
        "data": {
            "catalog_items": 8841823, "stored_item_dtype": "float16",
            "fit_queries": 50000, "validation_queries": 5000,
            "final_groups": {"dev-small": 1000, "trec19": 43, "trec20": 54},
            "token_cap": 512, "padding": "left, pad-to-multiple-of-eight",
        },
        "training": {
            "seed": 17, "epochs": 1, "microbatch": 8, "accumulation": 8, "effective_batch": 64,
            "optimizer_updates": 782, "last_block_queries": 16, "optimizer": "AdamW",
            "learning_rate": 2e-5, "betas": [0.9, 0.999], "epsilon": 1e-8,
            "weight_decay": 0, "gradient_clip_norm": 1, "temperature": 0.05,
            "checkpoint_every_updates": 100,
        },
        "smoke": {"queries": 100, "production_shaped_optimizer_steps": 5},
        "evaluation": {
            "arms": ["fresh pinned base", "same pinned base with final attached adapter"],
            "reranking": False, "quality_query_batch": 16,
            "encoding_batches": [1, 16], "pipeline_batch": 16, "warmup": 10, "samples": 50,
            "fresh_paired_contexts": 3,
            "metrics": ["MRR@10", "nDCG@10", "Recall@10", "Recall@100", "Recall@1000"],
        },
        "memory": {"maximum_observed_benchmark_gib": 48, "minimum_headroom_gib": 8},
        "admission": {
            "requires_completed_reviewed_durable_primary_evidence": True,
            "latest_start_utc": "2026-10-12T21:03:01.929Z",
            "maximum_wall_seconds": 21600, "engineering_ready_seconds": 7200,
            "maximum_gpu_seconds": 18000, "reserved_closure_seconds": 3600,
            "global_gpu_cutoff_utc": "2026-10-13T03:03:01.929Z",
            "global_hard_end_utc": "2026-10-13T09:03:01.929Z",
        },
    }
    for section, fields in expected.items():
        for field, value in fields.items():
            actual = protocol[section][field]
            if actual != value or type(actual) is not type(value):
                raise ValueError(f"The fixed public protocol changed: {section}.{field}")
    negatives = protocol["training"]["negatives"]
    for name, value in (
        ("probes", 64), ("retrieval_depth", 1000), ("consider_prefix", 100),
        ("hard_count", 12), ("uniform_count", 64),
    ):
        if type(negatives[name]) is not int or negatives[name] != value:
            raise ValueError("The fixed negative-mining protocol changed.")


def read_input_paths(source: Path) -> dict[str, Path]:
    values = read_json(source)
    if set(values) != set(INPUT_KEYS):
        raise ValueError("Inputs must explicitly provide every declared private path, and no substitutes.")
    result: dict[str, Path] = {}
    for key, value in values.items():
        if (
            not isinstance(value, str) or not value.strip() or "\x00" in value
            or ".." in Path(value).parts or ".." in PureWindowsPath(value).parts
        ):
            raise ValueError("Explicit source paths must be nonempty and have no traversal.")
        path = Path(value)
        if not path.is_absolute():
            path = source.parent / path
        if not (path.is_dir() if key == "cache" else path.is_file()):
            raise ValueError(f"Required explicit input is unavailable: {key}")
        result[key] = path.resolve()
    if len(set(result.values())) != len(result):
        raise ValueError("Mismapped source paths alias distinct required inputs.")
    return result


def checked_item_bank(
    vectors: Path, ids_file: Path, receipt: dict[str, Any], spec: dict[str, Any],
    rows: int = 8841823,
) -> np.ndarray:
    dimension = max(spec["dimensions"])
    representation = {
        "model": spec["model"], "revision": spec["revision"],
        "specification_sha256": configuration_digest(spec), "stored_dimension": dimension,
        "stored_dtype": "float16", "normalization": STORED_NORMALIZATION,
    }
    if (
        receipt.get("complete") is not True or receipt["rows"] != rows
        or receipt["stored_dimension"] != dimension or receipt["native_dimension"] != dimension
        or receipt["stored_dtype"] != "float16" or receipt["representation"] != representation
        or receipt["mrl_representations"] != {}
        or receipt["vectors_sha256"] != sha256(vectors) or receipt["ids_sha256"] != sha256(ids_file)
    ):
        raise ValueError("Item bank differs from the original complete pinned representation/source.")
    position = 0
    for shard in receipt["source_shards"]:
        if (
            shard["start"] != position or type(shard["rows"]) is not int or shard["rows"] <= 0
            or not isinstance(shard["receipt_sha256"], str) or len(shard["receipt_sha256"]) != 64
        ):
            raise ValueError("Item-bank source row coverage has changed.")
        position += shard["rows"]
    if position != rows:
        raise ValueError("Item-bank source row coverage is incomplete.")
    identifiers = np.load(ids_file, mmap_mode="r", allow_pickle=False)
    if identifiers.shape != (rows,) or identifiers.dtype != np.int64:
        raise ValueError("Item IDs do not match the full source row mapping.")
    for start in range(0, rows, 65536):
        if not np.array_equal(identifiers[start:start + 65536], np.arange(start, min(rows, start + 65536))):
            raise ValueError("Item IDs do not match the exact source row mapping.")
    items = np.load(vectors, mmap_mode="r", allow_pickle=False)
    checked_vectors(items, rows, dimension)
    return items


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=PHASES)
    for name in ("inputs", "protocol", "admission", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--deadline", required=True, type=utc)
    parser.add_argument("--session", type=int, choices=(0, 1, 2))
    return parser


@dataclass
class PilotContext:
    paths: dict[str, Path]
    protocol: dict[str, Any]
    deadline: dt.datetime
    output: Path
    binding: dict[str, Any]
    specification: dict[str, Any]
    panels: dict[str, dict[str, Any]]
    positives: list[list[int]]
    items: np.ndarray
    fit_queries: np.ndarray
    validation_queries: np.ndarray
    partition: Any
    model_smoke: dict[str, Any]
    control_paths: dict[str, Path] = field(default_factory=dict)
    control_hashes: dict[str, str] = field(default_factory=dict)


def load_context(args: argparse.Namespace) -> PilotContext:
    from importlib.metadata import version
    from rrbench import semantic_query_adaptation as core
    from rrbench.ann import Partition
    from rrbench.semantic_ann import validate_partition
    from rrbench.semantic_models import validate_model_files, validate_smoke_contract
    from rrbench.semantic_profile_contracts import validate_quality_cohort

    check_deadline(args.deadline)
    entry_hashes = {
        "inputs": sha256(args.inputs), "protocol": sha256(args.protocol), "admission": sha256(args.admission),
    }
    protocol = read_json(args.protocol)
    validate_protocol(protocol)
    admission = read_json(args.admission)
    protocol_hash = entry_hashes["protocol"]
    validate_admission(
        admission, protocol, protocol_hash, admission["sources"], args.phase, args.deadline,
    )
    paths = read_input_paths(args.inputs)
    if any(
        path == args.output.resolve() or args.output.resolve() in path.parents
        or (key == "cache" and path in args.output.resolve().parents)
        for key, path in paths.items()
    ):
        raise ValueError("Private output must not contain, replace or be inside an input/cache.")
    hashes = {}
    for key, path in paths.items():
        check_deadline(args.deadline)
        if key != "cache":
            hashes[key] = sha256(path)
    deadline = validate_admission(
        admission, protocol, protocol_hash, hashes, args.phase, args.deadline,
    )
    if hashes["items"] != PINNED_ITEMS:
        raise ValueError("Pilot must retain the exact original full Qwen0.6 canonical item bank.")
    specification = read_json(paths["registry"])["models"]["qwen06"]
    if configuration_digest(specification) != PINNED_SPECIFICATION:
        raise ValueError("The original pinned Qwen0.6 specification changed.")
    smoke = read_json(paths["model_smoke"])
    rows = [row for row in smoke["models"] if row["key"] == "qwen06"]
    if smoke.get("complete") is not True or len(rows) != 1:
        raise ValueError("A unique successful pinned original model smoke is required.")
    validate_smoke_contract(specification, rows[0])
    validate_model_files(specification, paths["cache"], rows[0]["model_files"])
    panels = {key: read_json(paths[key + "_panel"]) for key in ("fit", "validation", "final")}
    positives = validate_panel(panels["fit"], 100000, 8841823)
    validate_panel(panels["validation"], 5000, 8841823)
    validate_quality_cohort(panels["final"])
    validate_panel(panels["final"], 1097, 8841823)
    if (
        panels["fit"]["name"] != "fit" or panels["validation"]["name"] != "validation"
        or panels["fit"]["binary_threshold"] != 1 or panels["validation"]["binary_threshold"] != 1
        or "evaluation_groups" in panels["fit"] or "evaluation_groups" in panels["validation"]
    ):
        raise ValueError("Fitting and validation must be the original train-sourced official panels.")
    require_disjoint_panels(panels["fit"], panels["validation"], panels["final"])
    query_arrays = {}
    for key in ("fit", "validation"):
        query_receipt = read_json(paths[key + "_query_receipt"])
        if (
            query_receipt["vectors_sha256"] != hashes[key + "_queries"]
            or query_receipt["panel_sha256"] != hashes[key + "_panel"]
        ):
            raise ValueError("Frozen query receipt differs from the initial source snapshot.")
        query_arrays[key] = checked_queries(
            paths[key + "_queries"], query_receipt, paths[key + "_panel"], panels[key], specification,
        )
    item_receipt = read_json(paths["item_receipt"])
    if item_receipt["vectors_sha256"] != hashes["items"] or item_receipt["ids_sha256"] != hashes["item_ids"]:
        raise ValueError("Frozen item receipt differs from the initial source snapshot.")
    items = checked_item_bank(paths["items"], paths["item_ids"], item_receipt, specification)
    if read_json(paths["partition_receipt"])["partition_sha256"] != hashes["partition"]:
        raise ValueError("Original shared partition source checksum changed.")
    with np.load(paths["partition"], allow_pickle=False) as values:
        if set(values.files) != {"centroids", "assignments", "order", "offsets"}:
            raise ValueError("Shared partition has an unexpected payload schema.")
        partition = Partition(*(values[key] for key in ("centroids", "assignments", "order", "offsets")))
    validate_partition(partition, 8841823, 1024, 16384)
    selected = panels["fit"]["query_ids"][:50000]
    panels["fit"] = {
        **panels["fit"], "query_ids": selected, "queries": panels["fit"]["queries"][:50000],
        "qrels": {query: panels["fit"]["qrels"][query] for query in selected},
    }
    implementation_paths = {
        "runner": Path(__file__), "core": Path(core.__file__),
        **{
            name: Path(__file__).with_name(name + ".py")
            for name in ("semantic_models", "semantic_ir", "semantic_ann", "semantic_profile_contracts")
        },
    }
    control_paths = {
        "inputs": args.inputs, "protocol": args.protocol, "admission": args.admission,
        **implementation_paths,
    }
    control_hashes = {name: sha256(path) for name, path in control_paths.items()}
    if any(control_hashes[name] != expected for name, expected in entry_hashes.items()):
        raise ValueError("Input map, protocol or private admission changed while validating sources.")
    binding = {
        "schema_version": 1, "sources": hashes, "inputs_sha256": control_hashes["inputs"],
        "protocol_sha256": control_hashes["protocol"], "admission_sha256": control_hashes["admission"],
        "implementation": {name: control_hashes[name] for name in implementation_paths},
        "runtime_versions": {
            "torch": str(torch.__version__), "numpy": str(np.__version__),
            **{name: version(name) for name in ("transformers", "tokenizers", "huggingface_hub")},
        },
        "specification_sha256": configuration_digest(specification),
        "fit_order_sha256": configuration_digest({"query_ids": selected}),
        "canonical_items": "Original stored normalized FP16 values; no item re-encoding or re-normalization.",
    }
    check_deadline(deadline)
    return PilotContext(
        paths, protocol, deadline, args.output, binding, specification, panels, positives[:50000],
        items, query_arrays["fit"][:50000], query_arrays["validation"], partition, rows[0],
        control_paths=control_paths, control_hashes=control_hashes,
    )


DEPENDENCIES = {
    "audit": (), "mine": ("audit",), "smoke": ("audit", "mine"),
    "train": ("audit", "mine", "smoke"), "evaluate": ("audit", "train"),
    "runtime": ("audit", "train", "evaluate"),
}


def phase_binding(context: PilotContext, phase: str, session: int | None = None) -> dict[str, Any]:
    return {
        **context.binding, "phase": phase, "session": session,
        "dependencies": {
            name: sha256(context.output / name / "completed.json") for name in DEPENDENCIES[phase]
        },
    }


def check_dependencies(context: PilotContext, phase: str) -> None:
    for name in DEPENDENCIES[phase]:
        checked_phase(context.output / name, phase_binding(context, name))


def finish_pilot_phase(
    context: PilotContext, root: Path, binding: dict[str, Any], details: dict[str, Any],
) -> dict[str, Any]:
    verify_file_hashes(
        {name: path for name, path in context.paths.items() if name != "cache"},
        context.binding["sources"], context.deadline,
    )
    verify_file_hashes(context.control_paths, context.control_hashes, context.deadline)
    verify_file_hashes(
        {name: context.output / name / "completed.json" for name in binding["dependencies"]},
        binding["dependencies"], context.deadline,
    )
    check_dependencies(context, binding["phase"])
    if "cache" in context.paths:
        from rrbench.semantic_models import validate_model_files

        validate_model_files(context.specification, context.paths["cache"], context.model_smoke["model_files"])
    check_deadline(context.deadline)
    return finish_phase(root, binding, details)


def group_manifest(context: PilotContext) -> dict[str, Any]:
    return {
        name: {
            "query_ids": panel["query_ids"],
            "groups": execution_groups(panel["query_ids"]),
            "panel_sha256": context.binding["sources"][name + "_panel"],
            "batch": 16, "label_independent": True,
            "count": len(panel["query_ids"]),
            "policy": "One immutable ID-hash partition; tail retained for quality; shared by both arms and every context.",
        } for name, panel in context.panels.items() if name != "fit"
    }


def run_audit(context: PilotContext, tokenizer: Any | None = None) -> dict[str, Any]:
    root, binding = context.output / "audit", phase_binding(context, "audit")
    prior = begin_phase(root, binding)
    if prior is not None:
        return prior
    if tokenizer is None:
        from transformers import AutoTokenizer
        from rrbench.semantic_models import pinned_snapshot

        tokenizer = AutoTokenizer.from_pretrained(
            pinned_snapshot(context.specification, context.paths["cache"]),
            local_files_only=True, trust_remote_code=False, padding_side="left",
        )
    groups = group_manifest(context)
    publish_json(root, "groups.json", groups)
    audits = {}
    for name, panel in context.panels.items():
        actual_groups = (
            [list(range(start, min(start + 8, len(panel["queries"])))) for start in range(0, len(panel["queries"]), 8)]
            if name == "fit" else groups[name]["groups"]
        )
        audits[name], lengths = token_audit(
            tokenizer, panel["queries"], actual_groups, deadline=context.deadline,
        )
        publish_array(root, name + "-token-lengths.npy", lengths.astype(np.uint32))
    rendered = {
        "training": f"Instruct: {QUERY_INSTRUCTION}\nQuery:{context.panels['fit']['queries'][0]}",
        "evaluation": f"Instruct: {QUERY_INSTRUCTION}\nQuery:{context.panels['final']['queries'][0]}",
        "thinking": "N/A", "finish_reason": "N/A", "chat_template": "not used",
        "privacy": "Actual raw strings; private audit artifact, not a public report.",
    }
    publish_json(root, "rendered-samples.json", rendered)
    publish_json(root, "audit.json", audits)
    print("PRIVATE_ACTUAL_QUERY_TEMPLATE_SAMPLES", json.dumps(rendered, ensure_ascii=True), flush=True)
    if not all(row["pass"] for row in audits.values()):
        raise ValueError("Actual instructed query overflow fails preflight; no cap change, filter or silent truncation.")
    return finish_pilot_phase(context, root, binding, {
        "all_inputs_pass": True, "lengths": audits,
        "fit_query_ids": context.panels["fit"]["query_ids"],
        "all_fitting_positives": sum(len(values) for values in context.positives),
        "groups_sha256": sha256(root / "groups.json"), "deadline_utc": context.deadline.isoformat(),
        "scope": "CPU-only complete actual-input/source audit; not a GPU engineering certificate.",
    })


def smoke_rows(step: int) -> list[int]:
    if type(step) is not int or not 0 <= step < 5:
        raise ValueError("Smoke has exactly five production-shaped optimizer steps over100 real queries.")
    return [(step * 64 + offset) % 100 for offset in range(64)]


def assert_quality_replay(
    actual: tuple[np.ndarray, np.ndarray, np.ndarray],
    expected: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, Any]:
    query, scores, ids = actual
    reference, reference_scores, reference_ids = expected
    if (
        query.dtype != np.float16 or reference.dtype != np.float16
        or scores.dtype != np.float32 or ids.dtype != np.int64
        or query.ndim != 2 or scores.ndim != 2 or scores.shape != ids.shape
        or len(query) != len(scores) or query.shape != reference.shape
        or scores.shape != reference_scores.shape or ids.shape != reference_ids.shape
        or not np.isfinite(query).all() or not np.isfinite(scores).all()
        or not np.array_equal(query, reference) or not np.array_equal(ids, reference_ids)
        or not np.allclose(scores, reference_scores, rtol=1e-5, atol=1e-5)
    ):
        raise ValueError("Raw replay changed its exact canonical queries, topk IDs or scores in the same batch scope.")
    return {
        "canonical_queries_exact": True, "topk_ids_exact": True,
        "scores_exact": bool(np.array_equal(scores, reference_scores)),
        "maximum_score_absolute_difference": float(np.max(np.abs(scores - reference_scores))),
        "queries": len(query),
    }


def gpu_device() -> torch.device:
    from rrbench.runtime import configure, require_cuda

    configure(17, 4)
    random.seed(17)
    device = require_cuda()
    torch.cuda.reset_peak_memory_stats(device)
    return device


def observe_memory(device: torch.device) -> dict[str, Any]:
    from rrbench.semantic_cascade import device_telemetry

    free, _ = torch.cuda.mem_get_info(device)
    value = {
        **device_telemetry(device), "cuda_free_bytes": free,
        "torch_allocated_bytes": torch.cuda.memory_allocated(device),
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "torch_reserved_bytes": torch.cuda.memory_reserved(device),
        "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    value["peak_sensitivity_bytes"] = max(
        0, value["torch_peak_allocated_bytes"] - value["torch_allocated_bytes"],
    )
    validate_memory_sample(value)
    return value


def run_mine(context: PilotContext) -> dict[str, Any]:
    from rrbench.semantic_ann import SemanticFaissANN

    check_dependencies(context, "mine")
    root, binding = context.output / "mine", phase_binding(context, "mine")
    prior = begin_phase(root, binding)
    if prior is not None:
        read_mined(context)
        return prior
    device = gpu_device()
    memory = [observe_memory(device)]
    check_deadline(context.deadline)
    ann = SemanticFaissANN(context.items, context.partition, gpu=True, half=True)
    memory.append(observe_memory(device))
    count = len(context.fit_queries)
    ids = np.empty((count, 1000), dtype=np.int64)
    scores = np.empty((count, 1000), dtype=np.float32)
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, count, 64):
            check_deadline(context.deadline)
            queries = torch.tensor(
                np.array(context.fit_queries[start:start + 64], dtype=np.float32),
                device=device, dtype=torch.float32,
            )
            actual_scores, actual_ids = ann.search(queries, k=1000, probes=64)
            selected = actual_ids.detach().cpu().numpy()
            values = actual_scores.detach().cpu().numpy()
            checked_retrieval(selected, values, len(queries), len(context.items), allow_missing=True)
            ids[start:start + len(queries)] = selected
            scores[start:start + len(queries)] = values
            if start % 2048 == 0:
                memory.append(observe_memory(device))
                print("FIXED_QUERY_MINING", start + len(queries), count, flush=True)
    memory.append(observe_memory(device))
    negatives = sample_negatives(
        context.panels["fit"]["query_ids"], context.positives, ids[:, :100], len(context.items),
    )
    for name, values in {
        "retrieved_ids": ids, "retrieved_scores": scores,
        "first100": ids[:, :100].copy(), **negatives,
    }.items():
        publish_array(root, name + ".npy", values)
    publish_json(root, "query-ids.json", {"query_ids": context.panels["fit"]["query_ids"]})
    publish_json(root, "memory.json", {"samples": memory})
    check_deadline(context.deadline)
    return finish_pilot_phase(context, root, binding, {
        "queries": count, "catalog_items": len(context.items), "dimension": context.items.shape[1],
        "backend": "SemanticFaissANN(gpu=True,half=True)", "partition_clusters": 16384,
        "probes": 64, "retrieval_depth": 1000, "consider_prefix": 100,
        "seed": 17, "all_retrieved_source_ids_checked": True,
        "missing_retrieved_slots": int((ids < 0).sum()),
        "all_positives_retained": int(negatives["positive_mask"].sum()),
        "hard_negatives": int((negatives["hard_ids"] >= 0).sum()),
        "uniform_negatives": int(negatives["uniform_ids"].size),
        "elapsed_seconds": time.perf_counter() - started,
        "policy": "Per-query fixed samples; all P excluded, no held-out labels, in-batch negatives or refresh.",
        "truth": "Retrieved and uniform alternatives are weak negatives, not proven irrelevance.",
        "deadline_utc": context.deadline.isoformat(),
    })


def read_mined(context: PilotContext) -> dict[str, np.ndarray]:
    root = context.output / "mine"
    receipt = checked_phase(root, phase_binding(context, "mine"))
    count = len(context.panels["fit"]["query_ids"])
    if (
        receipt["details"]["queries"] != count
        or receipt["details"]["backend"] != "SemanticFaissANN(gpu=True,half=True)"
        or receipt["details"]["probes"] != 64 or receipt["details"]["retrieval_depth"] != 1000
        or read_json(root / "query-ids.json")["query_ids"] != context.panels["fit"]["query_ids"]
    ):
        raise ValueError("Mined negatives do not cover the exact fixed fitting query/source contract.")
    retrieved = np.load(root / "retrieved_ids.npy", mmap_mode="r", allow_pickle=False)
    scores = np.load(root / "retrieved_scores.npy", mmap_mode="r", allow_pickle=False)
    prefix = np.load(root / "first100.npy", mmap_mode="r", allow_pickle=False)
    checked_retrieval(retrieved, scores, count, len(context.items), allow_missing=True)
    if (
        retrieved.shape != (count, 1000) or scores.shape != retrieved.shape or scores.dtype != np.float32
        or not np.array_equal(prefix, retrieved[:, :100])
    ):
        raise ValueError("Mined first100/all-retrieved source checks failed.")
    expected = sample_negatives(
        context.panels["fit"]["query_ids"], context.positives, prefix, len(context.items),
    )
    result = {}
    for name, reference in expected.items():
        actual = np.load(root / (name + ".npy"), mmap_mode="r", allow_pickle=False)
        if actual.dtype != reference.dtype or not np.array_equal(actual, reference):
            raise ValueError("Mined P/H/U IDs or positive/valid masks changed from fixed per-query samples.")
        result[name] = actual
    return result


def new_encoder(
    context: PilotContext, device: torch.device, adapted: bool,
) -> tuple[SemanticEncoder, dict[str, Any] | None]:
    from transformers import AutoModel, AutoTokenizer
    from rrbench import semantic_query_adaptation as core
    from rrbench.semantic_models import pinned_snapshot

    check_deadline(context.deadline)
    snapshot = pinned_snapshot(context.specification, context.paths["cache"])
    # Use the existing formatting/encoder type, but load only the already-attested local snapshot.
    encoder = SemanticEncoder.__new__(SemanticEncoder)
    encoder.specification = context.specification
    encoder.device, encoder.is_qwen = device, True
    encoder.tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, padding_side="left",
    )
    encoder.model = AutoModel.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device).eval()
    encoder.model.requires_grad_(False)
    if any(parameter.dtype != torch.bfloat16 for parameter in encoder.model.parameters()):
        raise ValueError("The original base must load entirely in the fixed BF16 parameter dtype.")
    encoder.native_dimension = encoder.model.config.hidden_size
    if encoder.native_dimension != 1024:
        raise ValueError("Local original encoder dimension differs from the fixed1024 recipe.")
    encoder.parameters = sum(parameter.numel() for parameter in encoder.model.parameters())
    metadata = core.attach_query_adapters(encoder.model, rank=16, alpha=32) if adapted else None
    encoder.model.eval()
    if adapted:
        parameters = core.adapter_parameters(encoder.model)
        if (
            {id(value) for value in parameters}
            != {id(value) for value in encoder.model.parameters() if value.requires_grad}
            or any(value.dtype != torch.float32 for value in parameters)
        ):
            raise ValueError("Only FP32 query q_proj/v_proj adapter parameters may be trainable.")
    return encoder, metadata


def new_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    from rrbench.semantic_query_adaptation import adapter_parameters

    return torch.optim.AdamW(
        adapter_parameters(model), lr=2e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
        foreach=False, fused=False,
    )


def batch_losses(
    encoder: SemanticEncoder, item_gpu: torch.Tensor, negatives: dict[str, np.ndarray],
    texts: list[str], rows: list[int],
) -> torch.Tensor:
    from rrbench.semantic_query_adaptation import forward_queries, positive_mass_loss

    if encoder.model.training or any(module.training for module in encoder.model.modules()):
        raise ValueError("The frozen base and query adapters must remain in eval mode with dropout disabled.")
    candidates = torch.tensor(negatives["candidate_ids"][rows], device=encoder.device, dtype=torch.int64)
    positive = torch.tensor(negatives["positive_mask"][rows], device=encoder.device, dtype=torch.bool)
    valid = torch.tensor(negatives["valid_mask"][rows], device=encoder.device, dtype=torch.bool)
    queries = forward_queries(encoder, [texts[row] for row in rows])
    return positive_mass_loss(
        queries, item_gpu[candidates.clamp_min(0)], positive, valid, temperature=0.05,
    )


def tensor_digest(values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(str((tuple(values.shape), values.dtype)).encode())
    flat = values.detach().reshape(-1)
    for start in range(0, flat.numel(), 8 << 20):
        block = flat[start:start + (8 << 20)].cpu().contiguous().view(torch.uint8)
        digest.update(block.numpy().tobytes())
    return digest.hexdigest()


def frozen_model_digest(model: torch.nn.Module) -> str:
    return configuration_digest({
        name: tensor_digest(value) for name, value in model.named_parameters() if not value.requires_grad
    })


def run_smoke(context: PilotContext) -> dict[str, Any]:
    from rrbench import semantic_query_adaptation as core

    check_dependencies(context, "smoke")
    root, binding = context.output / "smoke", phase_binding(context, "smoke")
    prior = begin_phase(root, binding)
    if prior is not None:
        return prior
    if len(context.panels["fit"]["query_ids"]) != 50000:
        raise ValueError("Real smoke must select the first100 of the exact50000 fitting cohort.")
    negatives = read_mined(context)
    device = gpu_device()
    memory = [observe_memory(device)]
    baseline, _ = new_encoder(context, device, adapted=False)
    encoder, metadata = new_encoder(context, device, adapted=True)
    texts = context.panels["fit"]["queries"]
    initial = core.adapter_state(encoder.model)
    parity = []
    with torch.no_grad():
        for batch in (8, 16):
            for start in range(0, 100, batch):
                check_deadline(context.deadline)
                selected = texts[start:min(100, start + batch)]
                base = core.forward_queries(baseline, selected)
                attached = core.forward_queries(encoder, selected)
                if not torch.equal(base, attached):
                    raise ValueError("Zero-adapter smoke differs from a fresh pinned baseline on identical groups.")
                parity.append({"start": start, "rows": len(selected), "batch": batch, "canonical_exact": True})
    del baseline
    item_gpu = bank_tensor(context.items, device, torch.float16)
    item_before, base_before = tensor_digest(item_gpu), frozen_model_digest(encoder.model)
    optimizer = new_optimizer(encoder.model)
    parameters = core.adapter_parameters(encoder.model)
    observations: list[dict[str, Any]] = []
    for step in range(5):
        record = optimizer_step(
            parameters, optimizer, smoke_rows(step),
            lambda rows: batch_losses(encoder, item_gpu, negatives, texts, rows),
            deadline=context.deadline,
        )
        observations.append(record)
        memory.append(observe_memory(device))
        if step == 1:
            publish_checkpoint(root / "checkpoints", make_checkpoint(
                core.adapter_state(encoder.model), optimizer, binding, 2, 128,
                [row["loss"] for row in observations], total_queries=320,
            ))
    trained = make_checkpoint(
        core.adapter_state(encoder.model), optimizer, binding, 5, 320,
        [row["loss"] for row in observations], total_queries=320,
    )
    if tensor_tree_equal(initial, trained["adapters"]):
        raise ValueError("Five real smoke optimizer steps did not update any adapter.")
    if any(value.grad is not None for value in encoder.model.parameters() if not value.requires_grad):
        raise ValueError("Frozen base acquired gradients during real smoke.")
    resumed, resumed_metadata = new_encoder(context, device, adapted=True)
    if resumed_metadata != metadata:
        raise ValueError("Resumed smoke adapter target-module inventory changed.")
    resumed_optimizer = new_optimizer(resumed.model)
    saved, _ = read_checkpoint(root / "checkpoints", binding, total_queries=320)
    core.load_adapter_state(resumed.model, saved["adapters"])
    resumed_optimizer.load_state_dict(saved["optimizer"])
    restore_rng(saved["rng"])
    resumed_losses = list(saved["loss_history"])
    for step in range(2, 5):
        record = optimizer_step(
            core.adapter_parameters(resumed.model), resumed_optimizer, smoke_rows(step),
            lambda rows: batch_losses(resumed, item_gpu, negatives, texts, rows),
            deadline=context.deadline,
        )
        resumed_losses.append(record["loss"])
    resumed_state = make_checkpoint(
        core.adapter_state(resumed.model), resumed_optimizer, binding, 5, 320,
        resumed_losses, total_queries=320,
    )
    if not tensor_tree_equal(trained, resumed_state):
        raise ValueError("Adapter/optimizer/RNG checkpoint reload and resumed continuation disagree.")
    if (
        frozen_model_digest(encoder.model) != base_before
        or frozen_model_digest(resumed.model) != base_before
        or tensor_digest(item_gpu) != item_before
        or sha256(context.paths["items"]) != context.binding["sources"]["items"]
    ):
        raise ValueError("Frozen base weights or canonical item bank changed during smoke.")
    memory.append(observe_memory(device))
    publish_checkpoint(root / "checkpoints", trained)
    publish_json(root, "observations.json", {
        "optimizer_steps": observations, "zero_adapter_parity": parity, "memory": memory,
        "query_ids": context.panels["fit"]["query_ids"][:100],
        "base_before_after_sha256": base_before, "item_before_after_sha256": item_before,
    })
    check_deadline(context.deadline)
    return finish_pilot_phase(context, root, binding, {
        "unique_real_queries": 100, "optimizer_steps": 5, "continuation_replayed_steps": 3,
        "effective_batch": 64, "microbatch": 8, "adapter_inventory": metadata,
        "checks": {
            "fresh_zero_adapter_exact_parity": True, "finite_gradients_and_updates": True,
            "base_and_items_unchanged": True, "optimizer_rng_resume_exact": True,
            "all_source_and_target_identities": True,
        },
        "smoke_state_discarded_for_main": True,
        "scope": "Correctness only; main must restart from the original base and seed17, never this smoke state.",
        "deadline_utc": context.deadline.isoformat(),
    })


def _smoke_required(context: PilotContext) -> None:
    receipt = checked_phase(context.output / "smoke", phase_binding(context, "smoke"))
    details = receipt["details"]
    checks = {
        "fresh_zero_adapter_exact_parity", "finite_gradients_and_updates",
        "base_and_items_unchanged", "optimizer_rng_resume_exact", "all_source_and_target_identities",
    }
    if (
        details["unique_real_queries"] != 100 or details["optimizer_steps"] != 5
        or details["smoke_state_discarded_for_main"] is not True
        or set(details["checks"]) != checks or not all(value is True for value in details["checks"].values())
    ):
        raise ValueError("Main fitting requires all real100-query, five-step smoke gates.")


def run_train(context: PilotContext) -> dict[str, Any]:
    from rrbench import semantic_query_adaptation as core

    check_dependencies(context, "train")
    _smoke_required(context)
    root, binding = context.output / "train", phase_binding(context, "train")
    existed = (root / "run-config.json").exists()
    prior = begin_phase(root, binding, resumable=True)
    if prior is not None:
        read_checkpoint(root / "checkpoints", binding, require_final=True)
        return prior
    saved: dict[str, Any] | None = None
    if existed:
        saved, _ = read_checkpoint(root / "checkpoints", binding)
    negatives = read_mined(context)
    texts = context.panels["fit"]["queries"]
    blocks = core.optimizer_blocks(len(texts), effective_batch=64)
    if len(texts) != 50000 or len(blocks) != 782 or blocks[-1] != (49984, 50000):
        raise ValueError("Fixed fitting must be exactly50000 queries,782 updates and a16-query final block.")
    device = gpu_device()
    memory = [observe_memory(device)]
    encoder, metadata = new_encoder(context, device, adapted=True)
    optimizer = new_optimizer(encoder.model)
    item_gpu = bank_tensor(context.items, device, torch.float16)
    memory.append(observe_memory(device))
    first, losses = 0, []
    if saved is not None:
        core.load_adapter_state(encoder.model, saved["adapters"])
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
        first, losses = saved["completed_steps"], list(saved["loss_history"])
    started = time.perf_counter()
    for step in range(first, len(blocks)):
        start, stop = blocks[step]
        record = optimizer_step(
            core.adapter_parameters(encoder.model), optimizer, list(range(start, stop)),
            lambda rows: batch_losses(encoder, item_gpu, negatives, texts, rows),
            deadline=context.deadline,
        )
        losses.append(record["loss"])
        check_deadline(context.deadline)
        if (step + 1) % 100 == 0 or step + 1 == 782:
            state = make_checkpoint(
                core.adapter_state(encoder.model), optimizer, binding, step + 1, stop, losses,
            )
            publish_checkpoint(root / "checkpoints", state)
            memory.append(observe_memory(device))
            print("FIXED_QUERY_ADAPTATION_STEP", step + 1, stop, record["loss"], flush=True)
    final, path = read_checkpoint(root / "checkpoints", binding, require_final=True)
    if any(value.grad is not None for value in encoder.model.parameters() if not value.requires_grad):
        raise ValueError("Only adapters may receive gradients during main fitting.")
    publish_json(root, "training.json", {
        "loss_history": final["loss_history"], "memory": memory,
        "segment_start_step": first, "segment_seconds": time.perf_counter() - started,
        "seed": 17, "order": "Unchanged first50000 frozen fitting rows; no reshuffle.",
        "final_block": {"start": 49984, "stop": 50000, "microbatches": 2, "per_query_weight": 1 / 16},
        "effective_full_block": {"queries": 64, "microbatch": 8, "accumulation": 8},
    })
    check_deadline(context.deadline)
    return finish_pilot_phase(context, root, binding, {
        "queries": 50000, "optimizer_steps": 782, "completed_epochs": 1, "seed": 17,
        "adapter_inventory": metadata, "checkpoint": path.relative_to(root).as_posix(),
        "checkpoint_sha256": sha256(path), "checkpoint_policy": "Only final completed epoch admissible for evaluation.",
        "smoke_state_reused": False, "deadline_utc": context.deadline.isoformat(),
        "scope": "One fixed query-only recipe; no validation selection, rescue, base update or item re-encoding.",
    })


def final_adapter(context: PilotContext) -> dict[str, Any]:
    root, binding = context.output / "train", phase_binding(context, "train")
    receipt = checked_phase(root, binding)
    state, path = read_checkpoint(root / "checkpoints", binding, require_final=True)
    if (
        receipt["details"]["queries"] != 50000 or receipt["details"]["optimizer_steps"] != 782
        or receipt["details"]["completed_epochs"] != 1
        or artifact_path(root, receipt["details"]["checkpoint"]) != path
        or receipt["details"]["checkpoint_sha256"] != sha256(path)
    ):
        raise ValueError("Only the source-bound50000-query/782-update final adapter can be evaluated.")
    return state


def observation(
    vectors: torch.Tensor, scores: torch.Tensor, ids: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (
        vectors.ndim != 2 or vectors.dtype != torch.float32 or not torch.isfinite(vectors).all()
        or not torch.equal(vectors, vectors.half().float())
    ):
        raise ValueError("Actual raw queries are not canonical FP16-rounded FP32 values.")
    return (
        vectors.detach().half().cpu().numpy(), scores.detach().float().cpu().numpy(),
        ids.detach().long().cpu().numpy(),
    )


def raw_pipeline(
    encoder: SemanticEncoder, item_gpu: torch.Tensor, texts: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from rrbench.semantic_query_adaptation import forward_queries

    with torch.no_grad():
        queries = forward_queries(encoder, texts)
        scores, ids = exact_search(item_gpu, queries)
    return queries, scores, ids


def run_evaluate(context: PilotContext) -> dict[str, Any]:
    from rrbench.semantic_query_adaptation import load_adapter_state
    from rrbench.semantic_ir import panel_metrics

    check_dependencies(context, "evaluate")
    state = final_adapter(context)
    final_checkpoint_hash = read_json(context.output / "train" / "completed.json")["details"]["checkpoint_sha256"]
    root, binding = context.output / "evaluate", phase_binding(context, "evaluate")
    prior = begin_phase(root, binding)
    if prior is not None:
        return prior
    groups = read_json(context.output / "audit" / "groups.json")
    if groups != group_manifest(context):
        raise ValueError("Immutable label-independent evaluation groups changed.")
    device = gpu_device()
    memory = [observe_memory(device)]
    item_gpu = bank_tensor(context.items, device, torch.float32)
    base, _ = new_encoder(context, device, adapted=False)
    adapted, metadata = new_encoder(context, device, adapted=True)
    load_adapter_state(adapted.model, state["adapters"])
    memory.append(observe_memory(device))
    quality: dict[str, Any] = {}
    vector_receipts = {}
    for population in ("validation", "final"):
        panel = context.panels[population]
        manifest = groups[population]
        quality[population] = {}
        for arm, encoder in (("base", base), ("adapter", adapted)):
            count = len(panel["query_ids"])
            vectors = np.empty((count, 1024), dtype=np.float16)
            scores = np.empty((count, 1000), dtype=np.float32)
            ids = np.empty((count, 1000), dtype=np.int64)
            seen = np.zeros(count, dtype=np.bool_)
            for index, rows in enumerate(manifest["groups"]):
                check_deadline(context.deadline)
                actual = observation(*raw_pipeline(
                    encoder, item_gpu, [panel["queries"][row] for row in rows],
                ))
                checked_retrieval(actual[2], actual[1], len(rows), len(context.items))
                if seen[rows].any():
                    raise ValueError("Exact evaluation omitted, duplicated or returned invalid source rows.")
                vectors[rows], scores[rows], ids[rows] = actual
                seen[rows] = True
                if index % 20 == 0:
                    memory.append(observe_memory(device))
            if not seen.all():
                raise ValueError("Exact evaluation did not cover every original query.")
            names = {}
            for name, values in (("queries", vectors), ("scores", scores), ("ids", ids)):
                path = publish_array(root, f"{population}-{arm}-{name}.npy", values)
                names[name] = {"file": path.name, "sha256": sha256(path)}
            vector_receipts[population + "-" + arm] = {
                "complete": True, "query_ids": panel["query_ids"], "rows": count,
                "arm": arm, "model": context.specification["model"],
                "revision": context.specification["revision"],
                "specification_sha256": context.binding["specification_sha256"],
                "query_instruction": QUERY_INSTRUCTION,
                "adapter_checkpoint_sha256": final_checkpoint_hash if arm == "adapter" else None,
                "source_panel_sha256": context.binding["sources"][population + "_panel"],
                "group_manifest_sha256": sha256(context.output / "audit" / "groups.json"),
                "dimension": 1024, "query_dtype": "float16", "retrieval_math": "FP32; TF32 off",
                "catalog_items": len(context.items), "topk": 1000, "files": names,
            }
            quality[population][arm] = panel_metrics(panel, ids)
    effects = {
        "validation": paired_bootstrap(
            quality["validation"]["base"]["per_query"], quality["validation"]["adapter"]["per_query"],
        ),
        "final": {
            name: paired_bootstrap(
                quality["final"]["base"]["by_group"][name]["per_query"],
                quality["final"]["adapter"]["by_group"][name]["per_query"],
            ) for name in ("dev-small", "trec19", "trec20")
        },
    }
    publish_json(root, "raw-query-receipts.json", vector_receipts)
    publish_json(root, "per-query-quality.json", quality)
    publish_json(root, "paired-effects.json", effects)
    publish_json(root, "memory.json", {"samples": memory + [observe_memory(device)]})
    check_deadline(context.deadline)
    return finish_pilot_phase(context, root, binding, {
        "adapter_inventory": metadata, "validation_queries": 5000,
        "final_queries": {"dev-small": 1000, "trec19": 43, "trec20": 54},
        "groups_sha256": sha256(context.output / "audit" / "groups.json"),
        "final_checkpoint_only": True, "query_batch": 16, "catalog_uploads": 1,
        "base_path": "Fresh pinned base with no adapter modules, not zero-LoRA computation.",
        "retrieval": "Exhaustive whole-bank FP32 torch.mm+topk1000, TF32 disabled, scorer-none.",
        "effects": effects, "deadline_utc": context.deadline.isoformat(),
        "scope": "Post-test exploratory; validation diagnostic only; no NI qualification or untouched-test claim.",
    })


def quality_reference(
    context: PilotContext, groups: dict[str, Any],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    root = context.output / "evaluate"
    checked_phase(root, phase_binding(context, "evaluate"))
    receipts = read_json(root / "raw-query-receipts.json")
    panel = context.panels["final"]
    final_checkpoint_hash = read_json(context.output / "train" / "completed.json")["details"]["checkpoint_sha256"]
    count = len(panel["query_ids"])
    result = {}
    for arm in ("base", "adapter"):
        receipt = receipts["final-" + arm]
        if (
            receipt.get("complete") is not True or receipt["query_ids"] != panel["query_ids"]
            or receipt["arm"] != arm or receipt["model"] != context.specification["model"]
            or receipt["revision"] != context.specification["revision"]
            or receipt["specification_sha256"] != context.binding["specification_sha256"]
            or receipt["query_instruction"] != QUERY_INSTRUCTION
            or receipt["adapter_checkpoint_sha256"] != (final_checkpoint_hash if arm == "adapter" else None)
            or receipt["rows"] != count or receipt["dimension"] != 1024
            or receipt["source_panel_sha256"] != context.binding["sources"]["final_panel"]
            or receipt["group_manifest_sha256"] != sha256(context.output / "audit" / "groups.json")
            or receipt["catalog_items"] != len(context.items) or receipt["topk"] != 1000
            or groups["final"]["query_ids"] != panel["query_ids"]
        ):
            raise ValueError("RAW replay requires exact final query IDs, original panels and batch16 groups.")
        arrays = []
        for name in ("queries", "scores", "ids"):
            expected = receipt["files"][name]
            path = artifact_path(root, expected["file"])
            if sha256(path) != expected["sha256"]:
                raise ValueError("RAW reference payload bytes changed.")
            arrays.append(np.load(path, mmap_mode="r", allow_pickle=False))
        vectors, scores, ids = arrays
        checked_vectors(vectors, count, 1024)
        checked_retrieval(ids, scores, count, len(context.items))
        if (
            ids.shape != (count, 1000) or (ids < 0).any()
            or scores.dtype != np.float32 or scores.shape != ids.shape or not np.isfinite(scores).all()
        ):
            raise ValueError("RAW exact quality reference lacks full finite top1000 coverage.")
        result[arm] = (vectors, scores, ids)
    return result


def timed_operation(
    operation: Callable[[], Any], device: torch.device,
) -> tuple[Any, float, list[dict[str, Any]]]:
    before = observe_memory(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter_ns()
    actual = operation()
    torch.cuda.synchronize(device)
    milliseconds = (time.perf_counter_ns() - started) / 1e6
    after = observe_memory(device)
    if not math.isfinite(milliseconds) or milliseconds <= 0:
        raise ValueError("Actual timed operation did not produce a positive finite duration.")
    return actual, milliseconds, [before, after]


def runtime_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != 100 or {row["arm"] for row in records} != {"base", "adapter"}:
        raise ValueError("Runtime requires exactly50 actual paired observations per arm.")
    ordered = {
        arm: sorted((row for row in records if row["arm"] == arm), key=lambda row: row["sample"])
        for arm in ("base", "adapter")
    }
    if any([row["sample"] for row in rows] != list(range(50)) for rows in ordered.values()):
        raise ValueError("Runtime sample identities are missing or duplicated.")
    if [row["group"] for row in ordered["base"]] != [row["group"] for row in ordered["adapter"]]:
        raise ValueError("Paired runtime arms used different immutable groups.")
    samples = {arm: [float(row["milliseconds"]) for row in rows] for arm, rows in ordered.items()}
    if any(not math.isfinite(value) or value <= 0 for values in samples.values() for value in values):
        raise ValueError("Runtime sample durations must be actual positive finite measurements.")
    return {
        **{
            arm: {"samples_ms": values, "sample_count": 50, "mean_ms": float(np.mean(values))}
            for arm, values in samples.items()
        },
        "paired_mean_difference_ms": float(np.mean(np.asarray(samples["adapter"]) - samples["base"])),
        "scope": "Mean finite batch duration, not serving QPS or a tail SLO.",
    }


def canonical_array(values: torch.Tensor) -> np.ndarray:
    if (
        values.ndim != 2 or values.dtype != torch.float32 or not torch.isfinite(values).all()
        or not torch.equal(values, values.half().float())
    ):
        raise ValueError("Actual encoding is not finite canonical FP16-rounded FP32.")
    return values.detach().half().cpu().numpy()


def save_replays(
    root: Path, stem: str, rows: list[list[int]],
    values: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    if len(rows) != len(values) or not rows:
        raise ValueError("Replay evidence must align every recorded fixed group.")
    publish_array(root, stem + "-rows.npy", np.asarray(rows, dtype=np.int64))
    for column, name in enumerate(("queries", "scores", "ids")):
        publish_array(root, stem + "-" + name + ".npy", np.stack([row[column] for row in values]))


def run_runtime(context: PilotContext, session: int) -> dict[str, Any]:
    from rrbench.runtime import environment_manifest
    from rrbench.semantic_query_adaptation import forward_queries, load_adapter_state

    check_dependencies(context, "runtime")
    state = final_adapter(context)
    groups = read_json(context.output / "audit" / "groups.json")
    if groups != group_manifest(context):
        raise ValueError("Runtime cannot re-partition quality queries or substitute historical caches.")
    manifest = groups["final"]
    schedule = runtime_schedule(manifest["groups"], session)
    reference = quality_reference(context, groups)
    root = context.output / f"runtime-{session}"
    binding = phase_binding(context, "runtime", session)
    prior = begin_phase(root, binding)
    if prior is not None:
        return prior
    for other in (0, 1, 2):
        identity = context.output / f"runtime-{other}" / "process.json"
        if other != session and identity.exists() and read_json(identity)["process_context_id"] == PROCESS_CONTEXT_ID:
            raise ValueError("Each paired runtime context requires a genuinely fresh process.")
    publish_json(root, "process.json", {
        "pid": os.getpid(), "process_context_id": PROCESS_CONTEXT_ID, "session": session,
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    })
    publish_json(root, "schedule.json", {
        **manifest, **schedule, "session": session,
        "warmup_batch_count": 10, "measured_batch_count": 50, "probe_batch_count_per_arm": 100,
        "measured_query_observations_per_arm": 800, "probe_query_observations_per_arm": 1600,
        "independence": "Repeated fixed original-query groups, not three independent query cohorts.",
    })
    device = gpu_device()
    memory = [observe_memory(device)]
    item_gpu = bank_tensor(context.items, device, torch.float32)
    base, _ = new_encoder(context, device, adapted=False)
    adapted, metadata = new_encoder(context, device, adapted=True)
    load_adapter_state(adapted.model, state["adapters"])
    encoders = {"base": base, "adapter": adapted}
    memory.append(observe_memory(device))
    panel = context.panels["final"]
    singleton_rows = [
        manifest["groups"][index][0] for index in schedule["warmup"] + schedule["measured"]
    ]
    singleton_indices = {row: index for index, row in enumerate(singleton_rows)}
    singleton_reference = {}
    for arm, encoder in encoders.items():
        actual = []
        with torch.no_grad():
            for row in singleton_rows:
                check_deadline(context.deadline)
                actual.append(canonical_array(forward_queries(encoder, [panel["queries"][row]]))[0])
        singleton_reference[arm] = np.stack(actual)
        publish_array(root, f"batch1-{arm}-own-reference.npy", singleton_reference[arm])
    publish_json(root, "batch1-reference-scope.json", {
        "rows": singleton_rows, "query_ids": [panel["query_ids"][row] for row in singleton_rows],
        "scope": "Own batch1 repeat only; not compared with batch16 quality vectors or reported as batch1 quality.",
    })
    records: dict[str, list[dict[str, Any]]] = {}
    warmup_records = []
    summaries = {}
    for condition, batch in (("encode-b1", 1), ("encode-b16", 16), ("pipeline-b16", 16)):
        records[condition] = []
        measured_values: dict[str, list[Any]] = {"base": [], "adapter": []}
        measured_rows: dict[str, list[list[int]]] = {"base": [], "adapter": []}
        for stage in ("warmup", "measured"):
            for sample, group_index in enumerate(schedule[stage]):
                rows = manifest["groups"][group_index][:batch]
                texts = [panel["queries"][row] for row in rows]
                order = paired_order(session, sample)
                for arm in order:
                    check_deadline(context.deadline)
                    encoder = encoders[arm]

                    def operation() -> Any:
                        with torch.no_grad():
                            if condition == "pipeline-b16":
                                return raw_pipeline(encoder, item_gpu, texts)
                            return forward_queries(encoder, texts)

                    if stage == "measured":
                        value, elapsed, measured_memory = timed_operation(operation, device)
                        memory.extend(measured_memory)
                    else:
                        value = operation()
                        elapsed = None
                        memory.append(observe_memory(device))
                    if batch == 1:
                        actual_singleton = canonical_array(value)
                        expected_singleton = singleton_reference[arm][[singleton_indices[rows[0]]]]
                        if not np.array_equal(actual_singleton, expected_singleton):
                            raise ValueError("Batch1 encoding changed from its own same-query batch1 repeat.")
                        evidence: Any = actual_singleton
                        parity = {"own_batch1_repeat_exact": True, "queries": 1}
                    else:
                        if condition == "pipeline-b16":
                            evidence = observation(*value)
                        else:
                            with torch.no_grad():
                                scores, ids = exact_search(item_gpu, value)
                            evidence = observation(value, scores, ids)
                        expected = tuple(array[rows] for array in reference[arm])
                        parity = assert_quality_replay(evidence, expected)
                    record = {
                        "arm": arm, "sample": sample, "group": group_index, "rows": rows,
                        "query_ids": [panel["query_ids"][row] for row in rows],
                        "condition": condition, "batch": batch, "order": list(order), "parity": parity,
                    }
                    if stage == "measured":
                        record["milliseconds"] = elapsed
                        records[condition].append(record)
                        measured_values[arm].append(evidence)
                        measured_rows[arm].append(rows)
                    else:
                        warmup_records.append(record)
        summaries[condition] = runtime_summary(records[condition])
        for arm in ("base", "adapter"):
            stem = condition + "-" + arm
            if batch == 1:
                publish_array(root, stem + "-queries.npy", np.stack(measured_values[arm]))
                publish_array(root, stem + "-rows.npy", np.asarray(measured_rows[arm], dtype=np.int64))
            else:
                save_replays(root, stem, measured_rows[arm], measured_values[arm])
        print("PAIRED_RUNTIME_CONDITION", session, condition, json.dumps(summaries[condition]), flush=True)
    probes = []
    probe_values: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {"base": [], "adapter": []}
    probe_rows: dict[str, list[list[int]]] = {"base": [], "adapter": []}
    for sample, index in enumerate(schedule["probes"]):
        rows = manifest["groups"][index]
        for arm in paired_order(session, sample):
            check_deadline(context.deadline)
            evidence = observation(*raw_pipeline(
                encoders[arm], item_gpu, [panel["queries"][row] for row in rows],
            ))
            parity = assert_quality_replay(evidence, tuple(array[rows] for array in reference[arm]))
            probes.append({
                "arm": arm, "sample": sample, "group": index, "rows": rows,
                "query_ids": [panel["query_ids"][row] for row in rows], "parity": parity,
            })
            probe_values[arm].append(evidence)
            probe_rows[arm].append(rows)
            memory.append(observe_memory(device))
    for arm in ("base", "adapter"):
        save_replays(root, "probes-" + arm, probe_rows[arm], probe_values[arm])
    memory.append(observe_memory(device))
    memory_summary = {
        "sample_count": len(memory),
        "maximum_observed_used_bytes": max(row["memory_used_bytes"] for row in memory),
        "minimum_cuda_free_bytes": min(row["cuda_free_bytes"] for row in memory),
        "maximum_torch_peak_allocated_bytes": max(row["torch_peak_allocated_bytes"] for row in memory),
        "maximum_peak_sensitivity_bytes": max(row["peak_sensitivity_bytes"] for row in memory),
        "scope": "Finite benchmark-device samples plus Torch peak sensitivity; administrative context is parent-owned.",
    }
    publish_json(root, "raw-samples.json", {"measured": records, "warmup": warmup_records, "probes": probes})
    publish_json(root, "memory.json", {"samples": memory, "summary": memory_summary})
    publish_json(root, "environment.json", environment_manifest())
    check_deadline(context.deadline)
    return finish_pilot_phase(context, root, binding, {
        "session": session, "process_context_id": PROCESS_CONTEXT_ID,
        "warmup": 10, "samples": 50, "probe_replays_per_arm": 100,
        "summaries": summaries, "memory": memory_summary, "adapter_inventory": metadata,
        "catalog_uploads": 1, "catalog_upload_timed": False, "models_and_bank_resident": True,
        "groups_sha256": sha256(context.output / "audit" / "groups.json"),
        "scorer": "none", "deadline_utc": context.deadline.isoformat(),
        "base_path": "Fresh pinned model with no adapter wrapper/zero-LoRA overhead.",
        "scope": (
            "Batch1 encoding cost diagnostic; batch16 actual RAW-to-resident-fullbank-top1000 matched to quality. "
            "Three fresh contexts reuse one fixed query partition, not three independent query samples. "
            "No model-parameter-only, untouched-test, energy, MFU, QPS or p99 claim."
        ),
    })


def execute_phase(context: PilotContext, phase: str, session: int | None) -> dict[str, Any]:
    if phase == "runtime":
        if session is None:
            raise ValueError("Runtime requires an explicit fresh-process --session 0|1|2.")
        return run_runtime(context, session)
    if session is not None:
        raise ValueError("--session is only valid for runtime.")
    runners = {
        "audit": run_audit, "mine": run_mine, "smoke": run_smoke,
        "train": run_train, "evaluate": run_evaluate,
    }
    return runners[phase](context)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argument_parser()
    args = parser.parse_args(argv)
    if (args.phase == "runtime") != (args.session is not None):
        parser.error("--session is required exactly for --phase runtime")
    context = load_context(args)
    try:
        execute_phase(context, args.phase, args.session)
    except (RuntimeError, TimeoutError) as error:
        if not isinstance(error, TimeoutError) and not memory_failure(error):
            raise
        root = context.output / (f"runtime-{args.session}" if args.phase == "runtime" else args.phase)
        failure = {
            "complete": False, "status": "deadline_exceeded" if isinstance(error, TimeoutError) else "unsupported_memory",
            "reason": str(error), "error_type": type(error).__name__,
            "binding": phase_binding(context, args.phase, args.session),
            "time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "scope": "Explicit bounded failure, never successful pilot evidence or permission to escalate.",
        }
        if isinstance(error, MemoryGateFailure):
            failure["failed_observation"] = error.observation
        publish_json(root, "failure.json", failure)
        raise
    print("QUERY_ADAPTATION_PHASE_COMPLETE", args.phase, args.session, flush=True)


if __name__ == "__main__":
    main()
