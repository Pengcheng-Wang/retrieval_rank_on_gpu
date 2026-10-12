from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rrbench.io import bind_run_config, sha256, write_json


def build_plan(
    registry: dict[str, Any], corpora: dict[str, Path], output: Path, workers: int,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("Encoding worker count must be positive.")
    worker_order = [
        node * 8 + gpu for gpu in range(8) for node in range(workers // 8)
    ] if workers % 8 == 0 else list(range(workers))
    manifests = {}
    for name, directory in corpora.items():
        manifest = json.loads((directory / "corpus.json").read_text(encoding="utf-8"))
        position = 0
        for shard in manifest["shards"]:
            if shard["start"] != position or shard["rows"] <= 0:
                raise ValueError("Corpus shard ranges must cover contiguous unique rows.")
            if sha256(directory / shard["file"]) != shard["sha256"]:
                raise ValueError("Corpus shard checksum changed before scheduling.")
            position += shard["rows"]
        if position != manifest["rows"]:
            raise ValueError("Corpus shard counts do not cover the full declared catalog.")
        manifests[name] = manifest
    tasks = []
    models = sorted(
        ((key, spec) for key, spec in registry["models"].items() if "dimensions" in spec),
        key=lambda item: (item[0] != "qwen8", item[0] != "qwen06", item[0]),
    )
    for model, spec in models:
        for corpus in spec["corpora"]:
            if corpus not in corpora:
                continue
            for shard in manifests[corpus]["shards"]:
                key = f"{corpus}-{model}-{shard['start']:09d}"
                tasks.append({
                    "key": key, "model": model, "corpus": corpus,
                    "source": str(corpora[corpus] / shard["file"]),
                    "source_sha256": shard["sha256"],
                    "output": str(output / corpus / model / key),
                    "start": shard["start"], "rows": shard["rows"],
                    "batch_size": 32 if model == "qwen8" else 64,
                    "worker": worker_order[len(tasks) % workers],
                })
    if not tasks:
        raise ValueError("No declared semantic model/corpus tasks are available.")
    return {
        "workers": workers, "tasks": tasks,
        "rows_per_corpus": {name: manifest["rows"] for name, manifest in manifests.items()},
        "registry": registry,
        "policy": "Finite disjoint shard assignments; publish only checked complete-shard receipts.",
    }


def worker_tasks(plan: dict[str, Any], worker: int) -> list[dict[str, Any]]:
    if not 0 <= worker < plan["workers"]:
        raise ValueError("Encoding worker index is outside the frozen plan.")
    return [task for task in plan["tasks"] if task["worker"] == worker]


def validate_completed(directory: Path, rows: int, start: int) -> dict[str, Any]:
    completed = json.loads((directory / "completed.json").read_text(encoding="utf-8"))
    if not completed["complete"] or completed["rows"] != rows:
        raise ValueError("Embedding completion receipt does not cover every expected row.")
    ids_path = directory / "ids.npy"
    if sha256(ids_path) != completed["ids_sha256"]:
        raise ValueError("Embedding IDs checksum differs from the complete receipt.")
    ids = np.load(ids_path, mmap_mode="r")
    if ids.dtype != np.int64 or ids.shape != (rows,) or not np.array_equal(ids, np.arange(start, start + rows)):
        raise ValueError("Embedding IDs do not match the exact global shard row mapping.")
    representations = [{
        "file": "vectors.npy", "dimension": completed.get("stored_dimension", completed["native_dimension"]),
        "sha256": completed["vectors_sha256"],
    }, *completed["mrl_representations"].values()]
    for representation in representations:
        path = directory / representation["file"]
        if sha256(path) != representation["sha256"]:
            raise ValueError("Embedding vector checksum differs from its complete receipt.")
        vectors = np.load(path, mmap_mode="r")
        if vectors.dtype != np.float16 or vectors.shape != (rows, representation["dimension"]):
            raise ValueError("Embedding representation shape or storage dtype changed.")
        for position in range(0, rows, 8192):
            block = np.asarray(vectors[position:position + 8192], dtype=np.float32)
            if not np.isfinite(block).all() or not np.allclose(np.linalg.norm(block, axis=1), 1, atol=0.002):
                raise ValueError("Embedding shard contains invalid or nonnormalized rows.")
    return completed


def merge_corpus(tasks: list[dict[str, Any]], output: Path, dimension: int) -> dict[str, Any]:
    ordered = sorted(tasks, key=lambda task: task["start"])
    receipts, position = [], 0
    for task in ordered:
        if task["start"] != position:
            raise ValueError("Merged corpus cannot omit, duplicate or reorder any global row.")
        receipt = validate_completed(Path(task["output"]), task["rows"], task["start"])
        if receipt["native_dimension"] == dimension:
            filename = "vectors.npy"
        else:
            representation = receipt["mrl_representations"].get(str(dimension))
            if representation is None:
                raise ValueError("Requested merged dimension is not a measured stored representation.")
            filename = representation["file"]
        receipts.append((task, receipt, filename))
        position += task["rows"]
    if position == 0:
        raise ValueError("Cannot publish an empty merged semantic catalog.")
    model_identity = {
        name: receipts[0][1].get(name)
        for name in ("model", "revision", "specification_sha256")
    }
    if any(
        {name: receipt.get(name) for name in model_identity} != model_identity
        for _, receipt, _ in receipts
    ):
        raise ValueError("Merged semantic shards use different encoder/model specifications.")
    binding = {
        "dimension": dimension, "rows": position,
        "sources": [
            {"start": task["start"], "rows": task["rows"], "receipt_sha256": sha256(Path(task["output"]) / "completed.json")}
            for task, _, _ in receipts
        ],
    }
    bind_run_config(output, binding)
    if (output / "completed.json").exists():
        return validate_completed(output, position, 0)
    temporary = output / "vectors.partial.npy"
    matrix = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float16, shape=(position, dimension),
    )
    for task, _, filename in receipts:
        source = np.load(Path(task["output"]) / filename, mmap_mode="r")
        matrix[task["start"]:task["start"] + task["rows"]] = source
    matrix.flush()
    del matrix
    temporary.replace(output / "vectors.npy")
    np.save(output / "ids.npy", np.arange(position, dtype=np.int64))
    result = {
        "complete": True, "rows": position, "native_dimension": receipts[0][1]["native_dimension"],
        "stored_dimension": dimension, "stored_dtype": "float16",
        "vectors_sha256": sha256(output / "vectors.npy"), "ids_sha256": sha256(output / "ids.npy"),
        "mrl_representations": {}, "source_shards": binding["sources"],
        "normalization": "Unmodified normalized FP16 shard vectors; exact oracle uses these stored values.",
        "encoder_native_dimension": receipts[0][1]["native_dimension"],
        "representation": {
            **model_identity, "stored_dimension": dimension, "stored_dtype": "float16",
            "normalization": "FP32 unit normalization then canonicalFP16storage; MRLprefix re-normalized beforestorage.",
        },
    }
    write_json(output / "completed.json", result)
    return result


def run_worker(plan: dict[str, Any], worker: int, cache: Path, smoke: Path) -> None:
    from rrbench.semantic_models import (
        encode_shard, validate_encoding_contract, validate_model_files, validate_smoke_contract,
    )

    verified_smoke = json.loads(smoke.read_text(encoding="utf-8"))
    if not verified_smoke["complete"]:
        raise ValueError("Full real-text semantic model smoke must complete before encoding.")
    smoke_models = {row["key"]: row for row in verified_smoke["models"]}
    files_checked = set()
    for task in worker_tasks(plan, worker):
        spec = plan["registry"]["models"][task["model"]]
        verified = smoke_models[task["model"]]
        validate_smoke_contract(spec, verified)
        if task["model"] not in files_checked:
            validate_model_files(spec, cache, verified["model_files"])
            files_checked.add(task["model"])
        if sha256(Path(task["source"])) != task["source_sha256"]:
            raise ValueError("Scheduled text shard changed after the plan was frozen.")
        print(f"SHARD_START worker={worker} task={task['key']}", flush=True)
        encode_shard(
            Path(task["source"]), Path(task["output"]), spec, cache, task["batch_size"],
        )
        completed = validate_completed(Path(task["output"]), task["rows"], task["start"])
        validate_encoding_contract(spec, completed)
        print(f"SHARD_VERIFIED worker={worker} task={task['key']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finite audited semantic encoding shards and merge.")
    commands = parser.add_subparsers(dest="operation", required=True)
    plan_command = commands.add_parser("plan")
    plan_command.add_argument("--registry", required=True)
    plan_command.add_argument("--corpus", action="append", required=True, help="name=audited-directory")
    plan_command.add_argument("--output-root", required=True)
    plan_command.add_argument("--workers", type=int, required=True)
    plan_command.add_argument("--output", required=True)
    worker_command = commands.add_parser("worker")
    worker_command.add_argument("--plan", required=True)
    worker_command.add_argument("--worker", type=int, required=True)
    worker_command.add_argument("--cache", required=True)
    worker_command.add_argument("--smoke", required=True)
    worker_command.add_argument("--start-gate")
    worker_command.add_argument("--lease-owner")
    worker_command.add_argument("--parent", type=int)
    worker_command.add_argument("--parent-start", type=int)
    merge_command = commands.add_parser("merge")
    merge_command.add_argument("--plan", required=True)
    merge_command.add_argument("--corpus", required=True)
    merge_command.add_argument("--model", required=True)
    merge_command.add_argument("--dimension", type=int, required=True)
    merge_command.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.operation == "plan":
        registry = json.loads(Path(args.registry).read_text(encoding="utf-8"))
        corpora = {name: Path(path) for name, path in (value.split("=", 1) for value in args.corpus)}
        plan = build_plan(registry, corpora, Path(args.output_root), args.workers)
        path = Path(args.output)
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Existing encoding plan has different tasks; use a distinct version.")
        write_json(path, plan)
        print(f"ENCODING_PLAN_FROZEN tasks={len(plan['tasks'])} workers={args.workers}", flush=True)
    else:
        if args.operation == "worker" and args.start_gate:
            from rrbench.process_lease import guard_worker_parent, wait_for_start_gate

            if args.lease_owner is None or args.parent is None or args.parent_start is None:
                parser.error("start-gate requires lease-owner, parent and parent-start")
            guard_worker_parent(args.parent, args.parent_start)
            wait_for_start_gate(
                Path(args.start_gate), args.lease_owner, args.worker, args.parent, args.parent_start,
            )
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        if args.operation == "worker":
            run_worker(plan, args.worker, Path(args.cache), Path(args.smoke))
        else:
            tasks = [
                task for task in plan["tasks"]
                if task["corpus"] == args.corpus and task["model"] == args.model
            ]
            result = merge_corpus(tasks, Path(args.output), args.dimension)
            print(f"FULL_ENCODING_MERGED rows={result['rows']} dim={args.dimension}", flush=True)


if __name__ == "__main__":
    main()
