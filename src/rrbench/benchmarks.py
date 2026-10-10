from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rrbench.ann import FaissANN, FaissHNSW, SilverANN, train_partition
from rrbench.exact import exact_topk
from rrbench.filters import BloomFilter, InvertedFilter, evaluate, make_features, parse_expression
from rrbench.io import append_jsonl, bind_run_config, write_json
from rrbench.metrics import bootstrap_mean_ci, latency_summary, recall_at_k
from rrbench.models import make_model
from rrbench.runtime import configure, environment_manifest, measure, require_cuda
from rrbench.synthetic import generate_catalog


@torch.inference_mode()
def run_ann(args: Any) -> None:
    import faiss

    configure(args.seed, args.threads)
    faiss.omp_set_num_threads(args.threads)
    gpu_requested = any(not name.endswith("-cpu") for name in args.backends)
    cuda_available = torch.cuda.is_available()
    device = require_cuda() if gpu_requested else torch.device("cuda" if cuda_available else "cpu")
    if max(args.ks) > args.items or max(args.probes) > args.clusters:
        raise ValueError("k and nprobe must fit the catalog and cluster count.")
    if max(args.batches) > args.queries or args.repeats <= 0:
        raise ValueError("Batch sizes must fit the query count; repeats must be positive.")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = {name: getattr(args, name) for name in (
        "items", "dimension", "queries", "seed", "distribution", "clusters",
        "ks", "probes", "batches", "warmup", "iterations", "repeats", "threads", "backends",
    )}
    bind_run_config(output, {"experiment": "ann", **config})
    write_json(output / "manifest.json", {
        "experiment": "synthetic ANN", "config": config, "environment": environment_manifest(),
        "metric": "inner_product", "upstream_commit": "22c2007ed5d28ec8304de081cc2393778708ad12",
        "latency_scope": "full local ANN: probing, metadata, scoring, top-k, ID remapping; no RPC",
        "queries_residency": "CPU for CPU indexes, GPU for GPU indexes; transfers excluded",
        "oracle_device": device.type,
    })
    items, queries = generate_catalog(
        output / "data", args.items, args.dimension, args.queries, args.seed, args.distribution
    )
    query_gpu = torch.as_tensor(queries, device=device)
    truth_path = output / "truth.npy"
    if truth_path.exists():
        truth = np.load(truth_path)
        if truth.ndim != 2 or truth.shape[0] != args.queries or truth.shape[1] < max(args.ks):
            raise ValueError("Existing ground truth has the wrong shape.")
        truth = truth[:, : max(args.ks)]
    else:
        _, truth_tensor = exact_topk(items, query_gpu, max(args.ks))
        truth = truth_tensor.cpu().numpy()
        np.save(truth_path, truth)
    partition = train_partition(items, args.clusters, args.seed, output / "index")
    journal = output / "ann.jsonl"
    if journal.exists():
        import json

        completed = {
            (row["backend"], row["k"], row["probes"], row["batch_size"], row["repeat"])
            for line in journal.read_text().splitlines() if line.strip()
            for row in [json.loads(line)] if row.get("status") in ("ok", "unsupported")
        }
    else:
        completed = set()
    for name in args.backends:
        if cuda_available:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        free_before = torch.cuda.mem_get_info()[0] if cuda_available else 0
        started = time.perf_counter()
        if name.startswith("silver-"):
            backend = SilverANN(items, partition, name.removeprefix("silver-"))
        elif name == "faiss-hnsw-cpu":
            backend = FaissHNSW(items)
        elif name in ("faiss-ivf-cpu", "faiss-ivf-gpu", "faiss-ivf-gpu-fp16"):
            backend = FaissANN(items, partition, gpu="gpu" in name, half=name.endswith("fp16"))
        else:
            raise ValueError(f"Unknown ANN backend: {name}")
        if cuda_available:
            torch.cuda.synchronize()
        build_seconds = time.perf_counter() - started
        free_after = torch.cuda.mem_get_info()[0] if cuda_available else 0
        resident = max(0, free_before - free_after)
        query_values = query_gpu if backend.cuda else torch.from_numpy(queries)
        for k in args.ks:
            for probes in args.probes:
                unsupported = backend.cuda and name.startswith("faiss") and (k > 2048 or probes > 2048)
                quality = None
                if not unsupported:
                    predicted = []
                    for start in range(0, len(queries), max(args.batches)):
                        _, ids = backend.search(query_values[start : start + max(args.batches)], k, probes)
                        predicted.append(ids.cpu().numpy())
                    quality = recall_at_k(np.concatenate(predicted), truth, k)
                for batch in args.batches:
                    for repeat in range(args.repeats):
                        key = (name, k, probes, batch, repeat)
                        if key in completed:
                            continue
                        row: dict[str, object] = {
                            "backend": name, "k": k, "probes": probes,
                            "batch_size": batch, "repeat": repeat, "seed": args.seed,
                            "items": args.items, "dimension": args.dimension,
                            "distribution": args.distribution, "build_seconds": build_seconds,
                            "gpu_resident_delta_bytes": resident,
                            "effort_parameter": "efSearch" if name == "faiss-hnsw-cpu" else "nprobe",
                            "effective_effort": max(k, probes) if name == "faiss-hnsw-cpu" else probes,
                        }
                        if unsupported:
                            row.update(status="unsupported", reason="Classic FAISS GPU k/nprobe limit is 2048.")
                        else:
                            start = repeat * batch % max(1, len(queries) - batch + 1)
                            query = query_values[start : start + batch]
                            if cuda_available:
                                torch.cuda.reset_peak_memory_stats()
                            _, samples = measure(
                                lambda: backend.search(query, k, probes),
                                args.warmup, args.iterations, backend.cuda,
                            )
                            if quality is None:
                                raise RuntimeError("Missing ANN quality measurements.")
                            row.update(
                                status="ok", recall=float(quality.mean()),
                                recall_ci95=bootstrap_mean_ci(quality, args.seed),
                                per_query_recall=quality.tolist(),
                                torch_peak_allocated_bytes=torch.cuda.max_memory_allocated() if cuda_available else 0,
                                **latency_summary(samples, batch),
                            )
                        append_jsonl(journal, row)
                        print(f"{name} N={args.items} k={k} probes={probes} batch={batch} repeat={repeat} {row['status']}", flush=True)
        del backend
        gc.collect()
        if cuda_available:
            torch.cuda.empty_cache()
    write_json(output / "completed.json", {"complete": True, "experiment": "synthetic ANN"})


@torch.inference_mode()
def run_filter(args: Any) -> None:
    configure(args.seed, args.threads)
    device = require_cuda()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    features_cpu = torch.from_numpy(make_features(args.items, args.seed, args.features, args.values))
    features_gpu = features_cpu.to(device)
    inverted = InvertedFilter(features_cpu.numpy())
    expressions = [
        "1:1", "(1:1 OR 1:2) AND (2:1 OR 2:2)",
        "1:1 AND NOT 2:1", "NOT (1:1 OR 2:2)", "(1:1 OR 2:2) AND NOT (3:1 AND 3:2)",
    ]
    expressions = expressions[: args.query_types]
    write_json(output / "manifest.json", {
        "experiment": "synthetic filtering", "items": args.items, "features": args.features,
        "values_per_feature": args.values, "seed": args.seed, "expressions": expressions,
        "environment": environment_manifest(),
        "correctness_contract": "conservative positive Bloom + exact verification; raw NOT audited separately",
    })
    for multiplier in args.multipliers:
        started = time.perf_counter()
        bloom = BloomFilter(features_gpu, multiplier=multiplier)
        torch.cuda.synchronize()
        build_seconds = time.perf_counter() - started
        for expression in expressions:
            predicate = parse_expression(expression)
            exact_cpu = evaluate(predicate, features_cpu)
            exact_gpu = exact_cpu.to(device)
            plans = bloom.plans([expression])
            raw_plans = bloom.plans([expression], conservative=False)
            raw = bloom.search(raw_plans)[0]
            conservative = bloom.search(plans)[0]
            raw_false_negatives = int((exact_gpu & ~raw).sum().item())
            conservative_false_negatives = int((exact_gpu & ~conservative).sum().item())
            if conservative_false_negatives:
                raise RuntimeError("Conservative Bloom prefilter lost eligible items.")
            nonmatches = int((~exact_gpu).sum().item())
            fp = int((conservative & ~exact_gpu).sum().item())
            operations = {
                "exact-inverted-cpu": (lambda: inverted.search(predicate), False),
                "exact-forward-cpu": (lambda: evaluate(predicate, features_cpu), False),
                "exact-forward-gpu": (lambda: evaluate(predicate, features_gpu), True),
                "silver-bloom-packed": (lambda: bloom.search(plans, packed=True), True),
                "silver-bloom-exact-bool": (
                    lambda: bloom.search(plans) & evaluate(predicate, features_gpu)[None, :], True
                ),
            }
            for name, (operation, cuda) in operations.items():
                result, samples = measure(operation, args.warmup, args.iterations, cuda)
                if name == "silver-bloom-exact-bool":
                    torch.testing.assert_close(result[0], exact_gpu)
                elif name == "exact-inverted-cpu":
                    np.testing.assert_array_equal(result, exact_cpu.numpy())
                append_jsonl(output / "filter.jsonl", {
                    "backend": name, "expression": expression, "multiplier": multiplier,
                    "items": args.items, "build_seconds": build_seconds,
                    "index_bytes": bloom.index.numel() * bloom.index.element_size() + bloom.offsets.numel() * 8,
                    "raw_bloom_false_negatives": raw_false_negatives,
                    "conservative_false_negatives": conservative_false_negatives,
                    "conservative_false_positives": fp,
                    "conservative_false_positive_rate": fp / nonmatches if nonmatches else 0.0,
                    "eligible_count": int(exact_cpu.sum()), "status": "ok",
                    **latency_summary(samples, 1),
                })
                print(f"filter {name} N={args.items} multiplier={multiplier} query={expression}", flush=True)
        del bloom
        torch.cuda.empty_cache()
    write_json(output / "completed.json", {"complete": True, "experiment": "filtering"})


@torch.inference_mode()
def run_rank_timing(args: Any) -> None:
    configure(args.seed, args.threads)
    device = require_cuda()
    output = Path(args.output)
    write_json(output / "manifest.json", {
        "experiment": "ranker runtime only", "trained": False,
        "models": args.models, "environment": environment_manifest(),
        "interpretation": "Random weights establish execution cost only; no recommendation-quality conclusions.",
        "feature_budget": "user ID, item ID, two dense synthetic counts",
    })
    for name in args.models:
        model = make_model(name, args.users, args.items, device, args.seed).eval()
        for candidates in args.candidates:
            for batch in args.batches:
                pair_count = candidates * batch
                users = torch.randint(args.users, (batch,), device=device).repeat_interleave(candidates)
                items = torch.randint(args.items, (pair_count,), device=device)
                dense = torch.rand(pair_count, 2, device=device)
                operation = lambda: model(users, items, dense).reshape(batch, candidates).topk(min(100, candidates), dim=1)
                prediction, samples = measure(operation, args.warmup, args.iterations, True)
                if not torch.isfinite(prediction.values).all():
                    raise RuntimeError(f"{name} produced non-finite scores.")
                append_jsonl(output / "rank_timing.jsonl", {
                    "backend": name, "candidates": candidates, "batch_size": batch,
                    "pairs": pair_count, "trained": False, "status": "ok",
                    "parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "model_class": getattr(model, "model_class", f"{type(model).__module__}.{type(model).__name__}"),
                    **latency_summary(samples, batch),
                })
                print(f"rank {name} candidates={candidates} batch={batch}", flush=True)
        del model
        torch.cuda.empty_cache()
    write_json(output / "completed.json", {"complete": True, "experiment": "rank timing"})
