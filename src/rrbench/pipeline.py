from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rrbench.ann import FaissANN, SilverANN, train_partition
from rrbench.exact import exact_topk
from rrbench.filters import evaluate, make_features, parse_expression
from rrbench.io import append_jsonl, write_json
from rrbench.metrics import latency_summary, recall_at_k
from rrbench.models import make_model
from rrbench.runtime import configure, environment_manifest, measure, require_cuda
from rrbench.synthetic import generate_catalog


@torch.inference_mode()
def run_pipeline(args: Any) -> None:
    import faiss

    configure(args.seed, args.threads)
    faiss.omp_set_num_threads(args.threads)
    device = require_cuda()
    if max(args.candidates) * args.oversample > 2048:
        raise ValueError("FAISS pipeline oversampling must stay within its 2048 top-k limit.")
    if max(args.batches) > args.queries or max(args.probes) > args.clusters:
        raise ValueError("Batch sizes and probes must fit queries and clusters.")
    output = Path(args.output)
    items, queries = generate_catalog(
        output / "data", args.items, 128, args.queries, args.seed, "clustered"
    )
    partition = train_partition(items, args.clusters, args.seed, output / "index")
    features = make_features(args.items, args.seed + 1)
    features_cpu, features_gpu = torch.from_numpy(features), torch.tensor(features, device=device)
    expression = " OR ".join(f"1:{value}" for value in range(1, 17))
    predicate = parse_expression(expression)
    eligible = evaluate(predicate, features_gpu)
    query_gpu = torch.tensor(queries, device=device)
    _, truth = exact_topk(
        items, query_gpu, max(args.candidates), eligible[None, :].expand(args.queries, -1)
    )
    truth_numpy = truth.cpu().numpy()
    write_json(output / "manifest.json", {
        "experiment": "local retrieval-filter-rerank pipeline", "trained": False,
        "environment": environment_manifest(), "items": args.items, "queries": args.queries,
        "filter": expression, "eligible_fraction": float(eligible.float().mean()),
        "feature_budget": "user/item IDs and two dense values for all rerankers",
        "faiss_policy": f"post-filter exact verification with {args.oversample}x ANN oversampling",
        "silver_policy": "filter before top-k; exact verification before candidate truncation",
        "latency_scope": "GPU query to GPU output, including CPU transfers where required; no RPC or user tower",
        "filter_plan_preparation": "Expression parsing and plan upload are outside timing.",
        "interpretation": "runtime plus geometric candidate recall only; random rerank weights have no quality meaning",
    })
    for backend_name in args.backends:
        if backend_name.startswith("silver-"):
            representation = backend_name.removeprefix("silver-")
            mode = "full" if representation.endswith("-full") else "partial"
            representation = representation.removesuffix("-full")
            backend = SilverANN(items, partition, representation, features, mode)
        else:
            backend = FaissANN(items, partition, gpu="gpu" in backend_name)
        for model_name in args.models:
            ranker = make_model(model_name, args.queries, args.items, device, args.seed).eval()
            for candidate_count in args.candidates:
                for probes in args.probes:
                    for batch in args.batches:
                        texts = [expression] * batch
                        plans = backend.bloom.plans(texts) if isinstance(backend, SilverANN) else None
                        dense = torch.ones(batch * candidate_count, 2, device=device)
                        user_ids = torch.arange(batch, device=device).repeat_interleave(candidate_count)

                        def retrieve() -> tuple[torch.Tensor, torch.Tensor]:
                            if isinstance(backend, SilverANN):
                                return backend.search(query_gpu[:batch], candidate_count, probes, texts, plans)
                            scores, ids = backend.search(
                                query_gpu[:batch], candidate_count * args.oversample, probes
                            )
                            metadata = features_gpu if backend.cuda else features_cpu
                            accepted = evaluate(predicate, metadata[ids.long().clamp_min(0)])
                            scores = scores.masked_fill((ids < 0) | ~accepted, -torch.inf)
                            chosen_scores, positions = scores.topk(candidate_count, dim=1)
                            chosen_ids = ids.gather(1, positions).masked_fill(~torch.isfinite(chosen_scores), -1)
                            return chosen_scores.to(device), chosen_ids.to(device)

                        def pipeline() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                            _, candidate_ids = retrieve()
                            values = ranker(user_ids, candidate_ids.clamp_min(0).reshape(-1), dense)
                            scores = values.reshape(batch, candidate_count).masked_fill(candidate_ids < 0, -torch.inf)
                            final_scores, positions = scores.topk(min(100, candidate_count), dim=1)
                            return final_scores, candidate_ids.gather(1, positions), candidate_ids

                        result, samples = measure(pipeline, args.warmup, args.iterations, True)
                        _, _, candidates = result
                        recall = recall_at_k(candidates.cpu().numpy(), truth_numpy[:batch], candidate_count)
                        append_jsonl(output / "pipeline.jsonl", {
                            "backend": backend_name, "ranker": model_name, "candidates": candidate_count,
                            "probes": probes, "batch_size": batch, "items": args.items,
                            "status": "ok", "trained": False, "candidate_recall": float(recall.mean()),
                            "candidate_fill_fraction": float((candidates >= 0).float().mean()),
                            "filtering_implementation": (
                                "native-gpu-full-mask" if backend_name.endswith("-full")
                                else "native-gpu-partial-mask" if backend_name.startswith("silver-")
                                else "faiss-post-filter"
                            ),
                            "verification_implementation": (
                                "batched-exact-predicate-v2" if backend_name.startswith("silver-")
                                else "batched-faiss-postfilter-v1"
                            ),
                            **latency_summary(samples, batch),
                        })
                        print(f"pipeline {backend_name}+{model_name} k={candidate_count} probes={probes} batch={batch}", flush=True)
            del ranker
            gc.collect()
            torch.cuda.empty_cache()
        del backend
        gc.collect()
        torch.cuda.empty_cache()
    write_json(output / "completed.json", {"complete": True, "experiment": "local pipeline"})
