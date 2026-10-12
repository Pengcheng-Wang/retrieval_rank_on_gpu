from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from rrbench.ann import Partition, SilverANN
from rrbench.filters import evaluate, exact_candidate_mask, make_features, parse_expression
from rrbench.io import write_json
from rrbench.metrics import latency_summary
from rrbench.runtime import configure, environment_manifest, measure, require_cuda


@torch.inference_mode()
def profile_pipeline(args: Any) -> None:
    configure(args.seed, args.threads)
    device = require_cuda()
    directory = Path(args.input)
    items = np.load(directory / "data" / "items.npy", mmap_mode="r")
    queries = torch.tensor(np.load(directory / "data" / "queries.npy")[:16], device=device)
    arrays = np.load(directory / "index" / "partition.npz")
    partition = Partition(*(arrays[key] for key in ("centroids", "assignments", "order", "offsets")))
    metadata = make_features(len(items), args.seed + 1)
    backend = SilverANN(items, partition, "fp32", metadata)
    expression = " OR ".join(f"1:{value}" for value in range(1, 17))
    texts = [expression] * len(queries)
    plans = backend.bloom.plans(texts)
    selected, lengths, maximum = backend.probes(queries, 24)

    def scoring():
        return backend.score(queries, selected, lengths, maximum, plans)

    scores, ids = scoring()

    def sequential_exact():
        accepted = []
        for row, text in enumerate(texts):
            feature_rows = backend.features[ids[row].long().clamp_min(0)]
            accepted.append(evaluate(parse_expression(text), feature_rows))
        return torch.stack(accepted)

    def batched_exact():
        feature_rows = backend.features[ids.long().clamp_min(0)]
        return exact_candidate_mask(feature_rows, texts)

    torch.testing.assert_close(sequential_exact(), batched_exact())
    result = {
        "experiment": "exact-verification root-cause profile",
        "items": len(items), "batch": len(queries), "probes": 24,
        "candidate_capacity": maximum, "environment": environment_manifest(),
        "same_predicate_verified": True, "stages": {},
    }
    for name, operation in (
        ("fused_partial_scoring", scoring),
        ("sequential_exact_verification", sequential_exact),
        ("batched_exact_verification", batched_exact),
        ("complete_retrieval_filter", lambda: backend.search(queries, 512, 24, texts, plans)),
    ):
        _, samples = measure(operation, args.warmup, args.iterations, True)
        result["stages"][name] = latency_summary(samples, len(queries))
    write_json(Path(args.output), result)
    print({name: row["latency_mean_ms"] for name, row in result["stages"].items()})
