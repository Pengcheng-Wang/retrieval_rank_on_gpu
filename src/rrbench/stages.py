from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import torch

from rrbench.ann import SilverANN, train_partition
from rrbench.io import append_jsonl, write_json
from rrbench.metrics import latency_summary
from rrbench.quantization import quantize_symmetric
from rrbench.runtime import configure, environment_manifest, measure, require_cuda
from rrbench.synthetic import generate_catalog


@torch.inference_mode()
def run_stages(args: Any) -> None:
    configure(args.seed, args.threads)
    device = require_cuda()
    output = Path(args.output)
    items, queries = generate_catalog(
        output / "data", args.items, args.dimension, args.queries, args.seed, args.distribution
    )
    partition = train_partition(items, args.clusters, args.seed, output / "index")
    write_json(output / "stages_manifest.json", {
        "experiment": "isolated candidate-scoring stages", "environment": environment_manifest(),
        "scope": "query quantization and centroid probing prepared outside timing; no top-k; not full ANN",
        "naive_reference": "same native-selected IDs gathered into [batch, candidates, dimension], followed by bmm",
    })
    for representation in ("fp32", "fp16", "int8-global"):
        backend = SilverANN(items, partition, representation)
        for probes in args.probes:
            for batch in args.batches:
                query = torch.tensor(queries[:batch], device=device)
                selected, lengths, maximum = backend.probes(query, probes)
                if representation.startswith("int8"):
                    query_values = quantize_symmetric(query, "rowwise").values
                else:
                    query_values = query.to(backend.embeddings.dtype)

                def native():
                    return torch.ops.st.fused_kmean_ann(
                        backend.offsets, selected, lengths, backend.embeddings, query_values, maximum
                    )

                scores, ids = native()
                operations = {"fused-native": native}
                if not representation.startswith("int8"):
                    def gather():
                        vectors = backend.embeddings[ids.long().clamp_min(0)]
                        values = torch.bmm(vectors, query_values.unsqueeze(-1)).squeeze(-1)
                        return values.masked_fill(ids < 0, -torch.inf)

                    reference = gather()
                    valid = ids >= 0
                    torch.testing.assert_close(
                        scores[valid], reference[valid],
                        atol=0.002 if representation == "fp16" else 1e-4, rtol=0.005,
                    )
                    operations["torch-gather-bmm"] = gather
                for method, operation in operations.items():
                    for repeat in range(args.repeats):
                        torch.cuda.reset_peak_memory_stats()
                        baseline = torch.cuda.memory_allocated()
                        _, samples = measure(operation, args.warmup, args.iterations, True)
                        append_jsonl(output / "stages.jsonl", {
                            "backend": method, "representation": representation,
                            "items": args.items, "dimension": args.dimension, "probes": probes,
                            "batch_size": batch, "repeat": repeat,
                            "candidate_capacity": maximum, "status": "ok",
                            "torch_incremental_peak_bytes": max(0, torch.cuda.max_memory_allocated() - baseline),
                            **latency_summary(samples, batch),
                        })
                        print(f"stage {method} {representation} N={args.items} probes={probes} batch={batch} repeat={repeat}", flush=True)
        del backend
        gc.collect()
        torch.cuda.empty_cache()
