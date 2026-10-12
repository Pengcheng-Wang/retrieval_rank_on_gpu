from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rrbench.ann import FaissANN, SilverANN, train_partition
from rrbench.exact import exact_topk
from rrbench.io import append_jsonl, bind_run_config, sha256, write_json
from rrbench.metrics import recall_at_k, single_positive_metrics
from rrbench.models import make_model
from rrbench.movielens import load_movielens
from rrbench.quality import Features
from rrbench.runtime import configure, environment_manifest, require_cuda


@torch.inference_mode()
def run_quality_ann(args: Any) -> None:
    import faiss

    configure(args.seed, args.threads)
    faiss.omp_set_num_threads(args.threads)
    device = require_cuda()
    output, checkpoint_root = Path(args.output), Path(args.checkpoints)
    output.mkdir(parents=True, exist_ok=True)
    bind_run_config(output, {
        "seed": args.seed, "eval_users": args.eval_users, "clusters": args.clusters,
        "probes": args.probes, "candidates": args.candidates, "backends": args.backends,
        "checkpoints": {
            name: sha256(checkpoint_root / f"{name}_s{args.seed}.pt")
            for name in ("mf", "mlp", "torchrec-dlrm", "torchrec-dcn")
        },
    })
    if (output / "completed.json").exists():
        print("Frozen-checkpoint ANN-quality evaluation already completed.", flush=True)
        return
    data = load_movielens(Path(args.data))
    features = Features(data, device)
    rng = np.random.default_rng(20261010)
    selected_users = np.sort(rng.choice(len(data.eval_users), min(args.eval_users, len(data.eval_users)), replace=False))
    users_array = data.eval_users[selected_users]
    targets = data.test_targets[selected_users]
    catalog = torch.tensor(data.catalog, device=device)
    if not 0 < max(args.candidates) <= len(catalog) or max(args.probes) > args.clusters:
        raise ValueError("Candidate and probe counts must fit the frozen catalog and partition.")
    seen = torch.tensor(data.test_seen[users_array][:, data.catalog], device=device)
    users = torch.tensor(users_array, device=device)
    models = {}
    for name in ("mf", "mlp", "torchrec-dlrm", "torchrec-dcn"):
        checkpoint = checkpoint_root / f"{name}_s{args.seed}.pt"
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if state["model"] != name or state["seed"] != args.seed:
            raise ValueError("Frozen checkpoint identity does not match the experiment.")
        model = make_model(name, data.users, data.items, device, args.seed).eval()
        model.load_state_dict(state["state_dict"])
        models[name] = model
    item_vectors = models["mf"].item.weight[catalog].cpu().numpy()
    queries = models["mf"].user.weight[users].detach()
    partition = train_partition(item_vectors, args.clusters, args.seed, output / "index")
    _, exact_ids = exact_topk(item_vectors, queries, max(args.candidates), ~seen)
    true_exact = exact_ids.cpu().numpy()
    covered = np.isin(targets, data.catalog) & ~data.test_seen[users_array, targets]
    write_json(output / "manifest.json", {
        "experiment": "trained recommendation pipeline with real ANN backends",
        "environment": environment_manifest(), "seed": args.seed,
        "evaluation_users": len(users), "catalog": len(catalog),
        "eligible_target_coverage": float(covered.mean()),
        "checkpoint_policy": "Previously validation-selected checkpoints; no test-dependent retraining/tuning.",
        "faiss_policy": "4x post-filter oversampling capped to catalog and classic GPU's 2048 limit",
        "silver_policy": "Exact query-time seen-item eligibility before ANN top-k",
        "models": {name: getattr(model, "model_class", f"{type(model).__module__}.{type(model).__name__}") for name, model in models.items()},
    })
    for backend_name in args.backends:
        if backend_name.startswith("silver-"):
            backend = SilverANN(item_vectors, partition, backend_name.removeprefix("silver-"))
        else:
            backend = FaissANN(item_vectors, partition, gpu="gpu" in backend_name)
        for probes in args.probes:
            for candidate_count in args.candidates:
                all_candidates = []
                for start in range(0, len(users), 16):
                    query = queries[start : start + 16]
                    eligible = ~seen[start : start + len(query)]
                    if isinstance(backend, SilverANN):
                        _, ids = backend.search(query, candidate_count, probes, eligible=eligible)
                    else:
                        retrieval_k = min(len(catalog), candidate_count * 4, 2048 if backend.cuda else len(catalog))
                        scores, ids = backend.search(query, retrieval_k, probes)
                        scores, ids = scores.to(device), ids.long().to(device)
                        accepted = eligible.gather(1, ids.clamp_min(0))
                        scores.masked_fill_((ids < 0) | ~accepted, -torch.inf)
                        chosen, positions = scores.topk(candidate_count, dim=1)
                        ids = ids.gather(1, positions).masked_fill(~torch.isfinite(chosen), -1)
                    all_candidates.append(ids)
                row_ids = torch.cat(all_candidates).long()
                candidates = catalog[row_ids.clamp_min(0)].masked_fill(row_ids < 0, -1)
                candidate_numpy = candidates.cpu().numpy()
                geometric_recall = recall_at_k(row_ids.cpu().numpy(), true_exact, candidate_count)
                for model_name, model in models.items():
                    rankings = []
                    for start in range(0, len(users), 16):
                        batch_users = users[start : start + 16]
                        batch_items = candidates[start : start + len(batch_users)]
                        pair_users = batch_users.repeat_interleave(candidate_count)
                        pair_items = batch_items.clamp_min(0).reshape(-1)
                        score = model(pair_users, pair_items, features.get(pair_users, pair_items))
                        score = score.reshape(len(batch_users), candidate_count).masked_fill(batch_items < 0, -torch.inf)
                        positions = score.argsort(dim=1, descending=True, stable=True)
                        rankings.append(batch_items.gather(1, positions).cpu().numpy())
                    append_jsonl(output / "quality_ann.jsonl", {
                        "backend": backend_name, "ranker": model_name, "seed": args.seed,
                        "probes": probes, "candidates": candidate_count, "status": "ok",
                        "candidate_geometric_recall": float(geometric_recall.mean()),
                        "candidate_hit_rate": float((candidate_numpy == targets[:, None]).any(axis=1).mean()),
                        "candidate_fill": float((candidate_numpy >= 0).mean()),
                        "quality": single_positive_metrics(np.concatenate(rankings), targets, [10, 20, 100]),
                    })
                print(f"trained pipeline seed={args.seed} {backend_name} probes={probes} k={candidate_count}", flush=True)
        del backend
        gc.collect()
        torch.cuda.empty_cache()
    write_json(output / "completed.json", {"complete": True, "experiment": "trained ANN pipeline"})
