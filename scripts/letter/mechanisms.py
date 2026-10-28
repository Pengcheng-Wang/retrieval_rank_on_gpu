"""Declared reconstruction of LETTER's code-distribution and CF-transfer analyses."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from protocol import file_sha256, require_hash, write_json


def validate_codes(codes: np.ndarray, sizes: list[int] | None = None) -> None:
    if codes.ndim != 2 or min(codes.shape) < 1 or codes.dtype.kind not in "iu" or (codes < 0).any():
        raise ValueError("Identifiers must be a nonempty integer item-by-level matrix")
    if sizes is not None and (
        len(sizes) != codes.shape[1] or any(size < 1 for size in sizes)
        or any((codes[:, level] >= size).any() for level, size in enumerate(sizes))
    ):
        raise ValueError("Identifier index is outside its level-specific codebook")


def nearest_neighbors(embeddings: np.ndarray, similarity: str, block_size: int = 256) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float64)
    if (values.ndim != 2 or len(values) < 2 or values.shape[1] < 1
            or not np.isfinite(values).all() or block_size < 1):
        raise ValueError("CF neighbors require at least two finite item vectors")
    if similarity == "cosine":
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError("Cosine similarity is undefined for a zero CF vector")
        values = values / norms
    elif similarity != "dot":
        raise ValueError("Declare cosine or dot similarity explicitly")
    result = np.empty(len(values), dtype=np.int64)
    for start in range(0, len(values), block_size):
        stop = min(start + block_size, len(values))
        scores = values[start:stop] @ values.T
        if not np.isfinite(scores).all():
            raise ValueError("CF neighbor similarities overflowed")
        scores[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        result[start:stop] = np.argmax(scores, axis=1)
    return result


def code_similarity(codes: np.ndarray, neighbors: np.ndarray) -> dict:
    validate_codes(codes)
    if (neighbors.shape != (len(codes),) or neighbors.dtype.kind not in "iu"
            or (neighbors < 0).any() or (neighbors >= len(codes)).any()
            or np.any(neighbors == np.arange(len(codes)))):
        raise ValueError("Each item must have one distinct, in-catalog CF neighbor")
    equal = codes == codes[neighbors]
    intersection = equal.sum(axis=1)
    levels = codes.shape[1]
    return {
        "items": len(codes), "levels": levels,
        "position_overlap_mean": float((intersection / levels).mean()),
        "jaccard_mean": float((intersection / (2 * levels - intersection)).mean()),
        "prefix_overlap_mean": float((np.cumprod(equal, axis=1).sum(axis=1) / levels).mean()),
        "matching_positions_histogram": np.bincount(intersection, minlength=levels + 1).tolist(),
        "token_identity": "level-qualified; equal numeric values at different levels do not match",
    }


def assignment_statistics(codes: np.ndarray, sizes: list[int], group_size: int = 15) -> dict:
    validate_codes(codes, sizes)
    if group_size < 1:
        raise ValueError("Code-frequency group size must be positive")
    levels = []
    for level, size in enumerate(sizes):
        counts = np.bincount(codes[:, level], minlength=size)
        order = np.argsort(-counts, kind="stable")
        frequencies = counts[counts > 0] / len(codes)
        levels.append({
            "level": level, "codebook_size": size, "utilized_codes": int((counts > 0).sum()),
            "counts_by_code": counts.tolist(), "sorted_code_ids": order.tolist(),
            "grouped_assignment_counts": [int(counts[order[start:start + group_size]].sum())
                                          for start in range(0, size, group_size)],
            "entropy_bits": float(-(frequencies * np.log2(frequencies)).sum()),
        })
    return {"items": len(codes), "group_size": group_size, "levels": levels}


def quantized_vectors(codebooks: list[np.ndarray], codes: np.ndarray) -> np.ndarray:
    validate_codes(codes, [len(book) for book in codebooks])
    if (not codebooks or any(book.ndim != 2 or book.dtype != np.float32
                            or book.shape[1] != codebooks[0].shape[1]
                            or not np.isfinite(book).all() for book in codebooks)):
        raise ValueError("Codebooks must be finite, equal-dimensional float32 matrices")
    vectors = np.zeros((len(codes), codebooks[0].shape[1]), dtype=np.float32)
    for level, book in enumerate(codebooks):
        vectors += book[codes[:, level]]
    if not np.isfinite(vectors).all():
        raise ValueError("Residual codebook sum is nonfinite")
    return vectors


def codebook_pca(codebook: np.ndarray) -> dict:
    values = np.asarray(codebook, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError("PCA requires finite codebook vectors")
    centered = values - values.mean(axis=0, keepdims=True)
    _, singular, components = np.linalg.svd(centered, full_matrices=False)
    count = min(3, len(singular))
    axes = components[:count].copy()
    for axis in axes:
        if axis[np.argmax(np.abs(axis))] < 0:
            axis *= -1
    coordinates = np.zeros((len(values), 3), dtype=np.float64)
    coordinates[:, :count] = centered @ axes.T
    variance = singular**2
    ratios = np.zeros(3, dtype=np.float64)
    if variance.sum() > 0:
        ratios[:count] = variance[:count] / variance.sum()
    return {
        "coordinates": coordinates.tolist(), "explained_variance_ratio": ratios.tolist(),
        "normalization": "center_only_no_spherical_rescaling",
        "variance_status": "nonzero" if variance.sum() > 0 else "degenerate_zero_variance",
        "sign_convention": "Largest-absolute loading in each component is nonnegative",
    }


def replace_item_embeddings(model, vectors: np.ndarray) -> None:
    import torch

    parameter = model.item_emb.weight
    if (vectors.shape != (parameter.shape[0] - 1, parameter.shape[1])
            or vectors.dtype != np.float32 or not np.isfinite(vectors).all()):
        raise ValueError("Replacement must cover exactly the real float32 CF item rows")
    with torch.no_grad():
        parameter[1:].copy_(torch.as_tensor(vectors, device=parameter.device))


def load_teacher_model_state(path: Path, expected_sha256: str) -> dict:
    import torch

    require_hash(path, expected_sha256)
    # This harness wrote the checkpoint; Torch 1.13's restricted reader lacks BINFLOAT.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if (set(state) != {"model", "epoch", "validation"} or not isinstance(state["model"], dict)
            or not all(torch.is_tensor(value) and torch.isfinite(value).all()
                       for value in state["model"].values())):
        raise ValueError("Unexpected or nonfinite harness CF teacher checkpoint")
    return state["model"]


def rank_cf(model, interactions: dict, length: int, device: str, output: Path) -> dict:
    import torch
    from cf import teacher_sequences

    users, train, valid, test = teacher_sequences(interactions)
    sums = {f"{metric}@{k}": 0.0 for metric in ("recall", "ndcg") for k in (5, 10)}
    predictions_by_user = []
    model.eval()
    with torch.inference_mode():
        for start in range(1, len(users) + 1, 128):
            batch = list(range(start, min(start + 128, len(users) + 1)))
            sequences = np.zeros((len(batch), length), dtype=np.int32)
            for row, user in enumerate(batch):
                history = (train[user] + [valid[user]])[-length:]
                sequences[row, -len(history):] = history
            scores = model.log2feats(sequences)[:, -1] @ model.item_emb.weight[1:].T
            if not torch.isfinite(scores).all():
                raise ValueError("CF transfer produced nonfinite full-catalog scores")
            top = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :10].cpu().numpy() + 1
            for user, predictions in zip(batch, top):
                predictions_by_user.append({
                    "user": users[user - 1], "target_item": test[user] - 1,
                    "top10": (predictions - 1).tolist(),
                })
                ranks = np.flatnonzero(predictions == test[user])
                rank = int(ranks[0]) + 1 if len(ranks) else None
                for k in (5, 10):
                    if rank is not None and rank <= k:
                        sums[f"recall@{k}"] += 1
                        sums[f"ndcg@{k}"] += 1 / math.log2(rank + 1)
    if [row["user"] for row in predictions_by_user] != users:
        raise ValueError("CF transfer lost or reordered the original evaluation users")
    write_json(output, predictions_by_user)
    return {"users": len(users), "metrics": {key: value / len(users) for key, value in sums.items()},
            "seen_item_exclusion": False, "tie_rule": "original_item_id_ascending",
            "history_limit": length, "device": device, "predictions_sha256": file_sha256(output)}


def analyze(args) -> None:
    import torch
    from bank_tokenizers import checkpoint_path, read_state
    from cf import load_model_class
    from prepare import read_release
    from tokenizer_run import format_codes

    if args.output.exists():
        raise FileExistsError("Mechanism-analysis outputs are immutable")
    manifest = json.loads(args.manifest.read_text())
    interactions, items, _ = read_release(args.upstream, manifest, args.dataset)
    config = json.loads((args.tokenizer_run / "run-config.json").read_text())
    done = json.loads((args.tokenizer_run / "completed.json").read_text())
    latest = json.loads((args.tokenizer_run / "latest-state.json").read_text())
    if (done["status"] != "complete" or done["synthetic_fixture"] is not False
            or config["dataset"] != args.dataset or latest["epoch"] != config["epochs"]
            or config["epochs"] != 20000 or config["track_id"] != done["track_id"]
            or config["track_id"] != "letter-openllama7b-v2-method-replication"):
        raise ValueError("Mechanism analysis requires a complete real 20,000-epoch tokenizer")
    require_hash(args.tokenizer_run / "run-config.json", done["run_config_sha256"])
    final_state = read_state(checkpoint_path(args.tokenizer_run, latest["checkpoint"]), config)
    if (final_state["epoch"] + 1 != config["epochs"]
            or final_state["selected_checkpoint"] != done["selected_checkpoint"]):
        raise ValueError("The sealed final checkpoint disagrees with the epoch budget or selection")
    del final_state
    selected = checkpoint_path(args.tokenizer_run, done["selected_checkpoint"])
    state = read_state(selected, config)
    if state["epoch"] + 1 != done["selected_epoch"]:
        raise ValueError("Selected tokenizer epoch differs from the completion receipt")
    export = json.loads((args.tokenizer_run / "export/export.json").read_text())
    if done["export"] != export:
        raise ValueError("Tokenizer completion is bound to a different exported identifier bank")
    require_hash(args.tokenizer_run / "export/index.json", export["index_sha256"])
    codes = np.load(args.tokenizer_run / "export/codes.npy", allow_pickle=False)
    validate_codes(codes, config["codes"])
    if (len(codes) != len(items) or format_codes(codes) !=
            json.loads((args.tokenizer_run / "export/index.json").read_text())):
        raise ValueError("Exported IDs, integer codes and original item rows differ")
    teacher_done = json.loads((args.teacher / "completed.json").read_text())
    teacher_config = json.loads((args.teacher / "run-config.json").read_text())
    if (teacher_done["status"] != "complete" or teacher_done["smoke"] is not False
            or teacher_done["test_evaluated"] is not False or teacher_config["dataset"] != args.dataset
            or teacher_config.get("kind") != "training-only-BCE-teacher-reconstruction"
            or teacher_config["input_sha256"] != manifest["datasets"][args.dataset]["sha256"]["inter"]):
        raise ValueError("CF neighbor/transfer analysis requires the declared training-only teacher")
    require_hash(args.teacher / "run-config.json", teacher_done["config_sha256"])
    require_hash(args.teacher / "item-embeddings.pt", config["cf_sha256"])
    require_hash(args.teacher / "item-embeddings.pt", teacher_done["item_embeddings_sha256"])
    cf = torch.load(args.teacher / "item-embeddings.pt", map_location="cpu", weights_only=True).numpy()
    books = [state["model"][f"rq.vq_layers.{level}.embedding.weight"].cpu().numpy()
             for level in range(codes.shape[1])]
    vectors = quantized_vectors(books, codes)
    args.output.mkdir(parents=True)
    np.save(args.output / "quantized-item-embeddings.npy", vectors, allow_pickle=False)
    neighbor_report = {}
    for similarity in ("cosine", "dot"):
        neighbors = nearest_neighbors(cf, similarity)
        np.save(args.output / f"cf-neighbors-{similarity}.npy", neighbors, allow_pickle=False)
        neighbor_report[similarity] = code_similarity(codes, neighbors)
    report = {
        "status": "complete", "kind": "declared_mechanism_reconstruction", "dataset": args.dataset,
        "track_id": config["track_id"], "exact_original_llama1_reproduction": False,
        "cf_transfer_requested": args.rank_replacement,
        "selection": {"epoch": done["selected_epoch"], "rule": config["selection"]},
        "assignment_distribution": assignment_statistics(codes, config["codes"]),
        "first_codebook_pca": codebook_pca(books[0]), "cf_neighbor_overlap": neighbor_report,
        "definitions": {
            "neighbors": "One non-self nearest item; ascending original ID breaks exact ties; float64 similarity.",
            "primary_similarity": "Cosine; dot-product is an explicitly reported sensitivity analysis.",
            "primary_overlap": "Matching level-qualified tokens divided by identifier length.",
            "overlap_sensitivity": "Token-set Jaccard and normalized common-prefix length also reported.",
            "paper_ambiguity": "The paper does not fully specify neighbor normalization or overlap formula.",
            "quantized_vectors": "Sequential float32 sum of codebook rows indexed by the final exported IDs.",
            "replacement": "Raw quantized vectors replace real CF item rows only; no normalization or refitting.",
        },
        "provenance": {
            "tokenizer_run_config_sha256": file_sha256(args.tokenizer_run / "run-config.json"),
            "index_sha256": export["index_sha256"], "cf_embeddings_sha256": config["cf_sha256"],
            "selected_state_sha256": file_sha256(selected / "state.pt"),
            "analysis_sha256": file_sha256(Path(__file__)),
        },
    }
    if args.rank_replacement:
        if args.sasrec is None:
            raise ValueError("CF transfer ranking requires the pinned native SASRec source")
        model_args = SimpleNamespace(**{**teacher_config["model"], "device": args.device})
        model = load_model_class(args.sasrec)(len(interactions), len(items), model_args).to(args.device)
        model.load_state_dict(
            load_teacher_model_state(args.teacher / "best.pt", teacher_done["teacher_sha256"]),
            strict=True,
        )
        if not torch.equal(model.item_emb.weight[1:].detach().cpu(), torch.as_tensor(cf)):
            raise ValueError("Selected CF model rows differ from the regularization/neighbor bank")
        report["cf_original_reference"] = rank_cf(
            model, interactions, model_args.maxlen, args.device,
            args.output / "cf-original-predictions.json",
        )
        replace_item_embeddings(model, vectors)
        report["cf_quantized_replacement"] = rank_cf(
            model, interactions, model_args.maxlen, args.device,
            args.output / "cf-quantized-predictions.json",
        )
    report["files_sha256"] = {
        path.name: file_sha256(path) for path in sorted(args.output.iterdir()) if path.is_file()
    }
    write_json(args.output / "completed.json", report)
    print(json.dumps({"status": "complete", "dataset": args.dataset,
                      "cosine_neighbor_overlap": neighbor_report["cosine"]["position_overlap_mean"],
                      "cf_quantized_replacement": report.get("cf_quantized_replacement")}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("tokenizer-run", "teacher", "upstream", "manifest", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--dataset", choices=("Instruments", "Beauty", "Yelp"), required=True)
    parser.add_argument("--rank-replacement", action="store_true")
    parser.add_argument("--sasrec", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
