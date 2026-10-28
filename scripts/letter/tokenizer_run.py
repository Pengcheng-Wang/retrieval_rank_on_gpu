"""Managed lifecycle for the pinned author RQ-VAE, without changing its losses."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import random
import signal
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from protocol import bind_run_config, file_sha256, require_hash, write_json
from rrbench.io import append_jsonl


SOURCE_HASHES = {
    "models/rqvae.py": "797b61e5c9037b8649d90c746f41f51bd60c0c922b46d7233374e4da65f45c00",
    "models/rq.py": "67dc71c5e227889bfe06c70a7f516e70770b480f7a8e6b519023c7ffdc4bd363",
    "models/vq.py": "ebb0d92623abc38a3cd4952255e2d8ded098ac1472f13a62e234591040010516",
    "models/layers.py": "fa91c96d5dea456f09e2d872f62d9063a50111d0beba167994a27754b9090536",
    "trainer.py": "e9554af6530187376e1dfed9b3462e9817552fe53a7c326b8e5e2fe31451fea2",
    "datasets.py": "82fc717dd2650b748b5fc942757b8bfad56958a6808a9344bd4442272c54d66b",
    "utils.py": "c362cfe5ef615b54ba5c400e38161780ad726dc8a501f3e60e97c765f3c95776",
}


def cluster_bounds(codes: int, clusters: int) -> tuple[int, int]:
    if clusters < 1 or codes < 2 * clusters:
        raise ValueError("Diversity clustering needs at least two codes per cluster")
    return min(codes // (2 * clusters), 10), max(6 * clusters, math.ceil(codes / clusters))


def validate_spec(codes, epsilons, clusters: int, dimension: int, items: int) -> None:
    if not 1 <= len(codes) <= 8 or len(codes) != len(epsilons):
        raise ValueError("Codebook and epsilon levels must match, with at most eight levels")
    if dimension < 1 or any(not math.isfinite(e) or e < 0 for e in epsilons):
        raise ValueError("Invalid code dimension or Sinkhorn epsilon")
    for count in codes:
        cluster_bounds(count, clusters)
        if items < 2 * count:
            raise ValueError("Insufficient items for the author's constrained initialization")
        if items > 4 * min(items // (2 * count), 50) * count:
            raise ValueError("Item count exceeds the author's constrained initialization capacity")


def validate_embeddings(semantic, collaborative, dimension: int) -> None:
    if (semantic.ndim != 2 or collaborative.ndim != 2
            or semantic.shape[0] != collaborative.shape[0] or collaborative.shape[1] != dimension):
        raise ValueError("Semantic/CF embeddings must be row-aligned matrices")
    if not np.isfinite(semantic).all() or not np.isfinite(collaborative).all():
        raise ValueError("Input embeddings must be finite")
    if collaborative.dtype != np.float32:
        raise ValueError("The author CF loss expects float32 collaborative embeddings")


def collision_groups(codes: np.ndarray) -> list[list[int]]:
    groups = defaultdict(list)
    for item, row in enumerate(codes.tolist()):
        groups[tuple(row)].append(item)
    return [items for items in groups.values() if len(items) > 1]


def format_codes(codes: np.ndarray) -> dict[str, list[str]]:
    if codes.ndim != 2 or codes.shape[1] > 8 or codes.dtype.kind not in "iu" or (codes < 0).any():
        raise ValueError("Invalid residual-code matrix")
    return {str(i): [f"<{chr(ord('a') + level)}_{code}>"
                     for level, code in enumerate(row)] for i, row in enumerate(codes.tolist())}


def fresh_output_directory(
    directory: Path, completion_marker: str, recovery_root: Path | None = None,
) -> Path | None:
    displaced = None
    if directory.exists():
        if not directory.is_dir() or directory.is_symlink() or (directory / completion_marker).exists():
            raise FileExistsError("Refusing to overwrite a complete or redirected artifact")
        recovery_root = recovery_root or directory.parent
        recovery_root.mkdir(parents=True, exist_ok=True)
        displaced = recovery_root / f"{directory.name}.interrupted-{time.time_ns()}"
        directory.rename(displaced)
        print(f"QUARANTINED_INTERRUPTED_ARTIFACT {displaced}", flush=True)
    directory.mkdir(parents=True, exist_ok=False)
    return displaced


def encoder_track(receipt: dict, method_spec: Path | None) -> str:
    if method_spec is None:
        if (receipt.get("encoder") != "LLaMA1-7B"
                or receipt.get("track_id") not in (None, "letter-original-llama1")
                or receipt.get("exact_original_llama1_reproduction", True) is not True):
            raise ValueError("The original-encoder gate cannot admit an alternative backbone")
        return "letter-original-llama1"
    from download_openllama import MODEL_ID, REVISION, validate_track

    spec = json.loads(method_spec.read_text(encoding="utf-8"))
    validate_track(spec)
    if (receipt.get("encoder") != "OpenLLaMA-7B-v2" or receipt.get("track_id") != spec["track_id"]
            or receipt.get("model_id") != MODEL_ID or receipt.get("revision") != REVISION
            or receipt.get("exact_original_llama1_reproduction") is not False
            or receipt.get("smoke") is not False
            or receipt.get("method_spec_sha256") != file_sha256(method_spec)):
        raise ValueError("Alternative encoder receipt does not match the explicitly declared method track")
    return spec["track_id"]


def verify_real_inputs(args, semantic: np.ndarray) -> str:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    reference = manifest["datasets"][args.dataset]
    for kind in ("item", "inter"):
        require_hash(args.upstream / "data" / args.dataset / f"{args.dataset}.{kind}.json",
                     reference["sha256"][kind])
    receipt = json.loads(args.embedding_receipt.read_text(encoding="utf-8"))
    method_spec = getattr(args, "method_spec", None)
    encoder_track(receipt, method_spec)
    if (receipt.get("status") != "pass"
            or receipt.get("item_order") != "row_equals_zero_based_item_id"
            or semantic.shape != (reference["items"], 4096)
            or receipt.get("item_json_sha256") != reference["sha256"]["item"]
            or not receipt.get("model_files_sha256") or not receipt.get("tokenizer_files_sha256")
            or receipt.get("token_audit", {}).get("status") != "pass"):
        raise ValueError("A complete declared-encoder embedding/row-order audit is required")
    if method_spec is not None:
        audit = receipt["token_audit"]
        if (audit.get("records") != 2 * reference["items"]
                or audit.get("strict_clean") != audit["records"] or audit.get("over_cap") != 0):
            raise ValueError("The compatible encoder requires an all-field, no-truncation audit")
        require_hash(args.embedding_receipt.parent / "run-config.json", receipt["run_config_sha256"])
    require_hash(args.semantic, receipt["embeddings_sha256"])
    if receipt["shape"] != list(semantic.shape) or receipt["dataset"] != args.dataset:
        raise ValueError("Embedding audit dataset/shape mismatch")
    cf_hash = file_sha256(args.cf)
    if (method_spec is None and args.dataset == "Instruments"
            and cf_hash == manifest["sources"]["letter"]["released_instruments_cf_sha256"]):
        return "author_released_item_tensor_training_provenance_unavailable"
    teacher_receipt = json.loads((args.cf.parent / "completed.json").read_text(encoding="utf-8"))
    teacher_config = json.loads((args.cf.parent / "run-config.json").read_text(encoding="utf-8"))
    if (teacher_receipt.get("status") != "complete" or teacher_receipt.get("smoke") is not False
            or teacher_receipt.get("test_evaluated") is not False
            or teacher_config.get("kind") != "training-only-BCE-teacher-reconstruction"
            or teacher_receipt["item_embeddings_sha256"] != cf_hash
            or teacher_config["dataset"] != args.dataset
            or teacher_config["input_sha256"] != reference["sha256"]["inter"]):
        raise ValueError("CF teacher identity, fitting split or item-bank checksum mismatch")
    require_hash(args.cf.parent / "run-config.json", teacher_receipt["config_sha256"])
    return "training_only_BCE_teacher_reconstruction"


def load_author(upstream: Path, destination: Path):
    source = upstream / "RQ-VAE"
    receipts = {}
    for name, expected in SOURCE_HASHES.items():
        require_hash(source / name, expected)
        content = (source / name).read_bytes()
        if name == "models/vq.py":
            changes = [
                (b"self.constrained_km(data, 256)", b"self.constrained_km(data, self.n_e)", 1),
                (b"target_numbers = list(range(10))", b"target_numbers = sorted(set(labels))", 1),
                (b'print(f"Sinkhorn Algorithm returns nan/inf values.")',
                 b'raise FloatingPointError("Sinkhorn Algorithm returns nan/inf values.")', 2),
            ]
            for old, new, count in changes:
                if content.count(old) != count:
                    raise ValueError("Pinned quantizer patch context does not match")
                content = content.replace(old, new)
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != content:
            raise ValueError("The private tokenizer source stage has changed")
        if not target.exists():
            target.write_bytes(content)
        receipts[name] = {"original": expected, "executed": hashlib.sha256(content).hexdigest()}
    write_json(destination / "source-receipt.json", receipts)
    for name in ("models", "trainer", "datasets", "utils"):
        if name in sys.modules:
            raise RuntimeError(f"Tokenizer module conflict: {name}; use a fresh process")
    sys.path.insert(0, str(destination.resolve()))
    models = importlib.import_module("models.rqvae")
    trainer_module = importlib.import_module("trainer")
    dataset_module = importlib.import_module("datasets")
    trainer_module.tqdm = lambda iterable, **kwargs: iterable
    return models.RQVAE, trainer_module.Trainer, dataset_module.EmbDataset, receipts


def cluster_codebook(data, clusters: int):
    import torch
    from k_means_constrained import KMeansConstrained

    low, high = cluster_bounds(len(data), clusters)
    estimator = KMeansConstrained(n_clusters=clusters, size_min=low, size_max=high,
                                 max_iter=10, n_init=10, n_jobs=10, verbose=False)
    estimator.fit(data)
    labels = estimator.labels_.tolist()
    counts = Counter(labels)
    if len(counts) != clusters or min(counts.values()) < 2:
        raise ValueError("Invalid diversity clusters: missing or singleton positive groups")
    return torch.from_numpy(estimator.cluster_centers_), labels


def save_state(directory: Path, trainer, epoch: int, spec: dict, selected: str | None,
               recovery_root: Path):
    import torch

    fresh_output_directory(directory, "completed.json", recovery_root)
    flags = [layer.initted for layer in trainer.model.rq.vq_layers]
    if not all(flags):
        raise ValueError("Cannot checkpoint an incompletely initialized quantizer")
    state = {
        "epoch": epoch, "spec": spec, "model": trainer.model.state_dict(),
        "optimizer": trainer.optimizer.state_dict(), "initted": flags,
        "best_loss": trainer.best_loss, "best_collision_rate": trainer.best_collision_rate,
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
        "selected_checkpoint": selected,
    }
    temporary = directory / "state.pt.tmp"
    torch.save(state, temporary)
    temporary.replace(directory / "state.pt")
    write_json(directory / "completed.json", {
        "status": "complete", "kind": "tokenizer_epoch_checkpoint", "epoch": epoch,
        "state_sha256": file_sha256(directory / "state.pt"),
    })


def load_state(directory: Path, trainer, spec: dict):
    import torch

    receipt = json.loads((directory / "completed.json").read_text(encoding="utf-8"))
    require_hash(directory / "state.pt", receipt["state_sha256"])
    state = torch.load(directory / "state.pt", map_location="cpu")
    if state["spec"] != spec or len(state["initted"]) != len(trainer.model.rq.vq_layers):
        raise ValueError("Tokenizer checkpoint specification mismatch")
    if not all(state["initted"]):
        raise ValueError("Checkpoint does not contain initialized codebooks")
    trainer.model.load_state_dict(state["model"], strict=True)
    trainer.optimizer.load_state_dict(state["optimizer"])
    for layer in trainer.model.rq.vq_layers:
        layer.initted = True
    trainer.best_loss, trainer.best_collision_rate = state["best_loss"], state["best_collision_rate"]
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state


def export_indices(model, dataset, clusters: int, output: Path, seed: int, recovery_root: Path) -> dict:
    import torch
    from torch.utils.data import DataLoader

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model.eval()
    labels = {str(i): cluster_codebook(layer.embedding.weight.detach().cpu().numpy(), clusters)[1]
              for i, layer in enumerate(model.rq.vq_layers)}
    with torch.no_grad():
        parts = [model.get_indices(x.cuda(), labels, use_sk=False).cpu().numpy()
                 for x, _ in DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)]
        codes = np.concatenate(parts)
        before = len(codes) - len(set(map(tuple, codes.tolist())))
        for layer in model.rq.vq_layers[:-1]:
            layer.sk_epsilon = 0.0
        if model.rq.vq_layers[-1].sk_epsilon == 0:
            model.rq.vq_layers[-1].sk_epsilon = 0.003
        retries = 0
        for _ in range(20):
            groups = collision_groups(codes)
            if not groups:
                break
            for items in groups:
                xs, _ = dataset[items]
                codes[items] = model.get_indices(xs.cuda(), labels, use_sk=True).cpu().numpy()
            retries += 1
    if (output / "export.json").is_file():
        existing = json.loads((output / "export.json").read_text(encoding="utf-8"))
        require_hash(output / "index.json", existing["index_sha256"])
        if (not np.array_equal(np.load(output / "codes.npy", allow_pickle=False), codes)
                or json.loads((output / "index.json").read_text(encoding="utf-8")) != format_codes(codes)):
            raise ValueError("A completed export differs from the deterministic reconstruction")
        return existing
    fresh_output_directory(output, "export.json", recovery_root)
    write_json(output / "index.json", format_codes(codes))
    np.save(output / "codes.npy", codes, allow_pickle=False)
    result = {
        "items": len(codes), "levels": codes.shape[1], "collision_retries": retries,
        "excess_collision_items_before": before,
        "excess_collision_items_after": len(codes) - len(set(map(tuple, codes.tolist()))),
        "largest_group": max(Counter(map(tuple, codes.tolist())).values()),
        "utilization_by_level": [len(np.unique(codes[:, i])) for i in range(codes.shape[1])],
        "index_sha256": file_sha256(output / "index.json"), "seed": seed,
    }
    write_json(output / "export.json", result)
    return result


def train(args) -> None:
    import torch
    from torch.utils.data import DataLoader

    if not torch.cuda.is_available():
        raise RuntimeError("Tokenizer execution requires CUDA")
    if (args.output / "completed.json").exists():
        raise FileExistsError("Tokenizer attempt already completed")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    semantic = np.load(args.semantic, allow_pickle=False)
    collaborative = torch.load(args.cf, map_location="cpu", weights_only=True).detach().numpy()
    validate_embeddings(semantic, collaborative, args.dimension)
    validate_spec(args.codes, args.epsilons, args.clusters, args.dimension, len(semantic))
    cf_provenance = "synthetic_fixture" if args.synthetic else verify_real_inputs(args, semantic)
    method_spec = getattr(args, "method_spec", None)
    if args.synthetic and method_spec is not None:
        raise ValueError("Synthetic fixtures must not be labeled as a real backbone-method track")
    model_type, trainer_type, dataset_type, source_receipts = load_author(
        args.upstream, args.output / "author-rq",
    )
    spec = {
        "dataset": args.dataset, "synthetic_fixture": args.synthetic, "seed": args.seed,
        "track_id": ("synthetic_fixture" if args.synthetic else
                     encoder_track(json.loads(args.embedding_receipt.read_text()), method_spec)),
        "method_spec_sha256": file_sha256(method_spec) if method_spec is not None else None,
        "embedding_receipt_sha256": file_sha256(args.embedding_receipt) if not args.synthetic else None,
        "claim_type": ("synthetic_correctness_fixture" if args.synthetic else
                       "compatible_backbone_method_replication" if method_spec is not None else
                       "original_encoder_tokenizer_reconstruction"),
        "cf_provenance": cf_provenance,
        "semantic_sha256": file_sha256(args.semantic), "cf_sha256": file_sha256(args.cf),
        "semantic_shape": list(semantic.shape), "cf_shape": list(collaborative.shape),
        "codes": args.codes, "epsilons": args.epsilons, "dimension": args.dimension,
        "layers": args.layers, "clusters": args.clusters, "alpha": args.alpha, "beta": args.beta,
        "learning_rate": 0.001, "weight_decay": 0.0001, "batch_size": args.batch_size,
        "epochs": args.epochs, "eval_step": min(args.eval_step, args.epochs),
        "source": source_receipts, "harness_sha256": file_sha256(Path(__file__)),
        "num_workers": 0, "selection": "first minimum collision rate at scheduled evaluations",
    }
    bind_run_config(args.output, spec)
    dataset = dataset_type(str(args.semantic))
    model = model_type(
        in_dim=semantic.shape[1], num_emb_list=args.codes, e_dim=args.dimension, layers=args.layers,
        dropout_prob=0, bn=False, loss_type="mse", quant_loss_weight=1,
        kmeans_init=True, kmeans_iters=100, sk_epsilons=args.epsilons, sk_iters=50,
        alpha=args.alpha, beta=args.beta, n_clusters=args.clusters, sample_strategy="all",
        cf_embedding=collaborative,
    )

    class ManagedTrainer(trainer_type):
        def constrained_km(self, data, n_clusters=10):
            return cluster_codebook(data, args.clusters)

        def _check_nan(self, loss):
            if not torch.isfinite(loss):
                raise FloatingPointError("Tokenizer training loss is not finite")

    original_args = SimpleNamespace(
        lr=0.001, learner="AdamW", weight_decay=0.0001, epochs=args.epochs,
        eval_step=args.eval_step, device="cuda:0", ckpt_dir=str(args.output / "author-layout"),
        data_path=str(args.semantic), num_workers=0,
    )
    trainer = ManagedTrainer(original_args, model)
    if len(model.rq.vq_layers) != len(args.codes):
        raise ValueError("Actual codebook count differs from requested levels")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    first_epoch, selected = 0, None
    recovery_root = args.output.parent / "interrupted" / args.output.name
    if args.resume:
        state = load_state(args.resume, trainer, spec)
        first_epoch, selected = state["epoch"] + 1, state["selected_checkpoint"]
    else:
        trainer.vq_init()
    draining = [False]
    signal.signal(signal.SIGTERM, lambda sig, frame: draining.__setitem__(0, True))
    started = time.monotonic()
    for epoch in range(first_epoch, args.epochs):
        epoch_started = time.monotonic()
        total, reconstruction, collaborative_loss, last_quantized_loss = trainer._train_epoch(loader, epoch)
        trainer.best_loss = min(trainer.best_loss, total)
        log = {
            "epoch": epoch + 1, "sum_batch_total_loss": total,
            "sum_batch_reconstruction": reconstruction, "sum_batch_cf": collaborative_loss,
            "last_batch_quantized_loss": last_quantized_loss,
            "training_seconds": time.monotonic() - epoch_started,
        }
        due = (epoch + 1) % trainer.eval_step == 0
        directory = args.output / "checkpoints" / f"epoch-{epoch + 1:05d}"
        if due:
            collision = trainer._valid_epoch(loader)
            log["collision_rate"] = collision
            if collision < trainer.best_collision_rate:
                trainer.best_collision_rate = collision
                selected = str(directory.relative_to(args.output))
        stop = draining[0] or (args.stop_after_epoch is not None and epoch + 1 >= args.stop_after_epoch)
        if due or stop or (epoch + 1) % args.checkpoint_every == 0 or epoch + 1 == args.epochs:
            save_state(directory, trainer, epoch, spec, selected, recovery_root)
            write_json(args.output / "latest-state.json", {"epoch": epoch + 1, "checkpoint": str(directory)})
        append_jsonl(args.output / "epochs.jsonl", log)
        print(json.dumps(log), flush=True)
        if stop:
            return
    if selected is None:
        raise RuntimeError("No scheduled tokenizer checkpoint was selected")
    selected_state = load_state(args.output / selected, trainer, spec)
    export = export_indices(model, dataset, args.clusters, args.output / "export", args.seed, recovery_root)
    write_json(args.output / "completed.json", {
        "status": "complete", "synthetic_fixture": args.synthetic,
        "track_id": spec["track_id"], "claim_type": spec["claim_type"],
        "selected_epoch": selected_state["epoch"] + 1, "selected_checkpoint": selected,
        "export": export, "wall_seconds_this_invocation": time.monotonic() - started,
        "run_config_sha256": file_sha256(args.output / "run-config.json"),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("upstream", "semantic", "cf", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--embedding-receipt", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--method-spec", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--codes", nargs="+", type=int, default=[256] * 4)
    parser.add_argument("--epsilons", nargs="+", type=float, default=[0, 0, 0, 0.003])
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--layers", nargs="+", type=int, default=[2048, 1024, 512, 256, 128, 64])
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--beta", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-step", type=int, default=2000)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-epoch", type=int)
    args = parser.parse_args()
    if not args.synthetic and (args.embedding_receipt is None or args.manifest is None):
        parser.error("--embedding-receipt and --manifest are required for real experiments")
    if min(args.epochs, args.batch_size, args.eval_step, args.checkpoint_every) < 1:
        parser.error("Training and checkpoint intervals must be positive")
    if any(not math.isfinite(v) or v < 0 for v in (args.alpha, args.beta)):
        parser.error("Regularization strengths must be finite and nonnegative")
    for field in ("upstream", "semantic", "cf", "output"):
        setattr(args, field, getattr(args, field).resolve())
    train(args)


if __name__ == "__main__":
    main()
