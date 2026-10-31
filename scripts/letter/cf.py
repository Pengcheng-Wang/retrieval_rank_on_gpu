"""Training-only BCE teacher reconstruction around the author's named SASRec."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import signal
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from prepare import read_release
from protocol import bind_run_config, file_sha256, require_hash, write_json
from rrbench.io import append_jsonl


MODEL_SHA256 = "9925988b4f7f8f567bb304722e0dc6f95d3836e3500a7f52fa9b3e720107e7e8"


def teacher_sequences(interactions: dict[str, list[int]]):
    users = sorted(interactions, key=int)
    train, valid, test = {}, {}, {}
    for u, original in enumerate(users, 1):
        sequence = interactions[original]
        if len(sequence) < 5:
            raise ValueError("The released teacher input must have at least five events")
        shifted = [i + 1 for i in sequence]
        train[u], valid[u], test[u] = shifted[:-2], shifted[-2], shifted[-1]
    return users, train, valid, test


class TeacherSampler:
    """Same draws as one upstream worker, with explicit serializable RNG state."""

    def __init__(self, train, items: int, batch_size: int, max_length: int, seed: int):
        self.train, self.items, self.batch_size, self.max_length = train, items, batch_size, max_length
        if not train or min(items, batch_size, max_length) <= 0:
            raise ValueError("Invalid teacher sampler dimensions")
        if any(len(s) <= 1 or len(set(s)) >= items for s in train.values()):
            raise ValueError("Teacher sampling needs training positives and an unseen negative")
        self.rng = np.random.RandomState(seed)

    def state(self):
        return self.rng.get_state()

    def restore(self, state):
        self.rng.set_state(state)

    def next_batch(self):
        rows = []
        for _ in range(self.batch_size):
            user = self.rng.randint(1, len(self.train) + 1)
            history = self.train[user]
            sequence = np.zeros(self.max_length, dtype=np.int32)
            positive = np.zeros_like(sequence)
            negative = np.zeros_like(sequence)
            seen = set(history)
            next_item = history[-1]
            for position, item in zip(range(self.max_length - 1, -1, -1), reversed(history[:-1])):
                sequence[position] = item
                positive[position] = next_item
                sample = self.rng.randint(1, self.items + 1)
                while sample in seen:
                    sample = self.rng.randint(1, self.items + 1)
                negative[position] = sample
                next_item = item
            rows.append((user, sequence, positive, negative))
        return tuple(np.asarray(column) for column in zip(*rows))


def export_item_rows(weights, item_count: int):
    if len(weights.shape) != 2 or weights.shape[0] != item_count + 1:
        raise ValueError("Teacher embedding shape does not match padding plus catalog")
    return weights[1:item_count + 1]


def load_model_class(source: Path):
    require_hash(source / "model.py", MODEL_SHA256)
    spec = importlib.util.spec_from_file_location("letter_sasrec_author_model", source / "model.py")
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load the pinned SASRec model")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SASRec


def validation_scores(model, train, valid, max_length: int, output: Path | None = None):
    import torch

    model.eval()
    totals = {f"{m}@{k}": 0.0 for m in ("recall", "ndcg") for k in (5, 10)}
    per_user = []
    with torch.inference_mode():
        for offset in range(0, len(train), 256):
            users = list(range(offset + 1, min(offset + 257, len(train) + 1)))
            histories = np.zeros((len(users), max_length), dtype=np.int32)
            for row, u in enumerate(users):
                tail = train[u][-max_length:]
                histories[row, -len(tail):] = tail
            features = model.log2feats(histories)[:, -1]
            scores = features @ model.item_emb.weight[1:].T
            if not torch.isfinite(scores).all():
                raise ValueError("Nonfinite teacher validation scores")
            top = torch.topk(scores, 10, dim=-1).indices.cpu().numpy() + 1
            for u, predictions in zip(users, top):
                ranks = np.flatnonzero(predictions == valid[u])
                rank = int(ranks[0]) + 1 if len(ranks) else None
                for k in (5, 10):
                    if rank is not None and rank <= k:
                        totals[f"recall@{k}"] += 1
                        totals[f"ndcg@{k}"] += 1 / math.log2(rank + 1)
                if output is not None:
                    per_user.append({"teacher_user": u, "target": valid[u],
                                     "top10": predictions.tolist(), "rank": rank})
    if output is not None:
        write_json(output, per_user)
    return {k: v / len(train) for k, v in totals.items()}


def train_teacher(args) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Teacher training requires CUDA")
    if (args.output / "completed.json").exists():
        raise FileExistsError("Teacher attempt is already complete")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    interactions, items, _ = read_release(args.upstream, manifest, args.dataset)
    users, train, valid, _ = teacher_sequences(interactions)
    model_args = SimpleNamespace(
        device="cuda:0", hidden_units=32, maxlen=50, num_blocks=2,
        num_heads=1, dropout_rate=0.5,
    )
    model = load_model_class(args.sasrec)(len(users), len(items), model_args).cuda()
    for parameter in model.parameters():
        if parameter.ndim >= 2:
            torch.nn.init.xavier_normal_(parameter)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98))
    sampler = TeacherSampler(train, len(items), 128, 50, args.seed)
    config = {
        "kind": "training-only-BCE-teacher-reconstruction",
        "dataset": args.dataset, "seed": args.seed, "epochs": args.epochs,
        "model": vars(model_args), "batch_size": 128, "learning_rate": 0.001,
        "loss": "BCEWithLogitsLoss positive + negative, nonpadding positions",
        "optimizer_betas": [0.9, 0.98], "l2_emb": 0.0,
        "sampling": "synchronous single-worker-equivalent distribution, state restored exactly",
        "selection": "maximum full-catalog validation NDCG@10 every 20 epochs and final epoch",
        "seen_item_exclusion": False, "source_model_sha256": MODEL_SHA256,
        "input_sha256": manifest["datasets"][args.dataset]["sha256"]["inter"],
        "harness_sha256": file_sha256(Path(__file__)), "smoke": args.smoke,
    }
    bind_run_config(args.output, config)
    write_json(args.output / "user-map.json", users)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_metric, first_epoch = -1.0, 1
    if args.resume:
        markers = sorted(args.output.glob("resume-*.json"), reverse=True)
        if not markers:
            raise ValueError("No complete teacher checkpoint is available")
        resume_marker = markers[0]
        resume_file = resume_marker.with_suffix(".pt")
        require_hash(resume_file, json.loads(resume_marker.read_text())["sha256"])
        state = torch.load(resume_file, map_location="cuda")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        sampler.restore(state["sampler"])
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
        first_epoch, best_metric = state["epoch"] + 1, state["best_metric"]
    steps = 5 if args.smoke else len(users) // 128
    max_epoch = 1 if args.smoke else args.epochs
    started = time.monotonic()
    draining = [False]
    signal.signal(signal.SIGTERM, lambda sig, frame: draining.__setitem__(0, True))
    for epoch in range(first_epoch, max_epoch + 1):
        epoch_started = time.monotonic()
        model.train()
        loss_sum = 0.0
        for _ in range(steps):
            u, seq, pos, neg = sampler.next_batch()
            pos_logits, neg_logits = model(u, seq, pos, neg)
            mask = torch.as_tensor(pos != 0, device="cuda")
            loss = criterion(pos_logits[mask], torch.ones_like(pos_logits[mask]))
            loss = loss + criterion(neg_logits[mask], torch.zeros_like(neg_logits[mask]))
            if not torch.isfinite(loss):
                raise ValueError("Teacher training loss is not finite")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        log = {"epoch": epoch, "mean_loss": loss_sum / steps,
               "seconds": time.monotonic() - epoch_started}
        if epoch % 20 == 0 or epoch == max_epoch:
            metrics = validation_scores(model, train, valid, 50)
            log["validation"] = metrics
            if metrics["ndcg@10"] > best_metric:
                best_metric = metrics["ndcg@10"]
                temporary = args.output / "best.pt.tmp"
                torch.save({"model": model.state_dict(), "epoch": epoch, "validation": metrics}, temporary)
                temporary.replace(args.output / "best.pt")
        state = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "sampler": sampler.state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(), "epoch": epoch, "best_metric": best_metric,
        }
        resume_file = args.output / f"resume-{epoch:05d}.pt"
        resume_marker = resume_file.with_suffix(".json")
        temporary = resume_file.with_suffix(".pt.tmp")
        torch.save(state, temporary)
        temporary.replace(resume_file)
        write_json(resume_marker, {"epoch": epoch, "sha256": file_sha256(resume_file)})
        for older in sorted(args.output.glob("resume-*.json"), reverse=True)[2:]:
            older.with_suffix(".pt").unlink()
            older.unlink()
        append_jsonl(args.output / "epochs.jsonl", log)
        print(json.dumps(log), flush=True)
        if draining[0] or (args.stop_after_epoch is not None and epoch >= args.stop_after_epoch):
            return
    selected = torch.load(args.output / "best.pt", map_location="cuda")
    model.load_state_dict(selected["model"], strict=True)
    rows = export_item_rows(model.item_emb.weight.detach().cpu(), len(items))
    if rows.shape != (len(items), 32) or not torch.isfinite(rows).all():
        raise ValueError("Invalid collaborative item bank")
    torch.save(rows, args.output / "item-embeddings.pt")
    write_json(args.output / "completed.json", {
        "status": "complete", "kind": config["kind"], "smoke": args.smoke,
        "epoch": selected["epoch"], "validation": selected["validation"],
        "wall_seconds": time.monotonic() - started, "item_shape": list(rows.shape),
        "teacher_sha256": file_sha256(args.output / "best.pt"),
        "item_embeddings_sha256": file_sha256(args.output / "item-embeddings.pt"),
        "config_sha256": file_sha256(args.output / "run-config.json"),
        "test_evaluated": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("upstream", "sasrec", "manifest", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--dataset", choices=("Instruments", "Beauty", "Yelp"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=201)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int)
    train_teacher(parser.parse_args())


if __name__ == "__main__":
    main()
