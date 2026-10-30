"""GPU preflight for identical Figure 8 temperature initialization and data order."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from author import default_args, load_backend
from fresh_ids import validate_audit, verify_stage
from fresh_pipeline import require_real_stage
from prepare import make_tiger_tokenizer, read_release
from protocol import file_sha256, write_json
from run import training_options


def _tensor_bytes(tensor) -> bytes:
    import torch

    value = tensor.detach().cpu().contiguous()
    return value.view(torch.uint8).numpy().tobytes()


def model_state_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def batch_sha256(batch: dict) -> str:
    import torch

    digest = hashlib.sha256()
    for name in sorted(batch):
        value = batch[name]
        if torch.is_tensor(value):
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def probe(args) -> None:
    import torch
    from transformers import T5Config, Trainer, TrainingArguments

    if not torch.cuda.is_available():
        raise RuntimeError("Figure8 temperature preflight requires CUDA")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 2 or not 0 < args.temperature < float("inf"):
        raise ValueError(
            "Figure8 temperature preflight requires two ranks and a positive temperature"
        )
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group("nccl")
    staged = verify_stage(args.stage)
    require_real_stage(staged)
    validate_audit(args.stage, args.audit)
    author = load_backend(args.stage / "upstream", "tiger")
    author.utils.set_seed(args.seed)
    manifest = json.loads((args.stage / "manifest.json").read_text())
    _, _, indices = read_release(args.stage / "upstream", manifest, staged["dataset"])
    tokenizer = make_tiger_tokenizer(author, indices)
    upstream_args = default_args(author, args.stage / "upstream", staged["dataset"])
    train_data, valid_data = author.utils.load_datasets(upstream_args)
    config = T5Config.from_pretrained(
        author.directory / "ckpt" / "TIGER", local_files_only=True,
    )
    config.vocab_size = len(tokenizer)
    model = author.model.LETTER(config)
    model.set_hyper(args.temperature)
    model.resize_token_embeddings(len(tokenizer))
    model.to(torch.device("cuda", local_rank))
    model.config.use_cache = False
    initial_model_sha256 = model_state_sha256(model)
    options = training_options(
        seed=args.seed,
        batch_size=args.batch_size,
        effective_batch=args.effective_batch,
        world_size=world_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output / f"trainer-rank-{rank}"), **options,
        ),
        train_dataset=train_data,
        eval_dataset=valid_data,
        tokenizer=tokenizer,
        data_collator=author.collator.Collator(upstream_args, tokenizer),
    )
    loader = trainer.get_train_dataloader()
    if hasattr(loader, "set_epoch"):
        loader.set_epoch(0)
    iterator = iter(loader)
    batches = [batch_sha256(next(iterator)) for _ in range(args.batches)]
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / f"rank-{rank}.json"
    if output.exists():
        raise FileExistsError("Figure8 temperature probe receipts are immutable")
    write_json(output, {
        "status": "pass",
        "kind": "figure8_temperature_probe_v1",
        "temperature": args.temperature,
        "rank": rank,
        "world_size": world_size,
        "stage_sha256": file_sha256(args.stage / "stage.json"),
        "audit_sha256": file_sha256(args.audit / "audit.json"),
        "initial_model_sha256": initial_model_sha256,
        "ordered_batch_sha256": batches,
        "microbatches_per_epoch": len(loader),
        "training_options": options,
    })
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--effective-batch", type=int, default=1024)
    args = parser.parse_args()
    if not 1 <= args.batches <= 10:
        parser.error("Use 1..10 ordered preflight batches")
    probe(args)


if __name__ == "__main__":
    main()
