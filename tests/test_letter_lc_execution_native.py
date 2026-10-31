"""Production-sized native prompt/sampler replay without loading another GPU model."""

import hashlib
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "letter"))


@pytest.mark.skipif(not os.environ.get("LETTER_UPSTREAM"), reason="Pinned native runtime required")
def test_production_native_prompt_stream_cold_replay(tmp_path):
    import numpy as np
    import torch
    from accelerate.data_loader import BatchSamplerShard, DataLoaderShard, skip_first_batches
    from torch.utils.data import BatchSampler, RandomSampler
    from transformers import TrainingArguments
    from author import default_args, load_backend
    from lc_checkpoint import assert_identical
    from lc_run import epoch_geometry, trainer_class
    from protocol import file_sha256

    upstream = Path(os.environ["LETTER_UPSTREAM"])
    author = load_backend(upstream, "lc-rec")
    native = default_args(author, upstream, "Instruments")
    dataset = author.data.SeqRecDataset(native, mode="train", sample_num=0)
    assert len(dataset) == 131837

    def collate(rows):
        return [(row["input_ids"], row["labels"]) for row in rows]

    trainer = trainer_class()(
        model=torch.nn.Linear(1, 1),
        args=TrainingArguments(
            output_dir=str(tmp_path / "loader-construction"), no_cuda=True, report_to=[],
            per_device_train_batch_size=8, remove_unused_columns=False, disable_tqdm=True,
        ),
        train_dataset=dataset, data_collator=collate,
    )
    loader_class = type(trainer.get_train_dataloader())
    batches = epoch_geometry(len(dataset), 8, 4, 2)["microbatches_per_epoch"]
    assert batches == 4120
    results = []

    def reseed():
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

    def rng():
        return {"python": random.getstate(), "numpy": np.random.get_state(),
                "cpu": torch.random.get_rng_state()}

    def restore(state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.random.set_rng_state(state["cpu"])

    def make_loader(rank):
        generator = torch.Generator().manual_seed(42)
        sampler = BatchSamplerShard(
            BatchSampler(RandomSampler(dataset, generator=generator), batch_size=8, drop_last=False),
            num_processes=4, process_index=rank, split_batches=False, even_batches=True,
        )
        loader = loader_class(dataset, batch_sampler=sampler, collate_fn=collate, num_workers=0)
        loader._lc_before_fetch = None
        loader._lc_owner = SimpleNamespace(
            args=SimpleNamespace(process_index=rank), _lc_preserve_collated_rng=False,
        )
        assert len(loader) == batches
        return loader, generator

    def update(digest, batch):
        return hashlib.sha256(
            bytes.fromhex(digest) + json.dumps(batch, ensure_ascii=False).encode(),
        ).hexdigest()

    for rank in range(4):
        reseed()
        loader, generator = make_loader(rank)
        digest, count, cuts = "0" * 64, 0, {}
        for epoch in range(2):
            for batch in loader:
                count += 1
                digest = update(digest, batch)
                if count in (2, batches, batches + 2):
                    cuts[count] = {"rng": rng(), "digest": digest}
        expected = {"rng": rng(), "digest": digest, "sampler": generator.get_state(), "count": count}
        for cut, state in cuts.items():
            directory = tmp_path / f"rank-{rank}-cut-{cut}"
            directory.mkdir()
            torch.save(state["rng"], directory / f"rng_state_{rank}.pth")
            reseed()
            base, resumed_generator = make_loader(rank)
            epochs, skip = divmod(cut, batches)
            for epoch in range(epochs):
                for batch in base:
                    pass
            if skip:
                first = skip_first_batches(base, skip)
                assert isinstance(first, DataLoaderShard)
                first.__class__ = loader_class
                first._lc_before_fetch = directory
                first._lc_owner = base._lc_owner
            else:
                restore(state["rng"])
                first = base
            observed, count = state["digest"], cut
            restore_cpu_once = bool(skip)
            for epoch in range(epochs, 2):
                for batch in (first if epoch == epochs else base):
                    if restore_cpu_once:
                        # Trainer restores Torch/CUDA after collation, preserving advanced prompt RNG.
                        torch.random.set_rng_state(state["rng"]["cpu"])
                        restore_cpu_once = False
                    observed = update(observed, batch)
                    count += 1
            actual = {"rng": rng(), "digest": observed,
                      "sampler": resumed_generator.get_state(), "count": count}
            assert_identical(expected, actual, f"production.rank_{rank}.cut_{cut}")
            results.append({"rank": rank, "cut_microbatch": cut, "ordered_inputs_sha256": observed,
                            "final_microbatches": count, "status": "pass"})
    report = {"status": "pass", "dataset": "Instruments", "training_prefixes": len(dataset),
              "world_size": 4, "micro_batch": 8, "accumulation": 2, "microbatches_per_epoch": batches,
              "runner_sha256": file_sha256(ROOT / "scripts" / "letter" / "lc_run.py"),
              "test_source_sha256": file_sha256(Path(__file__)),
              "scope": "Native production-sized ordered rendered inputs and sampler/Python/NumPy/CPU RNG",
              "cases": results}
    (tmp_path / "production-loader.json").write_text(json.dumps(report, indent=2) + "\n")
    destination = os.environ.get("LC_PRODUCTION_REPORT")
    if destination:
        path = Path(destination).resolve()
        assert path.is_relative_to((ROOT / ".local" / "letter").resolve())
        with path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
