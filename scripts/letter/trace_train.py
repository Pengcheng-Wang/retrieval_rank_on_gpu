"""Gate-only instrumentation of the exact ordered tensors seen by author T5."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import sys
from pathlib import Path

from author import load_backend
from protocol import file_sha256, write_json


def batch_digest(inputs: dict) -> str:
    import torch

    digest = hashlib.sha256()
    for name in sorted(inputs):
        value = inputs[name]
        if torch.is_tensor(value):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(name.encode())
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--trace-dir", type=Path, required=True)
    trace, remaining = parser.parse_known_args()
    try:
        upstream = Path(remaining[remaining.index("--upstream") + 1]).resolve()
        output = Path(remaining[remaining.index("--output") + 1]).resolve()
    except (ValueError, IndexError):
        parser.error("--upstream and --output are required")
    author = load_backend(upstream, "tiger")
    original = author.model.LETTER.forward
    rank = int(os.environ.get("RANK", "0"))
    trace.trace_dir.mkdir(parents=True, exist_ok=True)
    trace_file = trace.trace_dir / f"rank-{rank}.jsonl"
    with trace_file.open("x", encoding="utf-8"):
        pass
    if rank == 0:
        instrumentation = {
            "kind": "gate_only_ordered_training_tensor_hashes",
            "source_sha256": file_sha256(Path(__file__)),
        }
        marker = output / "instrumentation.json"
        if marker.exists() and json.loads(marker.read_text()) != instrumentation:
            raise ValueError("Gate instrumentation changed between invocations")
        write_json(marker, instrumentation)

    @functools.wraps(original)
    def traced(model, *args, **kwargs):
        if model.training:
            if args:
                raise ValueError("Gate instrumentation expects the Trainer's named input tensors")
            with trace_file.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "digest": batch_digest(kwargs), "batch_size": len(kwargs["input_ids"]),
                }) + "\n")
        return original(model, *args, **kwargs)

    author.model.LETTER.forward = traced
    sys.argv = [sys.argv[0]] + remaining
    import run
    run.main()


if __name__ == "__main__":
    main()
