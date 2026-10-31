"""One source-hash-guarded, resume-only correction in isolated Transformers 4.31."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


ORIGINAL_SHA256 = "24a579e7af77c3699020ac16c1b4ad16d64aa584dafe52a90d0807c381174f31"
BEFORE = (
    b"            for epoch in range(epochs_trained):\n"
    b"                for _ in train_dataloader:\n"
    b"                    break\n"
)
AFTER = BEFORE.replace(b"                    break\n", b"                    pass\n")


def patched_trainer(source: bytes) -> bytes:
    if hashlib.sha256(source).hexdigest() == ORIGINAL_SHA256 and source.count(BEFORE) == 1:
        return source.replace(BEFORE, AFTER)
    if source.count(AFTER) == 1:
        reverted = source.replace(AFTER, BEFORE)
        if hashlib.sha256(reverted).hexdigest() == ORIGINAL_SHA256:
            return source
    raise ValueError("Trainer source checksum differs from the verified 4.31.0 implementation")


def trainer_path() -> Path:
    spec = importlib.util.find_spec("transformers")
    if spec is None or spec.origin is None:
        raise ImportError("The isolated Transformers package is not installed")
    path = Path(spec.origin).parent / "trainer.py"
    if sys.prefix == sys.base_prefix or not path.resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("Only a separate virtual environment may receive the compatibility patch")
    return path


def require_resume_patch() -> str:
    source = trainer_path().read_bytes()
    if patched_trainer(source) != source:
        raise RuntimeError("Install the verified resume-only Trainer patch before training")
    return hashlib.sha256(source).hexdigest()


def main() -> None:
    path = trainer_path()
    source = path.read_bytes()
    patched = patched_trainer(source)
    if patched != source:
        temporary = path.with_suffix(".letter-patch.tmp")
        temporary.write_bytes(patched)
        temporary.replace(path)
    print(json.dumps({
        "original_sha256": ORIGINAL_SHA256,
        "patched_sha256": require_resume_patch(),
        "change": "Fully exhaust completed epochs during resume-only sampler replay",
    }))


if __name__ == "__main__":
    main()
