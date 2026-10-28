import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "letter"
sys.path.insert(0, str(SCRIPTS))


def test_training_batch_digest_observes_sample_order_and_labels():
    assert (SCRIPTS / "trace_train.py").exists(), "Training-batch trace gate is missing"
    from trace_train import batch_digest
    inputs = {"input_ids": torch.tensor([[1, 2], [3, 4]]), "labels": torch.tensor([[5], [6]])}
    original = batch_digest(inputs)
    assert original == batch_digest({k: v.clone() for k, v in inputs.items()})
    assert original != batch_digest({k: v.flip(0) for k, v in inputs.items()})
    assert original != batch_digest({**inputs, "labels": torch.tensor([[6], [5]])})
