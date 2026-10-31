import importlib
import os
import random
import sys
import types
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "letter"))


@pytest.fixture(scope="module")
def rq_classes(tmp_path_factory):
    source = os.environ.get("LETTER_UPSTREAM")
    if source is None:
        pytest.skip("Explicit isolated Linux/CUDA LETTER environment is required")
    import torch
    from tokenizer_run import load_author

    assert torch.cuda.is_available()
    upstream = Path(source)
    patched, _, _, _ = load_author(upstream, tmp_path_factory.mktemp("rq-stage"))
    package = types.ModuleType("letter_original_rq")
    package.__path__ = [str(upstream / "RQ-VAE" / "models")]
    sys.modules[package.__name__] = package
    original = importlib.import_module("letter_original_rq.rqvae").RQVAE
    return original, patched


def test_default_tokenizer_parameter_plumbing_preserves_loss_and_indices(rq_classes):
    import torch

    original_class, patched_class = rq_classes
    cf = np.random.RandomState(7).normal(size=(8, 32)).astype(np.float32)
    kwargs = dict(in_dim=16, num_emb_list=[256] * 4, e_dim=32, layers=[64],
                  kmeans_init=False, sk_epsilons=[0, 0, 0, 0.003], sk_iters=50,
                  alpha=0.02, beta=0.0001, cf_embedding=cf)
    torch.manual_seed(42)
    original = original_class(**kwargs).cuda()
    patched = patched_class(**kwargs).cuda()
    patched.load_state_dict(original.state_dict(), strict=True)
    xs = torch.randn(8, 16, device="cuda")
    labels = {str(i): [j % 10 for j in range(256)] for i in range(4)}
    random.seed(42)
    out_a, quant_a, ids_a, dense_a = original(xs, labels)
    loss_a = original.compute_loss(out_a, quant_a, torch.arange(8), dense_a, xs)[0]
    random.seed(42)
    out_b, quant_b, ids_b, dense_b = patched(xs, labels)
    loss_b = patched.compute_loss(out_b, quant_b, torch.arange(8), dense_b, xs)[0]
    torch.testing.assert_close(ids_a, ids_b, rtol=0, atol=0)
    torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)
    loss_a.backward()
    loss_b.backward()
    for (name_a, pa), (name_b, pb) in zip(original.named_parameters(), patched.named_parameters()):
        assert name_a == name_b
        torch.testing.assert_close(pa.grad, pb.grad, rtol=0, atol=0)


def test_cf_gradient_remains_indirect_through_encoder(rq_classes):
    import torch

    _, model_class = rq_classes
    cf = np.random.RandomState(9).normal(size=(8, 32)).astype(np.float32)
    model = model_class(in_dim=16, num_emb_list=[256] * 4, e_dim=32, layers=[64],
                        kmeans_init=False, sk_epsilons=[0] * 4, cf_embedding=cf,
                        alpha=0.02, beta=0.0001).cuda()
    labels = {str(i): [j % 10 for j in range(256)] for i in range(4)}
    xs = torch.randn(8, 16, device="cuda")
    out, quant, _, dense = model(xs, labels)
    total, cf_loss, reconstruction, quant_loss = model.compute_loss(out, quant, torch.arange(8), dense, xs)
    torch.testing.assert_close(total, reconstruction + quant_loss + 0.02 * cf_loss)
    encoder = list(model.encoder.parameters())
    codebooks = [layer.embedding.weight for layer in model.rq.vq_layers]
    decoder = list(model.decoder.parameters())
    grads = torch.autograd.grad(cf_loss, encoder + codebooks + decoder, allow_unused=True)
    assert sum(g.abs().sum().item() for g in grads[:len(encoder)] if g is not None) > 0
    assert all(g is None for g in grads[len(encoder):])


def test_default_constrained_initialization_matches_original(rq_classes):
    import torch

    original_class, patched_class = rq_classes
    kwargs = dict(in_dim=16, num_emb_list=[256] * 4, e_dim=32, layers=[64],
                  kmeans_init=True, sk_epsilons=[0, 0, 0, 0.003], sk_iters=50,
                  cf_embedding=np.zeros((512, 32), dtype=np.float32))
    torch.manual_seed(42)
    original = original_class(**kwargs).cuda().eval()
    patched = patched_class(**kwargs).cuda().eval()
    patched.load_state_dict(original.state_dict(), strict=True)
    xs = torch.randn(512, 16, device="cuda")
    np.random.seed(42)
    with torch.no_grad():
        original.vq_initialization(xs)
    np.random.seed(42)
    with torch.no_grad():
        patched.vq_initialization(xs)
    for left, right in zip(original.rq.vq_layers, patched.rq.vq_layers):
        assert left.initted and right.initted
        torch.testing.assert_close(left.embedding.weight, right.embedding.weight, rtol=0, atol=0)
