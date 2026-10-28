"""GPU contracts; run explicitly in the isolated LETTER environment."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "letter"))


@pytest.fixture(scope="module")
def author():
    from author import load_backend

    upstream = os.environ.get("LETTER_UPSTREAM")
    if upstream is None:
        pytest.skip("Set LETTER_UPSTREAM in the validated Linux/CUDA LETTER environment")
    import torch
    assert torch.cuda.is_available(), "The GPU contract has no CPU fallback"
    return load_backend(Path(upstream), "tiger")


def test_author_temperature_loss_matches_independent_oracle(author):
    import torch
    from transformers import T5Config

    model = author.model.LETTER(T5Config(vocab_size=16, d_model=8, d_ff=16, num_heads=2,
                                        d_kv=4, num_layers=1, decoder_start_token_id=0))
    logits = torch.randn(2, 5, 16, device="cuda", requires_grad=True)
    labels = torch.tensor([[2, 3, 4, 1, -100], [5, 6, 7, 8, 1]], device="cuda")
    for temperature in (1.0, 0.7):
        model.set_hyper(temperature)
        actual = model.ranking_loss(logits, labels)
        oracle = torch.nn.functional.cross_entropy(
            (logits / temperature).reshape(-1, 16), labels.reshape(-1), ignore_index=-100,
        )
        torch.testing.assert_close(actual, oracle)
        actual_grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
        oracle_grad = torch.autograd.grad(oracle, logits, retain_graph=True)[0]
        torch.testing.assert_close(actual_grad, oracle_grad)
        assert actual_grad[0, 4].abs().sum().item() == 0


def test_author_forward_matches_stock_t5_at_unit_temperature(author, tmp_path):
    import torch
    from transformers import T5Config, T5ForConditionalGeneration

    config = T5Config(vocab_size=32, d_model=16, d_ff=32, num_heads=2, d_kv=8,
                      num_layers=2, dropout_rate=0.0, decoder_start_token_id=0)
    torch.manual_seed(42)
    custom = author.model.LETTER(config).cuda().eval()
    stock = T5ForConditionalGeneration(config).cuda().eval()
    stock.load_state_dict(custom.state_dict(), strict=True)
    inputs = torch.tensor([[3, 4, 1, 0], [5, 6, 7, 1]], device="cuda")
    labels = torch.tensor([[8, 9, 1], [10, 11, 1]], device="cuda")
    actual = custom(input_ids=inputs, attention_mask=inputs.ne(0), labels=labels)
    expected = stock(input_ids=inputs, attention_mask=inputs.ne(0), labels=labels)
    torch.testing.assert_close(actual.loss, expected.loss, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual.logits, expected.logits, rtol=1e-6, atol=1e-6)
    custom.save_pretrained(tmp_path)
    reloaded = T5ForConditionalGeneration.from_pretrained(tmp_path).cuda().eval()
    result = reloaded(input_ids=inputs, attention_mask=inputs.ne(0), labels=labels)
    torch.testing.assert_close(result.logits, actual.logits, rtol=1e-6, atol=1e-6)


def test_fixed_beam_generation_uses_valid_full_prefixes(author):
    import torch
    from transformers import T5Config, T5ForConditionalGeneration
    from protocol import PrefixTrie
    from evaluate import generate_fixed

    model = T5ForConditionalGeneration(T5Config(
        vocab_size=64, d_model=16, d_ff=32, num_heads=2, d_kv=8,
        num_layers=1, dropout_rate=0.0, decoder_start_token_id=0,
    )).cuda().eval()
    candidates = [[0, 2 + i, 30 + (i % 3), 1] for i in range(25)]
    trie = PrefixTrie(candidates)
    with torch.inference_mode():
        result = generate_fixed(
            model, {"input_ids": torch.tensor([[3, 1]], device="cuda")},
            lambda b, prefix: trie.allowed(prefix.tolist()), 20,
        )
    sequences = [tuple(row) for row in result.sequences.tolist()]
    assert len(sequences) == len(set(sequences)) == 20
    assert set(sequences) <= {tuple(row) for row in candidates}
