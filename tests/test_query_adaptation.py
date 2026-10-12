from __future__ import annotations

import copy
import importlib
import importlib.util
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn


def adaptation():
    assert importlib.util.find_spec("rrbench.semantic_query_adaptation"), "Query-only adaptation core is not implemented."
    return importlib.import_module("rrbench.semantic_query_adaptation")


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)

    def forward(self, value):
        return self.q_proj(value) + self.v_proj(value)


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=2, hidden_size=4)
        self.embedding = nn.Embedding(64, 4)
        self.layers = nn.ModuleList([nn.Module() for _ in range(2)])
        for layer in self.layers:
            layer.self_attn = Attention()
        self.extra = nn.Linear(4, 4, bias=False)

    def forward(self, input_ids, attention_mask, use_cache=False):
        assert use_cache is False
        value = self.embedding(input_ids)
        for layer in self.layers:
            value = layer.self_attn(value)
        return SimpleNamespace(last_hidden_state=value)


def test_zero_adapter_preserves_base_output_and_freezes_base_parameters():
    module = adaptation()
    torch.manual_seed(17)
    model = Transformer()
    baseline = copy.deepcopy(model)
    inputs = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(inputs)
    expected = baseline(inputs, mask).last_hidden_state.detach()
    info = module.attach_query_adapters(model, rank=2, alpha=4)
    assert info["target_count"] == 4 and info["trainable_parameters"] == 64
    assert torch.equal(model(inputs, mask).last_hidden_state, expected)
    assert len(module.adapter_parameters(model)) == 8
    assert all(not parameter.requires_grad for name, parameter in model.named_parameters() if "lora_" not in name)


def test_enabled_adapter_updates_but_disabled_path_is_the_unchanged_base():
    module = adaptation()
    torch.manual_seed(17)
    model = Transformer()
    baseline = copy.deepcopy(model)
    module.attach_query_adapters(model, rank=2, alpha=4)
    parameters = module.adapter_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=0.01, weight_decay=0)
    inputs = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(inputs)
    loss = model(inputs, mask).last_hidden_state.square().mean()
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in parameters)
    before = module.adapter_state(model)
    optimizer.step()
    assert any(not torch.equal(before[name], value) for name, value in module.adapter_state(model).items())
    module.set_adapters_enabled(model, False)
    assert torch.equal(model(inputs, mask).last_hidden_state, baseline(inputs, mask).last_hidden_state)


def test_adapter_state_contains_no_base_weights_and_reload_preserves_output():
    module = adaptation()
    torch.manual_seed(17)
    first = Transformer()
    second = copy.deepcopy(first)
    module.attach_query_adapters(first, rank=2, alpha=4)
    module.attach_query_adapters(second, rank=2, alpha=4)
    for parameter in module.adapter_parameters(first):
        parameter.data.add_(0.01)
    state = module.adapter_state(first)
    assert len(state) == 8 and all("lora_" in name for name in state)
    module.load_adapter_state(second, state)
    inputs = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(inputs)
    assert torch.equal(first(inputs, mask).last_hidden_state, second(inputs, mask).last_hidden_state)


@pytest.mark.parametrize("mutation", ("missing", "extra", "shape", "nonfinite"))
def test_adapter_reload_rejects_incompatible_or_partial_state_atomically(mutation):
    module = adaptation()
    model = Transformer()
    module.attach_query_adapters(model, rank=2, alpha=4)
    original = module.adapter_state(model)
    state = {name: value.clone() for name, value in original.items()}
    key = next(iter(state))
    if mutation == "missing":
        del state[key]
    elif mutation == "extra":
        state["base.weight"] = torch.zeros(1)
    elif mutation == "shape":
        state[key] = torch.zeros(1)
    else:
        state[key].fill_(float("nan"))
    with pytest.raises(ValueError):
        module.load_adapter_state(model, state)
    assert all(torch.equal(value, module.adapter_state(model)[name]) for name, value in original.items())


@pytest.mark.parametrize("problem", ("missing_target", "unexpected_target", "duplicate_install"))
def test_target_inventory_fails_closed_instead_of_expanding_scope(problem):
    module = adaptation()
    model = Transformer()
    if problem == "missing_target":
        del model.layers[1].self_attn.v_proj
    elif problem == "unexpected_target":
        model.extra.q_proj = nn.Linear(4, 4)
    else:
        module.attach_query_adapters(model, rank=2, alpha=4)
    with pytest.raises(ValueError):
        module.attach_query_adapters(model, rank=2, alpha=4)


def test_bfloat16_base_keeps_float32_adapter_parameters_and_gradients():
    module = adaptation()
    model = Transformer().to(torch.bfloat16)
    module.attach_query_adapters(model, rank=2, alpha=4)
    parameters = module.adapter_parameters(model)
    assert all(parameter.dtype == torch.float32 for parameter in parameters)
    inputs = torch.tensor([[1, 2, 3]])
    model(inputs, torch.ones_like(inputs)).last_hidden_state.float().square().mean().backward()
    assert all(parameter.grad is not None and parameter.grad.dtype == torch.float32 for parameter in parameters)
    assert all(not parameter.requires_grad for name, parameter in model.named_parameters() if "lora_" not in name)


def test_positive_set_mass_retains_every_positive_and_ignores_only_declared_padding():
    module = adaptation()
    query = torch.tensor([[1.0, 0.0]], requires_grad=True)
    items = torch.tensor([[[1.0, 0.0], [0.5, 0.0], [-1.0, 0.0], [100.0, 0.0]]])
    positive = torch.tensor([[True, True, False, False]])
    valid = torch.tensor([[True, True, True, False]])
    loss = module.positive_mass_loss(query, items, positive, valid)
    scores = torch.tensor([20.0, 10.0, -20.0])
    expected = torch.logsumexp(scores, 0) - torch.logsumexp(scores[:2], 0)
    assert loss.shape == (1,) and torch.allclose(loss[0], expected)
    loss.sum().backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()


@pytest.mark.parametrize("problem", ("no_positive", "positive_padding", "nonfinite", "trainable_items"))
def test_loss_rejects_invalid_training_records(problem):
    module = adaptation()
    query = torch.tensor([[1.0, 0.0]], requires_grad=True)
    items = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    positive = torch.tensor([[True, False]])
    valid = torch.tensor([[True, True]])
    if problem == "no_positive":
        positive.fill_(False)
    elif problem == "positive_padding":
        valid[0, 0] = False
    elif problem == "nonfinite":
        query = torch.tensor([[float("nan"), 0.0]], requires_grad=True)
    else:
        items.requires_grad_(True)
    with pytest.raises(ValueError):
        module.positive_mass_loss(query, items, positive, valid)


def test_optimizer_blocks_preserve_the_final_sixteen_query_denominator():
    module = adaptation()
    blocks = module.optimizer_blocks(50000, 64)
    assert len(blocks) == 782 and blocks[0] == (0, 64) and blocks[-1] == (49984, 50000)
    assert sum(stop - start for start, stop in blocks) == 50000
    losses = torch.arange(16, dtype=torch.float32, requires_grad=True)
    total = sum(losses[start:start + 8].sum() / 16 for start in (0, 8))
    total.backward()
    assert torch.equal(losses.grad, torch.full((16,), 1 / 16))


@pytest.mark.parametrize("count,batch", ((0, 64), (1, 0), (True, 64), (16, 1.0)))
def test_optimizer_blocks_reject_invalid_counts(count, batch):
    with pytest.raises(ValueError):
        adaptation().optimizer_blocks(count, batch)


class Tokenizer:
    def __init__(self):
        self.formatted = None

    def __call__(self, texts, padding, truncation, max_length, return_tensors, pad_to_multiple_of):
        assert padding is True and truncation is False and return_tensors == "pt" and pad_to_multiple_of == 8
        self.formatted = texts
        lengths = [len(text.split()) for text in texts]
        width = math.ceil(max(lengths) / 8) * 8
        ids, masks = [], []
        for length in lengths:
            ids.append([0] * (width - length) + list(range(1, length + 1)))
            masks.append([0] * (width - length) + [1] * length)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}


def encoder():
    from rrbench.semantic_models import SemanticEncoder

    value = SimpleNamespace(
        model=Transformer(), device=torch.device("cpu"), tokenizer=Tokenizer(), is_qwen=True,
        specification={"max_tokens": 16, "pooling": "last valid token", "query_instruction": "task"},
    )
    value.format_texts = lambda texts, query: SemanticEncoder.format_texts(value, texts, query)
    return value


def test_query_forward_uses_pinned_rendering_last_token_canonicalization_and_live_gradient():
    module = adaptation()
    assert hasattr(module, "forward_queries"), "Training needs a direct gradient-enabled query forward."
    value = encoder()
    module.attach_query_adapters(value.model, rank=2, alpha=4)
    vectors = module.forward_queries(value, ["one", "one two"])
    assert value.tokenizer.formatted == ["Instruct: task\nQuery:one", "Instruct: task\nQuery:one two"]
    assert vectors.shape == (2, 4) and vectors.dtype == torch.float32 and vectors.requires_grad
    assert torch.equal(vectors, vectors.half().float())
    vectors[:, 0].sum().backward()
    assert all(parameter.grad is not None for parameter in module.adapter_parameters(value.model))


def test_query_forward_does_not_silently_truncate_an_unaudited_overflow():
    module = adaptation()
    assert hasattr(module, "forward_queries"), "Training needs a bounded raw query forward."
    with pytest.raises(ValueError, match="cap"):
        module.forward_queries(encoder(), [" ".join(["word"] * 20)])


def test_zero_adapter_raw_forward_equals_fresh_base_with_identical_token_geometry():
    module = adaptation()
    assert hasattr(module, "forward_queries"), "Raw grouped baseline parity needs the training-compatible forward."
    first = encoder()
    second = encoder()
    second.model.load_state_dict(first.model.state_dict())
    module.attach_query_adapters(first.model, rank=2, alpha=4)
    assert torch.equal(module.forward_queries(first, ["one", "one two"]), module.forward_queries(second, ["one", "one two"]))
