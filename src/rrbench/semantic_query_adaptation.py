from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from rrbench.semantic_models import SemanticEncoder


class QueryLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if (
            not isinstance(base, nn.Linear) or type(rank) is not int or rank <= 0
            or isinstance(alpha, bool) or not math.isfinite(alpha) or alpha <= 0
        ):
            raise ValueError("Query adapter requires a linear target and positive finite rank/alpha.")
        self.base = base
        self.base.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / rank
        self.enabled = True
        self.in_features, self.out_features = base.in_features, base.out_features

    @property
    def weight(self) -> nn.Parameter:
        return self.base.weight

    @property
    def bias(self) -> nn.Parameter | None:
        return self.base.bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        original = self.base(inputs)
        if not self.enabled:
            return original
        residual = F.linear(F.linear(inputs.float(), self.lora_A), self.lora_B) * self.scale
        return original + residual.to(original.dtype)


def attach_query_adapters(model: nn.Module, rank: int = 16, alpha: float = 32) -> dict[str, Any]:
    config = getattr(model, "config", None)
    layers = getattr(config, "num_hidden_layers", None)
    if type(layers) is not int or layers <= 0:
        raise ValueError("Query-only adaptation requires the declared transformer-layer inventory.")
    targets = [(name, value) for name, value in model.named_modules() if name.rsplit(".", 1)[-1] in ("q_proj", "v_proj")]
    inventory: set[tuple[int, str]] = set()
    for name, value in targets:
        match = re.fullmatch(r"(?:.*\.)?layers\.(\d+)\.self_attn\.(q_proj|v_proj)", name)
        if match is None or not isinstance(value, nn.Linear):
            raise ValueError("Missing, unexpected or previously adapted query projection target.")
        key = (int(match.group(1)), match.group(2))
        if key in inventory:
            raise ValueError("Duplicate transformer query-adapter target.")
        inventory.add(key)
    expected = {(layer, projection) for layer in range(layers) for projection in ("q_proj", "v_proj")}
    if inventory != expected or len(targets) != 2 * layers:
        raise ValueError("Query-adapter target inventory differs from exactly q_proj/v_proj in every layer.")
    if type(rank) is not int or rank <= 0 or isinstance(alpha, bool) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Query-adapter rank and alpha must be positive finite values.")
    model.requires_grad_(False)
    for name, value in targets:
        parent_name, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        setattr(parent, attribute, QueryLoRALinear(value, rank, alpha))
    model.eval()
    return {
        "target_count": len(targets), "target_names": [name for name, _ in targets],
        "rank": rank, "alpha": alpha, "adapter_dtype": "float32",
        "trainable_parameters": sum(parameter.numel() for parameter in adapter_parameters(model)),
        "base_frozen": True, "scope": "Query-only attention q_proj/v_proj adapters; no merge, dropout or base quantization.",
    }


def adapter_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [
        parameter for module in model.modules() if isinstance(module, QueryLoRALinear)
        for parameter in (module.lora_A, module.lora_B)
    ]


def set_adapters_enabled(model: nn.Module, enabled: bool) -> None:
    if type(enabled) is not bool:
        raise ValueError("Adapter execution mode must be an explicit boolean.")
    modules = [module for module in model.modules() if isinstance(module, QueryLoRALinear)]
    if not modules:
        raise ValueError("Query model has no installed adapter inventory.")
    for module in modules:
        module.enabled = enabled


def adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    state = {
        name + "." + suffix: getattr(module, suffix).detach().cpu().clone()
        for name, module in model.named_modules() if isinstance(module, QueryLoRALinear)
        for suffix in ("lora_A", "lora_B")
    }
    if not state:
        raise ValueError("Query model has no adapter state to checkpoint.")
    return state


def load_adapter_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    parameters = {
        name + "." + suffix: getattr(module, suffix)
        for name, module in model.named_modules() if isinstance(module, QueryLoRALinear)
        for suffix in ("lora_A", "lora_B")
    }
    if not parameters or set(parameters) != set(state):
        raise ValueError("Adapter checkpoint has missing, unexpected or base-weight keys.")
    for name, parameter in parameters.items():
        value = state[name]
        if (
            not isinstance(value, torch.Tensor) or value.shape != parameter.shape
            or value.dtype != torch.float32 or not torch.isfinite(value).all()
        ):
            raise ValueError("Adapter checkpoint shape, dtype or finite-value contract differs.")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(state[name].to(parameter.device))


def positive_mass_loss(
    query_vectors: torch.Tensor, item_vectors: torch.Tensor, positive_mask: torch.Tensor,
    valid_mask: torch.Tensor | None = None, temperature: float = 0.05,
) -> torch.Tensor:
    if (
        query_vectors.ndim != 2 or item_vectors.ndim != 3 or not len(query_vectors)
        or item_vectors.shape[0] != len(query_vectors) or item_vectors.shape[2] != query_vectors.shape[1]
        or item_vectors.shape[1] == 0 or item_vectors.requires_grad
        or positive_mask.shape != item_vectors.shape[:2] or positive_mask.dtype != torch.bool
        or query_vectors.device != item_vectors.device or positive_mask.device != query_vectors.device
        or not torch.isfinite(query_vectors).all() or not torch.isfinite(item_vectors).all()
        or isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0
    ):
        raise ValueError("Positive-set mass requires aligned finite queries, frozen item vectors and boolean positives.")
    if valid_mask is None:
        valid_mask = torch.ones_like(positive_mask)
    if (
        valid_mask.shape != positive_mask.shape or valid_mask.dtype != torch.bool
        or valid_mask.device != positive_mask.device or not positive_mask.any(dim=1).all()
        or (positive_mask & ~valid_mask).any()
    ):
        raise ValueError("Every fitting query must retain all declared positives within its valid candidate set.")
    logits = (query_vectors.float().unsqueeze(1) * item_vectors.float()).sum(dim=2) / temperature
    denominator = torch.logsumexp(logits.masked_fill(~valid_mask, -torch.inf), dim=1)
    numerator = torch.logsumexp(logits.masked_fill(~positive_mask, -torch.inf), dim=1)
    return denominator - numerator


def optimizer_blocks(count: int, effective_batch: int = 64) -> list[tuple[int, int]]:
    if type(count) is not int or type(effective_batch) is not int or min(count, effective_batch) <= 0:
        raise ValueError("Optimizer blocks require positive integer query and effective-batch counts.")
    return [(start, min(count, start + effective_batch)) for start in range(0, count, effective_batch)]


def forward_queries(encoder: SemanticEncoder, texts: list[str]) -> torch.Tensor:
    from rrbench.semantic_models import normalize, pool_hidden

    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Query-only forward requires nonempty actual query strings.")
    limit = encoder.specification["max_tokens"]
    if type(limit) is not int or limit <= 0:
        raise ValueError("Query token cap must be a positive pinned integer.")
    formatted = encoder.format_texts(texts, query=True)
    inputs = encoder.tokenizer(
        formatted, padding=True, truncation=False, max_length=limit,
        return_tensors="pt", pad_to_multiple_of=8,
    )
    inputs = {name: value.to(encoder.device) for name, value in inputs.items()}
    mask = inputs["attention_mask"]
    if (
        mask.ndim != 2 or len(mask) != len(texts) or not mask.any(dim=1).all()
        or (mask.sum(dim=1) > limit).any() or mask.shape[1] > limit
    ):
        raise ValueError("Actual instructed query exceeds its audited token cap or token geometry.")
    output = encoder.model(**inputs, use_cache=False)
    vectors = normalize(pool_hidden(output.last_hidden_state, mask, encoder.specification["pooling"]))
    return vectors.to(torch.float16).float()
