from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SymmetricInt8:
    values: torch.Tensor
    scales: torch.Tensor
    mode: str

    def dequantize(self) -> torch.Tensor:
        return self.values.float() * self.scales


def quantize_symmetric(values: torch.Tensor, mode: str) -> SymmetricInt8:
    if values.ndim != 2 or not values.is_floating_point() or not torch.isfinite(values).all():
        raise ValueError("Quantization requires a finite floating-point matrix.")
    if mode == "rowwise":
        maxima = values.abs().amax(dim=1, keepdim=True)
    elif mode == "global":
        maxima = values.abs().amax().reshape(1, 1)
    else:
        raise ValueError("Quantization mode must be 'global' or 'rowwise'.")
    scales = torch.where(maxima > 0, maxima / 127.0, torch.ones_like(maxima))
    quantized = (values / scales).round().clamp(-127, 127).to(torch.int8)
    return SymmetricInt8(quantized.contiguous(), scales.float(), mode)
