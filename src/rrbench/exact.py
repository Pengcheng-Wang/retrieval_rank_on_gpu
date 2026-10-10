from __future__ import annotations

import numpy as np
import torch


@torch.inference_mode()
def exact_topk(
    items: np.ndarray | torch.Tensor,
    queries: torch.Tensor,
    k: int,
    eligible: torch.Tensor | None = None,
    chunk_size: int = 131072,
) -> tuple[torch.Tensor, torch.Tensor]:
    if queries.ndim != 2 or items.ndim != 2 or items.shape[1] != queries.shape[1]:
        raise ValueError("Item and query dimensions must match.")
    if not 0 < k <= len(items) or chunk_size <= 0:
        raise ValueError("Require 0 < k <= catalog size and a positive chunk size.")
    if eligible is not None and eligible.shape != (len(queries), len(items)):
        raise ValueError("Eligibility must have one Boolean value per query and item.")
    best_scores = torch.full((len(queries), k), -torch.inf, device=queries.device)
    best_ids = torch.full((len(queries), k), -1, dtype=torch.long, device=queries.device)
    for start in range(0, len(items), chunk_size):
        block = torch.as_tensor(
            np.array(items[start : start + chunk_size], copy=True)
            if isinstance(items, np.ndarray)
            else items[start : start + chunk_size],
            device=queries.device,
            dtype=torch.float32,
        )
        scores = queries.float() @ block.T
        if eligible is not None:
            scores = scores.masked_fill(~eligible[:, start : start + len(block)], -torch.inf)
        local_scores, local_ids = scores.topk(min(k, len(block)), dim=1)
        all_scores = torch.cat((best_scores, local_scores), dim=1)
        all_ids = torch.cat((best_ids, local_ids + start), dim=1)
        best_scores, positions = all_scores.topk(k, dim=1)
        best_ids = all_ids.gather(1, positions)
    return best_scores, best_ids.masked_fill(~torch.isfinite(best_scores), -1)
