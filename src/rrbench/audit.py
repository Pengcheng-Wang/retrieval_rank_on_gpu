from __future__ import annotations

from pathlib import Path

import torch

from rrbench.filters import BloomFilter, evaluate, make_features, parse_expression
from rrbench.io import write_json
from rrbench.runtime import configure, environment_manifest, require_cuda


@torch.inference_mode()
def audit_upstream(output: Path) -> dict[str, object]:
    import silvertorch.ops._load_ops  # noqa: F401

    configure(17, 4)
    device = require_cuda()
    features = torch.tensor(make_features(4096, 17), device=device)
    bloom = BloomFilter(features, multiplier=1.1)
    text = "NOT 1:1"
    truth = evaluate(parse_expression(text), features)
    raw = bloom.search(bloom.plans([text], conservative=False))[0]
    conservative = bloom.search(bloom.plans([text], conservative=True))[0]
    if (truth & ~conservative).any():
        raise RuntimeError("Conservative NOT handling failed.")
    offsets = torch.tensor([0, 32], device=device)
    selected = torch.tensor([[0]], device=device)
    lengths = torch.tensor([[32]], device=device)
    values = torch.full((32, 128), 127, device=device, dtype=torch.int8)
    query = values[:1]
    divisor = torch.full((32,), 128, device=device, dtype=torch.float16)
    native_scaled, native_ids = torch.ops.st.fused_kmean_ann(
        offsets, selected, lengths, values, query, 32, per_embedding_scale=divisor
    )
    raw_scores, raw_ids = torch.ops.st.fused_kmean_ann(offsets, selected, lengths, values, query, 32)
    rescaled = raw_scores.float() / 128.0
    valid = raw_ids >= 0
    if not torch.isfinite(rescaled[valid]).all() or not (rescaled[valid] == 16129).all():
        raise RuntimeError("FP32 post-rescaling failed.")
    mask_count = torch.tensor([1], dtype=torch.int32)
    mask_first = torch.tensor([0], dtype=torch.int8)
    zero_mask = torch.tensor([0], dtype=torch.long)
    cpu_scores, cpu_ids = torch.ops.st.fused_kmean_ann_with_partial_masks(
        offsets.cpu(), selected.cpu(), lengths.cpu(), values.cpu(), query.cpu(), 32,
        mask_count, mask_first, zero_mask,
    )
    gpu_scores, gpu_ids = torch.ops.st.fused_kmean_ann_with_partial_masks(
        offsets, selected, lengths, values, query, 32,
        mask_count.to(device), mask_first.to(device), zero_mask.to(device),
    )
    boundary_features = torch.full((100, 1, 1), 2, dtype=torch.long, device=device)
    boundary_features[[0, 31, 32, 63, 64, 65, 99]] = 1
    boundary_bloom = BloomFilter(boundary_features)
    boundary_plans = boundary_bloom.plans(["1:1"])
    boundary_mask = boundary_bloom.search(boundary_plans, packed=True)
    expected_boundary = boundary_bloom.search(boundary_plans)[0].nonzero().flatten().tolist()
    boundary_offsets = torch.tensor([0, 100], device=device)
    boundary_lengths = torch.tensor([[100]], device=device)
    boundary_items = torch.randn(100, 16, device=device)
    boundary_query = torch.randn(1, 16, device=device)
    boundary_results = {}
    for target_device in ("cpu", "cuda"):
        _, boundary_ids = torch.ops.st.fused_kmean_ann(
            boundary_offsets.to(target_device), selected.to(target_device),
            boundary_lengths.to(target_device), boundary_items.to(target_device),
            boundary_query.to(target_device), 100,
            filtering_bit_mask=boundary_mask.to(target_device),
        )
        boundary_results[target_device] = sorted(boundary_ids[boundary_ids >= 0].tolist())
    if boundary_results["cuda"] != expected_boundary:
        raise RuntimeError("Native GPU full-mask convention did not match Bloom.")
    result = {
        "upstream_commit": "22c2007ed5d28ec8304de081cc2393778708ad12",
        "environment": environment_manifest(),
        "bloom_not": {
            "fixture_items": len(features), "multiplier": 1.1,
            "exact_matches": int(truth.sum()),
            "raw_false_negatives": int((truth & ~raw).sum()),
            "conservative_false_negatives": int((truth & ~conservative).sum()),
            "interpretation": "Forced collisions; not a production false-negative-rate estimate.",
        },
        "native_int8_half_divisor": {
            "raw_dot_product": 2064512, "divisor": 128,
            "native_nonfinite_valid_scores": int((~torch.isfinite(native_scaled[native_ids >= 0])).sum()),
            "fp32_rescale_nonfinite_scores": int((~torch.isfinite(rescaled[valid])).sum()),
            "fp32_rescale_expected_score": 16129,
            "adapter_policy": "Raw INT32 kernel outputs followed by FP32 item/query rescaling; cost included.",
        },
        "cpu_partial_mask": {
            "zero_mask_cpu_valid_items": int((cpu_ids >= 0).sum()),
            "zero_mask_gpu_valid_items": int((gpu_ids >= 0).sum()),
            "adapter_policy": "Independent exact Boolean oracle; CPU partial-mask implementation is not an oracle.",
        },
        "packed_full_mask": {
            "expected_synthetic_ids": expected_boundary,
            "cpu_synthetic_ids": boundary_results["cpu"],
            "gpu_synthetic_ids": boundary_results["cuda"],
            "gpu_full_mask_is_compatible": True,
            "policy": "Use native GPU full-mask baseline; CPU full-mask path has a different bit convention.",
        },
    }
    write_json(output, result)
    return result
