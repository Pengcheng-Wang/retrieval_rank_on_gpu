# Public SilverTorch source audit

_Pinned source: `22c2007ed5d28ec8304de081cc2393778708ad12`. Adaptations live outside the unchanged vendored code._

---

## 📦 Released and missing components

| Surface | Public release | Treatment in this repository |
| --- | --- | --- |
| Bloom build/search/query parser | C++/CUDA operators | Build and execute the original operators |
| Partial-response Bloom co-design | Operators and partial-mask metadata | Use the original GPU path with independent exact checks |
| Fused clustered item scoring | FP32/FP16/BF16/INT8 CUDA code | Add complete IVF search/selection/ID adapter |
| Filter `nn.Module` wrappers | Public Python modules | Preserve the source; low-level ops drive numerical benchmarks |
| KMeans publish pipeline | No complete production pipeline | One explicit shared spherical-KMeans benchmark partition |
| User/item trained production models | Not released | Separate public-data MF/MLP/TorchRec baselines |
| Evaluated MoL OverArch | Production weights/configuration absent | Document its role; do not relabel another model as MoL |
| Business value model | Rules and trained business context absent | No fabricated multi-objective lift |
| Production replay/TCO topology | Private data/services | No claim of production QPS/cost reproduction |

These boundaries follow the pinned source tree and the paper's component and
evaluation descriptions.[^1][^2]

## 🛡️ Approximate NOT can reject eligible items

The GPU query evaluator implements `NOT` by complementing the Bloom
membership word. A positive-membership collision is therefore transformed
into a false negative in a negative predicate.[^3]

If `A` is exact term membership and `B` is Bloom term membership, normally
`A` is a subset of `B`. It follows that `NOT B` is a subset of `NOT A`, not a
conservative superset. The public README's general no-false-negative
description must not be applied indiscriminately to negative expressions.

**Adaptation:** approximate only a conservative positive expression and then
evaluate the full exact predicate before final candidate selection. Positive
Bloom false positives are eliminated before consuming the returned candidate
budget. This trades extra exact work for a meaningful correctness contract.

`python -m rrbench.cli audit-upstream --output runs/audit.json` includes an
intentional low-width collision fixture and records raw versus conservative
false negatives. It is a correctness demonstration, not an estimated
production false-negative rate.

## 🔢 Per-item half-divisor overflow

In `index_mm_helpers.cuh`, the INT32-to-FP16 path with a half divisor casts
the integer result to half **before dividing**. A raw INT8 dot product can
exceed 65,504 even when its final scaled value is representable.[^4]

The fixture uses 128 components all equal to 127:
`128 * 127 * 127 = 2,064,512`. Division by 128 should produce 16,129.
Casting 2,064,512 to half first can produce infinity.

**Adaptation:** request raw INT32 scores and apply query/item scale correction
in FP32 outside the native kernel. The benchmark includes this work in the
complete search latency. The source is not patched to hide the difference
between the original and the adapted implementation.

## 🔍 CPU partial masks are not a correctness oracle

The public CPU `fused_kmean_ann_with_partial_masks` implementation discards its
partial-mask arguments and calls the unfiltered reference. Its multi-index
CPU path also omits per-item scale lists.[^5] An all-pass-mask test cannot
detect this mismatch.

Our audit passes an all-zero mask to CPU and GPU and counts valid returned
items. The benchmark uses an independently evaluated exact Boolean oracle;
it does not treat CPU/GPU agreement on this CPU path as proof of correct
eligibility.

## 🧩 Packed-bit conventions and incomplete APIs

Standalone Bloom decoding tests the most significant bit first. The CPU
fused full-mask lookup tests the least significant bit first.[^6] Passing
one representation directly to the other is not a reliable generic adapter.

The project uses both the real GPU partial-response path and the native GPU
full-mask baseline. An independent fixture confirmed the GPU full-mask path
matches Bloom's Boolean results; the CPU path does not. An initial
full-to-partial conversion adapter was superseded rather than used to inflate
the co-design speedup. Mixed-mask tests exercise IDs around 31/32 and 63/64,
tails, nonuniform cluster ranges, and external item reordering.

The fused scorer does not perform all of IVF retrieval: centroid probing,
global top-k, index capacity, and external IDs remain the caller's
responsibility.[^7] These operations are included in the reported full ANN
latency.

## 🔧 Build and source-integrity policy

The original Apache-2.0 source, copyright notices, license, and README remain
intact. `UPSTREAM.json` records the imported commit and an empty modification
list. No source is copied from an unofficial reproduction and labeled Meta.

On the supplied CUDA node, the build initially lacked discovery of the
NVIDIA wheel's `cusparse.h`. The build helper adds installed wheel include/lib
directories through process-scoped environment variables, then compiles for
A100 `sm80`. It does not change a base image or fetch another experiment's
environment.

Two released test files, `test_bloom_search_integration.py` and
`test_fresh_index_post_processing.py`, retain unclosed internal BUCK-loading
fragments and fail Python syntax parsing. Our independent suite and the valid
upstream ANN test file are executable; we do not silently edit the upstream
test corpus to manufacture a full upstream-suite pass.

[^1]: Meta. "SilverTorch pinned tree." https://github.com/meta-recsys/silvertorch/tree/22c2007ed5d28ec8304de081cc2393778708ad12
[^2]: Xue et al. "SilverTorch," SIGIR '26 / arXiv v5. https://doi.org/10.1145/3805712.3809755 ; https://arxiv.org/abs/2511.14881v5
[^3]: Meta. `bloom_index_search_cuda.cu`, NOT evaluation. https://github.com/meta-recsys/silvertorch/blob/22c2007ed5d28ec8304de081cc2393778708ad12/silvertorch/ops/csrc/bloom_index_search_cuda.cu#L448-L450
[^4]: Meta. `index_mm_helpers.cuh`, half-divisor postprocessing. https://github.com/meta-recsys/silvertorch/blob/22c2007ed5d28ec8304de081cc2393778708ad12/silvertorch/ops/csrc/index_mm_helpers.cuh#L47-L55
[^5]: Meta. `fused_kmean_ann.cpp`, CPU partial-mask and multi-index implementations. https://github.com/meta-recsys/silvertorch/blob/22c2007ed5d28ec8304de081cc2393778708ad12/silvertorch/ops/csrc/fused_kmean_ann.cpp#L229-L307
[^6]: Meta. Packed Bloom decoding and full-mask access. https://github.com/meta-recsys/silvertorch/blob/22c2007ed5d28ec8304de081cc2393778708ad12/silvertorch/ops/csrc/bloom_index_search.cpp#L231-L239 ; https://github.com/meta-recsys/silvertorch/blob/22c2007ed5d28ec8304de081cc2393778708ad12/silvertorch/ops/csrc/fused_kmean_ann.cpp#L74-L80
[^7]: Meta. Fused ANN operator documentation. https://github.com/meta-recsys/silvertorch/blob/22c2007ed5d28ec8304de081cc2393778708ad12/silvertorch/ops/csrc/fused_kmean_ann.cpp#L313-L400
