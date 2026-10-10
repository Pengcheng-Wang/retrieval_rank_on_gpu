# SilverTorch reproduction: measured findings

_Initial study completed 2026-10-10 on A100-SXM4 80GB devices. All timings below are local offline measurements._

---

The public fused scorer provides useful **memory and ANN latency gains**, but
the larger claim is conditional on workload geometry, exact filtering cost,
model features, and serving architecture. At a high-recall 20M operating point,
the complete SilverTorch FP32 adapter was about **2.7x faster than classic
FAISS GPU** and 13x faster than its matched FAISS CPU implementation on the
final idle-host confirmation. INT8
was not quality-equivalent on the dense clustered synthetic catalog.

The unchanged upstream source, independent numerical audit, complete sweep
tables, real TorchRec training, and frozen-checkpoint ANN comparisons are now
available in this repository. They do **not** reproduce Meta's private
production 23.7x QPS or 20.9x TCO results.[^1]

The initial protocol was selected from arXiv v2; the final source review
checked the published SIGIR version, arXiv v5. The relevant benchmark settings
are unchanged. V5 clarifies that the production user tower is HSTU, which
our MF adaptation does not implement.

## 📦 What was executed

| Track | Completed scope |
| --- | --- |
| ANN | 1M clustered and isotropic, 20M clustered, and 80M clustered catalogs |
| CPU library controls | Separate `faiss-cpu` IVFFlat and HNSW, plus a true no-CUDA smoke |
| Scoring breakdown | Native fused FP32/FP16/INT8 versus PyTorch gather-and-bmm at 20M |
| Filtering | 1M, 10M, and 40M items; exact CPU inverted/forward, GPU forward, raw Bloom, Bloom plus exact verification |
| Runtime reranking | Actual MF, interaction MLP, TorchRec DLRM, and TorchRec DLRM_DCN |
| Recommendation training | Four model families, three seeds each, 12 epochs, validation-selected checkpoints |
| Learned ANN pipeline | Six ANN representations/backends, three probe counts, three candidate caps, four frozen rankers, three seeds |
| Integration controls | Quiescent FAISS baselines, native full versus partial GPU masks, equivalent batched exact verification |

The numerical archive contains 1,017 ANN rows, 45 isolated-scoring rows, 200
filtering rows, 24 rank-timing rows, and 648 frozen learned-pipeline rows.
Current pipeline comparisons retain 180 valid operating-point rows; earlier
unvectorized/conversion controls remain explicitly **superseded**, not deleted.
All 12 validation-selected learned checkpoints were persisted.

Read the [generated tables](measured-tables.md) for representative operating
points and the [full CSV/raw artifacts](../results/2026-10-10) for the sweep.
The [protocol](benchmark-protocol.md) gives exact settings and limitations.

## 🔍 ANN gains depend on precision and recall

At 20M clustered vectors, batch 16, top-2048, and 64 probes:

| Backend | Mean batch ms | Geometric recall |
| --- | --- | --- |
| Matched FAISS IVFFlat CPU | 36.322 | 0.9999 |
| Classic FAISS IVFFlat GPU | 6.769 | 1.0000 |
| SilverTorch FP32 | 2.558 | 0.9998 |
| SilverTorch FP16 | 1.783 | 0.6376 |
| SilverTorch INT8-global | 1.601 | 0.5331 |
| SilverTorch INT8-rowwise | 1.806 | 0.6968 |

The FP32 comparison is close in recall and directly supports the measured
local ANN speedup. Calling the 1.601-ms INT8 result a 4.2x
**quality-equivalent** GPU speedup would be misleading. Neither INT8 arm nor
FP16 reached the 0.8 or 0.95 recall target within this declared probe grid.

A final resource check found keepalive processes had resumed on the other
nodes. They were left intact. Repeating the same FP32 anchor on idle node 0,
with the identical vectors, queries, ground truth, and partition, measured
32.988 ms for CPU FAISS, 6.779 ms for GPU FAISS, and 2.534 ms for SilverTorch,
with all recalls above 0.9998. This confirms approximately 2.68x GPU and 13.0x
CPU speedups without claiming that every earlier sweep was an isolated-host
run. The original sweep remains separately identifiable in the raw archive.

The 1M isotropic catalog behaves differently: every arm achieved about 0.249
recall at 64 probes. Here insufficient cluster coverage dominates the result;
similar recall does not demonstrate that quantization is harmless at a
high-recall operating point. Use the recall-constrained frontier table rather
than reading a single equal-probe speed ratio as a universal result.

### Larger inventory

At 80M, batch 16, top-2048, and 24 probes, FAISS GPU took 6.488 ms at recall
0.542; the INT8-global adapter took 2.053 ms at recall 0.345. This demonstrates
capacity and runtime scaling, not comparable retrieval quality.

The measured resident-memory deltas were about 45.4 GiB for FAISS GPU and
10.1 GiB for the INT8 adapter. Those deltas include library/allocator effects,
not just vector storage. An A100 80GB fits a single FP32 FAISS index that the
paper's A100 40GB setup had to shard; our results are therefore not a
reproduction of its server-count or cost model.[^1]

## 🧮 Fusion reduces intermediate memory

At 20M, batch 16, and 64 probes, the isolated FP32 fused scorer used about
93 MiB of incremental PyTorch peak allocation versus 2,970 MiB for gathering
the same candidate embeddings and applying `bmm`. Its scoring time was
2.000 ms versus 6.420 ms.

The FP16 fused-versus-gather comparison was 1.171 versus 5.473 ms and
71 versus 1,505 MiB. Native INT8 raw scoring took 0.746 ms, excluding query
quantization and post-rescaling.

These are **isolated scoring measurements**, not end-to-end retrieval or
serving throughput. They support the mechanism described in the paper:
avoid materializing a large candidate-embedding tensor.[^1]

## 🛡️ Fast Bloom masks are not exact rules

For a positive single-term query over 40M synthetic items, raw packed Bloom
took about 0.151 ms at multiplier 8.6. The exact GPU forward predicate took
2.541 ms, and the exact CPU posting-list reference took 7.098 ms.

That raw Bloom result admitted 619 false positives. Paying for exact
verification produced a 2.585-ms result with no false positives. Doubling
the adaptive signature width eliminated collisions in this particular query,
but the tested exact path still took about 2.659 ms. Neither zero collisions
in a fixture nor a fast approximate mask establishes exact policy safety for
arbitrary future queries.

The independent hazard fixtures reproduced four public-code behaviors:

| Fixture | Native result | Safe handling |
| --- | --- | --- |
| `NOT 1:1`, forced collisions | 382 eligible items falsely removed among 3,976 exact matches | Conservative prefilter plus exact predicate; zero lost matches |
| Half-divisor INT8 dot product | All 32 valid scores became nonfinite before division | Raw INT32 scoring and FP32 rescaling returned finite 16,129 |
| CPU partial-mask all-zero mask | 32 items returned despite the mask | GPU returned none; independent Boolean oracle required |
| CPU/GPU full-mask bit boundaries | CPU returned a different item set; GPU matched Bloom | Native GPU full-mask baseline validated independently |

The first two rates are deliberate adversarial fixtures, **not production
failure-rate estimates**. The [source audit](upstream-audit.md) links each
finding to the unchanged public code.[^2]

## 🔗 Full local pipelines do not universally beat FAISS GPU

Exact verification initially dominated the SilverTorch adapter. Profiling
the same predicate showed 11.161 ms for repeated per-query evaluation versus
1.774 ms for equivalent batched evaluation. Independent mask comparisons
verified identical eligibility semantics. No checking work was simply removed
from the timer.

After vectorization and quiescent baseline remeasurement, at 1M items,
512 candidates, 24 probes, batch 16, and an MLP reranker:

| Pipeline | Mean batch ms | Candidate geometric recall |
| --- | --- | --- |
| FAISS CPU + exact post-filter + MLP | 9.667 | 1.0000 |
| FAISS GPU + exact post-filter + MLP | 2.034 | 1.0000 |
| SilverTorch FP32 + partial filter + exact verification + MLP | 2.617 | 0.9999 |
| SilverTorch INT8 + partial filter + exact verification + MLP | 2.878 | 0.6935 |
| SilverTorch INT8 + native full filter + exact verification + MLP | 2.796 | 0.6935 |

SilverTorch was faster than the CPU pipeline but not the GPU pipeline at this
operating point. Partial filtering was not faster than the native full-mask
path here: the small inventory's full mask is cheap, while exact verification
and dynamic metadata still consume meaningful time.

This does not disprove Meta's integrated production result. These pipelines
are local eager implementations with no RPC service costs, random FP32
rerankers, one retrieval vector per request, and a different feature/query
distribution. They do show why a component speedup must not be promoted to a
whole-system claim without measuring every stage.

Query-plan parsing/upload is outside these timings, as are HSTU execution
and service queueing. Inputs are already prepared query vectors and filter
plans; this is not a raw-request serving latency measurement.

## 🧠 Real TorchRec models were trained, not substituted

The recommendation study used 959,054 fitting events, 6,033 evaluable users,
3,521 warm items, and the same fixed 2,000-user panel. Held-out target coverage
was 99.9%. Four additional sampled unobserved negatives per positive never
consulted future labels.

| Model | Mean test NDCG@10 | Across-seed SD | Test Recall@100 |
| --- | --- | --- | --- |
| MF dot-product retriever | 0.03079 | 0.00103 | 0.31783 |
| Interaction MLP | 0.02911 | 0.00089 | 0.30900 |
| Actual TorchRec DLRM | 0.03060 | 0.00197 | 0.32450 |
| Actual TorchRec DLRM_DCN | 0.03098 | 0.00073 | 0.32017 |

There is no convincing large rerank-quality gain in this small matched-feature
experiment. DCN's slightly higher average NDCG is not a demonstrated
statistically significant improvement. MF uses 128-dimensional embeddings
and about 1.25M parameters; the neural rankers use 32-dimensional sparse
embeddings and roughly 0.33-0.35M parameters. Features and examples are matched,
not model capacity or an individually tuned hyperparameter budget.

The TorchRec models use the real `DLRM`/`DLRM_DCN` classes,
`EmbeddingBagCollection`, and `KeyedJaggedTensor`.[^3] They are unsharded
baselines, not a claim about optimized table-batched distributed serving.
MovieLens is a research adaptation and a transductive per-user temporal split,
not Meta engagement data or a globally time-consistent deployment replay.[^4][^5]

### Learned vectors tolerate quantization differently

For the frozen learned MF vectors, 500 candidates, and all 64 clusters probed,
FP32 FAISS/SilverTorch geometric candidate recall was 1.0000. INT8-global
reached 0.9938 and INT8-rowwise 0.9957. The held-out target appeared in about
70.3% of candidate sets, and downstream NDCG changed very little.

This contrasts with the tightly clustered synthetic vectors. It supports
evaluating **both geometric fidelity and recommendation metrics on the actual
learned representations** before choosing a quantization strategy.

## 🎯 Adoption implications and remaining boundaries

The strongest supported takeaway is the fused memory primitive and the
high-recall FP32 ANN advantage. INT8's storage benefit is useful, but recall
needs representation-specific calibration. Exact eligibility is a first-class
cost and correctness constraint; Bloom cannot be treated as a policy oracle.

Keep learned-model comparisons separate from index comparisons. Reranking
quality depends on features, objectives, training and candidate availability.
A generic DLRM or MLP is not Meta's production MoL OverArch, and public
MovieLens ratings cannot reproduce its multi-task engagement/value model.[^1]

The public repository is preserved byte-for-byte for all 53 original files.
`scripts/verify_upstream.py` checks the source manifest without accessing
another checkout. The reproducible environment lock, sanitized numerical
journals, source checksums, and selected models have persistent copies.
Allocation access and operational progress remain local and ignored.

Next research questions are deliberately separate from the completed initial
study: independent synthetic-data seeds, compiled/BF16 TorchRec inference,
richer public recommendation features, and a clearly labeled public RAILS/MoL
adaptation. None should be silently relabeled as production SilverTorch.

[^1]: Xue et al. (2026). "SilverTorch: A Unified Model-based System to Democratize Large-Scale Recommendation on GPUs," sections 4-6, SIGIR '26 / arXiv v5. https://doi.org/10.1145/3805712.3809755 ; https://arxiv.org/abs/2511.14881v5
[^2]: Meta. Pinned SilverTorch source. https://github.com/meta-recsys/silvertorch/tree/22c2007ed5d28ec8304de081cc2393778708ad12
[^3]: Meta-PyTorch. TorchRec DLRM implementation. https://github.com/meta-pytorch/torchrec/blob/v1.8.0-rc1/torchrec/models/dlrm.py
[^4]: GroupLens. MovieLens 1M license and description. https://files.grouplens.org/datasets/movielens/ml-1m-README.txt ; "A Critical Study on Data Leakage in Recommender System Offline Evaluation." https://arxiv.org/abs/2010.11060
[^5]: F. Maxwell Harper and Joseph A. Konstan. (2015). "The MovieLens Datasets: History and Context." ACM Transactions on Interactive Intelligent Systems 5(4), Article 19, 1-19. https://doi.org/10.1145/2827872
