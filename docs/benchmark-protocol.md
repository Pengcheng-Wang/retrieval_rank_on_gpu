# Benchmark protocol

_Frozen interpretation and correctness contracts for the 2026-10-10 study._

---

## 🎯 Questions and non-goals

This study asks which parts of public SilverTorch improve local GPU retrieval
efficiency, how those improvements trade off against geometric ANN recall,
how exact filtering affects the full pipeline, and what common trained
rerankers achieve on a separate recommendation task.

It does not claim reproduction of private production traffic, MoL weights,
multi-task engagement objectives, business value rules, online A/B results,
or the paper's service-level QPS/TCO.[^1]

Source control for the protocol: initial reading used arXiv v2; the final
research notes checked v5, the published SIGIR '26 version. The core ANN,
filtering, hardware, traffic, and latency-budget definitions used for this
adaptation remain consistent. V5 explicitly identifies HSTU as the user tower,
which this MF-based public-data benchmark does not reproduce.

| Track | Data | Measurement | What it cannot prove |
| --- | --- | --- | --- |
| ANN | Normalized synthetic 128D vectors | Latency versus exhaustive FP32 nearest-neighbor recall | Recommendation engagement lift |
| Filtering | Synthetic multi-value attributes | Matching latency, false positives, false negatives, bytes | Private CPU inverted-index speed |
| Integrated pipeline | Synthetic vectors, actual model classes, random weights | GPU input to GPU output runtime and candidate recall | Trained recommendation quality |
| Recommendation | Research-only MovieLens 1M | Validation-selected held-out Recall/NDCG | Meta production ranking or deployment-time lift |

## ⚖️ ANN fairness

All IVF arms share one spherical KMeans quantizer, one set of fixed item
assignments, and the same original external IDs. FAISS `add_core` ingests those
assignments explicitly rather than independently rebuilding another
partition. SilverTorch sorts vectors by the same assignment and maps its
internal row IDs back to external IDs.

The complete SilverTorch search measurement includes centroid scoring,
selected-length lookup, dynamic output-size synchronization, fused item
scoring, global top-k, and ID remapping. A raw scoring operator is not
represented as a whole ANN engine.

| Arm | Representation | Search |
| --- | --- | --- |
| FAISS IVFFlat CPU | FP32 | Actual CPU FAISS IVF implementation |
| Classic FAISS IVFFlat GPU | FP32 | Actual GPU clone; cuVS explicitly disabled where configurable |
| SilverTorch FP32 | FP32 | Public fused scorer plus the complete adapter |
| SilverTorch FP16 | FP16 | Same partition and complete adapter |
| SilverTorch INT8-global | Global symmetric item scale; rowwise query scales | Raw INT32 kernel scores followed by FP32 rescaling |
| SilverTorch INT8-rowwise | Per-item symmetric scales; rowwise query scales | Same raw kernel with item-specific FP32 correction |
| Optional FAISS HNSW CPU | FP32 | Separate graph baseline; `efSearch`, not IVF probes |

The GPU-enabled FAISS wheel also exposes actual CPU indexes. The report
distinguishes this from a separate `faiss-cpu` wheel run; do not install both
packages into the same environment because both export `faiss`.

The GPU-enabled binary is community-built `faiss-gpu-cu12==1.14.1.post1`;
the independent CPU-wheel reference is `faiss-cpu==1.14.2`. Their version/build
differences remain visible rather than being hidden in a single library label.

The precision oracle uses FP32 products, no autocast, and TF32 disabled.
Ground-truth computation is chunked and outside timing. Exhaustive search of
dequantized representations separately checks quantization correctness.
Quantization includes zero-vector guards, explicit round-to-nearest behavior,
clamping to `[-127,127]`, and scale correction. It is not an undocumented
implementation of the paper's full global min/max calibration.[^1]

### Matrix

| Catalog | Distribution | Main k | Main probes | Batch | Repeats / warmup / samples |
| --- | --- | --- | --- | --- | --- |
| 1M | Clustered | 100, 1024, 2048, 4096 | 8, 24, 64, 256 | 1, 16 | 3 / 20 / 50 |
| 1M | Isotropic | 100, 1024, 2048 | 8, 24, 64, 256 | 16 | 3 / 20 / 50 |
| 20M | Clustered | 2048, 4096 | 8, 24, 64, 256 | 16 | 3 / 50 / 100 |
| 80M | Clustered | 1024, 2048 | 24 | 16 | 3 / 10 / 20 |

The 80M arm is a capacity/scaling adaptation, not a copy of a two-A100-40GB
production deployment. Classic FAISS GPU top-k greater than 2048 is recorded
as unsupported, never timed as a fallback.[^2]

The headline FP32 20M point was additionally repeated on a verified idle node,
at k=2048, probes=64, batch 16, 50 warmups, 100 samples, and three repeats.
It reuses the exact original vectors, queries, truth, and IVF partition and
has its own experiment ID; it is not pooled into the original sweep.

The additional 1M CPU-wheel/HNSW study uses k=100/1024/2048,
IVF probes=24/64/256, batch 16, and three repeats with 20 warmups and 50
samples. HNSW uses M=32, efConstruction=100, and
efSearch=max(k,requested effort); it has no IVF probing parameter.

## ⏱️ Timing and uncertainty

CUDA timing synchronizes before and after each operation. CPU indexes receive
CPU queries and GPU indexes receive GPU queries in standalone ANN measurement;
input/output transfers are excluded there and explicitly included in the
integrated pipeline.

Every numerical journal retains individual batch samples, warmup and repeat
configuration, mean/p50/p95/p99, query-batch size, and environment metadata.
Bootstrap intervals within one measurement describe its local samples, not
independent production runs. Report tables aggregate repeat means and show
repeat variation. They do not call a small-sample p99 a proven serving SLO.

`offline_queries_per_second = batch_size * 1000 / mean_batch_latency_ms`.
This is **not serving QPS** under a concurrency/queueing/p99 budget. The paper
uses a client-driven replay with increasing offered load; our benchmark does
not emulate that protocol with invented latency.[^1]

Index/representation bytes and GPU memory deltas are recorded separately from
PyTorch peak allocations. The latter cannot see all FAISS allocations. GPU
resident deltas may include allocator reservations and are not perfectly
isolated model-size measurements.

## 🛡️ Eligibility correctness

The CPU sorted-posting baseline is our exact NumPy inverted-index reference,
not Meta's internal optimized service. CPU/GPU forward baselines evaluate the
same Boolean syntax. Bloom-packed rows measure the approximation alone;
Bloom-plus-exact rows measure an actual exact result with verification.

Standalone filtering runs use 1M/10M items at Bloom multipliers 2/5/10 and a
40M paper-sized adaptation at multipliers 8.6/17.1. All have six attributes
with two sampled values each, five declared Boolean query types, one query per
timed batch, 10 warmups and 30 samples. The real source's adaptive index width
and measured bytes are recorded; no private production attribute/query
distribution is assumed.

Negative Bloom terms are unsafe to complement directly. The adapter converts
expressions to negation-normal form and replaces negative literals with `TRUE`
in the approximate prefilter. It then evaluates the entire exact expression
on scored items **before final top-k**. This preserves eligible candidates in
the selected IVF universe, even when Bloom collisions occur.

The complete pipeline compares FAISS post-filtering with explicit 4x
oversampling against SilverTorch pre-top-k filtering. It reports both
candidate fill and geometric candidate recall: these different retrieval
policies are not mislabeled as identical candidate sets. Reranker weights and
feature budgets are matched across the retrieval engines.

Filter-expression parsing and query-plan upload are prepared outside the
pipeline timer. The timed request input is an existing query vector plus
GPU-resident filter plans, not a raw user history or text query. User-tower
execution and RPC/queueing are also excluded.

Exact predicates are batched across requests sharing the same expression.
`pipeline-profile` measures and verifies the earlier per-row implementation
against that vectorization. Superseded pipeline rows stay in the raw journal
with an explicit status and are excluded from current aggregate tables.

## 🧪 Recommendation quality

MovieLens is a research-use dataset with redistribution and commercial-use
restrictions; source/prepared data remain ignored and are not published.[^3][^7]

| Choice | Definition |
| --- | --- |
| Positive | Rating at least 4 |
| Holdout | Last two distinct positive timestamps per user |
| Simultaneous ratings | Do not fabricate within-second chronology; pick one deterministic target per held-out timestamp |
| Fitting events | All that user's observed events strictly before validation time |
| Warm inventory | Items with at least one positive training interaction |
| Original negative labels | Observed training ratings below 4 |
| Additional negatives | Four uniform unobserved warm items per positive, with replacement |
| Negative exclusion | All observed training items; never consult held-out positives to blacklist samples |
| Features | User/item IDs plus log-normalized training-only user/item counts |
| Training seeds | 17, 23, 42 |
| Training budget | 12 epochs, Adam 0.003, batch 4096 |
| Selection | Best validation full-catalog NDCG@10 |
| Evaluation panel | Same fixed 2,000-user sample across models and seeds |
| Test | Once after checkpoint selection, with target coverage reported |
| Metrics | Recall/hit-rate and NDCG at 10, 20, 100 |
| Two-stage candidates | 100, 500, 1000 from the same seed's frozen MF retriever |

The per-user split is **transductive**, not a global-time deployment
simulation: other users' pooled fitting interactions may occur after the
evaluated user's event. This limitation is not hidden behind a "no temporal
leakage" claim.[^4]

At each query, exclude all observed items strictly before its timestamp. Test
history can include validation-period observations known by that test time;
weights and training count features remain frozen. Do not inject the target
into retrieved candidates. Retrieval misses count as zero for downstream
metrics. Coverage includes targets outside the warm inventory or already
seen at query time.

Full-catalog results are distinct from sampled-negative ranking, because
sampled ranking can reverse model comparisons.[^5] With a single held-out
positive, Recall@k equals hit rate; NDCG discounts its position.

TorchRec uses actual unsharded DLRM and DLRM_DCN implementations.[^6]
Two sparse tables on MovieLens are a deliberately small matched feature
budget, not evidence about a rich web-scale DLRM schema or distributed
embedding sharding.

### Learned vectors through real ANN

The frozen selected checkpoints are also evaluated through actual FAISS
CPU/GPU and SilverTorch FP32/FP16/INT8 retrieval. This is a distinct,
predeclared test comparison, not further training or checkpoint tuning on test.
It uses the same evaluation panel and warm item inventory, a shared 64-cluster
partition, probes 8/24/64, and candidate caps 100/500/1000. All 64 probes
isolate precision/filtering effects from cluster approximation.

SilverTorch applies exact query-time seen-item eligibility before its ANN
top-k. FAISS uses 4x post-filter oversampling, capped at the inventory size
and at 2048 on the classic GPU backend. Both candidate fill and geometric
candidate recall are reported alongside downstream Recall/NDCG. Every frozen
ranker sees the same candidate set for a given ANN arm; no held-out target is
injected into that set.

## ✅ Gates and persistence

The 100-row CUDA smoke checks complete ANN search against FP32 exact results,
INT8 against its dequantized representation, external-ID remapping, nonuniform
cluster boundaries, mixed eligibility, and actual TorchRec forward/backward.
Upstream ANN tests run separately from our independent oracles.

Known upstream numerical/eligibility hazards are reproduced in
`audit-upstream`, with forced-collision/nonfinite fixtures explicitly
distinguished from production rate estimates. The exact source snapshot is
unchanged.

Quality checkpoints use atomic file replacement; epoch-level optimizer state
allows resumption. Completed test evaluations have a completion marker and
are not repeated by relaunch. ANN journals skip previously completed
configuration/repeat keys. Failures remain explicit rather than silently
switching precision, CPU/GPU backend, or model class.

New runs bind their output directory to an immutable configuration before
execution. A completed historical artifact without that binding remains
read-only; it is not silently assigned new arguments on relaunch.

Node-local data and environments are ephemeral. Completed code, manifests,
small checkpoints, and numerical bundles are persisted by one designated
writer. Workers never write to the same shared experiment folder concurrently.
The local operational record identifies the resource and writer; no access
configuration belongs in git.

The current timing models are eager FP32 rerankers, with 128-dimensional MF
retrieval embeddings and 32-dimensional neural sparse embeddings. Input
features and training examples are matched, not parameter counts. This is not
a comparison of compiled BF16 production reranking graphs. Model parameter
counts are preserved in the numerical artifacts.

[^1]: Xue et al. (2026). "SilverTorch," sections 4 and 6, SIGIR '26 / arXiv v5. https://doi.org/10.1145/3805712.3809755 ; https://arxiv.org/abs/2511.14881v5
[^2]: Faiss developers. "Faiss on the GPU." https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU
[^3]: GroupLens. "MovieLens 1M usage license." https://files.grouplens.org/datasets/movielens/ml-1m-README.txt
[^4]: "A Critical Study on Data Leakage in Recommender System Offline Evaluation." https://arxiv.org/abs/2010.11060
[^5]: Krichene and Rendle. "On Sampled Metrics for Item Recommendation." https://research.google/pubs/on-sampled-metrics-for-item-recommendation/
[^6]: Meta-PyTorch. "TorchRec DLRM source." https://github.com/meta-pytorch/torchrec/blob/v1.8.0-rc1/torchrec/models/dlrm.py
[^7]: F. Maxwell Harper and Joseph A. Konstan. (2015). "The MovieLens Datasets: History and Context." ACM Transactions on Interactive Intelligent Systems 5(4), Article 19, 1-19. https://doi.org/10.1145/2827872
