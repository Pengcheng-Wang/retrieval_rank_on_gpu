# Controlled personalization and real-semantic profiling extension

_Bounded follow-up to the immutable real-corpus study at `69bbd5e`._

---

## 🎯 Questions and scope

The extension asks two questions: whether a small learned user representation
can recover useful personalized candidate signal against the frozen video
catalog, and which actual retrieval/ranking pipeline provides a defensible
quality, latency and memory tradeoff. It does not repeat corpus encoding or
claim to reproduce Meta's private production models or serving gains.[^1]

All original results remain unchanged. Reusing inspected test panels makes
this follow-up **post-test exploratory**, even though recipes, validation
decisions and final evaluations are frozen before the new outcomes.

| Workstream | Reused inventory | Intervention | Primary comparison |
| --- | --- | --- | --- |
| Personalized candidate signal | 1,798,875 Qwen0.6B-encoded videos | One shared user head over the same past exposure vectors | Equal-user known-positive Recall@1000 and wrong-user/popularity controls |
| Real-semantic characterization | 8,841,823 Qwen8-MRL1024 passages | Storage/backend, batch and measured stage boundaries | Complete search and actual-candidate ranking quality plus synchronized wall time |

The machine-readable recipe and absolute execution gates are in
[`semantic-extension-v1.json`](../configs/semantic-extension-v1.json).
The previous [real-corpus findings](real-semantic-results.md) define the
input, model, sparse-judgment and coverage limitations.

## 🧠 Controlled shared user representation

Item vectors remain fixed. A 658,433-parameter attention-pooling/residual head
maps up to 12 training exposure-item vectors to a normalized 1024D query.
Relative-recency embeddings permit order sensitivity. Zero-initialized output
attention and residual layers start the head at the same-history mean, rather
than silently changing history selection, catalog eligibility or item training.

The new fitting cohort selects at most 50,000 admitted training timestamp
blocks using label-independent, user-balanced hashes. Positive-target and
nonempty-history filtering occurs afterward, with all attrition reported.
Every history envelope must end before the target's lower bound; histories
are not outcome-selected and held-out exposures are never appended.

The fixed sampled objective is:

\[
\mathcal L_q =
\log\sum_{i\in P_q\cup L_q\cup U_q}\exp(s(q,i)/\tau)
-\log\sum_{i\in P_q}\exp(s(q,i)/\tau),\qquad \tau=0.05.
\]

Every covered fitting positive in \(P_q\) is retained. \(L_q\) contains up to
12 distinct logged nonpositive items, excluding positives. \(U_q\) contains
up to 64 distinct sampled catalog alternatives, excluding both previous sets.
Sampling is fixed per query/seed and never consults validation/test labels.
These are source-distinct **weak negatives**, not verified exhaustive
irrelevance or an unbiased engagement estimator.

| Control | What it distinguishes |
| --- | --- |
| Frozen raw-history Qwen0.6B | Existing text-history retrieval |
| Mean of the same exposure vectors | Learned aggregation versus changed item/history selection |
| Training-positive popularity | Personalized utility versus a strong nonpersonalized control |
| Recent-history items plus popularity | Recurrence versus novel-to-selected-history retrieval |
| Uniform catalog candidates | Sparse-hit baseline |
| Whole-user history derangement | Aligned personalization versus assigning the wrong user's history |
| Within-history order permutation | Temporal sensitivity with the same exposure multiset |

“Novel” means outside the fixed selected history, not historically never
exposed. Training-positive support strata are not total exposure-support
strata. They are secondary diagnostics and do not replace the original
conditional query/user denominators.

## 🚦 Fixed validation admission and final evaluation

The pilot is **seed17, epoch3, full-catalog exhaustive top1000**, evaluated
only on the unchanged validation panel. The proposed resource gate requires
at least 10 distinct hit users and an equal-user aligned-minus-wrong-user
Recall@1000 difference of at least 0.005. This is an information-value
threshold, not a power calculation.

Only a passing, on-time persisted decision admits seed23. Failure means no
rescue sweep; a missed pilot deadline is time-budget incomplete, not evidence
of no signal. Final test metrics cannot authorize more fitting. A second seed
that is not admitted is **not assessed by design**, not a missing experiment.

Primary metrics preserve all conditional panel rows, including zero-hit
users. Equal-user and query-weighted means are reported separately.
Comparators share item/query mappings and the same aggregation. The
catalog-positive event coverage is reported numerically alongside these
conditional results; incompatible means are never multiplied.

Any two-seed research qualification uses a mean of per-query seed metrics,
not a prediction ensemble or test-selected winner. Its fixed quality and
runtime thresholds remain exploratory qualification criteria, not causal
engagement uplift or deployment readiness.

## ⚙️ Real ANN profiles and actual-pipeline comparison

```mermaid
flowchart LR
    accTitle: Controlled Extension Evidence Gates
    accDescr: Source-bound frozen data feeds isolated candidate-signal and profiling experiments, then matched actual-pipeline results and reviewed publication.
    frozen_inputs["Frozen catalog and query contracts"] --> user_head["Controlled user-head experiment"]
    frozen_inputs --> ann_profile["Separate CPU and CUDA profiles"]
    ann_profile --> actual_pipeline["Actual retrieved candidates and same scorer"]
    user_head --> evidence["Full denominators and paired uncertainty"]
    actual_pipeline --> evidence
    evidence --> final_review["Reviewed immutable results"]
```

The ANN characterization retains exactly 16,384 shared IVF clusters.
FAISS/Silver FP32, actual scalar-half FAISS, Silver FP16 and rowwise INT8 are
tested at probes64/256, depth1000 and batches1/16. Three independently started
sessions use label-independent paired query schedules and counterbalanced
backend order. Batch64 is a conditional memory stress case, not an
unconditionally promised point.

Profiles run separately from primary wall-time samples. CPU self time and
CUDA device time can overlap or nest; their sum is not a request latency or
critical path. CUDA tracing requires actual exported kernel events. A
completed unsupported-profiler attempt is not successful GPU profiling.

The complete-pipeline comparisons use frozen semantic-MLP seed17 and BGE
scorers, depth1000 and prefix caps50/500. Each point retrieves its **own actual
candidates**, then scores the prefix before final top10. Its quality cannot
be replaced with earlier common-union quality.

The first pipeline smoke rejected comparison with historical batch32 query
vectors: identical model/token inputs under a different BF16 execution
geometry exceeded the declared vector tolerance. Replaying the original
32-query group reproduced its stored vectors exactly; repeating the same new
group also reproduced its vectors exactly. Matching padding or logical
positions alone did not remove cross-group drift. This isolates execution
geometry sensitivity without identifying a particular CUDA kernel cause.

The prospective
[`pipeline-v2 contract`](../configs/semantic-extension-pipeline-v2.json)
therefore computes quality with **actual raw forwards in immutable execution
groups**, then checks independent raw forwards in those identical groups.
The historical cache remains lineage/diagnostic evidence, never an online
substitute. Every one of the 1,097 quality rows is covered exactly once and
outputs are scattered back to canonical row order. Backend, scorer, cap,
boundary and reduced measurement budgets cannot change the group partition.

Initial parity checks preserve complete groups and cover at least100 queries
(112 at normal batch16); the exact100-row smoke has a separate manifest.
Every measured/pretrace group is checked too. The original vector tolerance
remains maxabs0.0001/mincos0.99999; there is no outcome-based relaxation.
Candidate/reference comparisons require identical execution-group hashes
and actual canonical-query hashes, not merely equal policy names.
Quality is scoped to its session/batch geometry, and repeated contexts do
not create additional independent query observations.

Before production, execution priority is prospectively amended to run the
nine designated primary MLP/cap50/batch16 cells first, preserving the rotated
backend order within each of the three contexts. The remaining63 cells keep
their original relative order and stable IDs. All72 remain in the denominator
irrespective of favorable, negative, unsupported or deadline-unfinished
outcomes; secondary work is not selected by primary quality. Actual order and
UTC requested/completed times are recorded rather than implying each restart
is one contiguous time block. The separate planned pipeline traces follow;
already completed ANN traces are not repeated.

Within-backend resident/local-cascade
candidate and final IDs must match exactly and scorer values must meet the
declared tolerance. FAISS can feed a resident neural scorer too.[^2]

## 📐 Interpretation and execution limits

Two predeclared batch16/cap50/probes64 MLP comparisons use the fixed 1,000 selected
dev queries for MRR@10 non-inferiority. TREC19/20 remain separate graded
panels. Paired per-query metric differences are averaged across the three
fixed session contexts before resampling those same1,000 query identities.
They are not treated as3,000 independent queries. A lower confidence bound
must exceed the specified negative margin;
an interval containing zero is not equivalence. Multiple primary comparisons
receive the declared adjustment; other intervals remain secondary exploratory
comparisons.

After the first FP32 MLP primary point failed its prescribed memory-budget
gate, the geometry-oracle input dependency is prospectively amended. All
six session/batch contexts use the same fixed FAISS scalar-half / MLP /
cap50 donor for their complete **actual query payload**, independent of
quality, speed or availability. This is a post-failure operational amendment,
not original preregistration or a replacement primary reference.

The donor must supply the full source-bound 1,097x1,024 FP16 query artifact,
ordered cohort and immutable execution groups. Partial or independently
unattested query files are unavailable; there is no fallback donor. Only query
vectors are copied, never donor ANN candidates or scores. The oracle still
exhaustively scores the whole canonical item bank using FP32 products with
TF32 disabled. Geometry joins require matching actual-query, group, catalog,
panel, raw-source and model identities. Missing/mismatched evidence is
unavailable, not zero or permission to change a point's queries.

An exhaustive oracle can exist even when a full FP32 encoder/index/MLP
layout cannot maintain its headroom gate. That does not repair the primary
comparison: under the declared three-valid-reference requirement, primary
non-inferiority is not assessed when the compatible FP32 reference set is
unsupported. Half-versus-INT8 and different-scorer comparisons remain
secondary, not replacement primaries.

Fresh FP32 resident timing is the reference. A deployment candidate must
demonstrate complete-pipeline benefit or quantified memory savings within
the latency tolerance, rather than an ANN-only speedup. Observed headroom
and finite no-OOM runs are admission evidence, not guarantees against unseen
peaks. No serving QPS, tail SLO, energy, MFU or fabricated RPC is inferred.

Only source/implementation/checkpoint-bound completed files can be reused.
An absolute cutoff bounds in-flight registered workers. Worker identity and
process completion are verified before original keeper restoration; empty
process argv during teardown is pending observation, not permission to
signal or certify successful drainage.

## 🔒 Cooperative resource handoff

The original ANN-only characterization retains its checked STOP/CONT lease.
The first full-pipeline launch failed before any production point because the
original supervisor did not remain stopped. A bounded observation confirmed
the transition, but the source of the independent continuation remains
unidentified. This failure is not a scientific timing or quality outcome.

The prospective pipeline resource-control amendment leaves that supervisor
running and uses its existing cooperative external-client behavior. One
registered, start-gated administrative CUDA context on an unused GPU makes
the native keeper withdraw; actual observed withdrawal is required before
any benchmark starts. The context is bound to the intended device UUID and
performs no periodic/background CUDA work after initialization.

| Resource | Purpose | Required evidence |
| --- | --- | --- |
| Benchmark GPU | Actual isolated encoder/index/scorer workload | Existing exclusive-client and headroom observations |
| Additional administrative GPU | Idle context for native cooperative handoff | Device UUID and actual sampled NVML allocation, separate from the one-byte tensor payload |
| Original supervisor | Unchanged cooperative yield and restoration | Exact process identity, source-policy hashes and operating environment |

Each real child is registered before a **fresh, nonreusable** start gate is
published. Parent-death protection precedes scientific-module import, and
the original absolute cutoff remains in force. Unexpected reservation loss
prevents further admission and drains only the verified real worker.
Cleanup drains real workers, then the administrative context, verifies both
absent, and passively waits for the original policy and all eight original
allocations to return. This path sends no keeper/supervisor signals and
changes no profile, grace marker or persistent policy.

The administrative context is an additional occupied physical device, not
another experimental arm or evidence of useful utilization. Its allocation
is recorded separately from benchmark memory. Matched pipeline arms/restarts
share this policy; earlier STOP-controlled ANN measurements are not silently
pooled as the same environment. Scientific source hashes, execution groups,
models and thresholds remain unchanged. Linux lifecycle checks and a
20-row resource-control CUDA smoke with native-restoration evidence gate
the full finite grid; this does not replace the model/query parity smoke.

[^1]: [Original study protocol](semantic-study-design.md) and [measured results](real-semantic-results.md).
[^2]: FAISS GPU interoperability. https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU
