# Controlled personalized candidate-signal results

_Completed Primary A of the bounded extension; fixed recipe and independent GPT-6-Astra review._

---

## 📌 Outcome

The learned shared user head showed a small exploratory indication of
user-history alignment, but **did not establish improvement over raw Qwen0.6B
queries and substantially underperformed matched training-positive popularity**.
Its validation expansion gate failed, so a second seed was not admitted.
Runtime was inexpensive; the stopping decision concerned information value,
not a failed memory or latency budget.

This result does not establish that semantic personalization is impossible
or identify a single cause of the original recommendation weakness. It closes
one controlled recipe on the existing retrospective, catalog-conditioned
benchmark.

## 🧠 Fixed experiment

| Contract | Executed setting |
| --- | --- |
| Item bank | All 1,798,875 original Qwen0.6B item vectors, frozen 1024D FP16 |
| User model | 658,433-parameter shared attention-pooling/residual head |
| History | Same last12 admitted training exposure vectors; no outcome-selected history or held-out appends |
| Fitting selection | 50,000 user-balanced, label-independent training timestamp blocks |
| Fitting population | 31,469 retained positive/history queries, 980 users |
| Reported attrition | 4,364 selected blocks without history; 14,167 without covered positives |
| Optimizer | Seed17, fixed three epochs, AdamW, learning rate0.001 |
| Objective | All covered query positives versus up to12 logged and64 distinct sampled weak negatives |
| Validation | All 4,136 unchanged conditional queries, 903 users |
| Final test | All 2,707 unchanged conditional queries, 819 users |
| Retrieval | Full-catalog exhaustive FP32 arithmetic over canonical stored item/head-query vectors |

All original exposure, clock-envelope, item-mapping and retrospective-caption
limitations remain in force.[^1] The new fitting loss fell from3.09537 to
2.98200 and2.93760. Falling fitting loss is not evidence of full-catalog
recommendation quality.

The original first-stage baselines and new controls use the same retained
queries. No final positive was injected into retrieval. Wrong-user input is
a whole-user frozen-history derangement, not a different label set.
Order permutation changes only the exposure-vector order.

## 📊 Matched final quality

These are **equal-user means** on the same819-user test population.
Query-weighted metrics are retained separately in the numerical artifacts.

| Candidate-producing arm | Equal-user Recall@1000 | Users with a known hit |
| --- | ---: | ---: |
| Raw-history Qwen0.6B | 0.003439 | 13 |
| Mean of the same history vectors | 0.000996 | 6 |
| Learned shared user head | 0.005751 | 23 |
| Learned head, wrong-user history | 0.002146 | 8 |
| Learned head, order-permuted history | 0.004835 | 20 |
| Training-positive popularity | 0.026773 | 97 |
| Recent-history plus popularity | 0.026163 | 97 |
| Uniform full-catalog candidates | 0.000651 | 2 |

The same-history mean is not the earlier positive-only, full-history averaging
diagnostic. Keeping the history multiset fixed separates learned aggregation
from that earlier joint intervention.

## 🧪 Paired uncertainty and stopping rule

| Learned minus comparator | Equal-user difference | Pointwise95% user-bootstrap interval |
| --- | ---: | --- |
| Raw Qwen0.6B queries | +0.002312 | [-0.000999, +0.005693] |
| Wrong-user history | +0.003605 | [+0.000976, +0.006406] |
| Order-permuted history | +0.000916 | [-0.000407, +0.002646] |
| Training-positive popularity | -0.021022 | [-0.028822, -0.013853] |

These secondary exploratory intervals use20,000 user-cluster resamples,
keeping all retained queries of each sampled user. They are not
multiplicity-corrected confirmatory evidence. The raw-query and ordering
comparisons do not establish improvement; the wrong-user contrast is a
limited alignment indication, not causal engagement lift.

The fixed seed17/epoch3 validation result had40 hit users, clearing the
proposed10-user support floor. However, its **equal-user aligned-minus-wrong
Recall@1000 difference was0.003276**, below the frozen0.005 expansion gate.
The persisted decision was on time and `go=False`. Its larger query-weighted
effect was not substituted to admit expansion.

The original pilot receipt's2,000-draw intervals and the later20,000-draw
secondary analysis remain distinct; neither changed admission. Test
metrics were non-actionable. Two-seed research qualification is therefore
**not assessed by design**, not a missing measurement.

## ⚡ Measured cost

The learned head's incremental batch16 encoding averaged1.214ms over50
measured batches after10 warmups. This includes history-ID transfer,
frozen-bank gathering, head execution, normalization and query rounding.
It excludes complete retrieval and is not a serving latency or tail SLO.

Maximum sampled device occupancy was approximately11GiB, including the
frozen item bank and exact-oracle working state. This is not the head-only
footprint or a continuous peak. Both the5ms incremental latency and24GiB
observed memory budgets passed.

## 🔎 Interpretation and provenance

The current recipe is noncompetitive with popularity, despite low measured
inference cost. User representation, feature support, sampled training
objective, frozen-history staleness and incomplete logged judgments remain
possible explanations; this experiment does not isolate their effects.

Only14.78% of all test-positive events belong to the frozen catalog.
That event coverage is reported alongside conditional user/query means,
not multiplied into them. “Novel” target strata mean outside the selected
last12 history, not historically never exposed.

Source, input, implementation, checkpoint, admission and output hashes bind
the [aggregate results](../results/2026-10-11-semantic-extension/personalization-summary.json),
[quality table](../results/2026-10-11-semantic-extension/personalization-quality.csv)
and [paired statistics](../results/2026-10-11-semantic-extension/personalization-statistics.json).
The full [extension protocol](semantic-extension-protocol.md) describes
the separate real-semantic profiling workstream, which is not completed by
this result.

[^1]: [Original real-corpus protocol](semantic-study-design.md), [original measured results](real-semantic-results.md), and KuaiRand sources: https://zenodo.org/records/10439422 ; https://zenodo.org/records/18159199
