# LETTER reproduction: measured results

_Updated 2026-10-29. Completed components are reported separately below; the full fresh-ID comparison and paper reproduction remain unfinished._

---

## 📊 Concise completed-results matrix

Every cell below is **Recall@10 / NDCG@10**. Released-ID T5 rows are means
over seeds42,17,23; conventional references use seed42 and explicitly
reconstructed settings, not recovered LETTER baseline recipes. Every saved
prediction in all36 cells was rechecked against the original held-out
targets and complete user coverage. Exact per-seed values, sample standard
deviations and prediction hashes are in the
[verified aggregate matrix](../results/2026-10-28-letter/status-matrix-20261028-1231.json).

| Method and metric identity | Instruments | Beauty | Yelp |
| --- | ---: | ---: | ---: |
| Paper LETTER-TIGER, code reference[^1] | 0.11220 / 0.08310 | 0.06720 / 0.03640 | 0.04260 / 0.02310 |
| Released-ID T5, temperature0.7, code | 0.10905 / 0.08135 | 0.06350 / 0.03423 | 0.04419 / 0.02340 |
| Released-ID T5, temperature1.0, code | 0.10706 / 0.08053 | 0.06547 / 0.03481 | 0.04210 / 0.02271 |
| Released-ID T5, temperature0.7, item diagnostic | 0.10868 / 0.08067 | 0.06329 / 0.03406 | 0.03961 / 0.02135 |
| Released-ID T5, temperature1.0, item diagnostic | 0.10653 / 0.07985 | 0.06509 / 0.03457 | 0.03836 / 0.02101 |
| BPR-MF, item | 0.07678 / 0.04161 | 0.02898 / 0.01258 | 0.06207 / 0.03454 |
| LightGCN, item | 0.08623 / 0.04518 | 0.03466 / 0.01551 | 0.06530 / 0.03564 |
| SASRec-BCE, item | 0.08288 / 0.04029 | 0.04338 / 0.01925 | 0.03881 / 0.02223 |
| HGN-CE, item | 0.10350 / 0.07474 | 0.05903 / 0.02966 | 0.05583 / 0.03163 |
| Caser, item | 0.07073 / 0.04064 | 0.03662 / 0.01785 | 0.02810 / 0.01361 |
| BERT4Rec, item | 0.09188 / 0.06720 | 0.05259 / 0.02833 | 0.03628 / 0.01774 |

Lower temperature improves the three-seed mean on Instruments and Yelp,
but reduces it on Beauty. Three seeds do not establish statistical
significance or a universal benefit. Yelp is especially sensitive to
identity: the temperature0.7 Recall@10 mean is0.04419 by code but0.03961
under the declared original-item collision diagnostic. Several conventional
references outperform that diagnostic on Yelp under these reconstructed
settings. The paper's central claim of improving a generative backbone
still requires our fresh-ID, same-backbone comparisons. Compare item
metrics with item metrics, not with the paper's code-string matching.

All three auditable CF teachers and all three compatible-backbone semantic
embedding banks are complete and durably acknowledged. Some full-budget
tokenizer and compatible LC-Rec components have now completed, with their
measurements separated below. The full native LC corpus audit admits
Instruments and Yelp but rejects Beauty's untouched auxiliary-task corpus:
168 prompt variants exceed2048 tokens,91 have no supervised response after
masking, and841 have prompt/full token-prefix mismatches. These are
prompt-variant counts, not unique users. Empty auxiliary targets are
also retained in the failure evidence. No silent filtering or recipe
change has been used to bypass this gate. The complete aggregate
[LC corpus audit](../results/2026-10-28-letter/lc-corpus-audit-openllama-released-id.json)
preserves every task group, length distribution and failure count.

A subsequent [CPU-only Beauty admission study](letter-beauty-admission.md)
also rejected conditioning auxiliary sampling on each item's valid native
templates. Five original items have **zero valid `index2item` templates**;
all 85 freshly rendered variants for those items failed the unchanged checks.
Their maximum length was only 560 tokens, so increasing context would not
repair these particular boundary/empty-response defects. No sampler, item
filter or Beauty LC training variant was admitted.

The original LLaMA1 track remains blocked on model access. The OpenLLaMA
track is a separately named compatible-method replication; these semantic
features were not used to obtain the released-ID scores above. The remainder
of this report preserves the initial single-seed findings and their
conditional paired-user intervals.

## 📊 Frozen 32-dimensional teacher references

The already-fitted BCE SASRec teachers now have separately recorded test
references for the later Table 3 embedding-replacement comparison. These are
**32-dimensional teachers**, not the 50-dimensional SASRec benchmark above.
No new weights were fitted or reselected, and no partial tokenizer or
quantized embedding was used.

| Dataset | Recall@10 | NDCG@10 | Users | Validation-selected epoch |
| --- | ---: | ---: | ---: | ---: |
| Instruments | 0.08130 | 0.04269 | 24,772 | 80 |
| Beauty | 0.04163 | 0.01931 | 22,363 | 180 |
| Yelp | 0.03299 | 0.01609 | 30,431 | 201 |

All three fits completed their fixed 201 epochs before this evaluation. A
100-user validation-only parity check passed for each model before any test
scoring. Evaluation covered every user and original catalog item, used history
50 including the validation interaction, excluded no seen items, and resolved
score ties by ascending original item ID. Scoring was CPU float32 with two
intra-op threads and one inter-op thread. Every saved prediction was checked
against its original target, and all four R/N@5/10 metrics were independently
recomputed.

[Exact metrics and provenance](../results/2026-10-28-letter/frozen-teacher-references-cpu-v1.json)
include checkpoint and prediction hashes. The predictions, pre-test protocol,
admission evidence and exact scorer are durably banked. The original fitting
receipts remain immutable; separate evaluation receipts record this test access.
Replacement comparisons reuse these frozen references with matching teacher
identities and scoring semantics. The first completed-tokenizer results below
are partial coverage; these references alone do not establish a tokenizer benefit.

## 📊 Completed compatible-backbone components

These are **seed42, OpenLLaMA-7B-v2 compatible-backbone measurements**, not
original-LLaMA1 reproduction. LC uses the authors' released ID bank, not our
newly fitted tokenizer. BIGRec is the separately declared 1,024-prefix few-shot
reference, not an all-data baseline.

| Experiment | Dataset | Metric identity | Recall@10 | NDCG@10 |
| --- | --- | --- | ---: | ---: |
| LC, native position-set constraints | Instruments | Code | 0.11610 | 0.08972 |
| LC, full-prefix-trie correction | Instruments | Code | 0.11618 | 0.08977 |
| LC, either policy | Instruments | Item diagnostic | 0.11590 | 0.08902 |
| BIGRec, native title generation + frozen-base L2 | Yelp | Original item | 0.01666 | 0.01289 |

LC completed all four epochs / 8,240 updates and selected checkpoint6,180
solely from validation loss. Both fixed20-beam evaluations covered all24,772
users. Independent verification checked every target, beam, score, item
expansion, per-rank checksum and final-run/evaluation bank authorization.
Native constraints emitted1,837 invalid codes among495,440 beams; the full
trie emitted zero. Aggregate item metrics are identical, so format validity
should not be presented as a demonstrated item-quality gain.

BIGRec Yelp completed50 epochs, then four-beam/128-token generation and
single-best-title grounding for all30,431 users against all20,033 original
items. Every saved query's full-catalog L2 ranking was independently recomputed,
without new model generation or encoding. No users were removed:35 first-beam
answers reached the length limit. Title-equivalence Recall@10 is0.01840, a
separate diagnostic rather than a replacement for original-item Recall@10.

## 📉 First raw embedding-replacement results

Both Instruments tokenizers below completed the entire20,000-epoch budget.
The full model's collision-selected checkpoint is epoch18,000; diversity-only
selected epoch20,000. Their actual exported IDs index the selected codebooks.
Raw residual-order vector sums replace only the frozen32D teacher's real item
rows, without normalization, rescaling or refitting.

| Instruments condition | Recall@10 | NDCG@10 |
| --- | ---: | ---: |
| Original frozen teacher, reused reference | 0.08130 | 0.04269 |
| Full tokenizer's raw quantized replacement | 0.02156 | 0.01033 |
| Diversity-only tokenizer's raw quantized replacement | 0.00521 | 0.00239 |

**Both replacements perform worse than the original teacher** in this declared
reconstruction. The full tokenizer performs better than diversity-only, but
this does not reproduce a gain over the teacher or establish a generative-model
gain. Every per-user metric and the selected-codebook vector sum were independently
verified; the original teacher was not scored again.

Post-hoc median item-vector norms are1.259 for the teacher,8.708 for the full
replacement and3.614 for diversity-only. This describes a distribution difference;
it is not a causal test or permission to retune the reported experiment. The
compatible semantic encoder, reconstructed teacher and declared analysis
conventions remain limitations. Other tokenizer conditions and datasets are
not inferred from these two results.

[Verified component metrics and provenance](../results/2026-10-29-letter/verified-components-v1.json)
retain exact values, source/checkpoint/prediction hashes, coverage and the
separate diagnostic identities. Raw predictions and learned vectors remain
private and durably banked.

## 📊 Instruments released-ID replay

The first LETTER-TIGER run produces Recall@10 **0.10976** and NDCG@10
**0.08202**, compared with the paper's 0.1122 and 0.0831. This is a
**downstream replay using the authors' released semantic IDs**, not a claim
that the tokenizer training or complete paper has been reproduced.[^1]

| Result | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Paper reference, not measured here | 0.09090 | 0.11220 | 0.07630 | 0.08310 |
| Measured temperature0.7, code metric | 0.08861 | 0.10976 | 0.07524 | 0.08202 |
| Measured temperature1.0, code metric | 0.08833 | 0.10980 | 0.07486 | 0.08175 |
| Temperature0.7, original-item diagnostic | 0.08816 | 0.10932 | 0.07451 | 0.08130 |
| Temperature1.0, original-item diagnostic | 0.08792 | 0.10907 | 0.07422 | 0.08101 |

All four paper-style metrics fall within the predeclared
`max(0.002, 5% of paper value)` approximate-agreement band. That is an
engineering check, **not statistical equivalence**. Recall@10 is about 2.17%
below the published point; NDCG@10 is about 1.30% below it.

The run used seed42, temperature0.7, LR `5e-4`, the original small T5
configuration, two workers with microbatch256 and accumulation2, and
validation-loss selection. It selected epoch72 and stopped at epoch92
after patience20. Evaluation covered **all 24,772 test users exactly once**
with 20 beams, full-prefix catalog constraints and no seen-item exclusion.

The selected model and complete predictions are durably banked. An independent
pass over every prediction verified the original held-out target, finite
descending scores, 20 distinct valid codes, complete user coverage and both
sets of aggregate metrics. Exact values and artifact hashes are in the
[measurement record](../results/2026-10-28-letter/instruments-seed42-tau07.json).

## 📏 Temperature-control comparison

The matched ordinary-CE arm also selected epoch72 and stopped at epoch92.
It uses the **same full LETTER ID bank**, not semantic-only TIGER IDs.
Its Recall@10 is 0.10980 and NDCG@10 is 0.08175; the paper's corresponding
ordinary-loss ablation reports 0.1092 and 0.0819.[^2]

In this seed, lowering temperature from 1.0 to 0.7 yields **one fewer
Recall@10 hit**, alongside a small positive NDCG point difference.
None of the four marginal paired-user intervals excludes zero:

| Code metric | Difference, 0.7 minus 1.0 | Marginal 95% interval |
| --- | ---: | --- |
| Recall@5 | +0.000283 | [-0.001979, +0.002383] |
| Recall@10 | -0.000040 | [-0.002382, +0.002383] |
| NDCG@5 | +0.000378 | [-0.001109, +0.001823] |
| NDCG@10 | +0.000277 | [-0.001079, +0.001647] |

This does **not establish a temperature benefit in this seed**; it also
does not prove equivalence or absence of an effect. The analysis uses the
predeclared 2,000 paired-user percentile bootstrap replicates, seed20261027,
with complete matched users/targets and no exclusions. Outcomes are bounded
and concentrated at zero, so no normal-distribution assumption is imposed.
Intervals are unadjusted and conditional on these trained models; they
exclude training-seed uncertainty and do not separately model cross-user
dependence through shared items.

The [complete comparison](../results/2026-10-28-letter/instruments-seed42-temperature-comparison.json)
includes original-item diagnostics, win/tie/loss counts, matched-control checks
and prediction hashes. The [control record](../results/2026-10-28-letter/instruments-seed42-tau10.json)
retains its metrics and timing limitations. Seeds17/23 were already declared
before these results; their purpose is to measure training variability, not
to retune these arms using the test outcomes.

## 📊 Beauty released-ID replay

The frozen seed42, temperature0.7 run yields Recall@10 **0.06421** and
NDCG@10 **0.03463**, both below the paper's reported points.[^1]

| Result | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Paper reference, not measured here | 0.04310 | 0.06720 | 0.02860 | 0.03640 |
| Measured temperature0.7, code metric | 0.04074 | 0.06421 | 0.02708 | 0.03463 |
| Measured temperature1.0, code metric | 0.04329 | 0.06775 | 0.02837 | 0.03625 |
| Temperature0.7, original-item diagnostic | 0.04047 | 0.06394 | 0.02685 | 0.03441 |
| Temperature1.0, original-item diagnostic | 0.04275 | 0.06739 | 0.02804 | 0.03597 |

Recall@5 **misses the advisory agreement band**: the absolute gap is
0.002363, exceeding `max(0.002, 0.05 * 0.0431) = 0.002155`. The other three
metrics fall within that engineering band, which is not an equivalence
test. No threshold was relaxed and no test-guided retraining was performed.

The model selected epoch77 and stopped at97. Evaluation covered all
**22,363 users** once, including 46 targets in collision groups. It uses
the declared continuation-safe Beauty geometry, two workers with
microbatch512 and accumulation1; this is not an unchanged replay of the
original microbatch configuration. The independent audit recomputed every
code/item metric, checked the validation-minimum selection and verified the
durable model and prediction hashes. See the [complete measurement](../results/2026-10-28-letter/beauty-seed42-tau07.json).

The [ordinary-loss control](../results/2026-10-28-letter/beauty-seed42-tau10.json)
selected epoch84 and stopped at104. All four point estimates favor
temperature1.0. Under the same predeclared paired-user bootstrap:

| Code metric | Difference, 0.7 minus 1.0 | Marginal 95% interval |
| --- | ---: | --- |
| Recall@5 | -0.002549 | [-0.005054, -0.000179] |
| Recall@10 | -0.003533 | [-0.006484, -0.000804] |
| NDCG@5 | -0.001297 | [-0.002991, +0.000365] |
| NDCG@10 | -0.001619 | [-0.003278, -0.000040] |

This is a **negative fixed-temperature contrast**, not evidence of a
general temperature benefit: temperature0.7 has 79 fewer Recall@10 hits.
Three code-metric intervals exclude zero, but they are marginal,
unadjusted and conditional on this seed's two trained models. They do
not establish cross-seed or family-wise significance. Under original-item
diagnostics, only the Recall@10 interval excludes zero; the NDCG@10
interval includes it. All diagnostics are retained in the
[complete paired analysis](../results/2026-10-28-letter/beauty-seed42-temperature-comparison.json).
The result is not used to select a new recipe or to recover missing
dataset-specific author settings from test performance.

## 🔍 Identity and interpretation

There are 122 test targets in semantic-ID collision groups. The primary metric
matches generated **codes**, as the released evaluator does. The original-item
diagnostic expands each ranked code to ascending original item IDs, deduplicates
and takes the first K. It is a target-independent tie convention, not a learned
reranker and not the paper's evaluation definition.

Consequently, close agreement with the code metric does not establish
equivalent original-item ranking. The two definitions remain separate,
especially for Yelp, whose released IDs have substantially more collisions.
See the [source/data study](letter-study.md) for that audit.

The completed alternative semantic banks provide a further
[input-ambiguity check](../results/2026-10-28-letter/semantic-input-ambiguity-v1.json).
Bitwise-identical semantic vectors occur for48 Instruments items,15 Beauty
items and1,247 Yelp items; their duplicate groups contain122,32 and1,306
test targets respectively. A deterministic item-wise encoder cannot
recover missing identity from identical inputs. This does not imply that
every learned collision is unavoidable or that batch-coupled Sinkhorn
assignment is item-wise.

Title-only methods face greater Yelp ambiguity:6,127 catalog items share
titles, affecting8,508 test targets, and the largest title group contains149
items. Title-string hits must therefore remain separate from original-item
grounding; a title-to-last-ID dictionary is not valid ground truth.

The observed trainer invocation took about 55.2 minutes on two A100 80GB GPUs;
the evaluation loop took about 27.4 minutes on one GPU. These timings exclude
environment setup and the complete input audit; they are not production
latency/QPS measurements or a hardware comparison with the paper.

## 📋 Remaining coverage

The same-ID, temperature-one control is a separate ordinary-CE comparison;
it must not be relabeled the semantic-only TIGER baseline. The released-ID
matrix, its predeclared seeds and six conventional references now have
completed results. Full tokenizer reconstruction, LC-Rec, remaining
generative baselines and the paper's mechanism/ablation analyses are not
established by these cells.

Three train-only BCE SASRec teachers have been completed and durably banked
for the reconstruction phase. Original LLaMA1-7B access is still required
for the original semantic encoder and LC-Rec. Synthetic tokenizer/LoRA
correctness gates are not substituted for those real-data experiments.
A separately labeled Apache-licensed OpenLLaMA variant has passed full
semantic token audits and real7B encoding/continuation gates; it is **not**
an original-LLaMA1 reproduction or a recommendation-score result. Its
encoding and provenance contract is documented in the execution notes.

The [execution notes](letter-execution.md) describe the frozen source,
resume correction, declared geometry changes and storage contracts.
Operational progress and allocation details remain local and Git-ignored.

## 🔗 Reference

[^1]: Wang et al. "Learnable Item Tokenization for Generative Recommendation," arXiv 2405.07314v3, Table1. https://arxiv.org/html/2405.07314v3#S4.T1
[^2]: The same paper, Table2, variant(3). https://arxiv.org/html/2405.07314v3#S4.T2
