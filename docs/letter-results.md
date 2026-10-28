# LETTER reproduction: initial measured result

_2026-10-28. One completed seed42 cell; the broader reproduction is still running._

---

## 📊 Instruments released-ID replay

The first LETTER-TIGER run produces Recall@10 **0.10976** and NDCG@10
**0.08202**, compared with the paper's 0.1122 and 0.0831. This is a
**downstream replay using the authors' released semantic IDs**, not a claim
that the tokenizer training or complete paper has been reproduced.[^1]

| Result | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Paper reference, not measured here | 0.09090 | 0.11220 | 0.07630 | 0.08310 |
| Measured semantic-code metric | 0.08861 | 0.10976 | 0.07524 | 0.08202 |
| Same predictions, original-item diagnostic | 0.08816 | 0.10932 | 0.07451 | 0.08130 |

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

The observed trainer invocation took about 55.2 minutes on two A100 80GB GPUs;
the evaluation loop took about 27.4 minutes on one GPU. These timings exclude
environment setup and the complete input audit; they are not production
latency/QPS measurements or a hardware comparison with the paper.

## 📋 Remaining coverage

The same-ID, temperature-one control is a separate ordinary-CE comparison;
it must not be relabeled the semantic-only TIGER baseline. The Beauty and
Yelp pairs, seed replications, full tokenizer reconstruction, LC-Rec, and
the other paper baselines/analyses are not established by this cell.

Three train-only BCE SASRec teachers have been completed and durably banked
for the reconstruction phase. Original LLaMA1-7B access is still required
for the original semantic encoder and LC-Rec. Synthetic tokenizer/LoRA
correctness gates are not substituted for those real-data experiments.

The [execution notes](letter-execution.md) describe the frozen source,
resume correction, declared geometry changes and storage contracts.
Operational progress and allocation details remain local and Git-ignored.

## 🔗 Reference

[^1]: Wang et al. "Learnable Item Tokenization for Generative Recommendation," arXiv 2405.07314v3, Table1. https://arxiv.org/html/2405.07314v3#S4.T1
