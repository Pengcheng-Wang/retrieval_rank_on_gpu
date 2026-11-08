# GPT-6.1 Sol review of the industrial ranker extension

## Review identity and boundary

- Reviewer model: **GPT-6.1 Sol**
- Reasoning: **MAX**
- Context tier: **long context**
- Reviewed range: `6cf542c49517d0fcb33b30483724d4433c1f0553..29bd7c2d1ad10c9b47399e0a389e560bb015b5a5`
- Review scope: implementation, protocol, numerical aggregates, source
  provenance, scientific claims, and test coverage.

The exact requested GPT-6-Astra/MAX capability remained unavailable. This
review is an explicitly authorized substitute and is not labeled Astra. No
numeric one-million-token context claim is made.

## Findings and resolutions

### Training-only normalization

**Finding:** recency normalization scanned training, validation, and test
timestamps. A synthetic perturbation demonstrated that held-out timestamps
could change training features.

**Resolution:** `training_maximum_gap` now reads only training arrays. The data
receipt persists both the training normalizer and an all-split diagnostic. A
regression test changes held-out timestamps and verifies the training
normalizer is invariant. The measured MovieLens training maximum is
`67,337,688` seconds. The complete model/seed matrix was rerun.

### DIN labeling

**Finding:** the implementation softmax-normalizes attention, uses PReLU
instead of Dice, and omits mini-batch-aware regularization. Calling it simply
DIN could attribute a negative result to canonical DIN.

**Resolution:** all reports, protocol fields, and aggregate labels now call it
the **DIN-style normalized-attention adaptation** and list those deviations.
The unlicensed author repository remains research-only; no code was copied.

### Aggregate identity and protocol guards

**Finding:** set equality could accept duplicate run cells, and aggregation did
not bind every run to the exact data, candidates, hard negatives, protocol,
query populations, checkpoint, and log identity.

**Resolution:** final aggregation now requires exactly one record and one log
for every expected model/seed cell. It rejects diagnostic runs, duplicate
cells, population mismatches, and any input/protocol hash mismatch. Each
checkpoint and result receipt is hash-verified before publication. Regression
tests cover duplicate cells and mismatched input hashes.

### Observation-level provenance

**Finding:** aggregate metrics were internally consistent but could not be
independently reconstructed from committed per-query observations.

**Resolution:** the result package now includes:

- `query-ranks.npz`: validation/test query indices, target presence, and
  selected target rank for every model/seed;
- `run-provenance.json`: sanitized input/result/checkpoint hashes, package
  versions, model identities, selected epochs, residual weights, and
  training-normalization receipts;
- `rejected-protocols.json`: stable identifiers and reasons for every
  superseded experimental protocol.

Aggregation recomputes HR/NDCG from those observations and refuses publication
if any reported metric differs.

### Additional review corrections

- Diagnostic smoke validation/test users are now disjoint and explicitly
  marked diagnostic; final aggregation rejects them.
- Efficiency benchmarking reconstructs the selected checkpoint after releasing
  the training optimizer and gradients. Memory scope is stored with each
  measurement.
- The OneRec citation now points to arXiv `2502.18965`; the later OneRec
  Technical Report and Apache-2.0 OpenOneRec release are distinguished.
- HSTU chronology documentation now describes the retained 200-event history
  window instead of implying that every pre-last event is always observed.

## Independent numerical checks

The reviewer independently reproduced:

- all committed aggregate SHA-256 values;
- HSTU-large coverage counts at 50/100/200/500;
- unreachable-target rate;
- selected HR/NDCG and conditional/unconditional identities;
- all learning-gate means;
- 1,200 timing-sample summaries and throughput calculations;
- complete-run wall-time calculations;
- parameter counts;
- exact upstream HSTU and TransAct source identity.

The final review-corrected rerun still selected epoch 0 for every model and
seed. Thus the null-adoption conclusion is unchanged: none of these learned
residual configurations earned an additional ranking stage on this public
protocol.
