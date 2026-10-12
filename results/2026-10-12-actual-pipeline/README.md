# Complete actual-query pipeline evidence

This aggregate-only package completes the fixed 72 pipeline attempts,
108 boundary-condition outcomes, four separate CUDA profiles and six
actual-query geometry references over 8,841,823 real passages. It preserves
84 successful timing conditions, 24 unsupported outcomes, 2,688 unprofiled
batch durations and all 16 secondary scorer/cap contrasts.

The report is `docs/actual-pipeline-results.md`. Scientific and systems
interpretation were independently reviewed before publication. Original
partial-study artifacts remain unchanged.

Unsupported measurements are blank rather than zero. Both FP32/MLP primary
non-inferiority assessments remain not assessed. Secondary intervals are
pointwise, post-test exploratory and panel-specific; runtime views preserve
their sample counts, warmup differences and implementation scope.

Profile event counts and inclusive durations are not wall-time fractions,
MFU, energy or serving throughput. Memory observations are sampled layouts,
with the administrative GPU allocation recorded separately.

`checksums.json` seals all package files and the reviewed source-analysis
hashes. The secondary JSON is byte-identical to its reviewed producer output;
its `review_required` field records the producer's creation-time gate, not a
claim that publication proceeded without the subsequent independent review.

No raw query, passage or user identifiers, model checkpoints, credentials,
allocation-access configuration or private diagnostic paths are included.
