# LETTER execution harness

_Execution-stage implementation, 2026-10-28. Training results are not yet published._

---

## 📋 Current scope

The [study](letter-study.md) and
[reproduction protocol](superpowers/plans/2026-10-27-letter-reproduction.md)
remain the scientific reference. The implementation uses **unchanged,
SHA-256-verified author model, data, loss and collator modules** from the
pinned external source. Original orchestration adds complete input audits,
explicit runtime/configuration receipts, checkpoint continuation and
all-user evaluation; it does not vendor the unlicensed LETTER release.

The implemented paths are released-ID **TIGER replay**, a separate
**training-only BCE SASRec teacher reconstruction**, and the tokenizer's
audited training/resume/export lifecycle. Real-data tokenizer retraining,
LC-Rec and fresh semantic encoding are not represented as completed.
LLaMA1-7B-dependent stages require an authorized original
checkpoint; a publicly downloadable mirror does not establish access rights.
The mirror's card explicitly requires prior access from Meta.[^1]

## ⚙️ Isolated runtime

`environments/letter/pyproject.toml` and `uv.lock` describe the verified
Linux/Python 3.10 environment. Core versions are PyTorch `1.13.1+cu117`,
Transformers `4.31.0`, Accelerate `0.21.0`, PEFT `0.4.0` and bitsandbytes
`0.41.1`. CUDA 11 runtime/cuSPARSE wheels supply the legacy extension's
missing libraries without modifying a host image's CUDA installation.
Setuptools is pinned because the historical W&B import needs
`pkg_resources`.

The task-scoped `scripts/letter/env.sh` launches a clean process environment
with explicit GPU visibility and library paths. This prevents old
bitsandbytes from treating arbitrary host environment values as accessible
filesystem paths, and disables external experiment logging. No credentials,
allocation identities or user-specific paths are embedded in the launcher.

```bash
uv sync --project environments/letter --python 3.10 --locked
CUDA_VISIBLE_DEVICES=0 bash scripts/letter/env.sh \
  python scripts/letter/runtime_patch.py
```

Run this only in the experiment's isolated checkout. The compatibility
patcher refuses to modify a non-virtual-environment installation or unknown
Trainer source.

For independent jobs on one node, use
`--rdzv_backend=c10d --rdzv_endpoint=localhost:0` with a unique
`--rdzv_id` for each launch. In the pinned Torch version, `--standalone`
hardcodes `localhost:29400`; concurrent launchers can share that server,
creating an avoidable lifecycle dependency. This is separate from the
remote SSH transport.

## 🔄 Resume correction and admission boundary

A two-worker interruption test exposed two independent legacy behaviors:
floor-based reconstruction of epochs conflicts with accumulated microbatches
carried across nonintegral epochs, and replaying just one batch per completed
epoch fails to restore the distributed sampler's generator progression.[^2]

The implementation makes one **resume-only** dependency correction: fully
exhaust completed epochs during sampler replay rather than stopping after
their first batch. Fresh, uninterrupted training is unchanged. The original
Trainer source hash is
`24a579e7af77c3699020ac16c1b4ad16d64aa584dafe52a90d0807c381174f31`;
the patched hash is
`8590a1189d37370eda766efd19e5995ab885d787f8336d8727544ba42e502dd2`.
Each run binds the patched runtime identity.

The runner also requires the **actual prepared dataloader length** to be
divisible by gradient accumulation. It rejects unsafe geometry instead of
silently dropping data, changing scheduler math or labeling a warm start an
exact resume. Mid-epoch drains suppress artificial epoch-end validation.
Early-stopping patience is saved with each sealed checkpoint and restored.

| Released geometry | Microbatches per worker/epoch | Accumulation | Admission |
| --- | ---: | ---: | --- |
| Instruments, 2 workers x 256 | 258 | 2 | Integral |
| Beauty, 2 workers x 256 | 257 | 2 | Rejected |
| Yelp, 2 workers x 256 | 440 | 2 | Integral |
| Beauty, 4 workers x 256 | 129 | 1 | Not admitted under the bitwise continuation gate |
| Beauty, 2 workers x 512 | 129 | 1 | Admitted after full-data trace/state checks |

Beauty uses two workers with microbatch 512 and accumulation one. This
preserves **nominal effective batch 1,024** and worker count, not the old
microbatch trajectory. Its epoch padding changes from 171 to 683 repeated
examples and its 200-epoch update budget from 25,600 to 25,800. It is a new,
labeled configuration, never continuation of a 256-microbatch checkpoint.

The four-worker candidate had up to `2.384185791015625e-7` final-weight
differences after continuation. A fresh uninterrupted repeat was bitwise
identical, and RNG states agreed across the interrupted comparison. Reduction
ordering is a possible explanation, not an established root cause. Rather
than relax the gate after observing that discrepancy, the two-worker
alternative was checked on the full Beauty fitting data at interruptions
64, 129 and 130: ordered actual input/label tensors, model/optimizer/scheduler,
both ranks' RNG, validation sequence and callback state matched exactly.

The six declared released-ID cells are frozen in
[`letter-released-id-execution-v1.json`](../configs/letter-released-id-execution-v1.json).
Temperatures 0.7 and 1.0 are compared for each dataset at LR `5e-4`; Beauty
and Yelp use these prespecified transfer settings, not falsely claimed
recovered best author hyperparameters. The temperature-one arm is **not**
the semantic-only TIGER baseline because it uses the same full LETTER IDs.

GPU regression checks include interruption at the first-epoch middle, an
epoch boundary and a later-epoch middle. Model/optimizer/scheduler and both
ranks' RNG state match bitwise after continuation. Production-shaped data
and callback checks are separate gates, not implied by the tiny fixtures.
The Instruments two-worker check also passed on all 131,837 fitting records:
interrupting after step 130 preserved the next updates and both scheduled
validation evaluations in a two-epoch test.
Resume replay may become CPU-expensive late in training; that cost is
preferable to silently changing example order.

## 🧪 Input and result contracts

| Tool | Responsibility |
| --- | --- |
| `prepare.py` | Released-file checksums, complete split parity against author datasets, every actual T5-tokenized input/target, atomic SID tokens and EOS retention |
| `run.py` | Native T5 recipe, finite positive temperature, bound inputs/source/runtime, validation-loss selection, complete resume and failure receipts |
| `evaluate.py` | Saved tokenizer/model identity, fixed 20 beams, full-prefix trie, every user once, per-user codes/scores and separate original-item diagnostics |
| `cf.py` | Author-named BCE SASRec model, training-only sequences, explicit padding-row mapping, validation-only checkpoint selection and complete teacher state |
| `bank.py` / `bank_watch.py` | Independent local checkpoint snapshots, checksum verification, immutable durable artifacts and one designated writer/lock |
| `bank_remote.py` | Worker-local snapshots, explicit node-to-node copies, writer-only Gen2 exports and checksum-bound worker acknowledgments |
| `tokenizer_run.py` | Verified author RQ-VAE/trainer, minimal parameter plumbing, initialization-aware resume and native collision-resolution export |
| `trace_train.py` | Gate-only ordered input/label tensor hashes; not enabled for ordinary benchmark fits |

All three released T5 datasets have passed complete sequence/token audits.
This certifies **that path**, not LLaMA prompt tokenization. The primary metric
is still an exact semantic-code match; collisions are not silently resolved
by pretending each code uniquely identifies one catalog item. Original-item
diagnostics use the predeclared, target-independent ascending-ID tie rule.

CF training uses the historical author-named SASRec model unchanged. A
synchronous sampler reproduces one upstream worker's draw sequence and
exposes its RNG state, avoiding nondeterministic arrival from three
multiprocessing workers. This is a declared orchestration difference, not
recovery of the authors' missing teacher run. The recipe uses 32 dimensions,
history 50 and the other pinned defaults in the plan. It never uses the
upstream loop's periodic test feedback or sampled-negative evaluator to
select a teacher.

CPU contracts:

```powershell
python -m pytest tests\test_letter_protocol.py tests\test_letter_cf.py -q
```

GPU contracts additionally require the external verified snapshot:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/letter/env.sh \
  env LETTER_UPSTREAM="$LETTER_UPSTREAM" LETTER_SASREC="$LETTER_SASREC" \
  python -m pytest tests/test_letter_protocol.py tests/test_letter_cf.py \
  tests/test_letter_models.py -q
```

`LETTER_UPSTREAM` and `LETTER_SASREC` are caller-provided source directories;
no private allocation is selected by these commands.

The synthetic-weight LC-Rec API test must run in a **separate process** because
the authors' two backends reuse generic Python module names. It checks int8
LoRA, active SID embedding/head gradients, temperature-scaled loss, frozen
original modules and save/reload without downloading any LLaMA weights:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/letter/env.sh \
  env LETTER_UPSTREAM="$LETTER_UPSTREAM" \
  python -m pytest tests/test_letter_lora.py -q
```

This passed on the isolated GPU runtime, but it does not certify the full
LLaMA1-7B memory footprint, prompt corpus, ZeRO-2 execution or recommendation
quality.

## 🔬 Tokenizer lifecycle gates

The adapter verifies each author source file before making a private copy.
Only codebook-size initialization, dynamic diversity-label enumeration and
fail-fast handling of nonfinite Sinkhorn outputs are patched. The original
RQ-VAE losses and straight-through gradient routing remain unchanged.
Default initialization, losses, indices and parameter gradients matched the
unmodified source on GPU; CF gradients reach the encoder indirectly and do
not directly differentiate through codebook assignments.

The original training/evaluation epoch methods are reused. The adapter
records the source method's returned quantization loss as **last-batch** loss,
not an invented epoch average. Full epoch checkpoints include initialization
flags, optimizer and Python/NumPy/CPU/CUDA RNG states. Continuation skips
initialization, so trained centroids are not overwritten. Source DataLoader
workers are set to zero as an explicit operational choice.

On a 512-row **synthetic fixture**, interrupted and uninterrupted three-epoch
runs produced bitwise-equal state and identical exported IDs after the
authors' collision postprocessing. An additional actual eight-level run
with 128 codes/level and 20 diversity clusters emitted eight-token IDs,
confirming that requested structure reaches the model and exporter rather
than being silently truncated by `zip` or a six-prefix list.

Those are correctness gates, **not paper measurements**. Real execution
requires a complete original-encoder embedding receipt with model/tokenizer
hashes, full token-audit status, metadata hash, row order and embedding hash.
CF inputs must match either the exact released Instruments tensor or a
completed, dataset-bound teacher reconstruction. No alternative encoder
or synthetic matrix may silently enter a paper run.

## 💾 Operations and persistence

Allocation binding, SSH transport, process identities, node/GPU assignments,
storage destinations and `progress.md` are **local and Git-ignored**.
The user-requested hourly session automation checks progress and advances
admitted work; it does not authorize another allocation or cancelling the
current one.

Training/checkpoints are node-local. The designated exporter copies sealed
checkpoints to independent local snapshots before a serialized shared-storage
write. Complete SHA-256 receipts, not directory existence, establish durable
completion. Other nodes send their artifacts to the writer rather than
writing directly to the shared folder. That folder has an ownership and
single-writer README.

An unfinished run cannot be exported as a completed run. During training,
export only its sealed checkpoint snapshots; after a drain, resume from
those snapshots before producing any final-model or evaluation completion
receipt.

Source/model authorization, mathematical/evaluation validity, and operational
readiness are different gates. A passing CUDA smoke or completed teacher is
not a reproduced LETTER recommendation score. Negative outcomes, attempted
resume failures, protocol amendments and model-access blockers remain in the
private execution record rather than disappearing from the final analysis.

## 🔗 References

[^1]: HuggyLLaMA LLaMA1-7B model card and license at the inspected revision. https://huggingface.co/huggyllama/llama-7b/blob/4782ad278652c7c71b72204d462d6d01eaaf7549/README.md
[^2]: Transformers 4.31.0 Trainer resume arithmetic and epoch replay. https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/trainer.py#L1703-L1758 ; Accelerate 0.21.0 sampler preparation: https://github.com/huggingface/accelerate/blob/v0.21.0/src/accelerate/data_loader.py#L737-L746 ; PyTorch 1.13.1 RandomSampler: https://github.com/pytorch/pytorch/blob/v1.13.1/torch/utils/data/sampler.py#L117-L133
