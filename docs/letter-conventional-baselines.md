# LETTER conventional references

_Declared native-class reconstruction; small gates verified 2026-10-28. This is not a claim that the paper's baseline scores or missing recipes have been reproduced._

---

## 📋 Frozen recipe

Use [`letter-conventional-baselines-v1.json`](../configs/letter-conventional-baselines-v1.json).
It is **not recovered LETTER baseline configuration**. The unchanged native classes are
RecBole 1.2.0 (MIT),[^recbole] original SASRec (Apache-2.0),[^sasrec] and original Caser
(LGPL-3.0).[^caser] Exact revisions, source checksums, and declared adapter decisions are in the recipe.

| CLI model | Native reference | History / objective | Batch |
|---|---|---|---:|
| `bpr` | RecBole BPR, 64D | Every training event; BPR-MF, not MSE-MF[^letter] | 1024 |
| `lightgcn` | RecBole LightGCN, 64D, 2 layers | Binary train-only graph; native BPR + `1e-5` unsquared-norm regularizer | 1024 |
| `hgn` | RecBole HGN, 64D | 20 items; average pooling, CE, zero regularization | 256 |
| `bert4rec` | RecBole BERT4Rec, 64D, 2 layers/heads | 20 context items + target; masked CE | 256 |
| `sasrec` | Author-named pmixer SASRec, 50D | 50 items; native user sampler, nonpadding positive + negative BCE | 128 |
| `caser` | Original Caser, 50D | Native newest-first `L=5,T=3` windows; 3 total negatives; separate BCE means | 512 |

Adam `1e-3`; at most 200 epochs (SASRec 201), validation every 5 epochs and final,
earliest best validation NDCG@10, patience 10 validation checks. SASRec uses native
Xavier initialization and Adam `(0.9,0.98)`; it is **not the 32D teacher**. Caser retains
native initialization, including its nonzero padding embedding, and Adam weight decay `1e-6`.

HGN deliberately uses CE rather than the native BPR branch's differing embedding tables.
BERT masking is an explicit reconstruction: complete train-only prefixes, up to 21 real
items, right padding, no supervised mask at position zero; terminal masking probability
`0.5`, otherwise Bernoulli `0.2`, retaining at most four masks and forcing a terminal mask
when none was drawn. Training positions match native inference; inference supplies
`[0] + last20_history + right_padding`, length `len(history)+1`, before native first-column
removal. Caser scores batched candidates with `for_pred=False` and uses the actual last
five observed items, not the history portion of its final training window.

## 🔒 Protocol and receipts

`read_release` verifies input hashes; `teacher_sequences` preserves numeric original-user
ordering, shifts item IDs once, and splits `s[:-2] / s[-2] / s[-1]`. No new filtering,
timestamps, or remapping. Negatives and graph use training only. Test sequential context
appends validation without refitting. Ranking covers every user and real item, excludes
only padding, retains repeat targets, and breaks score ties by ascending original item ID.
Private predictions and R/N@5/10 use original IDs, never semantic codes.

```mermaid
flowchart LR
    accTitle: Frozen baseline evaluation protocol
    accDescr: Training uses validation to select a checkpoint before a separate single-use test evaluation.
    train["Train and checkpoint"] --> select["Select using validation"]
    select --> freeze["Freeze train-completed.json"]
    freeze --> test["Claim test once"]
    test --> receipt["Atomic completed.json"]
```

Epoch checkpoints contain model, Adam, sampler, Python/NumPy/Torch/CUDA RNG states and
selection history. SIGTERM drains the current epoch; `--stop-after-epoch 1` is a deterministic
gate. Resume only with identical code, source, recipe, runtime, data and user map. The two
latest checkpoints and their selected best are retained; incomplete unmarked writes are
ignored, but a malformed completed checkpoint is an error. A stale `train.lock` requires
operator inspection of its recorded PID; never automatically remove another process's lock.
`test-claimed.json` is irreversible, including after a failed test attempt: do not delete it
to rerun test. Only checksum-verified private checkpoints produced here should be loaded.

Hashes bind **exact deployed bytes**, including line endings. The initial Windows-origin
baseline script and recipe were deployed with CRLF; Git stores their LF-normalized form.
Normalization was verified to be the only content difference. Preserve the private deployed-byte
snapshot for continuation; do not overwrite an active deployment with a semantically equivalent
but differently hashed checkout.

## 🚀 Launch after coordinator approval

These are instructions, **not launches performed by the adapter task**. Full fits require
an explicitly assigned GPU on the existing authorized allocation. `ROOT` is that task's
existing root; send remote Bash through `.local\letter\aml.ps1`. Do not source `env.sh`,
change the baseline venv, or access another allocation.

```bash
set -euo pipefail
: "${ROOT:?existing authorized task root}" "${AUTHORIZED_GPU:?coordinator-assigned GPU}"
cd "$ROOT"
umask 077
mkdir -p baseline-private/home baseline-private/hf
export HOME="$ROOT/baseline-private/home" HF_HOME="$ROOT/baseline-private/hf"
export CUDA_VISIBLE_DEVICES="$AUTHORIZED_GPU" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export WANDB_DISABLED=true WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/project/src" CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$ROOT/project"
PY="$ROOT/project/environments/letter-baselines/.venv/bin/python"
COMMON=(--model bpr --dataset Instruments --seed 42 --device cuda:0
  --upstream "$ROOT/upstream" --manifest configs/letter-reproduction-v1.json
  --recipe configs/letter-conventional-baselines-v1.json
  --sasrec "$ROOT/sasrec"
  --caser "$ROOT/baseline-sources/caser_pytorch-c4e10e4c12f0fce8a0cba12008b3aff5d07a26d1"
  --output "$ROOT/runs/conventional/bpr-instruments-seed42")
"$PY" scripts/letter/baselines.py train "${COMMON[@]}"
# For an incomplete attempt only: same command plus --resume.
# Separately, after train-completed.json exists and selection is frozen:
"$PY" scripts/letter/baselines.py evaluate "${COMMON[@]}"
```

Change model/dataset/output together for another approved fit; never reuse an output
directory for another configuration. Bank private artifacts through the coordinator's
serialized storage writer; this adapter performs no Gen2 writes.

## 🧪 Validation

Local regression command: `python -m pytest -q tests\test_letter_baselines.py tests\test_letter_baselines_gpu.py tests\test_letter_cf.py tests\test_letter_protocol.py` — **76 passed, 7 skipped** (external Caser and opt-in GPU gates).

Pinned-runtime CPU command: `CUDA_VISIBLE_DEVICES='' "$PY" -m pytest -q tests/test_letter_baselines.py` — **49 passed**, with `LETTER_CASER` set to the pinned private source.

After verifying GPU1 has no processes/memory, set `CUDA_VISIBLE_DEVICES=1`,
`LETTER_BASELINE_GPU_GATE=1`, `LETTER_BASELINE_GATE_ROOT` to a new private task-root
directory, and `LETTER_SASREC`/`LETTER_CASER` to the source paths above. Run
`"$PY" -m pytest -q -s tests/test_letter_baselines_gpu.py` — **6 passed** in 124.76 s.
The fixture has 84 training events, 8 users and 32 items. Each native model passes
forward/loss/gradient/full-catalog checks, bitwise cold-process interrupted continuation,
checkpoint reload and single-use test evaluation. HGN also drains a real SIGTERM during
its first batch. Two warnings come from pinned Torch/TensorBoard's deprecated
`distutils` checks. GPU1 returned to zero usage; no full baseline fit was started.

## 📦 Checkpoint snapshots and banking

`bank_baselines.py` is a separate **CPU-only, node-0-only** helper; training code stays
frozen. A local immutable snapshot contains the latest complete epoch, its earlier
selected checkpoint and validation predictions, and exact config/user mapping files.
Restore its `payload/`, not the snapshot root: the root `completed.json` seals the
snapshot, not the training run. Only an already-frozen, matching `train-completed.json`
is included. After evaluation, the entire immutable run is captured, including predictions.

On node0, using the existing baseline Python and the coordinator's supplied paths:

```bash
RUN="$ROOT/runs/conventional/bpr-instruments-seed42"
CUDA_VISIBLE_DEVICES='' "$PY" scripts/letter/bank_baselines.py snapshot \
  --run "$RUN" --staging "$ROOT/baseline-snapshots"

# Shared-storage watcher: launch only when authorized; never invent a per-run lock.
: "${BANK_DEST:?coordinator banking destination}" "${BANK_LOCK:?existing COMMON bank.lock}"
CUDA_VISIBLE_DEVICES='' "$PY" scripts/letter/bank_baselines.py watch \
  --writer-host node-0 --runs "$RUN" --staging "$ROOT/baseline-snapshots" \
  --destination "$BANK_DEST" --lock-file "$BANK_LOCK" \
  --ack-root "$ROOT/baseline-acks" --interval 120
```

`--runs` accepts multiple approved directories with distinct basenames. Missing future
runs remain pending. Only expected live-checkpoint rotation `FileNotFoundError`s skip a
cycle; checksum/config errors are explicitly logged and never acknowledged. Owned partial
staging directories are quarantined; completed snapshots are verified, never overwritten.
Source files are copied locally before taking the common lock and calling `bank.copy_artifact`.
Unchanged, already-banked identities do not transfer again.

Before single-use test, wait for `<run-name>.model.json` in the acknowledgment directory
and match its `train_completed_sha256` against the source frozen receipt. The final
`<run-name>.evaluation.json` also binds `completed_sha256`. Acknowledgments include the
verified destination and banking-receipt hash; the watcher exits successfully only after
every supplied evaluation and its full archive are verified.

Verification: `python -m pytest -q tests\test_letter_baseline_bank.py` — **30 passed**
locally and in the pinned node0 runtime. A CPU-only integration against the completed
native-v4 SASRec gate restored epoch 2 with earlier best epoch 1, verified full-run model/test
acknowledgments, and confirmed unchanged banking does not rewrite receipts. Its destination
was a private node-local test directory: **no shared-storage writes or GPU work**.

[^recbole]: https://github.com/RUCAIBox/RecBole/tree/362d31f00af801d7d99bc635c902d1df1405e79d
[^sasrec]: https://github.com/pmixer/SASRec.pytorch/tree/29b9922adfc0dd1710a5f8fad8dddace2639fc76
[^caser]: https://github.com/graytowne/caser_pytorch/tree/c4e10e4c12f0fce8a0cba12008b3aff5d07a26d1
[^letter]: LETTER's MF baseline cites BPR: https://arxiv.org/html/2405.07314v3#bib.bib35
