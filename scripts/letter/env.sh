#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$PROJECT/environments/letter/.venv"
SITE="$VENV/lib/python3.10/site-packages"
STATE="$(dirname "$PROJECT")"
: "${CUDA_VISIBLE_DEVICES:?Set the explicitly allocated GPU IDs}"
mkdir -p "$STATE/runtime-home" "$STATE/hf-cache"

# Old bitsandbytes inspects arbitrary environment values as filesystem paths.
# A private runtime environment also prevents accidental external experiment logging.
exec env -i \
  HOME="$STATE/runtime-home" \
  PATH="$VENV/bin:/usr/local/bin:/usr/bin:/bin" \
  LD_LIBRARY_PATH="$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cusparse/lib:$SITE/torch/lib" \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  PYTHONPATH="$PROJECT/src" \
  PYTHONHASHSEED="${LETTER_SEED:-42}" \
  PYTHONUNBUFFERED=1 \
  HF_HOME="$STATE/hf-cache" \
  WANDB_MODE=disabled WANDB_DISABLED=true \
  TOKENIZERS_PARALLELISM=false \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  "$@"
