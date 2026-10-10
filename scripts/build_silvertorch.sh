#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 ]]; then
  echo "Usage: build_silvertorch.sh PYTHON UPSTREAM_DIRECTORY" >&2
  exit 2
fi
PYTHON=$1
UPSTREAM=$2
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.0}
export MAX_JOBS=${MAX_JOBS:-8}

# Wheel-based CUDA stacks keep development headers outside CUDA_HOME.
INCLUDES=$("$PYTHON" -c 'import glob,os,sys; print(":".join(dict.fromkeys(p for root in sys.path for p in glob.glob(os.path.join(root,"nvidia","*","include")) if os.path.isdir(p))))')
LIBRARIES=$("$PYTHON" -c 'import glob,os,sys; print(":".join(dict.fromkeys(p for root in sys.path for p in glob.glob(os.path.join(root,"nvidia","*","lib")) if os.path.isdir(p))))')
export CPATH="${INCLUDES}${CPATH:+:$CPATH}"
export LIBRARY_PATH="${LIBRARIES}${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="${LIBRARIES}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
uv pip install --python "$PYTHON" --no-deps --no-build-isolation -e "$UPSTREAM"
"$PYTHON" -c 'import silvertorch.ops._load_ops,torch; assert torch.cuda.is_available(); assert hasattr(torch.ops.st,"fused_kmean_ann"); assert hasattr(torch.ops.st,"bloom_index_search_batch"); print("SilverTorch CUDA operators ready")'
