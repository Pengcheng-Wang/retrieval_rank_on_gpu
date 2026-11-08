# HSTU isolated environment

_Pinned Linux CUDA environment for the paper-era public HSTU reproduction_

---

## 📋 Scope

This environment runs the unchanged source in
`third_party/generative-recommenders` at
`9afe62cccfbb1e893780313b89a2d43185c5f66d`. It is separate from the
SilverTorch GPU environment and from the later reviewed HSTU kernel source.

The lock resolves:

| Component | Version |
| --- | --- |
| Python | 3.12 |
| PyTorch | `2.13.0+cu129` |
| Triton | `3.7.1` |
| FBGEMM-GPU | `1.8.0+cu126` |
| NumPy | `2.2.6` |
| pandas | `3.0.5` |
| gin-config | `0.5.0` |

The mixed CUDA-minor wheel set is intentional and was accepted only after
real FBGEMM jagged operations, CPU forward/backward, and A100 GPU
forward/backward/evaluation completed. It must not be installed over a shared
base environment.

## 🔧 Create the environment

```bash
export UV_CACHE_DIR="$PWD/.local/hstu-uv-cache"
uv sync --project environments/hstu
```

The resolved environment is `environments/hstu/.venv`. On an ephemeral GPU
node, place both the environment and cache under task-owned node-local scratch.

## 🧪 Verify source and adapters

```bash
python scripts/verify_hstu_upstream.py
python -m pytest tests/test_hstu_adapter.py -q
```

From the vendored upstream directory, a CPU-applicable model smoke is:

```bash
export PYTHONPATH="$PWD/scripts/hstu:$PWD/third_party/generative-recommenders"
cd third_party/generative-recommenders
../../environments/hstu/.venv/bin/python ../../scripts/hstu/cpu_smoke.py \
  --models sasrec hstu hstu-large \
  --output ../../runs/hstu-cpu-smoke.json
```

## ⚠️ Upstream dependency deviation

The first public README lists broad unpinned packages including Apex. Apex is
not imported by the released MovieLens execution path and has no released
version/build contract in that snapshot, so this environment does not install
an arbitrary ABI-sensitive Apex build. Every package actually used by the
reproduction is resolved in `uv.lock`.
