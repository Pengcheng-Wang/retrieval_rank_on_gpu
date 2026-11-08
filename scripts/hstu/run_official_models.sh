#!/usr/bin/env bash
set -euo pipefail

: "${HSTU_STUDY_ROOT:?Set HSTU_STUDY_ROOT to the task-owned scratch directory.}"
: "${HSTU_DURABLE_ROOT:?Set HSTU_DURABLE_ROOT to the study durable directory.}"

REPO_ROOT="${HSTU_STUDY_ROOT}/repo"
UPSTREAM="${REPO_ROOT}/third_party/generative-recommenders"
PYTHON="${REPO_ROOT}/environments/hstu/.venv/bin/python"
RUN_ROOT="${HSTU_STUDY_ROOT}/runs/ml1m-full"
CONFIG_ROOT="${HSTU_STUDY_ROOT}/runtime-configs/ml1m-full"
MAX_WALL_SECONDS="${HSTU_MAX_WALL_SECONDS:-21600}"
POLL_SECONDS="${HSTU_POLL_SECONDS:-10}"

mkdir -p "${RUN_ROOT}" "${CONFIG_ROOT}" "${HSTU_DURABLE_ROOT}/checkpoints"
cd "${UPSTREAM}"

export PYTHONPATH="${REPO_ROOT}/scripts/hstu:${UPSTREAM}"
export HSTU_REPRO_SEED="${HSTU_REPRO_SEED:-42}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

models=(hstu hstu-large sasrec)
declare -A configs=(
    [hstu]="configs/ml-1m/hstu-sampled-softmax-n128-final.gin"
    [hstu-large]="configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin"
    [sasrec]="configs/ml-1m/sasrec-sampled-softmax-n128-final.gin"
)
declare -A prefixes=(
    [hstu]="HSTU-b2-h1-"
    [hstu-large]="HSTU-b8-h2-"
    [sasrec]="SASRec-d50-b2-h1-"
)
declare -A gpus=([hstu]=0 [hstu-large]=1 [sasrec]=2)
declare -A ports=([hstu]=23341 [hstu-large]=23342 [sasrec]=23343)
declare -A pids=()
declare -A exit_codes=()

for model in "${models[@]}"; do
    runtime_config="${CONFIG_ROOT}/${model}.gin"
    cp -f "${configs[$model]}" "${runtime_config}"
    printf "\ntrain_fn.save_ckpt_every_n = 5\n" >>"${runtime_config}"
    memory_file="${RUN_ROOT}/${model}-memory.csv"
    printf "utc,memory_used_mib,utilization_gpu_percent\n" >"${memory_file}"
    CUDA_VISIBLE_DEVICES="${gpus[$model]}" \
        "${PYTHON}" train.py \
        --gin_config_file="${runtime_config}" \
        --master_port="${ports[$model]}" \
        >"${RUN_ROOT}/${model}.log" 2>&1 &
    pids["${model}"]=$!
    printf "%s\n" "${pids[$model]}" >"${RUN_ROOT}/${model}.pid"
done

publish_checkpoint() {
    local checkpoint="$1"
    local base marker lock_dir
    base="$(basename "${checkpoint}")"
    marker="${RUN_ROOT}/published-${base}"
    [[ -e "${marker}" ]] && return 0
    if ! "${PYTHON}" - "${checkpoint}" <<'PY' >/dev/null 2>&1
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
required = {"epoch", "model_state_dict", "optimizer_state_dict"}
if not required.issubset(payload):
    raise SystemExit(2)
PY
    then
        return 0
    fi
    lock_dir="${HSTU_DURABLE_ROOT}/state/publish.lockdir"
    while ! mkdir "${lock_dir}" 2>/dev/null; do
        sleep 1
    done
    cp -f "${checkpoint}" "${HSTU_DURABLE_ROOT}/checkpoints/${base}.tmp"
    mv -f \
        "${HSTU_DURABLE_ROOT}/checkpoints/${base}.tmp" \
        "${HSTU_DURABLE_ROOT}/checkpoints/${base}"
    sha256sum \
        "${HSTU_DURABLE_ROOT}/checkpoints/${base}" \
        >"${HSTU_DURABLE_ROOT}/checkpoints/${base}.sha256.tmp"
    mv -f \
        "${HSTU_DURABLE_ROOT}/checkpoints/${base}.sha256.tmp" \
        "${HSTU_DURABLE_ROOT}/checkpoints/${base}.sha256"
    rmdir "${lock_dir}"
    touch "${marker}"
}

bank_available_checkpoints() {
    local checkpoint
    while IFS= read -r checkpoint; do
        publish_checkpoint "${checkpoint}"
    done < <(
        find ckpts/ml-1m-l200 -maxdepth 1 -type f \
            \( -name "HSTU-b2-h1-*-b128-*_ep*" \
            -o -name "HSTU-b8-h2-*-b128-*_ep*" \
            -o -name "SASRec-d50-b2-h1-*-b128-*_ep*" \) \
            -print | sort
    )
}

terminate_owned_children() {
    local model pid
    for model in "${models[@]}"; do
        pid="${pids[$model]}"
        if kill -0 "${pid}" 2>/dev/null; then
            kill -TERM "${pid}" 2>/dev/null || true
        fi
    done
}

handle_signal() {
    terminate_owned_children
    bank_available_checkpoints || true
    exit 143
}
trap handle_signal INT TERM

start_seconds="$(date +%s)"
while true; do
    active=0
    for model in "${models[@]}"; do
        pid="${pids[$model]}"
        state="$(ps -o stat= -p "${pid}" 2>/dev/null | tr -d ' ' || true)"
        if [[ -n "${state}" && "${state}" != Z* ]]; then
            active=$((active + 1))
            sample="$(
                nvidia-smi -i "${gpus[$model]}" \
                    --query-gpu=memory.used,utilization.gpu \
                    --format=csv,noheader,nounits | head -n1
            )"
            printf "%s,%s\n" \
                "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)" \
                "${sample}" \
                >>"${RUN_ROOT}/${model}-memory.csv"
        fi
    done
    bank_available_checkpoints || true
    if ((active == 0)); then
        break
    fi
    if (( $(date +%s) - start_seconds >= MAX_WALL_SECONDS )); then
        printf "maximum wall time reached: %s seconds\n" "${MAX_WALL_SECONDS}" \
            >"${RUN_ROOT}/timeout.txt"
        terminate_owned_children
        break
    fi
    sleep "${POLL_SECONDS}"
done

overall_exit=0
for model in "${models[@]}"; do
    set +e
    wait "${pids[$model]}"
    exit_codes["${model}"]=$?
    set -e
    printf "%s\n" "${exit_codes[$model]}" >"${RUN_ROOT}/${model}.exit"
    if ((exit_codes[$model] != 0)); then
        overall_exit=1
        continue
    fi
    exp_dir="$(
        find exps/ml-1m-l200 -mindepth 1 -maxdepth 1 -type d \
            -name "${prefixes[$model]}*b128-*" | sort | tail -n1
    )"
    "${PYTHON}" "${REPO_ROOT}/scripts/hstu/audit.py" parse-events \
        --input "${exp_dir}" \
        --output "${RUN_ROOT}/${model}-summary.json"
    "${PYTHON}" "${REPO_ROOT}/scripts/hstu/audit.py" summarize-memory \
        --input "${RUN_ROOT}/${model}-memory.csv" \
        --output "${RUN_ROOT}/${model}-resource.json" \
        --total-memory-mib 81920
done

bank_available_checkpoints
printf '{"hstu":%s,"hstu-large":%s,"sasrec":%s}\n' \
    "${exit_codes[hstu]}" \
    "${exit_codes[hstu-large]}" \
    "${exit_codes[sasrec]}" \
    >"${RUN_ROOT}/exit-codes.json"

exit "${overall_exit}"
