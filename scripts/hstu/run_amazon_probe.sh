#!/usr/bin/env bash
set -euo pipefail

: "${HSTU_STUDY_ROOT:?Set HSTU_STUDY_ROOT.}"

REPO="${HSTU_STUDY_ROOT}/repo"
UPSTREAM="${REPO}/third_party/generative-recommenders"
PYTHON="${REPO}/environments/hstu/.venv/bin/python"
RUN="${HSTU_STUDY_ROOT}/runs/amazon-probe"
CONFIG="${HSTU_STUDY_ROOT}/runtime-configs/amazon-probe"
MAX_WALL_SECONDS="${HSTU_MAX_WALL_SECONDS:-14400}"

mkdir -p "${RUN}" "${CONFIG}"
cd "${UPSTREAM}"
mkdir -p tmp
export PYTHONPATH="${REPO}/scripts/hstu:${UPSTREAM}"
export HSTU_REPRO_SEED=42
export PYTHONHASHSEED=42
export NCCL_DEBUG=WARN

if [[ ! -s tmp/amzn_books/sasrec_format.csv ]]; then
    timeout 7200 "${PYTHON}" -c \
        "from data.preprocessor import get_common_preprocessors; get_common_preprocessors()['amzn-books'].preprocess_rating()"
fi
"${PYTHON}" "${REPO}/scripts/hstu/audit.py" summarize-csv \
    --input tmp/amzn_books/sasrec_format.csv \
    --output "${RUN}/data-audit.json"
"${PYTHON}" "${REPO}/scripts/hstu/audit.py" select-smoke \
    --input tmp/amzn_books/sasrec_format.csv \
    --output "${RUN}/smoke-100.csv" \
    --receipt "${RUN}/smoke-100-receipt.json" \
    --users 100 --seed 42

full_data="tmp/amzn_books/sasrec_format.csv"
backup_data="tmp/amzn_books/sasrec_format.full.csv"
cp -f "${full_data}" "${backup_data}"
restore_data() {
    mv -f "${backup_data}" "${full_data}"
}
trap restore_data EXIT
cp -f "${RUN}/smoke-100.csv" "${full_data}"

for model in hstu sasrec; do
    source_config="configs/amzn-books/${model}-sampled-softmax-n512-final.gin"
    runtime_config="${CONFIG}/${model}.gin"
    grep -v "train_fn.item_feature_embedding_dim" \
        "${source_config}" >"${runtime_config}"
    printf "\ntrain_fn.num_epochs = 1\ntrain_fn.local_batch_size = 16\n" \
        >>"${runtime_config}"
    printf "train_fn.eval_batch_size = 16\ntrain_fn.eval_interval = 1000000\n" \
        >>"${runtime_config}"
    printf "train_fn.full_eval_every_n = 1\ncreate_data_loader.prefetch_factor = 2\n" \
        >>"${runtime_config}"
done

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" train.py \
    --gin_config_file="${CONFIG}/hstu.gin" --master_port=23541 \
    >"${RUN}/hstu.log" 2>&1 &
hstu_pid=$!
CUDA_VISIBLE_DEVICES=1 "${PYTHON}" train.py \
    --gin_config_file="${CONFIG}/sasrec.gin" --master_port=23542 \
    >"${RUN}/sasrec.log" 2>&1 &
sasrec_pid=$!
printf "%s\n" "${hstu_pid}" >"${RUN}/hstu.pid"
printf "%s\n" "${sasrec_pid}" >"${RUN}/sasrec.pid"

start="$(date +%s)"
is_running() {
    local state
    state="$(ps -o stat= -p "$1" 2>/dev/null | tr -d ' ' || true)"
    [[ -n "${state}" && "${state}" != Z* ]]
}
while is_running "${hstu_pid}" || is_running "${sasrec_pid}"; do
    if (( $(date +%s) - start >= MAX_WALL_SECONDS )); then
        kill -TERM "${hstu_pid}" "${sasrec_pid}" 2>/dev/null || true
        printf "timeout\n" >"${RUN}/status"
        exit 124
    fi
    sleep 10
done

set +e
wait "${hstu_pid}"
hstu_exit=$?
wait "${sasrec_pid}"
sasrec_exit=$?
set -e
printf '{"hstu":%s,"sasrec":%s}\n' "${hstu_exit}" "${sasrec_exit}" \
    >"${RUN}/exit-codes.json"
if ((hstu_exit != 0 || sasrec_exit != 0)); then
    exit 1
fi

for model in hstu sasrec; do
    if [[ "${model}" == "hstu" ]]; then
        prefix="HSTU-b4-h4-"
    else
        prefix="SASRec-d64-b4-h4-"
    fi
    exp_dir="$(
        find exps/amzn-books-l50 -mindepth 1 -maxdepth 1 -type d \
            -name "${prefix}*" | sort | tail -n1
    )"
    "${PYTHON}" "${REPO}/scripts/hstu/audit.py" parse-events \
        --input "${exp_dir}" --output "${RUN}/${model}-summary.json"
done
printf "ok\n" >"${RUN}/status"
