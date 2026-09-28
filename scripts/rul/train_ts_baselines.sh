#!/usr/bin/env bash
# =============================================================================
# SOTA Time-Series Baseline Training Script — XJTU Case-1
#
# Trains PatchTST, TimesNet, TimeMixer, TimeMixerPP on raw vibration signals.
# Results written to result_rul.txt in the same format as psdi_kd baselines.
#
# Usage:
#   bash scripts/rul/train_ts_baselines.sh [GPU_ID]   # default GPU=0
#
# Checkpoints: ./checkpoints/ts_rul_*
# Results:     ./results/ts_rul_*
# Summary:     result_rul.txt
# =============================================================================
set -e
export TOKENIZERS_PARALLELISM=false

GPU=${1:-0}
DATASET=xjtu
RAW_DATA=data/raw_data/XJTU-SY
MODEL_ID_PREFIX=RUL_XJTU_Case1

# ── Shared hyperparameters ────────────────────────────────────────────────────
BATCH_SIZE=64
NUM_WORKERS=8
TRAIN_EPOCHS=50
PATIENCE=10
D_MODEL=128
D_FF=512
E_LAYERS=3
N_HEADS=8
DROPOUT=0.1
LR=0.0001
WEIGHT_DECAY=1e-5
LRADJ=cosine
LOSS=SmoothL1
TS_WINDOW_LEN=2560
ENC_IN=2
USE_NORM=1

# ── Helper: run one model ─────────────────────────────────────────────────────
run_ts_model() {
    local MODEL=$1
    shift
    local EXTRA_ARGS="$@"
    local MODEL_ID="${MODEL_ID_PREFIX}_${MODEL}"

    echo ""
    echo "============================================================"
    echo "  Training: ${MODEL}"
    echo "  Dataset:  ${DATASET}  |  GPU: ${GPU}"
    echo "============================================================"

    python -u run_ts.py \
        --task_name ts_rul \
        --is_training 1 \
        --model "${MODEL}" \
        --model_id "${MODEL_ID}" \
        --dataset_name "${DATASET}" \
        --raw_data_path "${RAW_DATA}" \
        --ts_window_len "${TS_WINDOW_LEN}" \
        --ts_window_offset center \
        --normalize_signal true \
        --use_ts_cache true \
        --enc_in "${ENC_IN}" \
        --d_model "${D_MODEL}" \
        --d_ff "${D_FF}" \
        --e_layers "${E_LAYERS}" \
        --n_heads "${N_HEADS}" \
        --dropout "${DROPOUT}" \
        --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --train_epochs "${TRAIN_EPOCHS}" \
        --patience "${PATIENCE}" \
        --learning_rate "${LR}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --lradj "${LRADJ}" \
        --loss "${LOSS}" \
        --use_norm "${USE_NORM}" \
        --use_amp \
        --gpu "${GPU}" \
        --des baseline \
        ${EXTRA_ARGS}

    echo "Done: ${MODEL}"
    echo "============================================================"
}

# ── PatchTST ──────────────────────────────────────────────────────────────────
# Channel-independent patch Transformer
# N_patches = (2560 - 64) / 32 + 1 = 79
run_ts_model PatchTST \
    --patch_len 64 \
    --stride 32

# ── TimesNet ──────────────────────────────────────────────────────────────────
# 2-D temporal variation via FFT-based period detection + Inception conv
run_ts_model TimesNet \
    --top_k 5 \
    --num_kernels 6

# ── TimeMixer ─────────────────────────────────────────────────────────────────
# Multi-scale decomposable mixing (finest scale output)
# Scales: 2560, 1280, 640, 320  (M=3, factor=2)
run_ts_model TimeMixer \
    --down_sampling_layers 3 \
    --down_sampling_window 2 \
    --decomp_method moving_avg \
    --moving_avg 25

# ── TimeMixer++ ───────────────────────────────────────────────────────────────
# Extends TimeMixer with channel mixing gate + learnable scale weights
run_ts_model TimeMixerPP \
    --down_sampling_layers 3 \
    --down_sampling_window 2 \
    --decomp_method moving_avg \
    --moving_avg 25

echo ""
echo "============================================================"
echo "  All TS baselines completed."
echo "  Checkpoints : ./checkpoints/ts_rul_*"
echo "  Results     : ./results/ts_rul_*"
echo "  Summary     : result_rul.txt"
echo "============================================================"
