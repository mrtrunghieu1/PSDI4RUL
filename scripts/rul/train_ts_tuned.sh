#!/usr/bin/env bash
# =============================================================================
# Tuned TS Baseline Training Script — XJTU Case-1  (Waveform mode, L=2560)
#
# Chạy các variant tinh chỉnh hyperparameters cho PatchTST, TimeMixer,
# TimeMixerPP dựa trên phân tích kết quả baseline:
#
#   PatchTST baseline    MSE=0.0285, Score=1.33  (patch=64, stride=32)
#   TimeMixer baseline   MSE=0.0308, Score=1.42  (moving_avg=25)
#   TimeMixerPP          chưa có kết quả
#
# Chiến lược tinh chỉnh:
#   PatchTST  v2: patch_len=128, stride=64  (40 patches, capture 1 cycle ~200Hz)
#   PatchTST  v3: patch_len=32,  stride=16  (158 patches, finer resolution)
#   TimeMixer v2: moving_avg=101 (smooth ~4% waveform, better trend separation)
#   TimeMixer v3: moving_avg=201 + dft_decomp fallback
#   TimeMixer v4: d_model=256, d_ff=1024 (bigger capacity)
#   TimeMixerPP v1: baseline (moving_avg=25)
#   TimeMixerPP v2: moving_avg=101
#   TimeMixerPP v3: dft_decomp, top_k=5
#
# Usage:
#   bash scripts/rul/train_ts_tuned.sh [GPU_ID]   # default GPU=0
#
# Results appended to: result_rul.txt
# =============================================================================
set -e
export TOKENIZERS_PARALLELISM=false

GPU=${1:-0}
DATASET=xjtu
RAW_DATA=data/raw_data/XJTU-SY
MODEL_ID_PREFIX=RUL_XJTU_TSv2

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

# ── Helper ────────────────────────────────────────────────────────────────────
run_model() {
    local TAG=$1          # used as part of model_id
    local MODEL=$2
    shift 2
    local EXTRA_ARGS="$@"
    local MODEL_ID="${MODEL_ID_PREFIX}_${TAG}"

    echo ""
    echo "============================================================"
    echo "  Training: ${MODEL}  [${TAG}]"
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
        --des tuned \
        ${EXTRA_ARGS}

    echo "Done: ${MODEL} [${TAG}]"
    echo "============================================================"
}

# =============================================================================
# TimesNet variants
# =============================================================================
# Baseline: top_k=5, num_kernels=6
# v2: top_k=3  → fewer dominant periods, less noise from minor harmonics
# v3: top_k=7  → more periods, richer 2D representation
# v4: num_kernels=3 → lighter Inception block
# v5: d_model=256 → more capacity

echo "===== TimesNet variants ====="

run_model TimesNet_tk5_nk6 TimesNet \
    --top_k 5 \
    --num_kernels 6

run_model TimesNet_tk3_nk6 TimesNet \
    --top_k 3 \
    --num_kernels 6

run_model TimesNet_tk7_nk6 TimesNet \
    --top_k 7 \
    --num_kernels 6

run_model TimesNet_tk5_nk3 TimesNet \
    --top_k 5 \
    --num_kernels 3

run_model TimesNet_dm256 TimesNet \
    --top_k 5 \
    --num_kernels 6 \
    --d_model 256 \
    --d_ff 1024

# =============================================================================
# PatchTST variants
# =============================================================================
# Baseline: patch=64, stride=32  →  N_patches = (2560-64)/32 + 1 = 79
# v2:       patch=128, stride=64 →  N_patches = (2560-128)/64 + 1 = 39
#           Longer patch captures ~1 full vibration cycle at 32kHz/256Hz≈128samp
# v3:       patch=32, stride=16  →  N_patches = (2560-32)/16 + 1 = 158
#           Finer resolution, more patches, richer local patterns

echo "===== PatchTST variants ====="

run_model PatchTST_p128s64 PatchTST \
    --patch_len 128 \
    --stride 64

run_model PatchTST_p32s16 PatchTST \
    --patch_len 32 \
    --stride 16

# =============================================================================
# TimeMixer variants
# =============================================================================
# moving_avg=25  on L=2560 → smoothing window = 0.98% of signal (too small)
# moving_avg=101 → 3.9% → better trend separation for vibration
# moving_avg=201 → 7.8% → smoother trend
# d_model=256    → more capacity

echo "===== TimeMixer variants ====="

run_model TimeMixer_ma101 TimeMixer \
    --down_sampling_layers 3 \
    --down_sampling_window 2 \
    --decomp_method moving_avg \
    --moving_avg 101

run_model TimeMixer_ma201 TimeMixer \
    --down_sampling_layers 3 \
    --down_sampling_window 2 \
    --decomp_method moving_avg \
    --moving_avg 201

run_model TimeMixer_dft TimeMixer \
    --down_sampling_layers 3 \
    --down_sampling_window 2 \
    --decomp_method dft_decomp \
    --top_k 5

run_model TimeMixer_dm256 TimeMixer \
    --down_sampling_layers 3 \
    --down_sampling_window 2 \
    --decomp_method moving_avg \
    --moving_avg 101 \
    --d_model 256 \
    --d_ff 1024

# =============================================================================
# TimeMixerPP variants
# =============================================================================

echo "===== TimeMixerPP variants ====="

run_model TimeMixerPP_ma25 TimeMixerPP \
    --down_sampling_layers 3 \
    --down_sampling_window 2 \
    --decomp_method moving_avg \
    --moving_avg 25

run_model TimeMixerPP_ma101 TimeMixerPP \
    --down_sampling_layers 3 \
    --down_sampling_window 2 \
    --decomp_method moving_avg \
    --moving_avg 101

run_model TimeMixerPP_dft TimeMixerPP \
    --down_sampling_layers 3 \
    --down_sampling_window 2 \
    --decomp_method dft_decomp \
    --top_k 5

echo ""
echo "============================================================"
echo "  All tuned TS variants completed."
echo "  Results     : ./results/ts_rul_*"
echo "  Summary     : result_rul.txt"
echo "============================================================"
