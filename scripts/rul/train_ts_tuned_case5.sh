#!/usr/bin/env bash
# =============================================================================
# Tuned TS Baseline Training Script — PHM Case-5  (Waveform mode, L=2560)
#
# Dataset layout:
#   data/raw_data/PHM-2012/BearingX_Y/acc_NNNNN.csv
#   Each CSV: no header, 2560 rows × 6 cols — cols 4,5 = H,V vibration (g)
#
# Split (by condition group):
#   train: Bearing1_1..1_7  (Condition 1 — 25 kN radial load)
#   val:   Bearing2_1..2_4  (Condition 2 — 12 kN)
#   test:  Bearing3_2
#
# Run order (fast → slow):
#   PatchTST    — patch_len/stride variants
#   TimeMixer   — moving_avg / dft_decomp / d_model variants
#   TimeMixerPP — moving_avg / dft_decomp variants
#   TimesNet    — 2 representative variants (slow, runs last)
#
# Usage:
#   bash scripts/rul/train_ts_tuned_case5.sh [GPU_ID]   # default GPU=0
#
# Results appended to: result_rul.txt
# =============================================================================
set -e
export TOKENIZERS_PARALLELISM=false

GPU=${1:-0}
DATASET=phm_c5
RAW_DATA=data/raw_data/PHM-2012
MODEL_ID_PREFIX=RUL_PHM_C5_TSv2

# ── Shared hyperparameters ────────────────────────────────────────────────────
BATCH_SIZE=64
NUM_WORKERS=8
TRAIN_EPOCHS=50
PATIENCE=5
D_MODEL=128
D_FF=512
E_LAYERS=3
N_HEADS=8
DROPOUT=0.1
LR=0.0001
WEIGHT_DECAY=1e-5
LRADJ=cosine
LOSS=SmoothL1
# PHM files have exactly 2560 samples — use full file as one window
TS_WINDOW_LEN=2560
ENC_IN=2
USE_NORM=1

# ── Helper ────────────────────────────────────────────────────────────────────
run_model() {
    local TAG=$1
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
# PatchTST variants
# =============================================================================
# L=2560: patch=128→19 patches, patch=32→79 patches
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

# =============================================================================
# TimesNet variants  (slowest — runs last, 2 representative configs)
#   v1: tk5_nk6 — baseline config (top_k=5, num_kernels=6)
#   v2: tk3_nk6 — fewer dominant periods, less noise from minor harmonics
# =============================================================================
echo "===== TimesNet variants (2 representative) ====="

run_model TimesNet_v1_tk5_nk6 TimesNet \
    --top_k 5 \
    --num_kernels 6

run_model TimesNet_v2_tk3_nk6 TimesNet \
    --top_k 3 \
    --num_kernels 6

echo ""
echo "============================================================"
echo "  All PHM Case-5 tuned variants completed."
echo "  Results     : ./results/ts_rul_*"
echo "  Summary     : result_rul.txt"
echo "============================================================"
