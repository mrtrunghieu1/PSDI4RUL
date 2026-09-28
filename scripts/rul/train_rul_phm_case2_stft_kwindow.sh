#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false

# ============================================================================
# RUL Prediction — PHM Case 2 — STFT Spectrogram — K-Window Sweep
# ============================================================================
# Identical pipeline to train_rul_phm_case2_kwindow.sh (PSDI baseline),
# but uses STFT spectrogram images instead of PSDI images.
#
# Purpose: ablation/comparison — PSDI vs STFT vs Wavelet
#
# Data must be pre-generated with:
#   python scripts/data_processor/build_dataset_case2_spectrogram.py --mode stft
#
# Usage:
#   bash scripts/rul/train_rul_phm_case2_stft_kwindow.sh
# ============================================================================

# === Sweep grid ===
K_VALUES=(3 5 7)
L_VALUES=(1 3)
TEMPORAL_CONV_KERNEL=3

# === Shared configuration ===
model_name=psdi_kd
gpu=0
batch_size=32
num_workers=8
d_model=128

# Dataset — PHM case 2, STFT spectrogram
dataset_name=phm
root_path=data/processed_data/case2_spectrogram/stft

# Teacher model (Phase 1)
teacher_vlm_type=mae_base
mae_size=base
mae_pretrained_path=facebook/vit-mae-base
teacher_hidden_size=768
use_cls_token=true
finetune_vlm=false

# Student model (Phase 2)
student_vision_type=tiny_vit
student_hidden_size=128

# Distillation parameters  (same as PSDI baseline)
use_distillation=true
enable_adaptive_weights=true
init_temperature=4.0
distill_lr_ratio=0.05
num_alignment_scales=4
feature_alignment_dropout=0.1
loss_momentum=0.9
weight_regularization=0.001

# Loss weights
feature_w=0.1
fcst_w=1.0
recon_w=0.5
att_w=0.0

# Training  (same as PSDI baseline)
learning_rate=0.0001
teacher_pretrain_epochs=10
phase2_epochs=50
phase3_epochs=25
patience=7
dropout=0.1
weight_decay=0.01

# Phase 3 settings
phase3_qkv_mode=vision_q_phase_kv
phase3_residual_fusion=false
phase3_fusion_alpha_init=0.7
use_degradation_weighted_fcst=false
degradation_lambda=2.0
degradation_gamma=2.0
phase3_from_scratch=false

# Logging
mkdir -p logs/phm_case2_stft_kwindow
sweep_timestamp="$(date +%Y%m%d_%H%M%S)"
summary_file="logs/phm_case2_stft_kwindow/${sweep_timestamp}_SWEEP_SUMMARY.txt"

# Phase 2 checkpoint paths
phase2_setting="rul_RUL_PHM_C2_STFT_Student_Phase2_phase2_dm${d_model}_0_dst_${teacher_vlm_type}"
phase2_ckpt="./checkpoints/${phase2_setting}/checkpoint.pth"
phase2_teacher_ckpt="./checkpoints/${phase2_setting}_teacher/checkpoint.pth"

echo "============================================================"
echo "PHM Case 2 — STFT — K-Window Full Sweep"
echo "  K values: ${K_VALUES[*]}"
echo "  L values: ${L_VALUES[*]}"
echo "  Data: ${root_path}"
echo "  Summary: ${summary_file}"
echo "============================================================"

# ============================================================================
# Check if Phase 2 checkpoints exist — skip Phase 1/2 if so
# ============================================================================
if [ -f "$phase2_ckpt" ] && [ -f "$phase2_teacher_ckpt" ]; then
  echo ""
  echo "Phase 2 checkpoints found — skipping Phase 1 & 2."
  echo "  Student: ${phase2_ckpt}"
  echo "  Teacher: ${phase2_teacher_ckpt}"
  echo "============================================================"
else
  echo ""
  echo "Phase 2 checkpoints NOT found — training Phase 1 & 2 first."
  echo "============================================================"

  # ==========================================================================
  # PHASE 1: Train Teacher Model
  # ==========================================================================
  echo ""
  echo "============================================================"
  echo "PHASE 1: Training Teacher Model (MAE-Base + RUL Head) — STFT"
  echo "  Epochs: ${teacher_pretrain_epochs}"
  echo "============================================================"

  python -u run.py \
    --task_name rul \
    --is_training 1 \
    --root_path "${root_path}" \
    --dataset_name "${dataset_name}" \
    --model_id RUL_PHM_C2_STFT_Teacher_Phase1 \
    --model "${model_name}" \
    --data custom \
    --rul_phase 1 \
    --gpu "${gpu}" \
    --use_amp \
    --batch_size "${batch_size}" \
    --d_model "${d_model}" \
    --learning_rate "${learning_rate}" \
    --num_workers "${num_workers}" \
    --train_epochs "${teacher_pretrain_epochs}" \
    --patience "${patience}" \
    --dropout "${dropout}" \
    --teacher_vlm_type "${teacher_vlm_type}" \
    --mae_size "${mae_size}" \
    --mae_pretrained_path "${mae_pretrained_path}" \
    --teacher_hidden_size "${teacher_hidden_size}" \
    --use_cls_token "${use_cls_token}" \
    --finetune_vlm "${finetune_vlm}" \
    --weight_decay "${weight_decay}" \
    --des 'PHM_C2_STFT_Teacher_Phase1' 2>&1 | tee "logs/phm_case2_stft_kwindow/${sweep_timestamp}_phase1.log"

  echo ""
  echo "PHASE 1 COMPLETED"
  echo "============================================================"

  # ==========================================================================
  # PHASE 2: Train Student with Knowledge Distillation
  # ==========================================================================
  echo ""
  echo "============================================================"
  echo "PHASE 2: Training Student with Knowledge Distillation — STFT"
  echo "  Epochs: ${phase2_epochs}"
  echo "============================================================"

  python -u run.py \
    --task_name rul \
    --is_training 1 \
    --root_path "${root_path}" \
    --dataset_name "${dataset_name}" \
    --model_id RUL_PHM_C2_STFT_Student_Phase2 \
    --model "${model_name}" \
    --data custom \
    --rul_phase 2 \
    --gpu "${gpu}" \
    --use_amp \
    --batch_size "${batch_size}" \
    --d_model "${d_model}" \
    --learning_rate "${learning_rate}" \
    --num_workers "${num_workers}" \
    --train_epochs "${phase2_epochs}" \
    --patience "${patience}" \
    --dropout "${dropout}" \
    --teacher_vlm_type "${teacher_vlm_type}" \
    --mae_size "${mae_size}" \
    --mae_pretrained_path "${mae_pretrained_path}" \
    --teacher_hidden_size "${teacher_hidden_size}" \
    --use_cls_token "${use_cls_token}" \
    --finetune_vlm "${finetune_vlm}" \
    --teacher_pretrain_epochs "${teacher_pretrain_epochs}" \
    --student_vision_type "${student_vision_type}" \
    --student_hidden_size "${student_hidden_size}" \
    --use_distillation "${use_distillation}" \
    --enable_adaptive_weights "${enable_adaptive_weights}" \
    --init_temperature "${init_temperature}" \
    --distill_lr_ratio "${distill_lr_ratio}" \
    --num_alignment_scales "${num_alignment_scales}" \
    --feature_alignment_dropout "${feature_alignment_dropout}" \
    --loss_momentum "${loss_momentum}" \
    --weight_regularization "${weight_regularization}" \
    --feature_w "${feature_w}" \
    --fcst_w "${fcst_w}" \
    --recon_w "${recon_w}" \
    --att_w "${att_w}" \
    --weight_decay "${weight_decay}" \
    --des 'PHM_C2_STFT_Student_Phase2_KD' 2>&1 | tee "logs/phm_case2_stft_kwindow/${sweep_timestamp}_phase2.log"

  echo ""
  echo "PHASE 2 COMPLETED"
  echo "  Student ckpt: ${phase2_ckpt}"
  echo "  Teacher ckpt: ${phase2_teacher_ckpt}"
  echo "============================================================"

  if [ ! -f "$phase2_ckpt" ] || [ ! -f "$phase2_teacher_ckpt" ]; then
    echo "ERROR: Phase 2 checkpoints were not created. Aborting."
    exit 1
  fi
fi

# ============================================================================
# Helper function: run one Phase 3 experiment
# ============================================================================
run_phase3_experiment() {
  local k="$1"
  local l="$2"
  local tag="KW${k}_L${l}"

  local model_id="RUL_PHM_C2_STFT_P3_${tag}"
  local setting="rul_${model_id}_phase3_dm${d_model}_0_dst_${teacher_vlm_type}"
  local log="logs/phm_case2_stft_kwindow/${sweep_timestamp}_${tag}.log"

  if [ -f "results/${setting}/metrics.txt" ]; then
    echo "[SKIP] ${tag} — results already exist: results/${setting}/metrics.txt"
    return 0
  fi

  echo ""
  echo "============================================================"
  echo "PHASE 3 [STFT]: ${tag}   K=${k}, L=${l}"
  echo "============================================================"

  python -u run.py \
    --task_name rul \
    --is_training 1 \
    --root_path "${root_path}" \
    --dataset_name "${dataset_name}" \
    --model_id "${model_id}" \
    --model "${model_name}" \
    --data custom \
    --rul_phase 3 \
    --phase3_qkv_mode "${phase3_qkv_mode}" \
    --phase3_residual_fusion "${phase3_residual_fusion}" \
    --phase3_fusion_alpha_init "${phase3_fusion_alpha_init}" \
    --use_degradation_weighted_fcst "${use_degradation_weighted_fcst}" \
    --degradation_lambda "${degradation_lambda}" \
    --degradation_gamma "${degradation_gamma}" \
    --temporal_mono_w 0.0 \
    --temporal_slope_w 0.0 \
    --learn_temporal_weights false \
    --k_window_size "${k}" \
    --temporal_conv_layers "${l}" \
    --temporal_conv_kernel "${TEMPORAL_CONV_KERNEL}" \
    --gpu "${gpu}" \
    --use_amp \
    --batch_size "${batch_size}" \
    --d_model "${d_model}" \
    --learning_rate "${learning_rate}" \
    --num_workers "${num_workers}" \
    --train_epochs "${phase3_epochs}" \
    --patience "${patience}" \
    --dropout "${dropout}" \
    --teacher_vlm_type "${teacher_vlm_type}" \
    --mae_size "${mae_size}" \
    --mae_pretrained_path "${mae_pretrained_path}" \
    --teacher_hidden_size "${teacher_hidden_size}" \
    --use_cls_token "${use_cls_token}" \
    --finetune_vlm "${finetune_vlm}" \
    --teacher_pretrain_epochs "${teacher_pretrain_epochs}" \
    --student_vision_type "${student_vision_type}" \
    --student_hidden_size "${student_hidden_size}" \
    --use_distillation "${use_distillation}" \
    --enable_adaptive_weights "${enable_adaptive_weights}" \
    --init_temperature "${init_temperature}" \
    --distill_lr_ratio "${distill_lr_ratio}" \
    --num_alignment_scales "${num_alignment_scales}" \
    --feature_alignment_dropout "${feature_alignment_dropout}" \
    --loss_momentum "${loss_momentum}" \
    --weight_regularization "${weight_regularization}" \
    --feature_w "${feature_w}" \
    --fcst_w "${fcst_w}" \
    --recon_w "${recon_w}" \
    --att_w 0.01 \
    --phase2_ckpt_path "${phase2_ckpt}" \
    --phase2_teacher_ckpt_path "${phase2_teacher_ckpt}" \
    --skip_teacher_stage_in_phase3 true \
    --phase3_from_scratch "${phase3_from_scratch}" \
    --weight_decay "${weight_decay}" \
    --des "PHM_C2_STFT_${tag}" 2>&1 | tee "${log}"

  python tools/rul_eval_degradation.py \
    --result_dir "results/${setting}" \
    --test_npz "${root_path}/test.npz" \
    --dataset_name "${dataset_name}" 2>&1 | tee -a "${log}"

  echo "[DONE] ${tag}"
}

# ============================================================================
# PHASE 3 SWEEP
# ============================================================================
echo ""
echo "============================================================"
echo "Starting Phase 3 sweep [STFT]: K={${K_VALUES[*]}} x L={${L_VALUES[*]}}"
echo "============================================================"

total=$(( ${#K_VALUES[@]} * ${#L_VALUES[@]} ))
count=0

for k in "${K_VALUES[@]}"; do
  for l in "${L_VALUES[@]}"; do
    count=$((count + 1))
    echo ""
    echo ">>> Experiment ${count}/${total}: K=${k}, L=${l}"
    run_phase3_experiment "${k}" "${l}"
  done
done

# ============================================================================
# Summary table
# ============================================================================
echo ""
echo "============================================================"
echo "SWEEP COMPLETE [STFT] — Collecting results"
echo "============================================================"

{
  echo "============================================================"
  echo "PHM Case 2 — STFT — K-Window Sweep Summary"
  echo "Date: $(date)"
  echo "============================================================"
  echo ""
  printf "%-32s  %8s  %8s  %8s  %8s\n" "Setting" "MSE" "MAE" "RMSE" "Score"
  printf "%-32s  %8s  %8s  %8s  %8s\n" "--------------------------------" "--------" "--------" "--------" "--------"

  for metrics_file in results/rul_RUL_PHM_C2_STFT_P3_*/metrics.txt; do
    [ -f "$metrics_file" ] || continue
    tag=$(basename "$(dirname "$metrics_file")")
    short=$(echo "$tag" | sed 's/rul_RUL_PHM_C2_STFT_P3_//; s/_phase3_.*//')

    mse=$(grep -oP  'MSE:\s+\K[\d.]+' "$metrics_file" 2>/dev/null || echo "N/A")
    mae=$(grep -oP  'MAE:\s+\K[\d.]+' "$metrics_file" 2>/dev/null || echo "N/A")
    rmse=$(grep -oP 'RMSE:\s+\K[\d.]+'  "$metrics_file" 2>/dev/null || echo "N/A")
    score=$(grep -oP 'Score:\s+\K[\d.]+' "$metrics_file" 2>/dev/null || echo "N/A")

    printf "%-32s  %8s  %8s  %8s  %8s\n" "$short" "$mse" "$mae" "$rmse" "$score"
  done

  echo ""
  echo "============================================================"
} | tee "${summary_file}"

echo ""
echo "Summary saved to: ${summary_file}"
echo "============================================================"
