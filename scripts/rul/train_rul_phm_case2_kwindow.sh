#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false

# ============================================================================
# RUL Prediction — PHM (PRONOSTIA) Case 2 — Full Sweep K-Window
# ============================================================================
# Phase 1: Train teacher model (MAE-Base + RUL head)       [run once]
# Phase 2: Train student model with knowledge distillation [run once]
# Phase 3: K-Window temporal aggregation                   [sweep K x L]
#
# FPT/EOL are loaded automatically from utils/rul.py via dataset_name=phm
# ============================================================================
#
# Usage:
#   bash scripts/rul/train_rul_phm_case2_kwindow.sh
#
# Sweep grid (edit below to customize):
#   K_VALUES:  3 5 7 8 10       (window sizes)
#   L_VALUES:  1 2 3            (conv layers)
# ============================================================================

# === Sweep grid ===
K_VALUES=(3 5 7 8 10)
L_VALUES=(1 2 3)
TEMPORAL_CONV_KERNEL=3

# === Shared configuration ===
model_name=psdi_kd
gpu=0
batch_size=32
num_workers=8
d_model=128

# Dataset — PHM case 2
dataset_name=phm
root_path=data/processed_data/case2/denoise_off/image_bins_224

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

# Distillation parameters
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

# Training
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
mkdir -p logs/phm_case2_kwindow
sweep_timestamp="$(date +%Y%m%d_%H%M%S)"
summary_file="logs/phm_case2_kwindow/${sweep_timestamp}_SWEEP_SUMMARY.txt"

# Phase 2 checkpoint paths
phase2_setting="rul_RUL_PHM_C2_Student_Phase2_phase2_dm${d_model}_0_dst_${teacher_vlm_type}"
phase2_ckpt="./checkpoints/${phase2_setting}/checkpoint.pth"
phase2_teacher_ckpt="./checkpoints/${phase2_setting}_teacher/checkpoint.pth"

echo "============================================================"
echo "PHM Case 2 — K-Window Full Sweep"
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
  echo "PHASE 1: Training Teacher Model (MAE-Base + RUL Head) — PHM"
  echo "  Epochs: ${teacher_pretrain_epochs}"
  echo "============================================================"

  python -u run.py \
    --task_name rul \
    --is_training 1 \
    --root_path "${root_path}" \
    --dataset_name "${dataset_name}" \
    --model_id RUL_PHM_C2_Teacher_Phase1 \
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
    --des 'PHM_C2_Teacher_Phase1' 2>&1 | tee "logs/phm_case2_kwindow/${sweep_timestamp}_phase1.log"

  echo ""
  echo "PHASE 1 COMPLETED"
  echo "============================================================"

  # ==========================================================================
  # PHASE 2: Train Student with Knowledge Distillation
  # ==========================================================================
  echo ""
  echo "============================================================"
  echo "PHASE 2: Training Student with Knowledge Distillation — PHM"
  echo "  Epochs: ${phase2_epochs}"
  echo "============================================================"

  python -u run.py \
    --task_name rul \
    --is_training 1 \
    --root_path "${root_path}" \
    --dataset_name "${dataset_name}" \
    --model_id RUL_PHM_C2_Student_Phase2 \
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
    --des 'PHM_C2_Student_Phase2_KD' 2>&1 | tee "logs/phm_case2_kwindow/${sweep_timestamp}_phase2.log"

  echo ""
  echo "PHASE 2 COMPLETED"
  echo "  Student ckpt: ${phase2_ckpt}"
  echo "  Teacher ckpt: ${phase2_teacher_ckpt}"
  echo "============================================================"

  # Verify checkpoints were created
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

  local model_id="RUL_PHM_C2_P3_${tag}"
  local setting="rul_${model_id}_phase3_dm${d_model}_0_dst_${teacher_vlm_type}"
  local log="logs/phm_case2_kwindow/${sweep_timestamp}_${tag}.log"

  # Skip if results already exist
  if [ -f "results/${setting}/metrics.txt" ]; then
    echo "[SKIP] ${tag} — results already exist: results/${setting}/metrics.txt"
    return 0
  fi

  echo ""
  echo "============================================================"
  echo "PHASE 3: ${tag}"
  echo "  K=${k}, L=${l}"
  echo "  Log: ${log}"
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
    --des "PHM_C2_${tag}" 2>&1 | tee "${log}"

  # Evaluation
  python tools/rul_eval_degradation.py \
    --result_dir "results/${setting}" \
    --test_npz "${root_path}/test.npz" \
    --dataset_name "${dataset_name}" 2>&1 | tee -a "${log}"

  echo "[DONE] ${tag}"
}

# ============================================================================
# PHASE 3 SWEEP: K x L grid (NoMono — best from XJTU case1)
# ============================================================================
echo ""
echo "============================================================"
echo "Starting Phase 3 sweep: K={${K_VALUES[*]}} x L={${L_VALUES[*]}}"
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
echo "SWEEP COMPLETE — Collecting results"
echo "============================================================"

{
  echo "============================================================"
  echo "PHM Case 2 — K-Window Sweep Summary"
  echo "Date: $(date)"
  echo "============================================================"
  echo ""
  printf "%-28s  %8s  %8s  %8s  %8s\n" "Setting" "MSE" "MAE" "RMSE" "sMAPE"
  printf "%-28s  %8s  %8s  %8s  %8s\n" "----------------------------" "--------" "--------" "--------" "--------"

  for metrics_file in results/rul_RUL_PHM_C2_P3_*/metrics.txt; do
    [ -f "$metrics_file" ] || continue
    tag=$(basename "$(dirname "$metrics_file")")
    # Extract short name: rul_RUL_PHM_C2_P3_KW7_L2_phase3_... -> KW7_L2
    short=$(echo "$tag" | sed 's/rul_RUL_PHM_C2_P3_//; s/_phase3_.*//')

    mse=$(grep -oP 'MSE:\s+\K[\d.]+' "$metrics_file" 2>/dev/null || echo "N/A")
    mae=$(grep -oP 'MAE:\s+\K[\d.]+' "$metrics_file" 2>/dev/null || echo "N/A")
    rmse=$(grep -oP 'RMSE:\s+\K[\d.]+' "$metrics_file" 2>/dev/null || echo "N/A")
    score=$(grep -oP 'Score:\s+\K[\d.]+' "$metrics_file" 2>/dev/null || echo "N/A")

    printf "%-28s  %8s  %8s  %8s  %8s\n" "$short" "$mse" "$mae" "$rmse" "$score"
  done

  echo ""
  echo "============================================================"
} | tee "${summary_file}"

echo ""
echo "Summary saved to: ${summary_file}"
echo "============================================================"
