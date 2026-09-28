#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false

# Usage:
#   bash scripts/rul/train_rul_xjtu_phase3_kwindow.sh           # K=5 default
#   bash scripts/rul/train_rul_xjtu_phase3_kwindow.sh 3          # K=3
#   bash scripts/rul/train_rul_xjtu_phase3_kwindow.sh 7 3        # K=7, 3 conv layers
#
# K-Window experiments for Phase 3 RUL prediction.
# Builds on A5 (best ablation: mono_slope) and adds temporal context.

k_window="${1:-5}"
temporal_conv_layers="${2:-2}"
temporal_conv_kernel="${3:-3}"

# Shared configuration (same as ablation script)
model_name=psdi_kd
gpu=0
batch_size=32
num_workers=8
d_model=128
dataset_name=xjtu
root_path=data/processed_data/case1/denoise_off/image_bins_224
teacher_vlm_type=mae_base
mae_size=base
mae_pretrained_path=facebook/vit-mae-base
teacher_hidden_size=768
use_cls_token=true
finetune_vlm=false
student_vision_type=tiny_vit
student_hidden_size=128
use_distillation=true
enable_adaptive_weights=true
init_temperature=4.0
distill_lr_ratio=0.05
num_alignment_scales=4
feature_alignment_dropout=0.1
loss_momentum=0.9
weight_regularization=0.001
feature_w=0.1
fcst_w=1.0
recon_w=0.5
att_w=0.01
learning_rate=0.0001
phase3_epochs=25
patience=7
dropout=0.1
weight_decay=0.01
teacher_pretrain_epochs=10
phase3_from_scratch=false

# Phase 2 checkpoint (reuse existing)
phase2_setting="rul_RUL_XJTU_Student_Phase2_phase2_dm${d_model}_0_dst_${teacher_vlm_type}"
phase2_ckpt="./checkpoints/${phase2_setting}/checkpoint.pth"
phase2_teacher_ckpt="./checkpoints/${phase2_setting}_teacher/checkpoint.pth"

# Phase 3 settings
phase3_qkv_mode=vision_q_phase_kv
phase3_residual_fusion=false
phase3_fusion_alpha_init=0.7
use_degradation_weighted_fcst=false
degradation_lambda=2.0
degradation_gamma=2.0

# Model ID and description
model_id="RUL_XJTU_P3_KW${k_window}_L${temporal_conv_layers}"
des="rec_20260215_KW${k_window}"

if [ ! -f "$phase2_ckpt" ]; then
  echo "Phase2 checkpoint missing: $phase2_ckpt"
  exit 1
fi

if [ ! -f "$phase2_teacher_ckpt" ]; then
  echo "Phase2 teacher checkpoint missing: $phase2_teacher_ckpt"
  exit 1
fi

mkdir -p logs/kwindow
timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="logs/kwindow/${timestamp}_K${k_window}_L${temporal_conv_layers}.log"

echo "Running K-Window: K=${k_window}, ConvLayers=${temporal_conv_layers}, Kernel=${temporal_conv_kernel}"
echo "Log file: ${log_file}"

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
  --k_window_size "${k_window}" \
  --temporal_conv_layers "${temporal_conv_layers}" \
  --temporal_conv_kernel "${temporal_conv_kernel}" \
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
  --att_w "${att_w}" \
  --phase2_ckpt_path "${phase2_ckpt}" \
  --phase2_teacher_ckpt_path "${phase2_teacher_ckpt}" \
  --skip_teacher_stage_in_phase3 true \
  --phase3_from_scratch "${phase3_from_scratch}" \
  --weight_decay "${weight_decay}" \
  --des "${des}" 2>&1 | tee "${log_file}"

setting="rul_${model_id}_phase3_dm${d_model}_0_dst_${teacher_vlm_type}"
python tools/rul_eval_degradation.py \
  --result_dir "results/${setting}" \
  --test_npz "${root_path}/test.npz" \
  --dataset_name "${dataset_name}" 2>&1 | tee -a "${log_file}"

echo "Completed K-Window K=${k_window}"
echo "Results: results/${setting}"
