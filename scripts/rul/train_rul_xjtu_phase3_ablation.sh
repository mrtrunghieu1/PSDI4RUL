#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false

# Usage:
#   bash scripts/rul/train_rul_xjtu_phase3_ablation.sh A0_baseline
#   bash scripts/rul/train_rul_xjtu_phase3_ablation.sh A5_mono_slope true
# Variants:
#   A0_baseline, A1_no_att, A2_phaseq_visionkv, A3_residual_fusion,
#   A4_deg_weighted, A5_mono_slope

variant="${1:-A0_baseline}"
learn_temporal_weights="${2:-false}"  # Optional: true/false

# Shared configuration
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
learning_rate=0.0001
phase3_epochs=25
patience=7
dropout=0.1
weight_decay=0.01
teacher_pretrain_epochs=10
phase3_from_scratch=false
phase2_setting="rul_RUL_XJTU_Student_Phase2_phase2_dm${d_model}_0_dst_${teacher_vlm_type}"
phase2_ckpt="./checkpoints/${phase2_setting}/checkpoint.pth"
phase2_teacher_ckpt="./checkpoints/${phase2_setting}_teacher/checkpoint.pth"

# Ablation defaults
att_w=0.01
phase3_qkv_mode=vision_q_phase_kv
phase3_residual_fusion=false
phase3_fusion_alpha_init=0.7
use_degradation_weighted_fcst=false
degradation_lambda=2.0
degradation_gamma=2.0
temporal_mono_w=0.0
temporal_slope_w=0.0
model_id="RUL_XJTU_P3_ABL_A0_base"
des="rec_20260215_A0"

case "$variant" in
  A0_baseline)
    model_id="RUL_XJTU_P3_ABL_A0_base"
    des="rec_20260215_A0"
    ;;
  A1_no_att)
    model_id="RUL_XJTU_P3_ABL_A1_no_att"
    des="rec_20260215_A1"
    att_w=0.0
    ;;
  A2_phaseq_visionkv)
    model_id="RUL_XJTU_P3_ABL_A2_phaseq_visionkv"
    des="rec_20260215_A2"
    phase3_qkv_mode=phase_q_vision_kv
    ;;
  A3_residual_fusion)
    model_id="RUL_XJTU_P3_ABL_A3_residual_fusion"
    des="rec_20260215_A3"
    phase3_residual_fusion=true
    phase3_fusion_alpha_init=0.7
    ;;
  A4_deg_weighted)
    model_id="RUL_XJTU_P3_ABL_A4_deg_weighted"
    des="rec_20260215_A4"
    use_degradation_weighted_fcst=true
    degradation_lambda=2.0
    degradation_gamma=2.0
    ;;
  A5_mono_slope)
    model_id="RUL_XJTU_P3_ABL_A5_mono_slope"
    des="rec_20260215_A5"
    temporal_mono_w=0.2
    temporal_slope_w=0.1
    # Keep fixed by default for reproducibility; pass 2nd arg=true to learn them.
    ;;
  *)
    echo "Unknown ablation variant: $variant"
    exit 1
    ;;
esac

if [ ! -f "$phase2_ckpt" ]; then
  echo "Phase2 checkpoint missing: $phase2_ckpt"
  exit 1
fi

if [ ! -f "$phase2_teacher_ckpt" ]; then
  echo "Phase2 teacher checkpoint missing: $phase2_teacher_ckpt"
  exit 1
fi

mkdir -p logs/ablation
timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="logs/ablation/${timestamp}_${variant}.log"

echo "Running variant=${variant}"
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
  --temporal_mono_w "${temporal_mono_w}" \
  --temporal_slope_w "${temporal_slope_w}" \
  --learn_temporal_weights "${learn_temporal_weights}" \
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

echo "Completed ${variant}"
echo "Results: results/${setting}"
