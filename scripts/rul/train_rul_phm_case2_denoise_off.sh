#!/usr/bin/env bash
set -e

export TOKENIZERS_PARALLELISM=false

# ============================================================================
# RUL Prediction Training Script for PHM (PRONOSTIA) Bearing Dataset — Case 2
# ============================================================================
# Phase 1: Train teacher model (MAE-Base + RUL head)
# Phase 2: Train student model with knowledge distillation
# Phase 3: Cross-modal attention refinement (X + P)
# ============================================================================

# Basic configuration
model_name=psdi_kd
gpu=0
batch_size=32
num_workers=8
d_model=128

# Dataset configuration — PHM case 2
dataset_name=phm
root_path=data/processed_data/case2/denoise_off/image_bins_224

# Teacher model configuration (Phase 1)
teacher_vlm_type=mae_base
mae_size=base
mae_pretrained_path=facebook/vit-mae-base
teacher_hidden_size=768
use_cls_token=true
finetune_vlm=false

# Student model configuration (Phase 2)
student_vision_type=tiny_vit
student_hidden_size=128

# Distillation parameters (Phase 2)
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

# Training configuration
learning_rate=0.0001
train_epochs=50
phase3_epochs=25
teacher_pretrain_epochs=10
patience=7
dropout=0.1
weight_decay=0.01
run_phase3=true
phase3_att_w=0.01
phase3_from_scratch=false
phase3_qkv_mode=vision_q_phase_kv

# ============================================================================
# PHASE 1: Train Teacher Model Only
# ============================================================================
echo "============================================================"
echo "PHASE 1: Training Teacher Model (MAE-Base + RUL Head) — PHM"
echo "============================================================"
echo "Configuration:"
echo "  Dataset: ${dataset_name} (case 2)"
echo "  Model: ${teacher_vlm_type} (${mae_size})"
echo "  Hidden size: ${teacher_hidden_size}"
echo "  Freeze encoder: ${finetune_vlm}"
echo "  Epochs: ${teacher_pretrain_epochs}"
echo "  Batch size: ${batch_size}"
echo "  Learning rate: ${learning_rate}"
echo "============================================================"

python -u run.py \
  --task_name rul \
  --is_training 1 \
  --root_path ${root_path} \
  --dataset_name ${dataset_name} \
  --model_id RUL_PHM_Teacher_Phase1 \
  --model ${model_name} \
  --data custom \
  --rul_phase 1 \
  --gpu ${gpu} \
  --use_amp \
  --batch_size ${batch_size} \
  --d_model ${d_model} \
  --learning_rate ${learning_rate} \
  --num_workers ${num_workers} \
  --train_epochs ${teacher_pretrain_epochs} \
  --patience ${patience} \
  --dropout ${dropout} \
  --teacher_vlm_type ${teacher_vlm_type} \
  --mae_size ${mae_size} \
  --mae_pretrained_path ${mae_pretrained_path} \
  --teacher_hidden_size ${teacher_hidden_size} \
  --use_cls_token ${use_cls_token} \
  --finetune_vlm ${finetune_vlm} \
  --weight_decay ${weight_decay} \
  --des 'Teacher_Phase1'

echo ""
echo "============================================================"
echo "PHASE 1 COMPLETED"
echo "Teacher model saved in: ./checkpoints/RUL_PHM_Teacher_Phase1_teacher"
echo "============================================================"
echo ""

# ============================================================================
# PHASE 2: Train Student Model with Knowledge Distillation
# ============================================================================
echo "============================================================"
echo "PHASE 2: Training Student Model with Knowledge Distillation — PHM"
echo "============================================================"
echo "Configuration:"
echo "  Student: TinyViT (${student_hidden_size} hidden)"
echo "  Teacher: Frozen MAE-Base (${teacher_hidden_size} hidden)"
echo "  Distillation: Feature + Forecast + Reconstruction"
echo "  Adaptive weights: ${enable_adaptive_weights}"
echo "  Temperature: ${init_temperature}"
echo "  Epochs: ${train_epochs}"
echo "  Batch size: ${batch_size}"
echo "  Learning rate: ${learning_rate}"
echo "  Note: Phase Space (P) NOT used in Phase 2"
echo "============================================================"

python -u run.py \
  --task_name rul \
  --is_training 1 \
  --root_path ${root_path} \
  --dataset_name ${dataset_name} \
  --model_id RUL_PHM_Student_Phase2 \
  --model ${model_name} \
  --data custom \
  --rul_phase 2 \
  --gpu ${gpu} \
  --use_amp \
  --batch_size ${batch_size} \
  --d_model ${d_model} \
  --learning_rate ${learning_rate} \
  --num_workers ${num_workers} \
  --train_epochs ${train_epochs} \
  --patience ${patience} \
  --dropout ${dropout} \
  --teacher_vlm_type ${teacher_vlm_type} \
  --mae_size ${mae_size} \
  --mae_pretrained_path ${mae_pretrained_path} \
  --teacher_hidden_size ${teacher_hidden_size} \
  --use_cls_token ${use_cls_token} \
  --finetune_vlm ${finetune_vlm} \
  --teacher_pretrain_epochs ${teacher_pretrain_epochs} \
  --student_vision_type ${student_vision_type} \
  --student_hidden_size ${student_hidden_size} \
  --use_distillation ${use_distillation} \
  --enable_adaptive_weights ${enable_adaptive_weights} \
  --init_temperature ${init_temperature} \
  --distill_lr_ratio ${distill_lr_ratio} \
  --num_alignment_scales ${num_alignment_scales} \
  --feature_alignment_dropout ${feature_alignment_dropout} \
  --loss_momentum ${loss_momentum} \
  --weight_regularization ${weight_regularization} \
  --feature_w ${feature_w} \
  --fcst_w ${fcst_w} \
  --recon_w ${recon_w} \
  --att_w ${att_w} \
  --weight_decay ${weight_decay} \
  --des 'Student_Phase2_KD'

echo ""
echo "============================================================"
echo "PHASE 2 COMPLETED"
phase2_setting="rul_RUL_PHM_Student_Phase2_phase2_dm${d_model}_0_dst_${teacher_vlm_type}"
phase2_ckpt="./checkpoints/${phase2_setting}/checkpoint.pth"
phase2_teacher_ckpt="./checkpoints/${phase2_setting}_teacher/checkpoint.pth"
echo "Phase 2 setting: ${phase2_setting}"
echo "Student checkpoint: ${phase2_ckpt}"
echo "Teacher checkpoint: ${phase2_teacher_ckpt}"
echo "============================================================"
echo ""

# ============================================================================
# PHASE 3: Cross-Modal Attention (X + P)
# ============================================================================
if [ "${run_phase3}" = true ]; then
  echo "============================================================"
  echo "PHASE 3: Cross-Modal Attention Refinement — PHM"
  echo "============================================================"
  echo "Configuration:"
  echo "  Resume student from: ${phase2_ckpt}"
  echo "  Load teacher from: ${phase2_teacher_ckpt}"
  echo "  Input: X [B,2,224,224] + P [B,2,2560]"
  echo "  Epochs: ${phase3_epochs}"
  echo "  Initial att_w: ${phase3_att_w}"
  echo "============================================================"

  python -u run.py \
    --task_name rul \
    --is_training 1 \
    --root_path ${root_path} \
    --dataset_name ${dataset_name} \
    --model_id RUL_PHM_Student_Phase3 \
    --model ${model_name} \
    --data custom \
    --rul_phase 3 \
    --phase3_qkv_mode ${phase3_qkv_mode} \
    --gpu ${gpu} \
    --use_amp \
    --batch_size ${batch_size} \
    --d_model ${d_model} \
    --learning_rate ${learning_rate} \
    --num_workers ${num_workers} \
    --train_epochs ${phase3_epochs} \
    --patience ${patience} \
    --dropout ${dropout} \
    --teacher_vlm_type ${teacher_vlm_type} \
    --mae_size ${mae_size} \
    --mae_pretrained_path ${mae_pretrained_path} \
    --teacher_hidden_size ${teacher_hidden_size} \
    --use_cls_token ${use_cls_token} \
    --finetune_vlm ${finetune_vlm} \
    --teacher_pretrain_epochs ${teacher_pretrain_epochs} \
    --student_vision_type ${student_vision_type} \
    --student_hidden_size ${student_hidden_size} \
    --use_distillation ${use_distillation} \
    --enable_adaptive_weights ${enable_adaptive_weights} \
    --init_temperature ${init_temperature} \
    --distill_lr_ratio ${distill_lr_ratio} \
    --num_alignment_scales ${num_alignment_scales} \
    --feature_alignment_dropout ${feature_alignment_dropout} \
    --loss_momentum ${loss_momentum} \
    --weight_regularization ${weight_regularization} \
    --feature_w ${feature_w} \
    --fcst_w ${fcst_w} \
    --recon_w ${recon_w} \
    --att_w ${phase3_att_w} \
    --phase2_ckpt_path ${phase2_ckpt} \
    --phase2_teacher_ckpt_path ${phase2_teacher_ckpt} \
    --skip_teacher_stage_in_phase3 true \
    --phase3_from_scratch ${phase3_from_scratch} \
    --weight_decay ${weight_decay} \
    --des 'Student_Phase3_CrossModal'

  echo ""
  echo "============================================================"
  echo "PHASE 3 COMPLETED"
  echo "Results saved in: ./results/rul_RUL_PHM_Student_Phase3_phase3_dm${d_model}_0_dst_${teacher_vlm_type}"
  echo "============================================================"
  echo ""
fi

echo "Training completed! Check results:"
echo "  - Checkpoints: ./checkpoints/"
echo "  - Results: ./results/"
echo "  - Metrics: result_rul.txt"
echo "============================================================"
