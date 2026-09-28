#!/usr/bin/env bash
set -e

export TOKENIZERS_PARALLELISM=false

# ============================================================================
# RUL Prediction Training Script for XJTU Bearing Dataset
# ============================================================================
# Phase 1: Train teacher model (MAE-Base + RUL head)
# Phase 2: Train student model with knowledge distillation
# ============================================================================

# Basic configuration
model_name=psdi_kd
gpu=0
batch_size=32
num_workers=8
d_model=128

# Dataset configuration
dataset_name=xjtu
root_path=data/processed_data/case1/denoise_on/image_bins_224

# Teacher model configuration (Phase 1)
teacher_vlm_type=mae_base
mae_size=base                    # base (768), large (1024), huge (1280)
mae_pretrained_path=facebook/vit-mae-base
teacher_hidden_size=768          # MAE-base hidden size
use_cls_token=true               # Use CLS token for feature extraction
finetune_vlm=false               # Freeze MAE encoder, only train RUL head

# Student model configuration (Phase 2)
student_vision_type=tiny_vit     # TinyViT: 4 layers, 4 heads
student_hidden_size=128          # Student encoder hidden size

# Distillation parameters (Phase 2)
use_distillation=true
enable_adaptive_weights=true     # Learnable loss weights
init_temperature=4.0             # Distillation temperature
distill_lr_ratio=0.05            # Distillation params lr = main_lr * ratio
num_alignment_scales=4           # Multi-scale feature alignment
feature_alignment_dropout=0.1    # Dropout in feature aligner
loss_momentum=0.9                # Loss balancer momentum
weight_regularization=0.001      # Weight regularization coefficient

# Loss weights (initial values, will be learned if adaptive=true)
feature_w=0.1                    # Feature distillation loss
fcst_w=1.0                       # Forecast (main task) loss
recon_w=0.5                      # Reconstruction (mimic teacher) loss
att_w=0.0                        # Attention loss (not used in Phase 1 & 2)

# Training configuration
learning_rate=0.0001             # Reduced from 0.001 to prevent overfitting
train_epochs=50                  # Student training epochs
phase3_epochs=25                 # Phase 3 fine-tuning epochs
teacher_pretrain_epochs=10       # Teacher training epochs
patience=7                       # Early stopping patience
dropout=0.1
weight_decay=0.01                # L2 regularization to reduce overfitting
run_phase3=true                  # Set false to skip Phase 3
phase3_att_w=0.01                # Initial attention distillation weight (w4)
phase3_from_scratch=false        # true: train phase3 without loading phase2 ckpt
phase3_qkv_mode=vision_q_phase_kv # legacy best setting

# Phase control
# rul_phase:
#   1 = Train teacher only (supervised learning)
#   2 = Two-stage: Train teacher, then student with distillation
#   3 = Three-stage: Phase 2 + cross-modal attention with Phase Space

# ============================================================================
# PHASE 1: Train Teacher Model Only
# ============================================================================
echo "============================================================"
echo "PHASE 1: Training Teacher Model (MAE-Base + RUL Head)"
echo "============================================================"
echo "Configuration:"
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
  --model_id RUL_XJTU_DenoiseOn_Teacher_Phase1 \
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
echo "Teacher model saved in: ./checkpoints/RUL_XJTU_DenoiseOn_Teacher_Phase1_teacher"
echo "============================================================"
echo ""

# ============================================================================
# PHASE 2: Train Student Model with Knowledge Distillation
# ============================================================================
echo "============================================================"
echo "PHASE 2: Training Student Model with Knowledge Distillation"
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
  --model_id RUL_XJTU_DenoiseOn_Student_Phase2 \
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
phase2_setting="rul_RUL_XJTU_DenoiseOn_Student_Phase2_phase2_dm${d_model}_0_dst_${teacher_vlm_type}"
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
  echo "PHASE 3: Cross-Modal Attention Refinement (Optional)"
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
    --model_id RUL_XJTU_DenoiseOn_Student_Phase3 \
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
  echo "Results saved in: ./results/rul_RUL_XJTU_DenoiseOn_Student_Phase3_phase3_dm${d_model}_0_dst_${teacher_vlm_type}"
  echo "============================================================"
  echo ""
fi

echo "Training completed! Check results:"
echo "  - Checkpoints: ./checkpoints/"
echo "  - Results: ./results/"
echo "  - Metrics: result_rul.txt"
echo "============================================================"
