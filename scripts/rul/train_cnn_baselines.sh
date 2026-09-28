#!/usr/bin/env bash
set -e

export TOKENIZERS_PARALLELISM=false

# ============================================================================
# CNN Baseline Training Script for PSDI Image RUL Prediction
# ============================================================================
# Trains four standalone CNN baselines (Phase 1 only, no distillation):
#   - simple:          4-block CNN from scratch   (~0.5M params)
#   - vgg:             VGG-style CNN from scratch (~4M params)
#   - resnet18:        ResNet-18, ImageNet init   (~11M params)
#   - resnet34:        ResNet-34, ImageNet init   (~21M params)
#   - efficientnet_b0: EfficientNet-B0, ImageNet init (~5M params)
#
# Usage:
#   bash scripts/rul/train_cnn_baselines.sh [dataset]
#   dataset: xjtu (default) | phm_case2 | phm_case3
#
# Example:
#   bash scripts/rul/train_cnn_baselines.sh xjtu
# ============================================================================

# ----- Dataset selection (override with first CLI arg) -----
DATASET=${1:-xjtu}

case ${DATASET} in
  xjtu)
    root_path=data/processed_data/case1/denoise_off/image_bins_224
    dataset_name=xjtu
    model_id_prefix=RUL_XJTU_CNN
    ;;
  phm_case2)
    root_path=data/processed_data/case2/denoise_off/image_bins_224
    dataset_name=phm
    model_id_prefix=RUL_PHM_Case2_CNN
    ;;
  phm_case3)
    root_path=data/processed_data/case3/denoise_off/image_bins_224
    dataset_name=phm
    model_id_prefix=RUL_PHM_Case3_CNN
    ;;
  *)
    echo "Unknown dataset: ${DATASET}. Use: xjtu | phm_case2 | phm_case3"
    exit 1
    ;;
esac

# ----- Common hyper-parameters -----
model_name=psdi_kd
gpu=0
batch_size=32
num_workers=8
d_model=128
train_epochs=50
patience=7
dropout=0.1
weight_decay=1e-5

# CNN-specific learning rates:
#   From-scratch CNNs benefit from a higher LR; pretrained backbones need lower LR.
lr_scratch=0.001
lr_pretrained=0.0001

# Whether to finetune pretrained backbones (ResNet / EfficientNet).
# Set to 'false' to freeze backbone and only train RUL head.
finetune_cnn_backbone=true

# ============================================================================
# Helper function
# ============================================================================
run_cnn() {
  local cnn_type=$1
  local lr=$2
  local model_id="${model_id_prefix}_${cnn_type}"

  echo ""
  echo "============================================================"
  echo " Training CNN baseline: ${cnn_type}  (dataset=${DATASET})"
  echo "  root_path : ${root_path}"
  echo "  epochs    : ${train_epochs}  patience: ${patience}"
  echo "  lr        : ${lr}  batch_size: ${batch_size}"
  echo "  finetune  : ${finetune_cnn_backbone}"
  echo "============================================================"

  python -u run.py \
    --task_name rul \
    --is_training 1 \
    --root_path ${root_path} \
    --dataset_name ${dataset_name} \
    --model_id ${model_id} \
    --model ${model_name} \
    --data custom \
    --rul_phase 1 \
    --cnn_type ${cnn_type} \
    --finetune_cnn_backbone ${finetune_cnn_backbone} \
    --gpu ${gpu} \
    --use_amp \
    --batch_size ${batch_size} \
    --d_model ${d_model} \
    --learning_rate ${lr} \
    --num_workers ${num_workers} \
    --train_epochs ${train_epochs} \
    --patience ${patience} \
    --dropout ${dropout} \
    --weight_decay ${weight_decay} \
    --des "CNN_${cnn_type}_baseline"

  echo ""
  echo "Done: ${cnn_type} — checkpoint at ./checkpoints/rul_${model_id}_phase1_dm${d_model}_0"
  echo "============================================================"
}

# ============================================================================
# Run all CNN baselines sequentially
# ============================================================================

run_cnn simple         ${lr_scratch}
run_cnn vgg            ${lr_scratch}
run_cnn resnet18       ${lr_pretrained}
run_cnn resnet34       ${lr_pretrained}
run_cnn efficientnet_b0 ${lr_pretrained}

echo ""
echo "============================================================"
echo "All CNN baselines completed for dataset: ${DATASET}"
echo "Results: ./results/    Metrics: result_rul.txt"
echo "============================================================"
