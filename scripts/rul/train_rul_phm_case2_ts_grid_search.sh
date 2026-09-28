#!/usr/bin/env bash
# =============================================================================
# Teacher-Student Grid Search — PHM Case 2 — PSDI Images
# =============================================================================
# Compares all Teacher × Student combinations with fixed hyperparameters
# (enable_adaptive_weights=false) for fair comparison.
#
# Grid:
#   Teachers : mae_base | mae_large | efficientnet_b3 | clip_vit_b32
#   Students : tiny_vit | efficientnet_b0 | mobilenet_v3
#   K values : 7 9 11   (Phase 3 window sizes)
#   L values : 2 3      (temporal conv layers)
#
# Pipeline per pair:
#   Phase 1 → train teacher        (shared, skip if ckpt exists)
#   Phase 2 → train student + KD   (per pair, skip if ckpt exists)
#   Phase 3 → K×L sweep            (per pair, skip if result exists)
#
# To add a new case (e.g. Case 3), duplicate this script and adjust:
#   dataset_name, root_path, and the data split / FPT-EOF table.
#
# Requirements:
#   Run inside conda env that has numpy>=2.0 + torchvision installed.
#   The np2_convert env satisfies these (numpy 2.x, torch 2.8, torchvision 0.23).
#   Example:
#     conda activate np2_convert
#     bash scripts/rul/train_rul_phm_case2_ts_grid_search.sh [GPU_ID]
# =============================================================================
set -euo pipefail
export TOKENIZERS_PARALLELISM=false

# ── Device ──────────────────────────────────────────────────────────────────
gpu="${1:-0}"

# ── Dataset (PHM Case 2) ─────────────────────────────────────────────────────
dataset_name=phm
root_path=data/processed_data/case2/denoise_off/image_bins_224

# ── Sweep grid ───────────────────────────────────────────────────────────────
TEACHERS=(mae_base mae_large efficientnet_b3 clip_vit_b32)
STUDENTS=(tiny_vit efficientnet_b0 mobilenet_v3)
K_VALUES=(7 9 11)
L_VALUES=(2 3)
TEMPORAL_CONV_KERNEL=3

# ── Teacher metadata (hidden size + extra args per type) ─────────────────────
declare -A TEACHER_HIDDEN
TEACHER_HIDDEN[mae_base]=768
TEACHER_HIDDEN[mae_large]=1024
TEACHER_HIDDEN[efficientnet_b3]=1536
TEACHER_HIDDEN[clip_vit_b32]=768   # CLIP ViT-B/32 actual hidden = 768

declare -A TEACHER_EXTRA_ARGS
TEACHER_EXTRA_ARGS[mae_base]="--mae_size base --mae_pretrained_path facebook/vit-mae-base --use_cls_token true"
TEACHER_EXTRA_ARGS[mae_large]="--mae_size large --mae_pretrained_path facebook/vit-mae-large --use_cls_token true"
TEACHER_EXTRA_ARGS[efficientnet_b3]=""
TEACHER_EXTRA_ARGS[clip_vit_b32]=""

# ── Shared hyperparameters (FIXED — fair comparison for grid search) ──────────
model_name=psdi_kd
batch_size=32
num_workers=8
d_model=128
learning_rate=0.0001
teacher_pretrain_epochs=10
phase2_epochs=50
phase3_epochs=25
patience=7
dropout=0.1
weight_decay=0.01

# Distillation — FIXED weights (no adaptive learning) for fair comparison
use_distillation=true
enable_adaptive_weights=false      # Fixed for grid search; enable only for best pair
feature_w=0.1
fcst_w=1.0
recon_w=0.5
att_w=0.01
init_temperature=4.0
distill_lr_ratio=0.05
num_alignment_scales=4
feature_alignment_dropout=0.1
loss_momentum=0.9
weight_regularization=0.001
finetune_vlm=false

# Phase 3 settings
phase3_qkv_mode=vision_q_phase_kv
phase3_residual_fusion=false
phase3_fusion_alpha_init=0.7
use_degradation_weighted_fcst=false
degradation_lambda=2.0
degradation_gamma=2.0
phase3_from_scratch=false

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR="logs/phm_case2_ts_grid_search"
mkdir -p "${LOG_DIR}"
sweep_timestamp="$(date +%Y%m%d_%H%M%S)"
summary_file="${LOG_DIR}/${sweep_timestamp}_SWEEP_SUMMARY.txt"

echo "============================================================"
echo "PHM Case 2 — Teacher-Student Grid Search"
echo "  Teachers : ${TEACHERS[*]}"
echo "  Students : ${STUDENTS[*]}"
echo "  K values : ${K_VALUES[*]}"
echo "  L values : ${L_VALUES[*]}"
echo "  GPU      : ${gpu}"
echo "  Log dir  : ${LOG_DIR}"
echo "  Summary  : ${summary_file}"
echo "============================================================"

# =============================================================================
# Helper: teacher checkpoint path
# =============================================================================
teacher_ckpt_path() {
  local teacher="$1"
  echo "./checkpoints/rul_RUL_PHM_C2_T${teacher}_Phase1_phase1_dm${d_model}_0/checkpoint.pth"
}

# =============================================================================
# Helper: Phase-2 student checkpoint paths
# =============================================================================
phase2_student_ckpt_path() {
  local teacher="$1" student="$2"
  echo "./checkpoints/rul_RUL_PHM_C2_T${teacher}_S${student}_Phase2_phase2_dm${d_model}_0_dst_${teacher}/checkpoint.pth"
}

phase2_teacher_ckpt_path() {
  local teacher="$1" student="$2"
  echo "./checkpoints/rul_RUL_PHM_C2_T${teacher}_S${student}_Phase2_phase2_dm${d_model}_0_dst_${teacher}_teacher/checkpoint.pth"
}

# =============================================================================
# PHASE 1: Train teacher (once per teacher type)
# =============================================================================
run_phase1() {
  local teacher="$1"
  local ckpt; ckpt="$(teacher_ckpt_path "${teacher}")"

  if [ -f "${ckpt}" ]; then
    echo "[SKIP] Phase 1 ${teacher} — checkpoint exists: ${ckpt}"
    return 0
  fi

  local log="${LOG_DIR}/${sweep_timestamp}_P1_${teacher}.log"
  echo ""
  echo "============================================================"
  echo "PHASE 1: ${teacher}  (hidden=${TEACHER_HIDDEN[${teacher}]})"
  echo "============================================================"

  # shellcheck disable=SC2086
  python -u run.py \
    --task_name rul \
    --is_training 1 \
    --root_path "${root_path}" \
    --dataset_name "${dataset_name}" \
    --model_id "RUL_PHM_C2_T${teacher}_Phase1" \
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
    --teacher_vlm_type "${teacher}" \
    --teacher_hidden_size "${TEACHER_HIDDEN[${teacher}]}" \
    --finetune_vlm "${finetune_vlm}" \
    --weight_decay "${weight_decay}" \
    ${TEACHER_EXTRA_ARGS[${teacher}]} \
    --des "PHM_C2_T${teacher}_Phase1" 2>&1 | tee "${log}"

  echo "[DONE] Phase 1 ${teacher}"
}

# =============================================================================
# PHASE 2: Train student with KD (once per teacher×student pair)
# =============================================================================
run_phase2() {
  local teacher="$1" student="$2"
  local ckpt; ckpt="$(phase2_student_ckpt_path "${teacher}" "${student}")"
  local teacher_ckpt; teacher_ckpt="$(teacher_ckpt_path "${teacher}")"

  if [ -f "${ckpt}" ]; then
    echo "[SKIP] Phase 2 T=${teacher} S=${student} — checkpoint exists"
    return 0
  fi

  local log="${LOG_DIR}/${sweep_timestamp}_P2_T${teacher}_S${student}.log"
  echo ""
  echo "============================================================"
  echo "PHASE 2: Teacher=${teacher}  Student=${student}"
  echo "============================================================"

  # shellcheck disable=SC2086
  python -u run.py \
    --task_name rul \
    --is_training 1 \
    --root_path "${root_path}" \
    --dataset_name "${dataset_name}" \
    --model_id "RUL_PHM_C2_T${teacher}_S${student}_Phase2" \
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
    --teacher_vlm_type "${teacher}" \
    --teacher_hidden_size "${TEACHER_HIDDEN[${teacher}]}" \
    --finetune_vlm "${finetune_vlm}" \
    --teacher_pretrain_epochs "${teacher_pretrain_epochs}" \
    --student_vision_type "${student}" \
    --student_hidden_size 128 \
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
    ${TEACHER_EXTRA_ARGS[${teacher}]} \
    --des "PHM_C2_T${teacher}_S${student}_Phase2" 2>&1 | tee "${log}"

  echo "[DONE] Phase 2 T=${teacher} S=${student}"
}

# =============================================================================
# PHASE 3: K-window sweep (one experiment per K×L)
# =============================================================================
run_phase3() {
  local teacher="$1" student="$2" k="$3" l="$4"
  local tag="KW${k}_L${l}"
  local model_id="RUL_PHM_C2_T${teacher}_S${student}_P3_${tag}"
  local setting="rul_${model_id}_phase3_dm${d_model}_0_dst_${teacher}"
  local result_dir="results/${setting}"
  local log="${LOG_DIR}/${sweep_timestamp}_P3_T${teacher}_S${student}_${tag}.log"

  local p2_ckpt; p2_ckpt="$(phase2_student_ckpt_path "${teacher}" "${student}")"
  local p2_teacher_ckpt; p2_teacher_ckpt="$(phase2_teacher_ckpt_path "${teacher}" "${student}")"

  if [ -f "${result_dir}/metrics.txt" ]; then
    echo "[SKIP] Phase 3 T=${teacher} S=${student} ${tag} — results exist"
    return 0
  fi

  echo ""
  echo "============================================================"
  echo "PHASE 3: Teacher=${teacher}  Student=${student}  K=${k}  L=${l}"
  echo "============================================================"

  # shellcheck disable=SC2086
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
    --teacher_vlm_type "${teacher}" \
    --teacher_hidden_size "${TEACHER_HIDDEN[${teacher}]}" \
    --finetune_vlm "${finetune_vlm}" \
    --teacher_pretrain_epochs "${teacher_pretrain_epochs}" \
    --student_vision_type "${student}" \
    --student_hidden_size 128 \
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
    --phase2_ckpt_path "${p2_ckpt}" \
    --phase2_teacher_ckpt_path "${p2_teacher_ckpt}" \
    --skip_teacher_stage_in_phase3 true \
    --phase3_from_scratch "${phase3_from_scratch}" \
    --weight_decay "${weight_decay}" \
    ${TEACHER_EXTRA_ARGS[${teacher}]} \
    --des "PHM_C2_T${teacher}_S${student}_${tag}" 2>&1 | tee "${log}"

  echo "[DONE] Phase 3 T=${teacher} S=${student} ${tag}"
}

# =============================================================================
# MAIN LOOP
# =============================================================================
total_p3=$(( ${#TEACHERS[@]} * ${#STUDENTS[@]} * ${#K_VALUES[@]} * ${#L_VALUES[@]} ))
count=0

for teacher in "${TEACHERS[@]}"; do

  echo ""
  echo "============================================================"
  echo ">>> TEACHER: ${teacher}  (hidden=${TEACHER_HIDDEN[${teacher}]})"
  echo "============================================================"

  run_phase1 "${teacher}"

  for student in "${STUDENTS[@]}"; do

    echo ""
    echo ">>> PAIR: Teacher=${teacher}  Student=${student}"

    run_phase2 "${teacher}" "${student}"

    for k in "${K_VALUES[@]}"; do
      for l in "${L_VALUES[@]}"; do
        count=$((count + 1))
        echo ""
        echo ">>> Phase 3 [${count}/${total_p3}]: T=${teacher} S=${student} K=${k} L=${l}"
        run_phase3 "${teacher}" "${student}" "${k}" "${l}"
      done
    done

  done
done

# =============================================================================
# SUMMARY TABLE
# =============================================================================
echo ""
echo "============================================================"
echo "SWEEP COMPLETE — Collecting results"
echo "============================================================"

{
  echo "============================================================"
  echo "PHM Case 2 — Teacher-Student Grid Search Summary"
  echo "Date: $(date)"
  echo "Weights: enable_adaptive_weights=${enable_adaptive_weights}"
  echo "============================================================"
  echo ""
  printf "%-18s %-14s %3s %2s | %8s %8s %8s | %8s %8s %8s %10s\n" \
    "Teacher" "Student" "K" "L" "RMSE" "MAE" "Score" \
    "EncP(M)" "TotP(M)" "Mem(MiB)" "Speed(ms)"
  printf -- "%-18s %-14s %3s %2s-+-%8s %8s %8s-+-%8s %8s %8s %10s\n" \
    "------------------" "--------------" "---" "--" \
    "--------" "--------" "--------" \
    "--------" "--------" "--------" "----------"

  for teacher in "${TEACHERS[@]}"; do
    for student in "${STUDENTS[@]}"; do
      for k in "${K_VALUES[@]}"; do
        for l in "${L_VALUES[@]}"; do
          tag="KW${k}_L${l}"
          model_id="RUL_PHM_C2_T${teacher}_S${student}_P3_${tag}"
          setting="rul_${model_id}_phase3_dm${d_model}_0_dst_${teacher}"
          mfile="results/${setting}/metrics.txt"
          pfile="results/${setting}/model_profile.json"

          rmse="N/A"; mae="N/A"; score="N/A"
          enc_p="N/A"; tot_p="N/A"; mem="N/A"; spd="N/A"

          if [ -f "${mfile}" ]; then
            rmse=$(grep  -oP 'RMSE:\s+\K[\d.]+' "${mfile}"  2>/dev/null || echo "N/A")
            mae=$(grep   -oP 'MAE:\s+\K[\d.]+'  "${mfile}"  2>/dev/null || echo "N/A")
            score=$(grep -oP 'Score:\s+\K[\d.]+' "${mfile}" 2>/dev/null || echo "N/A")
          fi

          if [ -f "${pfile}" ] && command -v python3 &>/dev/null; then
            read -r enc_p tot_p mem spd < <(python3 - <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
s = d.get('student', d.get('teacher', {}))
enc_p = s.get('encoder_params_M', 'N/A')
tot_p = s.get('total_params_M',   'N/A')
mem   = s.get('mem_mib',          'N/A')
spd   = s.get('speed_s_per_iter', 'N/A')
if spd != 'N/A':
    spd = f"{float(spd)*1000:.2f}"
print(enc_p, tot_p, mem, spd)
PYEOF
            "${pfile}" 2>/dev/null) || true
          fi

          printf "%-18s %-14s %3s %2s | %8s %8s %8s | %8s %8s %8s %10s\n" \
            "${teacher}" "${student}" "${k}" "${l}" \
            "${rmse}" "${mae}" "${score}" \
            "${enc_p}" "${tot_p}" "${mem}" "${spd}"
        done
      done
    done
  done

  echo ""
  echo "============================================================"
} | tee "${summary_file}"

echo ""
echo "Summary saved to: ${summary_file}"
echo "============================================================"
