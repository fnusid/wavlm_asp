#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/home/sidcs/miniconda3/envs/mtse/bin/python"

DATASET_ROOT="/home/sidcs/datasets/LibriMix/LibriMix/memory_bank_longform_eval_v9_246810_fixed5_exact2_reappear2_500uniq"
OUT_DIR="/home/sidcs/codebase/wavlm_dual_embedding/analysis/v9_tse_vs_sep_longform"

DUAL_BASE_CKPT="/home/sidcs/model_ckpts/librispeech_asp_ft_wavlm_linear_dualemb_tr360/best-epoch=49-val_separation=0.000.ckpt"
DUAL_JOINT_CKPT="/home/sidcs/model_ckpts/pDCCRN_2sp_dpccn_joint_training_freezewavlm_indloss/best-epoch=19-val_separation=0.000.ckpt"
TEACHER_CKPT="/home/sidcs/model_ckpts/librispeech_asp_wavlm_tr360/best-epoch=62-val_separation=0.000.ckpt"
TSE_CKPT="/home/sidcs/model_ckpts/pDCCRN_2sp_dpccn_joint_training_freezewavlm_indloss/best-epoch=19-val_separation=0.000.ckpt"
SEP_CKPT="/home/sidcs/model_ckpts/convtasnet_2sp_sep_/best-epoch=14-val_separation=0.000.ckpt"

mkdir -p "$OUT_DIR"

"$PYTHON_BIN" /home/sidcs/codebase/wavlm_dual_embedding/analysis_scripts/eval_tse_vs_sep_longform.py \
  --dataset-root "$DATASET_ROOT" \
  --dual-base-ckpt "$DUAL_BASE_CKPT" \
  --dual-joint-ckpt "$DUAL_JOINT_CKPT" \
  --teacher-ckpt "$TEACHER_CKPT" \
  --tse-ckpt "$TSE_CKPT" \
  --sep-ckpt "$SEP_CKPT" \
  --out-dir "$OUT_DIR" \
  --add-noise

"$PYTHON_BIN" /home/sidcs/codebase/wavlm_dual_embedding/analysis_scripts/plot_tse_vs_sep_longform.py \
  --summary-json "$OUT_DIR/setting_summary.json" \
  --out-dir "$OUT_DIR/plots"
