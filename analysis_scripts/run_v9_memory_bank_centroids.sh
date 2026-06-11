#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/home/sidcs/miniconda3/envs/mtse/bin/python"
REPO_ROOT="/home/sidcs/codebase/wavlm_dual_embedding"

DATASET_ROOT="/home/sidcs/datasets/LibriMix/LibriMix/memory_bank_longform_eval_v9_246810_fixed5_exact2_reappear2_500uniq"
OUT_DIR="/home/sidcs/codebase/wavlm_dual_embedding/analysis/large_num_speakers_v9_centroids"

DUAL_BASE_CKPT="/home/sidcs/model_ckpts/librispeech_asp_ft_wavlm_linear_dualemb_tr360/best-epoch=49-val_separation=0.000.ckpt"
DUAL_JOINT_CKPT="/home/sidcs/model_ckpts/pDCCRN_2sp_dpccn_joint_training_freezewavlm_indloss/best-epoch=19-val_separation=0.000.ckpt"
TEACHER_CKPT="/home/sidcs/model_ckpts/librispeech_asp_wavlm_tr360/best-epoch=62-val_separation=0.000.ckpt"

THRESHOLD="0.5"

"${PYTHON_BIN}" "${REPO_ROOT}/analysis_scripts/eval_memory_bank_centroids.py" \
  --dataset-root "${DATASET_ROOT}" \
  --dual-base-ckpt "${DUAL_BASE_CKPT}" \
  --dual-joint-ckpt "${DUAL_JOINT_CKPT}" \
  --teacher-ckpt "${TEACHER_CKPT}" \
  --out-dir "${OUT_DIR}" \
  --threshold "${THRESHOLD}" \
  --add-noise

echo "[done] centroid evaluation written to ${OUT_DIR}"
echo "[summary] ${OUT_DIR}/setting_summary.json"
