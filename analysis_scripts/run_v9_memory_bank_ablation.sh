#!/usr/bin/env bash
set -euo pipefail

# PYTHON_BIN="/home/sidcs/miniconda3/bin/python3.13"
PYTHON_BIN="/home/sidcs/miniconda3/envs/mtse/bin/python"
REPO_ROOT="/home/sidcs/codebase/wavlm_dual_embedding"

DATASET_ROOT="/home/sidcs/datasets/LibriMix/LibriMix/memory_bank_longform_eval_v9_246810_fixed5_exact2_reappear2_500uniq"
RESULTS_DIR="/home/sidcs/codebase/wavlm_dual_embedding/analysis/large_num_speakers_v9_exact2_reappear2_500"
LOCAL_METADATA_CSV="/home/sidcs/datasets/LibriMix/LibriMix/Libriuni_06_08/Libri2Mix_ovl60to80/wav16k/min/metadata/mixture_test_mix_clean.csv"

DUAL_BASE_CKPT="/home/sidcs/model_ckpts/librispeech_asp_ft_wavlm_linear_dualemb_tr360/best-epoch=49-val_separation=0.000.ckpt"
DUAL_JOINT_CKPT="/home/sidcs/model_ckpts/pDCCRN_2sp_dpccn_joint_training_freezewavlm_indloss/best-epoch=19-val_separation=0.000.ckpt"
TEACHER_CKPT="/home/sidcs/model_ckpts/librispeech_asp_wavlm_tr360/best-epoch=62-val_separation=0.000.ckpt"

# v9 setup:
# - global speaker counts: [2, 4, 6, 8, 10]
# - 500 unique speaker combinations per setting
# - exact 5-second chunks
# - exactly 2 active speakers in every chunk
# - each speaker appears twice overall (middle ground between cover-once and long repeated tracking)
# - no extra random chunks

"${PYTHON_BIN}" "${REPO_ROOT}/tools/create_memory_bank_longform_dataset.py" \
  --local-metadata-csv "${LOCAL_METADATA_CSV}" \
  --output-root "${DATASET_ROOT}" \
  --settings 2 4 6 8 10 \
  --num-conversations 500 \
  --appearances-per-speaker 2 \
  --extra-random-chunks 0 \
  --chunk-sec 5.0 \
  --activity-mode exact2 \
  --unique-speaker-combos \
  --seed 44 \
  --force

"${PYTHON_BIN}" "${REPO_ROOT}/analysis_scripts/eval_memory_bank_ablation.py" \
  --dataset-root "${DATASET_ROOT}" \
  --dual-base-ckpt "${DUAL_BASE_CKPT}" \
  --dual-joint-ckpt "${DUAL_JOINT_CKPT}" \
  --teacher-ckpt "${TEACHER_CKPT}" \
  --out-dir "${RESULTS_DIR}" \
  --threshold 0.5 \
  --add-noise

"${PYTHON_BIN}" "${REPO_ROOT}/analysis_scripts/plot_memory_bank_ablation.py" \
  --summary-json "${RESULTS_DIR}/setting_summary.json" \
  --out-dir "${RESULTS_DIR}/plots"

echo "[done] v9 dataset, evaluation, and plots are ready."
