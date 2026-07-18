#!/bin/bash
# E040b: corrected-RA width chain 0.20 -> 0.30, sequential warm-start lineage.
# ra_w20 warm-starts the corrected 0.12 final; ra_w30 warm-starts the corrected
# 0.20 final. Avoid arms are NOT rerun (reused from E025). 1024 envs, 150M each.
set -u
cd /home/buzi/Desktop/RESEARCH/SAFE/DEVELOPMENT/safe_mjlab_zoo
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mjlab
export MUJOCO_GL=egl WANDB_MODE=online
CMN="--num-envs 1024 --steps 150000000 --lr 1e-4 --no-adaptive-lr --ent-coef 1e-3 --max-std 0.4 --target-kl 0.01 --out runs_e040b --wandb-project robot_safety_sandbox"

echo "[chase] START $(date)"

echo "[chase] launching ra_w20 (warm <- corrected 0.12 final)"
python -u examples/train.py --task go2_gap_split2_ra_w20 \
  --load runs_e040/go2_gap_split2_ra/final_model.zip $CMN \
  > runs_e040b/e040b_ra_w20.log 2>&1
echo "[chase] ra_w20 DONE $(date)"

W20_FINAL=runs_e040b/go2_gap_split2_ra_w20/final_model.zip
if [ ! -f "$W20_FINAL" ]; then
  echo "[chase] ABORT: $W20_FINAL not written (w20 failed) — not launching w30"; exit 1
fi

echo "[chase] launching ra_w30 (warm <- corrected 0.20 final)"
python -u examples/train.py --task go2_gap_split2_ra_w30 \
  --load "$W20_FINAL" $CMN \
  > runs_e040b/e040b_ra_w30.log 2>&1
echo "[chase] ra_w30 DONE $(date)"
echo "[chase] CHAIN COMPLETE $(date)"
