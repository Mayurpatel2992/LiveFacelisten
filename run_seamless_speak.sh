#!/usr/bin/env bash
# Seamless (UniLS protocol), step 2: Seamless speaking model -- Exp-5 recipe (Mimi + Moshi
# text, both Moshi tables trainable) on the agent's OWN track only; 108-D motion, exp loss
# on the 100 expression dims. Epochs capped at 600 steps (DualTalk-sized). Frozen
# main branch of Option B, starting point of Option A. 2-GPU DDP.
#   STAGE1=outputs/Seamless/seamless_vqvae_prior/<id> ./run_seamless_speak.sh
# (uses that run's best Stage-1 checkpoint unless STAGE1_CKPT is given)
set -euo pipefail
cd /home/mpate169/Mayur/LiveFace_listen_speak/project
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate liveface
# Shared 128-core machine: cap CPU threads per process (uncapped, ~16 jobs pushed the
# load average to ~850 on 2026-10-14 and every job on the node was killed).
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8} NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-8}
: "${STAGE1:?set STAGE1 to the Stage-1 run folder}"
STAGE1=$(realpath "$STAGE1")
STAGE1_CKPT=${STAGE1_CKPT:-$(ls "$STAGE1"/tensorboard/*/checkpoints/best-epoch=*.ckpt | tail -1)}
echo "Stage-1 checkpoint: $STAGE1_CKPT"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export MASTER_PORT=${MASTER_PORT:-29721}

python3.9 train_all.py \
  experiment=seamless_speak \
  state=new \
  data=seamless_pred_moshi_text \
  model=model_dualtalk_speak \
  model.folder_prior=$STAGE1 \
  model.version_prior=0 \
  "model.ckpt_path_prior='$STAGE1_CKPT'" \
  callback.best_epoch_ckpt.monitor=free_run_l1/val \
  ~callback.early_stopping \
  model.loss.exp_dims=100 \
  trainer.devices=2 \
  +trainer.limit_train_batches=600 \
  +trainer.strategy=ddp_find_unused_parameters_true \
  "$@" \
  2>&1 | tee -a outputs/seamless_speak_launch.log
