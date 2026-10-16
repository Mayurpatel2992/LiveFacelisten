#!/usr/bin/env bash
# Seamless step 1: Stage-1 VQ-VAE motion prior on UniLS-Talk Seamless Interaction
# (108-D exp 100 | head rot 3 | jaw 1 | gaze 4). Same model / loss config as the DualTalk
# prior (run_dualtalk_stage1.sh); exp loss on the 100 expression dims, jaw loss on the 8
# pose dims. A full pass is 142k tracks (8.9k steps on 2 GPUs), so an epoch is capped at
# 1000 steps (16k random tracks) to keep DualTalk-sized epochs for early stopping.
set -euo pipefail
cd /home/mpate169/Mayur/LiveFace_listen_speak/project
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate liveface
mkdir -p outputs
# Shared 128-core machine: cap CPU threads per process (uncapped, ~16 jobs pushed the
# load average to ~850 on 2026-10-14 and every job on the node was killed).
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8} NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-8}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export MASTER_PORT=${MASTER_PORT:-29711}

python3.9 train_all.py \
  experiment=seamless_vqvae_prior \
  state=new \
  data=seamless_prior \
  model=model_vqvae_prior \
  model.losses.exp_dims=100 \
  trainer.devices=2 \
  trainer.max_epochs=200 \
  +trainer.limit_train_batches=1000 \
  +trainer.strategy=ddp_find_unused_parameters_true \
  callback.early_stopping.patience=30 \
  "$@" \
  2>&1 | tee -a outputs/seamless_stage1_launch.log
