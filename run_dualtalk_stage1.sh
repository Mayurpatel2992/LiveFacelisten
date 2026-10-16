#!/usr/bin/env bash
# Listen+speak, step 1: Stage-1 VQ-VAE motion prior on DualTalk (53-D exp+jaw).
# Same model/loss config as the MEAD prior (outputs/MEAD/vqvae_prior/10etdx0z),
# trained on full DualTalk tracks (<= 750 frames), both speakers of each
# conversation. 2-GPU DDP alongside the MEAD runs.
set -euo pipefail
cd /home/mpate169/Mayur/LiveFace_listen_speak/project
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate probtalk3d
mkdir -p outputs
export MASTER_PORT=${MASTER_PORT:-29611}

python3.9 train_all.py \
  experiment=dualtalk_vqvae_prior \
  state=new \
  data=dualtalk_prior \
  model=model_vqvae_prior \
  trainer.devices=2 \
  trainer.max_epochs=200 \
  +trainer.strategy=ddp_find_unused_parameters_true \
  callback.early_stopping.patience=30 \
  "$@" \
  2>&1 | tee -a outputs/dualtalk_stage1_launch.log
