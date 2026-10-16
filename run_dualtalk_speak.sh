#!/usr/bin/env bash
# Listen+speak, step 2: DualTalk speaking model -- Exp-5 recipe (Mimi + Moshi
# text, both Moshi tables trainable) on the agent's OWN track only. Frozen
# main branch of Option B, starting point of Option A. 2-GPU DDP.
#   STAGE1=outputs/DualTalk/dualtalk_vqvae_prior/<id> ./run_dualtalk_speak.sh
# (uses that run's best Stage-1 checkpoint unless STAGE1_CKPT is given)
set -euo pipefail
cd /home/mpate169/Mayur/LiveFace_listen_speak/project
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate probtalk3d
: "${STAGE1:?set STAGE1 to the Stage-1 run folder}"
STAGE1=$(realpath "$STAGE1")
STAGE1_CKPT=${STAGE1_CKPT:-$(ls "$STAGE1"/tensorboard/*/checkpoints/best-epoch=*.ckpt | tail -1)}
echo "Stage-1 checkpoint: $STAGE1_CKPT"
export MASTER_PORT=${MASTER_PORT:-29621}

python3.9 train_all.py \
  experiment=dualtalk_speak \
  state=new \
  data=dualtalk_pred_moshi_text \
  model=model_dualtalk_speak \
  model.folder_prior=$STAGE1 \
  model.version_prior=0 \
  "model.ckpt_path_prior='$STAGE1_CKPT'" \
  callback.best_epoch_ckpt.monitor=free_run_l1/val \
  ~callback.early_stopping \
  trainer.devices=2 \
  +trainer.strategy=ddp_find_unused_parameters_true \
  "$@" \
  2>&1 | tee -a outputs/dualtalk_speak_launch.log
