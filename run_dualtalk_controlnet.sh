#!/usr/bin/env bash
# Listen+speak, step 3: Option B (ControlNet): frozen speaking model + trainable listening copy, gated.
# Fine-tunes from the trained DualTalk speaking model on own + partner tracks.
#   STAGE1=outputs/DualTalk/dualtalk_vqvae_prior/<id> \
#   SPEAK=outputs/DualTalk/dualtalk_speak/<id> ./run_dualtalk_controlnet.sh
# (best checkpoints of both runs unless STAGE1_CKPT / SPEAK_CKPT are given)
set -euo pipefail
cd /home/mpate169/Mayur/LiveFace_listen_speak/project
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate probtalk3d
: "${STAGE1:?set STAGE1 to the Stage-1 run folder}" "${SPEAK:?set SPEAK to the speaking-model run folder}"
STAGE1=$(realpath "$STAGE1"); SPEAK=$(realpath "$SPEAK")
STAGE1_CKPT=${STAGE1_CKPT:-$(ls "$STAGE1"/tensorboard/*/checkpoints/best-epoch=*.ckpt | tail -1)}
SPEAK_CKPT=${SPEAK_CKPT:-$(ls "$SPEAK"/tensorboard/*/checkpoints/best-epoch=*.ckpt | tail -1)}
echo "Stage-1: $STAGE1_CKPT"; echo "speaking model: $SPEAK_CKPT"
export MASTER_PORT=${MASTER_PORT:-29641}

python3.9 train_all.py \
  experiment=dualtalk_controlnet \
  state=new \
  data=dualtalk_pred_moshi_text \
  data.load_partner=true \
  model=model_dualtalk_controlnet \
  "model.duplex.speak_ckpt_path='$SPEAK_CKPT'" \
  model.folder_prior=$STAGE1 \
  model.version_prior=0 \
  "model.ckpt_path_prior='$STAGE1_CKPT'" \
  callback.best_epoch_ckpt.monitor=free_run_l1/val \
  ~callback.early_stopping \
  trainer.devices=2 \
  +trainer.strategy=ddp_find_unused_parameters_true \
  "$@" \
  2>&1 | tee -a outputs/dualtalk_controlnet_launch.log
