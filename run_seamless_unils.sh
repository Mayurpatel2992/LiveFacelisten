#!/usr/bin/env bash
# Seamless (UniLS protocol), step 3: Option A (UniLS-style): + own/partner causal cross-attention, LoRA on self-attn/FFN.
# Fine-tunes from the trained Seamless speaking model on own + partner tracks.
#   STAGE1=outputs/Seamless/seamless_vqvae_prior/<id> \
#   SPEAK=outputs/Seamless/seamless_speak/<id> ./run_seamless_unils.sh
# (best checkpoints of both runs unless STAGE1_CKPT / SPEAK_CKPT are given)
set -euo pipefail
cd /home/mpate169/Mayur/LiveFace_listen_speak/project
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate liveface
# Shared 128-core machine: cap CPU threads per process (uncapped, ~16 jobs pushed the
# load average to ~850 on 2026-10-14 and every job on the node was killed).
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8} NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-8}
: "${STAGE1:?set STAGE1 to the Stage-1 run folder}" "${SPEAK:?set SPEAK to the speaking-model run folder}"
STAGE1=$(realpath "$STAGE1"); SPEAK=$(realpath "$SPEAK")
STAGE1_CKPT=${STAGE1_CKPT:-$(ls "$STAGE1"/tensorboard/*/checkpoints/best-epoch=*.ckpt | tail -1)}
# final speaking-model weights (its best-epoch file was picked by a noisy 4-clip free-run
# L1 and stayed at epoch 0); fixed-budget final checkpoint, as in FacePlex / UniLS
SPEAK_CKPT=${SPEAK_CKPT:-$(ls -t "$SPEAK"/tensorboard/*/checkpoints/latest-epoch=*.ckpt | head -1)}
echo "Stage-1: $STAGE1_CKPT"; echo "speaking model: $SPEAK_CKPT"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export MASTER_PORT=${MASTER_PORT:-29731}

python3.9 train_all.py \
  experiment=seamless_unils \
  state=new \
  data=seamless_pred_moshi_text \
  data.load_partner=true \
  model=model_dualtalk_unils \
  "model.duplex.speak_ckpt_path='$SPEAK_CKPT'" \
  model.folder_prior=$STAGE1 \
  model.version_prior=0 \
  "model.ckpt_path_prior='$STAGE1_CKPT'" \
  callback.best_epoch_ckpt.monitor=total/val \
  ~callback.early_stopping \
  model.loss.exp_dims=100 \
  trainer.devices=2 \
  +trainer.limit_train_batches=600 \
  +trainer.strategy=ddp_find_unused_parameters_true \
  "$@" \
  2>&1 | tee -a outputs/seamless_unils_launch.log
