#!/usr/bin/env bash
# Speaking model on the chosen Stage-1 checkpoint (epoch 69), then Option A + B.
set -uo pipefail
cd /home/mpate169/Mayur/LiveFace_listen_speak/project
export STAGE1=outputs/DualTalk/dualtalk_vqvae_prior/2u9pgvce
export STAGE1_CKPT=$PWD/outputs/stage1_best_ep69.ckpt   # copy of "best-epoch=69 copy.ckpt"
log() { echo "[$(date '+%F %T')] $*"; }
log "speaking model, Stage-1 = $STAGE1_CKPT"
./run_dualtalk_speak.sh
SPEAK=$(ls -dt outputs/DualTalk/dualtalk_speak/*/ | head -1)
log "speaking model finished: $SPEAK"
tmux new-session -d -s ls_unils "STAGE1=$STAGE1 STAGE1_CKPT=$STAGE1_CKPT SPEAK=$SPEAK ./run_dualtalk_unils.sh"
tmux new-session -d -s ls_controlnet "STAGE1=$STAGE1 STAGE1_CKPT=$STAGE1_CKPT SPEAK=$SPEAK ./run_dualtalk_controlnet.sh"
log "launched Option A (ls_unils) and Option B (ls_controlnet)"
