#!/usr/bin/env bash
# Listen+speak orchestration (runs in tmux session `ls_pipeline`):
#   1. wait for Mimi codes + Whisper (tmux ls_mimi*/ls_whisper*), build the
#      Moshi text tokens, check every track has all features
#   2. wait for Stage-1 (tmux ls_stage1) to end -- early stopping, max epochs,
#      or `tmux kill-session -t ls_stage1` once you are happy with it
#   3. speaking model (run_dualtalk_speak.sh) on Stage-1's best checkpoint
#   4. Option A and Option B side by side (tmux ls_unils / ls_controlnet)
set -uo pipefail
cd /home/mpate169/Mayur/LiveFace_listen_speak/project
N=11526        # tracks: (4610 + 243 + 533 + 377) conversations x 2
log() { echo "[$(date '+%F %T')] $*"; }

running() { tmux list-sessions -F '#S' 2>/dev/null | grep -qx "$1"; }

# ---- 1. audio features ------------------------------------------------- #
while running ls_mimi0 || running ls_mimi1 || running ls_whisper0 || running ls_whisper1; do sleep 300; done
log "Mimi codes: $(ls datasets/dualtalk/mimi_codes | wc -l)/$N  Whisper: $(ls datasets/dualtalk/whisper | wc -l)/$N"
/home/mpate169/miniconda3/envs/miburi/bin/python pre_process/dualtalk_text_tokens.py --overwrite
for d in mimi_codes whisper moshi_text_codes; do
  n=$(ls datasets/dualtalk/$d | wc -l)
  if [ "$n" -lt "$N" ]; then log "ERROR: only $n/$N files in datasets/dualtalk/$d -- stopping"; exit 1; fi
done
log "audio features complete"

# ---- 2. Stage-1 ---------------------------------------------------------- #
while running ls_stage1; do sleep 600; done
STAGE1=$(ls -d outputs/DualTalk/dualtalk_vqvae_prior/*/ | head -1)
log "Stage-1 finished: $STAGE1 ($(ls "$STAGE1"/tensorboard/*/checkpoints/best-epoch=*.ckpt))"

# ---- 3. speaking model ---------------------------------------------------- #
log "starting speaking model"
STAGE1=$STAGE1 ./run_dualtalk_speak.sh
SPEAK=$(ls -dt outputs/DualTalk/dualtalk_speak/*/ | head -1)
log "speaking model finished: $SPEAK"

# ---- 4. Option A + Option B ----------------------------------------------- #
tmux new-session -d -s ls_unils "STAGE1=$STAGE1 SPEAK=$SPEAK ./run_dualtalk_unils.sh"
tmux new-session -d -s ls_controlnet "STAGE1=$STAGE1 SPEAK=$SPEAK ./run_dualtalk_controlnet.sh"
log "launched Option A (tmux ls_unils) and Option B (tmux ls_controlnet)"
