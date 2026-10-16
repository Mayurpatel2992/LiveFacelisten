#!/usr/bin/env bash
# Seamless orchestration (tmux session `si_pipeline`, log outputs/seamless_pipeline.log):
#   1. wait for Mimi codes + Whisper (tmux si_mimi*/si_whisper*), build the Moshi text
#      tokens + pack all codes, check every track has all features
#   2. wait for Stage-1 (tmux si_stage1) to end -- early stopping, max epochs, or
#      `tmux kill-session -t si_stage1` once you are happy with it
#   3. speaking model (run_seamless_speak.sh) on Stage-1's best checkpoint, GPUs 0,1
#   4. Option A (GPUs 0,1) and Option B (GPUs 4,5) side by side
#   5. test-set generation + UniLS metrics for the speaking model, A and B
set -uo pipefail
cd /home/mpate169/Mayur/LiveFace_listen_speak/project
E=/home/mpate169/miniconda3/envs
# Shared 128-core machine: cap CPU threads per process (uncapped, ~16 jobs pushed the
# load average to ~850 on 2026-10-14 and every job on the node was killed).
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4} NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-4}
log() { echo "[$(date '+%F %T')] $*"; }
running() { tmux list-sessions -F '#S' 2>/dev/null | grep -q "^$1"; }

# ---- 1. audio features ------------------------------------------------- #
while running si_mimi || running si_whisper; do sleep 300; done
for split in val test train; do
  N=$($E/liveface/bin/python3.9 -c "import json; print(len(json.load(open('datasets/seamless/index_$split.json'))))")
  for d in mimi_codes whisper; do
    n=$(ls datasets/seamless/$d/$split | wc -l)
    if [ "$n" -lt "$N" ]; then log "ERROR: only $n/$N $split files in datasets/seamless/$d -- stopping"; exit 1; fi
  done
done
$E/liveface/bin/python3.9 pre_process/seamless_pack_codes.py --splits val test train || { log "ERROR: packing failed"; exit 1; }
log "audio features complete"

# ---- 2. Stage-1 ---------------------------------------------------------- #
while running si_stage1; do sleep 600; done
STAGE1=$(ls -dt outputs/Seamless/seamless_vqvae_prior/*/ | head -1)
log "Stage-1 finished: $STAGE1 ($(ls "$STAGE1"/tensorboard/*/checkpoints/best-epoch=*.ckpt))"

# ---- 3. speaking model ---------------------------------------------------- #
log "starting speaking model"
STAGE1=$STAGE1 CUDA_VISIBLE_DEVICES=0,1 ./run_seamless_speak.sh trainer.enable_progress_bar=false
SPEAK=$(ls -dt outputs/Seamless/seamless_speak/*/ | head -1)
log "speaking model finished: $SPEAK"

# ---- 4. Option A + Option B ----------------------------------------------- #
tmux new-session -d -s si_unils "STAGE1=$STAGE1 SPEAK=$SPEAK CUDA_VISIBLE_DEVICES=0,1 ./run_seamless_unils.sh trainer.enable_progress_bar=false"
tmux new-session -d -s si_controlnet "STAGE1=$STAGE1 SPEAK=$SPEAK CUDA_VISIBLE_DEVICES=4,5 ./run_seamless_controlnet.sh trainer.enable_progress_bar=false"
log "launched Option A (tmux si_unils) and Option B (tmux si_controlnet)"
./run_seamless_eval.sh "$SPEAK" "6 7"
log "speaking model scored"
while running si_unils || running si_controlnet; do sleep 600; done

# ---- 5. evaluation -------------------------------------------------------- #
for exp in seamless_unils seamless_controlnet; do
  RUN=$(ls -dt outputs/Seamless/$exp/*/ | head -1)
  log "evaluating $RUN"
  ./run_seamless_eval.sh "$RUN" "0 1 4 5 6 7"
done
log "pipeline done; metrics in outputs/seamless_eval/*_metrics.txt"
