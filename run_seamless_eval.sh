#!/usr/bin/env bash
# Generate the Seamless test set (speaker A of the 3457 protocol pairs) with one trained
# run, one shard per GPU, then score it with seamless_metrics.py (UniLS metric code).
#   ./run_seamless_eval.sh outputs/Seamless/seamless_controlnet/<id> "0 1 4 5" [extra hydra args]
set -euo pipefail
cd /home/mpate169/Mayur/LiveFace_listen_speak/project
FOLDER=$(realpath "$1"); GPUS=($2); shift 2
E=/home/mpate169/miniconda3/envs
# Shared 128-core machine: cap CPU threads per process (uncapped, ~16 jobs pushed the
# load average to ~850 on 2026-10-14 and every job on the node was killed).
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4} NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-4}
mkdir -p outputs/seamless_eval
NAME=$(basename "$(dirname "$FOLDER")")_$(basename "$FOLDER")
pids=()
for s in "${!GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES=${GPUS[$s]} $E/liveface/bin/python3.9 evaluate_seamless.py folder="$FOLDER" device=0 \
    shard=$s num_shards=${#GPUS[@]} "$@" > outputs/seamless_eval/${NAME}_shard$s.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
OUT=$(grep -h "outputs:" outputs/seamless_eval/${NAME}_shard0.log | tail -1 | awk '{print $NF}')
CUDA_VISIBLE_DEVICES=${GPUS[0]} $E/ARTalk/bin/python seamless_metrics.py --pred "$OUT" | tee outputs/seamless_eval/${NAME}_metrics.txt
