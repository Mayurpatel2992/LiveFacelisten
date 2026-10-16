#!/bin/bash
# Mimi codes (GPU 4, 2 shards) + Whisper word timings (GPUs 5-7, 4 shards each) for
# every Seamless track; GPUs 2-3 left to other users. Each shard skips finished tracks,
# so this can be re-run after an interruption.  Then: pre_process/seamless_pack_codes.py
cd /home/mpate169/Mayur/LiveFace_listen_speak/project
E=/home/mpate169/miniconda3/envs
# Shared 128-core machine: cap CPU threads per process (uncapped, ~16 jobs pushed the
# load average to ~850 on 2026-10-14 and every job on the node was killed).
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4} NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-4}
mkdir -p outputs/seamless_features
for i in 0 1; do
  tmux new-session -d -s si_mimi$i "CUDA_VISIBLE_DEVICES=4 $E/mimi/bin/python pre_process/seamless_mimi_codes.py \
    --splits val test train --shard $i --num_shards 2 > outputs/seamless_features/mimi_$i.log 2>&1"
done
s=0
for g in 5 6 7; do
  for j in 0 1 2 3; do
    tmux new-session -d -s si_whisper$s "CUDA_VISIBLE_DEVICES=$g $E/liveface/bin/python3.9 pre_process/seamless_whisper.py \
      --splits val test train --shard $s --num_shards 12 > outputs/seamless_features/whisper_$s.log 2>&1"
    s=$((s + 1))
  done
done
