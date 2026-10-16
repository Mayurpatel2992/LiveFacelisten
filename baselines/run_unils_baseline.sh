#!/usr/bin/env bash
# Released UniLS checkpoint on the 3457 Seamless test clips (its eval.py loop, 2 shards on
# one GPU), then seamless_metrics.py on the saved predictions.
#   ./baselines/run_unils_baseline.sh [UniLSLoRAGen|UniLSLoRAGenS] [gpu]
set -euo pipefail
cd /home/mpate169/Mayur/LiveFace_listen_speak/project
CK=${1:-UniLSLoRAGen}; GPU=${2:-4}
E=/home/mpate169/miniconda3/envs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p outputs/seamless_eval
pids=()
for s in 0 1; do
  (cd deps/UniLS && CUDA_VISIBLE_DEVICES=$GPU $E/unils/bin/python ../../baselines/unils_generate.py \
     -r checkpoints/$CK.pt --shard $s --num_shards 2) > outputs/seamless_eval/unils_${CK}_shard$s.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
CUDA_VISIBLE_DEVICES=$GPU $E/ARTalk/bin/python seamless_metrics.py --pred results/seamless/unils_released/$CK/test \
  | tee outputs/seamless_eval/unils_${CK}_metrics.txt
