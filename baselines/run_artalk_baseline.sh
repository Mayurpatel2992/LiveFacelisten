#!/usr/bin/env bash
# Released ARTalk on the 3457 Seamless test clips (speaker A's own audio), then seamless_metrics.py.
#   ./baselines/run_artalk_baseline.sh [gpu]
set -euo pipefail
GPU=${1:-4}
E=/home/mpate169/miniconda3/envs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
L=/home/mpate169/Mayur/LiveFace_listen_speak/project
mkdir -p $L/outputs/seamless_eval
(cd /home/mpate169/Mayur/ARTalk && CUDA_VISIBLE_DEVICES=$GPU $E/ARTalk/bin/python $L/baselines/artalk_generate.py) \
  2>&1 | grep -v "Inferring\|Done!\|No style motion" > $L/outputs/seamless_eval/artalk_generate.log
cd $L && CUDA_VISIBLE_DEVICES=$GPU $E/ARTalk/bin/python seamless_metrics.py --pred results/seamless/artalk/ARTalk_wav2vec/test \
  | tee outputs/seamless_eval/artalk_metrics.txt
