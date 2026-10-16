#!/usr/bin/env bash
# Speaking model 2uxmfnxt, epoch-99 checkpoint: generate + score test and OOD.
set -uo pipefail
cd /home/mpate169/Mayur/LiveFace_listen_speak/project
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate probtalk3d
F=outputs/DualTalk/dualtalk_speak/2uxmfnxt
for s in test ood; do
  python3.9 evaluate_dualtalk.py folder=$F ckpt=$PWD/outputs/speak_best_ep99.ckpt split=$s tag=_ep99 device=${DEV:-0}
  python3.9 dualtalk_metrics.py --pred results/dualtalk/dualtalk_speak/2uxmfnxt/${s}_ep99 --split $s
done
