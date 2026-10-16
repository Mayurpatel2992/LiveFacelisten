#!/usr/bin/env bash
# Speaking model 2uxmfnxt ep99: why is generated motion flat? 100 test tracks per variant.
set -uo pipefail
cd /home/mpate169/Mayur/LiveFace_listen_speak/project
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate probtalk3d
F=outputs/DualTalk/dualtalk_speak/2uxmfnxt; CK=$PWD/outputs/speak_best_ep99.ckpt
run() { tag=$1; shift
  python3.9 evaluate_dualtalk.py folder=$F ckpt=$CK split=test max_tracks=100 device=1 tag=_diag_$tag "$@" > outputs/diag_$tag.log 2>&1
  python3.9 dualtalk_metrics.py --subset --pred results/dualtalk/dualtalk_speak/2uxmfnxt/test_diag_$tag --split test >> outputs/diag_$tag.log 2>&1; }
run greedy token_temperature=0 &
run t07 token_temperature=0.7 &
run t10 token_temperature=1.0 &
run greedy_steps25 token_temperature=0 sampling_steps=25 &
run tf_greedy teacher_forced=true token_temperature=0 &
run tf_t10 teacher_forced=true token_temperature=1.0 &
wait; echo ALL DONE
