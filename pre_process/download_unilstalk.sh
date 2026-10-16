#!/bin/bash
# Seamless Interaction part of UniLS-Talk (HF xg-chu/UniLSTalkDataset): FLAME 108-D
# motion + 16 kHz audio + speech_mask per track, one LMDB (~130 GB). hf_xet fetches
# chunks in parallel (~300 MB/s here vs ~8 MB/s for plain curl). Run in the `mimi` env
# (huggingface_hub 0.36 + hf_xet):
#   conda activate mimi && ./pre_process/download_unilstalk.sh
set -euo pipefail
HF_XET_HIGH_PERFORMANCE=1 python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["seamless_interaction_pairs.json", "seamless_interaction_stats.json",
          "data_lmdb/lock.mdb", "data_lmdb/data.mdb"]:
    print(hf_hub_download("xg-chu/UniLSTalkDataset", f"SeamlessInteractionTalk/{f}",
                          repo_type="dataset", local_dir="datasets/unilstalk"), flush=True)
PY
