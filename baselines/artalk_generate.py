"""Released ARTalk (wav2vec) on the Seamless test split: speaker A's own audio only.

ARTalk is a speaking model, so like FacePlex's ARTalk row it is driven by the agent's own
track for the whole clip (its listening stretches included). Its default inference
(ARTAvatarInferEngine.inference: no style motion, temperature 0, Savitzky-Golay smoothing)
gives 106-D motion = exp 100 | head rot 3 | jaw 3 (only the first jaw value is non-zero),
mapped to the protocol's 108-D: exp 100 | head rot 3 | jaw 1 | gaze 4 = 0 (no gaze).
Head rotation is zero-centred per clip, as the ground truth is (it only affects L_FID;
vertex metrics use no head pose, PDD uses std).

    cd ~/Mayur/ARTalk && python ~/Mayur/LiveFace_listen_speak/baselines/artalk_generate.py --shard 0 --num_shards 2
Run in the `ARTalk` env. Outputs: results/seamless/artalk/ARTalk_wav2vec/test/<trackA>.npy
"""
import argparse
import io
import json
import os
import sys
from pathlib import Path

import lmdb
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/unilstalk/SeamlessInteractionTalk"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--max_clips", type=int, default=None)
    args = ap.parse_args()

    sys.path.insert(0, os.getcwd())
    from inference import ARTAvatarInferEngine

    torch.manual_seed(42 + args.shard)
    engine = ARTAvatarInferEngine(load_gaga=False, fix_pose=False, device="cuda")
    txn = lmdb.open(str(DATA / "data_lmdb"), readonly=True, lock=False, readahead=False).begin()
    meta = json.load(open(DATA / "seamless_interaction_pairs.json"))["test"]
    pairs = [(Path(a).name, n) for a, b, n in meta.values() if n > 101]
    pairs = pairs[args.shard::args.num_shards][:args.max_clips]
    out_dir = ROOT / "results/seamless/artalk/ARTalk_wav2vec/test"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, (key, T) in enumerate(pairs):
        if (out_dir / f"{key}.npy").exists():
            continue
        w = np.load(io.BytesIO(txn.get(key.encode())), allow_pickle=False)["audio"].astype(np.float32)
        w = np.pad(w, (0, max(0, T * 640 - w.shape[0])))[:T * 640]
        with torch.no_grad():
            m = engine.inference(torch.from_numpy(w), clip_length=T).float().cpu().numpy()   # [T,106]
        if m.shape[0] != T:
            raise RuntimeError(f"{key}: ARTalk gave {m.shape[0]} frames, expected {T}")
        out = np.zeros((T, 108), dtype=np.float32)
        out[:, :103] = m[:, :103]
        out[:, 103] = m[:, 103]
        out[:, 100:103] -= out[:, 100:103].mean(axis=0, keepdims=True)
        np.save(out_dir / f"{key}.npy", out)
        if (i + 1) % 200 == 0:
            print(f"[{i + 1}/{len(pairs)}]", flush=True)
    print(f"done: {len(pairs)} clips -> {out_dir}")


if __name__ == "__main__":
    main()
