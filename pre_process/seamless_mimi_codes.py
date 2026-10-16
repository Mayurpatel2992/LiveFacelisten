"""Frozen-Mimi RVQ codes for every Seamless track (own + partner streams).

Same encoder path and level normalisation as pre_process/dualtalk_mimi_codes.py
(16 kHz -> 24 kHz -> MimiModel.encode -> first 8 codebooks, 12.5 Hz; the two tracks of a
pair DC-removed and divided by their joint peak). Audio comes from the UniLS-Talk LMDB,
cut / zero-padded to length * 640 samples like UniLS's loader.

Saved: datasets/seamless/mimi_codes/<split>/<track>.npy   int16 [8, T_mimi]
Run in the `mimi` env after seamless_prepare.py. Shard with --shard i --num_shards n.
"""
import argparse
import io
import json
import time
from pathlib import Path

import lmdb
import numpy as np
import torch
import torchaudio
from transformers import MimiModel

SRC = Path("datasets/unilstalk/SeamlessInteractionTalk")
ROOT = Path("datasets/seamless")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "test", "train"])
    ap.add_argument("--model", default="kyutai/mimi")
    ap.add_argument("--num_codebooks", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    args = ap.parse_args()

    txn = lmdb.open(str(SRC / "data_lmdb"), readonly=True, lock=False, readahead=False).begin()
    mimi = MimiModel.from_pretrained(args.model).to(args.device).eval()
    assert mimi.config.sampling_rate == 24000

    for split in args.splits:
        index = json.load(open(ROOT / f"index_{split}.json"))
        pairs = sorted((k, v[2], v[1]) for k, v in index.items() if v[3] == 1)[args.shard::args.num_shards]
        out_dir = ROOT / "mimi_codes" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"{split} shard {args.shard}/{args.num_shards}: {len(pairs)} pairs", flush=True)
        t0, deltas = time.time(), []
        for i, (a, b, T) in enumerate(pairs):
            dsts = [out_dir / f"{a}.npy", out_dir / f"{b}.npy"]
            if all(d.exists() for d in dsts):
                continue
            waves = []
            for key in (a, b):
                w = np.load(io.BytesIO(txn.get(key.encode())), allow_pickle=False)["audio"].astype(np.float32)
                w = np.pad(w, (0, max(0, T * 640 - w.shape[0])))[:T * 640]
                waves.append(w - w.mean())
            peak = max(np.abs(w).max() for w in waves) + 1e-8
            for w, dst in zip(waves, dsts):
                x = torchaudio.functional.resample(torch.from_numpy(w / peak), 16000, 24000)
                with torch.no_grad():
                    codes = mimi.encode(x.to(args.device)[None, None]).audio_codes[0, :args.num_codebooks]
                np.save(dst, codes.to(torch.int16).cpu().numpy())
                deltas.append(2 * codes.shape[1] - T)
            if (i + 1) % 2000 == 0:
                el = time.time() - t0
                print(f"  [{i + 1}/{len(pairs)}] {el:.0f}s, ~{el / (i + 1) * (len(pairs) - i - 1):.0f}s left", flush=True)
        if deltas:
            d = np.array(deltas)
            print(f"{split}: done in {time.time() - t0:.0f}s; 2*T_mimi - T: min={d.min()} "
                  f"median={np.median(d):.0f} max={d.max()}", flush=True)


if __name__ == "__main__":
    main()
