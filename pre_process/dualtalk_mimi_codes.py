"""Frozen-Mimi RVQ codes for every DualTalk track (own + partner streams).

Same encoder path as pre_process/precompute_mimi_codes.py (16 kHz -> 24 kHz ->
MimiModel.encode -> first 8 codebooks, 12.5 Hz), except for the level
normalisation: MEAD min-max scales each clip to [-1, 1], which here would blow
a mostly-silent listener track (partner bleed + noise) up to full scale. The two
tracks of a conversation are instead DC-removed and divided by their *joint*
peak, so the speaker/listener level difference is kept.

Saved: datasets/dualtalk/mimi_codes/<conv>_speakerX.npy   int16 [8, T_mimi]

Run in the `miburi` env. Shard across GPUs with --shard i --num_shards n.
"""
import argparse
import json
import os
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio
from transformers import MimiModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/dualtalk")
    ap.add_argument("--out_dir", default="datasets/dualtalk/mimi_codes")
    ap.add_argument("--model", default="kyutai/mimi")
    ap.add_argument("--num_codebooks", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--src_sr", type=int, default=16000)
    ap.add_argument("--mimi_sr", type=int, default=24000)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    with open(root / "splits" / "source_dirs.json") as f:
        src = json.load(f)
    jobs = []
    for split, d in src.items():
        for conv in (root / "splits" / split).read_text().split():
            jobs.append(root / "raw" / d / conv)
    jobs = sorted(set(jobs))[args.shard::args.num_shards]

    dev = args.device if torch.cuda.is_available() else "cpu"
    mimi = MimiModel.from_pretrained(args.model).to(dev).eval()
    assert mimi.config.sampling_rate == args.mimi_sr
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"shard {args.shard}/{args.num_shards}: {len(jobs)} conversations -> {out_dir}")

    deltas, t0, n_done = [], time.time(), 0
    for i, base in enumerate(jobs):
        names = [f"{base.name}_speaker1", f"{base.name}_speaker2"]
        dsts = [out_dir / f"{n}.npy" for n in names]
        if all(d.exists() for d in dsts) and not args.overwrite:
            continue
        waves = [librosa.load(str(base.parent / f"{n}.wav"), sr=args.src_sr)[0] for n in names]
        waves = [w - w.mean() for w in waves]
        peak = max(np.abs(w).max() for w in waves) + 1e-8
        for name, w, dst in zip(names, waves, dsts):
            x = torch.from_numpy((w / peak).astype("float32"))
            x = torchaudio.functional.resample(x, args.src_sr, args.mimi_sr).to(dev)[None, None]
            with torch.no_grad():
                codes = mimi.encode(x).audio_codes[0, :args.num_codebooks]
            codes = codes.to(torch.int16).cpu().numpy()                     # [K, T_mimi]
            np.save(dst, codes)
            T = np.load(base.parent / f"{name}.npz")["exp"].shape[0]
            deltas.append(2 * codes.shape[1] - T)
        n_done += 1
        if (i + 1) % 200 == 0:
            el = time.time() - t0
            print(f"  [{i + 1}/{len(jobs)}] {el:.0f}s, ~{el / n_done * (len(jobs) - i - 1):.0f}s left", flush=True)

    print(f"DONE {n_done} conversations in {time.time() - t0:.0f}s")
    if deltas:
        d = np.array(deltas)
        print(f"2*T_mimi - T_motion: min={d.min()} median={np.median(d):.0f} max={d.max()}  "
              f"|delta|<=2: {(np.abs(d) <= 2).mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
