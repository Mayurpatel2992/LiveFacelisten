"""Precompute frozen-Mimi RVQ code streams for every MEAD clip.

Mimi is frozen in all three experiments, so its output is a deterministic
function of the input waveform -- caching the codes is mathematically identical
to running a frozen Mimi encoder inside the training graph, and it keeps the
Python 3.9 LiveFace training env (transformers 4.10, no MimiModel) untouched.

Audio front-end is kept identical to the HuBERT baseline so the ablation only
changes the *encoder*, not the signal:

    wav (48 kHz on disk)
      -> librosa.load(sr=16000)                 # same as framework/data/mead.py
      -> minmax_scale to [-1, 1]                # same as collate.audio_normalize
      -> torchaudio resample 16 kHz -> 24 kHz   # Mimi's expected rate
      -> MimiModel.encode(...).audio_codes      # [1, 32, T_mimi] @ 12.5 Hz
      -> keep first `num_codebooks` (=8)        # MIBURI uses emb[:8]

Saved: datasets/mead/mimi_codes/<keyid>.npy   int16   shape [K, T_mimi]

Run in the `miburi` conda env.  ~30 min for the full MEAD set on one GPU.
"""
import argparse
import glob
import os
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio
from sklearn.preprocessing import minmax_scale
from transformers import MimiModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_dir", default="datasets/mead/wav")
    ap.add_argument("--motion_dir", default="datasets/mead/param",
                    help="Only clips with a matching motion file are encoded "
                         "(these are the only ones the dataset loads).")
    ap.add_argument("--out_dir", default="datasets/mead/mimi_codes")
    ap.add_argument("--model", default="kyutai/mimi")
    ap.add_argument("--num_codebooks", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--src_sr", type=int, default=16000,
                    help="Intermediate rate (matches the HuBERT-path audio).")
    ap.add_argument("--mimi_sr", type=int, default=24000)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    mimi = MimiModel.from_pretrained(args.model).to(dev).eval()
    fr = float(mimi.config.frame_rate)
    print(f"Mimi loaded: num_quantizers={mimi.config.num_quantizers} "
          f"codebook_size={mimi.config.codebook_size} frame_rate={fr} "
          f"sampling_rate={mimi.config.sampling_rate}")
    assert mimi.config.sampling_rate == args.mimi_sr

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stems = sorted(Path(p).stem for p in glob.glob(os.path.join(args.motion_dir, "*.npy")))
    if args.limit:
        stems = stems[:args.limit]
    print(f"{len(stems)} clips to encode -> {out_dir}")

    deltas = []          # 2*T_mimi - motion_len  (how much the held feature is cropped)
    n_done = n_skip = n_miss = 0
    t0 = time.time()
    for i, stem in enumerate(stems):
        dst = out_dir / f"{stem}.npy"
        if dst.exists() and not args.overwrite:
            n_skip += 1
            continue
        wav_path = os.path.join(args.wav_dir, f"{stem}.wav")
        if not os.path.exists(wav_path):
            n_miss += 1
            continue

        w, _ = librosa.load(wav_path, sr=args.src_sr)
        w = minmax_scale(w, feature_range=(-1, 1)).astype("float32")
        w = torch.from_numpy(w)
        if args.mimi_sr != args.src_sr:
            w = torchaudio.functional.resample(w, args.src_sr, args.mimi_sr)
        w = w.to(dev)[None, None]                        # [1, 1, samples]

        with torch.no_grad():
            codes = mimi.encode(w).audio_codes          # [1, 32, T_mimi]
        codes = codes[0, :args.num_codebooks].to(torch.int16).cpu().numpy()   # [K, T_mimi]
        np.save(dst, codes)
        n_done += 1

        m_len = int(np.load(os.path.join(args.motion_dir, f"{stem}.npy")).shape[1])
        deltas.append(2 * codes.shape[1] - m_len)

        if (i + 1) % 500 == 0:
            el = time.time() - t0
            print(f"  [{i + 1}/{len(stems)}] done={n_done} skip={n_skip} miss={n_miss} "
                  f"| {el:.0f}s | ~{el / max(n_done, 1) * (len(stems) - i - 1):.0f}s left")

    print(f"\nDONE  encoded={n_done} skipped={n_skip} missing_wav={n_miss}  "
          f"in {time.time() - t0:.0f}s")
    if deltas:
        d = np.array(deltas)
        print("Temporal alignment delta  (2*T_mimi - motion_len), over newly encoded clips:")
        print(f"  min={d.min()}  p05={np.percentile(d,5):.0f}  median={np.median(d):.0f}  "
              f"p95={np.percentile(d,95):.0f}  max={d.max()}")
        print(f"  |delta|<=2 : {(np.abs(d) <= 2).mean() * 100:.1f}%   "
              f"delta>=0 (feature covers motion): {(d >= 0).mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
