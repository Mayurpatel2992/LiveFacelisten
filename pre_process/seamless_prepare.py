"""Seamless Interaction (UniLS-Talk LMDB) -> packed motion / speech-mask arrays per split.

Pairs and tracks follow UniLS (deps/UniLS/core/data/sitalk_data.py): every split keeps
the pairs of seamless_interaction_pairs.json with length > 101 frames (so test = the
UniLS / FacePlex evaluation set). Each pair = two tracks of one 2-16 s window, A and B.
Per track:
    motion  108-D = exp 100 | head rot 3 | jaw 1 | gaze 4   (motioncode[:, :104], [106:108],
            [109:111]), head rotation zero-centred over the clip, raw (not normalised)
    vad     speech_mask, 1 = this track's speaker is talking (also the metric's mask)

Saved under datasets/seamless/:
    motion_<split>.npy   float32 [N_frames, 108]   (np.load(mmap_mode='r'))
    vad_<split>.npy      uint8   [N_frames]
    index_<split>.json   {track: [offset, length, partner, is_a]}
    norm.json            per-dim train 0.1 / 99.9 percentiles (lo / hi, min-max scaling)
                         + mean / std

    python3.9 pre_process/seamless_prepare.py --workers 16
Run in the `liveface` env (lmdb).
"""
import argparse
import io
import json
import time
from multiprocessing import Pool
from pathlib import Path

import lmdb
import numpy as np

SRC = Path("datasets/unilstalk/SeamlessInteractionTalk")
OUT = Path("datasets/seamless")
MIN_LEN = 101
_txn = None


def _open():
    global _txn
    if _txn is None:
        _txn = lmdb.open(str(SRC / "data_lmdb"), readonly=True, lock=False, readahead=False).begin()
    return _txn


def load_record(key):
    return np.load(io.BytesIO(_open().get(key.encode())), allow_pickle=False)


def select_motion(motioncode):
    m = np.concatenate([motioncode[:, :104], motioncode[:, 106:108], motioncode[:, 109:111]], axis=1)
    m = m.astype(np.float32)
    m[:, 100:103] -= m[:, 100:103].mean(axis=0, keepdims=True)
    return m


def work(job):
    split, key, offset, n = job
    rec = load_record(key)
    m, v = select_motion(rec["motioncode"]), rec["speech_mask"].reshape(-1)
    T = min(n, m.shape[0], v.shape[0])
    mot = np.load(OUT / f"motion_{split}.npy", mmap_mode="r+")
    vad = np.load(OUT / f"vad_{split}.npy", mmap_mode="r+")
    mot[offset:offset + T] = m[:T]
    vad[offset:offset + T] = (v[:T] > 0.5).astype(np.uint8)
    mot.flush(); vad.flush()
    return key, T, m.shape[0], v.shape[0], rec["audio"].shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--splits", nargs="+", default=["val", "test", "train"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(SRC / "seamless_interaction_pairs.json"))

    for split in args.splits:
        t0 = time.time()
        index, jobs, offset = {}, [], 0
        for a, b, n in meta[split].values():
            if n <= MIN_LEN:
                continue
            a, b = Path(a).name, Path(b).name
            for key, partner, is_a in ((a, b, 1), (b, a, 0)):
                assert key not in index, f"{key} in two pairs"
                index[key] = [offset, n, partner, is_a]
                jobs.append((split, key, offset, n))
                offset += n
        np.lib.format.open_memmap(OUT / f"motion_{split}.npy", "w+", np.float32, (offset, 108))
        np.lib.format.open_memmap(OUT / f"vad_{split}.npy", "w+", np.uint8, (offset,))
        print(f"{split}: {len(jobs)} tracks, {offset} frames ({offset / 25 / 3600:.1f} h)", flush=True)

        bad = []
        with Pool(args.workers) as pool:
            for i, (key, T, Tm, Tv, Ta) in enumerate(pool.imap_unordered(work, jobs, chunksize=16)):
                n = index[key][1]
                if T != n or Ta != n * 640:
                    bad.append((key, n, Tm, Tv, Ta))
                index[key][1] = T
                if (i + 1) % 20000 == 0:
                    print(f"  [{i + 1}/{len(jobs)}] {time.time() - t0:.0f}s", flush=True)
        json.dump(index, open(OUT / f"index_{split}.json", "w"))
        print(f"{split}: done in {time.time() - t0:.0f}s; {len(bad)} tracks whose motion / mask / "
              f"audio length differs from the listed one (first: {bad[:3]})", flush=True)

        if split == "train":
            mot = np.load(OUT / "motion_train.npy", mmap_mode="r")
            sub = np.asarray(mot[::10])
            lo, hi = np.percentile(sub, 0.1, axis=0), np.percentile(sub, 99.9, axis=0)
            norm = {"lo": lo.tolist(), "hi": hi.tolist(),
                    "mean": sub.mean(0).tolist(), "std": sub.std(0).tolist(),
                    "frames_used": int(sub.shape[0]), "layout": "exp 100 | head rot 3 | jaw 1 | gaze 4"}
            json.dump(norm, open(OUT / "norm.json", "w"), indent=1)
            print(f"norm.json: hi - lo min {np.min(hi - lo):.4g}, max {np.max(hi - lo):.4g}")


if __name__ == "__main__":
    main()
