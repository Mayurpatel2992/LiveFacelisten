"""DualTalk preprocessing, CPU part: splits, per-track speaking (VAD) labels, motion stats.

Input  (HF ZiqiaoPeng/DualTalk_Dataset, unzipped):
    datasets/dualtalk/raw/{train,test,ood}/<conv>_speaker{1,2}.{npz,wav,txt}
    npz: exp [T,50], pose [T,6] (0:3 head, 3:6 jaw), 25 fps; wav: 16 kHz mono,
    one separated track per speaker; both tracks of a conversation share T.

Output (datasets/dualtalk/):
    splits/{train,val,test,ood}   one conversation id per line. val = 5% of the
                                  DualTalk train conversations (seeded); like the
                                  DualTalk test set it shares videos with train.
    vad/<conv>_speakerX.npy       uint8 [T] at the 25 fps motion rate, 1 = this
                                  speaker is talking.
    stats.json                    exp/jaw percentiles on train (for the scaler).

VAD: 40 ms RMS per track. Levels are referenced to the louder track of the
conversation (per-track references inflate the bleed of a mostly-silent
listener). A frame is speech if it is within ABS_DB of that reference and
REL_DB louder than the partner track; median-filtered over 7 frames. On 150
train conversations its speech time correlates 0.84 with the word count of the
provided transcripts.
"""
import argparse
import glob
import json
import os
import random

import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter

FPS = 25
HOP = 16000 // FPS          # 640 samples = one motion frame
ABS_DB = 25.0
REL_DB = 6.0
MEDIAN = 7


def rms_db(x):
    n = len(x) // HOP
    x = x[:n * HOP].reshape(n, HOP)
    return 20 * np.log10(np.sqrt((x ** 2).mean(1)) + 1e-8)


def fit_len(v, T):
    return v[:T] if len(v) >= T else np.concatenate([v, np.zeros(T - len(v), v.dtype)])


def pair_vad(wav1, wav2, T):
    a, _ = sf.read(wav1, dtype="float32")
    b, _ = sf.read(wav2, dtype="float32")
    ea, eb = rms_db(a), rms_db(b)
    n = min(len(ea), len(eb))
    ea, eb = ea[:n], eb[:n]
    ref = max(np.percentile(ea, 95), np.percentile(eb, 95))
    A = (ea - ref > -ABS_DB) & (ea - eb > REL_DB)
    B = (eb - ref > -ABS_DB) & (eb - ea > REL_DB)
    A = median_filter(A.astype(np.uint8), MEDIAN)
    B = median_filter(B.astype(np.uint8), MEDIAN)
    return fit_len(A, T), fit_len(B, T)


def conversations(raw, split):
    return sorted({os.path.basename(p).rsplit("_speaker", 1)[0]
                   for p in glob.glob(os.path.join(raw, split, "*.npz"))})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/dualtalk")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    raw = os.path.join(args.root, "raw")

    # ---- splits ----
    split_dir = os.path.join(args.root, "splits")
    os.makedirs(split_dir, exist_ok=True)
    train = conversations(raw, "train")
    rng = random.Random(args.seed)
    val = sorted(rng.sample(train, int(round(args.val_frac * len(train)))))
    val_set = set(val)
    splits = {
        "train": [c for c in train if c not in val_set],
        "val": val,
        "test": conversations(raw, "test"),
        "ood": conversations(raw, "ood"),
    }
    src = {"train": "train", "val": "train", "test": "test", "ood": "ood"}
    for name, ids in splits.items():
        with open(os.path.join(split_dir, name), "w") as f:
            f.write("\n".join(ids) + "\n")
        print(f"split {name}: {len(ids)} conversations")
    with open(os.path.join(split_dir, "source_dirs.json"), "w") as f:
        json.dump(src, f, indent=1)

    # ---- VAD + stats ----
    vad_dir = os.path.join(args.root, "vad")
    os.makedirs(vad_dir, exist_ok=True)
    exp_all, jaw_all, speak_frac = [], [], {}
    for name, ids in splits.items():
        for i, conv in enumerate(ids):
            base = os.path.join(raw, src[name], conv)
            m1 = np.load(base + "_speaker1.npz")
            m2 = np.load(base + "_speaker2.npz")
            T = min(m1["exp"].shape[0], m2["exp"].shape[0])
            v1, v2 = pair_vad(base + "_speaker1.wav", base + "_speaker2.wav", T)
            np.save(os.path.join(vad_dir, conv + "_speaker1.npy"), v1)
            np.save(os.path.join(vad_dir, conv + "_speaker2.npy"), v2)
            speak_frac.setdefault(name, []).extend([v1.mean(), v2.mean()])
            if name == "train" and i % 4 == 0:            # stats on a 25% subsample
                for m in (m1, m2):
                    exp_all.append(m["exp"][:, :50])
                    jaw_all.append(m["pose"][:, 3:6])
        print(f"vad {name}: mean speaking fraction {np.mean(speak_frac[name]):.3f}")

    exp_all = np.concatenate(exp_all)
    jaw_all = np.concatenate(jaw_all)
    q = [0.1, 1, 50, 99, 99.9]
    stats = {
        "exp_percentiles": {str(p): float(np.percentile(exp_all, p)) for p in q},
        "jaw_percentiles": {str(p): np.percentile(jaw_all, p, axis=0).tolist() for p in q},
        "exp_dim_std_first10": exp_all[:, :10].std(0).tolist(),
        "speaking_fraction": {k: float(np.mean(v)) for k, v in speak_frac.items()},
        "vad": {"abs_db": ABS_DB, "rel_db": REL_DB, "median": MEDIAN},
    }
    with open(os.path.join(args.root, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
