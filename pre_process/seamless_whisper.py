"""Whisper word timestamps for every Seamless track (input to the Moshi text tokens).

Same model, options and JSON layout as pre_process/dualtalk_whisper.py (openai-whisper
small.en, word_timestamps, condition_on_previous_text=False). Audio from the UniLS-Talk
LMDB (16 kHz, cut to length * 640 samples, DC-removed and divided by the pair's joint
peak like the Mimi input). Tracks with no own speech in the dataset's
speech_mask (datasets/seamless/vad_<split>.npy) get an empty transcript without running
Whisper, which otherwise invents words on silence.

Saved: datasets/seamless/whisper/<split>/<track>.json
Run in the `liveface` env after seamless_prepare.py. Shard with --shard i --num_shards n.
"""
import argparse
import io
import json
import time
from pathlib import Path

import lmdb
import numpy as np
import whisper

SRC = Path("datasets/unilstalk/SeamlessInteractionTalk")
ROOT = Path("datasets/seamless")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "test", "train"])
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    args = ap.parse_args()

    txn = lmdb.open(str(SRC / "data_lmdb"), readonly=True, lock=False, readahead=False).begin()
    model = whisper.load_model(args.model, device=args.device)

    for split in args.splits:
        index = json.load(open(ROOT / f"index_{split}.json"))
        vad = np.load(ROOT / f"vad_{split}.npy", mmap_mode="r")
        keys = sorted(index)[args.shard::args.num_shards]
        out_dir = ROOT / "whisper" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"{split} shard {args.shard}/{args.num_shards}: {len(keys)} tracks", flush=True)
        t0, n_run, n_silent = time.time(), 0, 0
        for i, key in enumerate(keys):
            dst = out_dir / f"{key}.json"
            if dst.exists():
                continue
            off, T = index[key][:2]
            if not vad[off:off + T].any():
                dst.write_text(json.dumps({"text": "", "segments": [], "skipped": "no speech in speech_mask"}))
                n_silent += 1
                continue
            waves = []
            for k in (key, index[key][2]):
                w = np.load(io.BytesIO(txn.get(k.encode())), allow_pickle=False)["audio"].astype(np.float32)
                w = np.pad(w, (0, max(0, T * 640 - w.shape[0])))[:T * 640]
                waves.append(w - w.mean())
            # LMDB audio peaks around -36 dBFS and Whisper's log-mel is level dependent:
            # scale by the pair's joint peak, as for the Mimi codes
            w = waves[0] / (max(np.abs(x).max() for x in waves) + 1e-8)
            res = model.transcribe(w, language="en", word_timestamps=True,
                                   condition_on_previous_text=False, fp16=True, verbose=None)
            segs = [{"start": s["start"], "end": s["end"], "text": s["text"],
                     "no_speech_prob": s.get("no_speech_prob"), "avg_logprob": s.get("avg_logprob"),
                     "words": [{"word": x["word"], "start": x["start"], "end": x["end"],
                                "probability": x.get("probability")} for x in s.get("words", [])]}
                    for s in res["segments"]]
            dst.write_text(json.dumps({"text": res["text"], "segments": segs}))
            n_run += 1
            if (i + 1) % 2000 == 0:
                el = time.time() - t0
                print(f"  [{i + 1}/{len(keys)}] {el:.0f}s, ~{el / (i + 1) * (len(keys) - i - 1):.0f}s left", flush=True)
        print(f"{split}: {n_run} transcribed, {n_silent} silent, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
