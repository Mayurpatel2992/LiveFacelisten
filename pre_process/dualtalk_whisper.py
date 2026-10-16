"""Whisper word timestamps for every DualTalk track (input to the Moshi text tokens).

Same model and JSON layout as the MEAD transcripts (openai-whisper small.en,
datasets/mead/whisper_transcription/<key>.json):
    {"segments": [{"words": [{"word": " Yeah", "start": 1.2, "end": 1.5}, ...]}, ...]}

condition_on_previous_text=False: the separated tracks have long silent
stretches, where Whisper otherwise tends to loop or invent text.

Saved: datasets/dualtalk/whisper/<conv>_speakerX.json
Run in the `probtalk3d` env (has openai-whisper). Shard with --shard/--num_shards.
"""
import argparse
import json
import time
from pathlib import Path

import whisper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/dualtalk")
    ap.add_argument("--out_dir", default="datasets/dualtalk/whisper")
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    args = ap.parse_args()

    root = Path(args.root)
    src = json.loads((root / "splits" / "source_dirs.json").read_text())
    wavs = []
    for split, d in src.items():
        for conv in (root / "splits" / split).read_text().split():
            for s in ("speaker1", "speaker2"):
                wavs.append(root / "raw" / d / f"{conv}_{s}.wav")
    wavs = sorted(set(wavs))[args.shard::args.num_shards]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = whisper.load_model(args.model, device=args.device)
    print(f"shard {args.shard}/{args.num_shards}: {len(wavs)} tracks", flush=True)
    t0, n = time.time(), 0
    for i, wav in enumerate(wavs):
        dst = out_dir / f"{wav.stem}.json"
        if dst.exists():
            continue
        res = model.transcribe(str(wav), language="en", word_timestamps=True,
                               condition_on_previous_text=False, fp16=True, verbose=None)
        segs = [{"start": s["start"], "end": s["end"], "text": s["text"],
                 "no_speech_prob": s.get("no_speech_prob"), "avg_logprob": s.get("avg_logprob"),
                 "words": [{"word": w["word"], "start": w["start"], "end": w["end"],
                            "probability": w.get("probability")} for w in s.get("words", [])]}
                for s in res["segments"]]
        dst.write_text(json.dumps({"text": res["text"], "segments": segs}))
        n += 1
        if (i + 1) % 200 == 0:
            el = time.time() - t0
            print(f"  [{i + 1}/{len(wavs)}] {el:.0f}s, ~{el / n * (len(wavs) - i - 1):.0f}s left", flush=True)
    print(f"DONE {n} tracks in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
