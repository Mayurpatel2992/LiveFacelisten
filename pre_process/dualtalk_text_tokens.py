"""MIBURI-style interleaved Moshi text-token stream for every DualTalk track.

Reuses the MEAD pipeline (pre_process/precompute_moshi_text_tokens.py: Moshi
Inner-Monologue interleaver, text_padding=3, 12.5 Hz, length pinned to the
track's Mimi codes) with word timings from pre_process/dualtalk_whisper.py.

Bleed filter: a separated track still carries the partner's voice ~36 dB down,
which Whisper sometimes transcribes. A word is dropped when none of its frames
are speech for this track (vad/<key>.npy) AND it does not occur in the track's
own DualTalk transcript (<key>.txt). Overlapping backchannels ("yeah" while the
partner talks) usually fail the VAD but pass the transcript check, so they stay.

Saved: datasets/dualtalk/moshi_text_codes/<conv>_speakerX.npy   int32 [1, T_mimi]
Run in the `miburi` env (sentencepiece + torch) after Mimi codes and Whisper.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from interleaver import Interleaver
from precompute_moshi_text_tokens import (MAIN_SPEAKER, MIMI_FRAME_RATE, build_text_stream,
                                          load_tokenizer, whisper_json_to_alignments)

FPS = 25


def norm(w: str) -> str:
    return re.sub(r"[^a-z0-9']", "", w.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/dualtalk")
    ap.add_argument("--out_dir", default="datasets/dualtalk/moshi_text_codes")
    ap.add_argument("--spm", default=(
        "/home/mpate169/Mayur/LiveFace_train_moshi_abl_exps_20261012/assets_dep/kyutai_cache/"
        "models--kyutai--moshiko-pytorch-bf16/snapshots/"
        "2bfc9ae6e89079a5cc7ed2a68436010d91a3d289/tokenizer_spm_32k_3.model"))
    ap.add_argument("--split_path", default=None, help="default: <root>/splits")
    ap.add_argument("--skip_missing", action="store_true",
                    help="skip tracks whose Mimi codes / Whisper JSON are not there yet")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    sp = load_tokenizer(args.spm)
    interleaver = Interleaver(sp, audio_frame_rate=MIMI_FRAME_RATE, text_padding=3,
                              end_of_text_padding=0, zero_padding=-1, keep_main_only=True,
                              main_speaker_label=MAIN_SPEAKER, audio_delay=0.0, device="cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_path = Path(args.split_path) if args.split_path else root / "splits"
    src = json.loads((split_path / "source_dirs.json").read_text())

    kept = dropped = n = 0
    t0 = time.time()
    for split, d in src.items():
        for conv in (split_path / split).read_text().split():
            for s in ("speaker1", "speaker2"):
                key = f"{conv}_{s}"
                dst = out_dir / f"{key}.npy"
                if dst.exists() and not args.overwrite:
                    continue
                if args.skip_missing and not ((root / "mimi_codes" / f"{key}.npy").exists()
                                              and (root / "whisper" / f"{key}.json").exists()):
                    continue
                t_mimi = int(np.load(root / "mimi_codes" / f"{key}.npy").shape[1])
                vad = np.load(root / "vad" / f"{key}.npy")
                own_words = {norm(w) for w in (root / "raw" / d / f"{key}.txt").read_text().split()}
                align = whisper_json_to_alignments(
                    json.loads((root / "whisper" / f"{key}.json").read_text()))
                good = []
                for word, (st, en), spk in align:
                    if en <= st:
                        continue
                    a, b = int(st * FPS), max(int(st * FPS) + 1, int(np.ceil(en * FPS)))
                    if vad[a:b].any() or norm(word) in own_words:
                        good.append((word, (st, en), spk))
                    else:
                        dropped += 1
                kept += len(good)
                np.save(dst, build_text_stream(interleaver, good or None, t_mimi))
                n += 1
    print(f"DONE {n} tracks in {time.time() - t0:.0f}s; words kept={kept} dropped as bleed={dropped}")


if __name__ == "__main__":
    main()
