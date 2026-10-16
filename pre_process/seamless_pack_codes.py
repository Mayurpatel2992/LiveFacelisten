"""Moshi Inner-Monologue text streams + packing of the Mimi / text codes per split.

Text stream: the MEAD / DualTalk recipe (pre_process/precompute_moshi_text_tokens.py:
Interleaver, text_padding=3, 12.5 Hz, length pinned to the track's Mimi codes) on the
Whisper words of pre_process/seamless_whisper.py.
Bleed filter: a word is dropped when none of its frames, widened by +-5 frames (200 ms),
is own speech in the dataset's speech_mask. (DualTalk also kept words found in the
track's transcript; Seamless clips come without one.)

Saved under datasets/seamless/:
    mimi_<split>.npy    int16 [8, N_mimi]
    text_<split>.npy    int32 [N_mimi]
    codes_<split>.json  {track: [offset, T_mimi]}

Run in the `liveface` env (sentencepiece) after seamless_mimi_codes.py + seamless_whisper.py.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from interleaver import Interleaver
from precompute_moshi_text_tokens import (MAIN_SPEAKER, MIMI_FRAME_RATE, build_text_stream,
                                          load_tokenizer, whisper_json_to_alignments)

ROOT = Path("datasets/seamless")
FPS = 25
TOL = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "test", "train"])
    ap.add_argument("--spm", default="datasets/kyutai/tokenizer_spm_32k_3.model")
    ap.add_argument("--num_codebooks", type=int, default=8)
    args = ap.parse_args()

    sp = load_tokenizer(args.spm)
    interleaver = Interleaver(sp, audio_frame_rate=MIMI_FRAME_RATE, text_padding=3,
                              end_of_text_padding=0, zero_padding=-1, keep_main_only=True,
                              main_speaker_label=MAIN_SPEAKER, audio_delay=0.0, device="cpu")
    for split in args.splits:
        t0 = time.time()
        index = json.load(open(ROOT / f"index_{split}.json"))
        vad = np.load(ROOT / f"vad_{split}.npy", mmap_mode="r")
        keys = sorted(index)
        lens = {k: int(np.load(ROOT / "mimi_codes" / split / f"{k}.npy", mmap_mode="r").shape[1]) for k in keys}
        total = sum(lens.values())
        mimi = np.lib.format.open_memmap(ROOT / f"mimi_{split}.npy", "w+", np.int16, (args.num_codebooks, total))
        text = np.lib.format.open_memmap(ROOT / f"text_{split}.npy", "w+", np.int32, (total,))
        codes, off, kept, dropped = {}, 0, 0, 0
        for i, k in enumerate(keys):
            t_mimi = lens[k]
            mimi[:, off:off + t_mimi] = np.load(ROOT / "mimi_codes" / split / f"{k}.npy")[:args.num_codebooks]
            v_off, T = index[k][:2]
            v = vad[v_off:v_off + T]
            align = whisper_json_to_alignments(json.loads((ROOT / "whisper" / split / f"{k}.json").read_text()))
            good = []
            for word, (st, en), spk in align:
                if en <= st:
                    continue
                a = max(0, int(st * FPS) - TOL)
                b = max(int(st * FPS) + 1, int(np.ceil(en * FPS))) + TOL
                if v[a:b].any():
                    good.append((word, (st, en), spk))
                else:
                    dropped += 1
            kept += len(good)
            text[off:off + t_mimi] = build_text_stream(interleaver, good or None, t_mimi).reshape(-1)
            codes[k] = [off, t_mimi]
            off += t_mimi
            if (i + 1) % 20000 == 0:
                print(f"  [{i + 1}/{len(keys)}] {time.time() - t0:.0f}s", flush=True)
        mimi.flush(); text.flush()
        json.dump(codes, open(ROOT / f"codes_{split}.json", "w"))
        print(f"{split}: {len(keys)} tracks, {total} Mimi frames, words kept={kept} "
              f"dropped as bleed={dropped}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
