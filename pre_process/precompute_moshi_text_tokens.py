"""Precompute the MIBURI-style interleaved Moshi text-token stream for every MEAD clip.

This is the text half of the Moshi/Mimi conditioning ablation.  It follows Moshi's
*Inner Monologue* text/audio alignment exactly -- the interleaver in
``pre_process/interleaver.py`` is vendored verbatim from MIBURI
(``scripts/trainers/dataloaders/utils/interleaver.py``) which in turn is Moshi's
published algorithm.

Pipeline (mirrors MIBURI's BEAT-X HDF5 builder, `_load_models` + `_process_entry`):

    wav (48 kHz on disk)
      -> librosa.load(sr=16000)                         # same as the audio path
      -> Whisper word-level timestamps  (or a supplied WhisperX/whisper JSON)
      -> alignments = [(word, (start_s, end_s), "SPEAKER_MAIN"), ...]
      -> Interleaver(text_padding=3, end_of_text_padding=0, zero_padding=-1,
                     keep_main_only=True, audio_delay=0.0,
                     audio_frame_rate = 12.5)            # == mimi.frame_rate
      -> one text token per 80 ms slot, on the SAME 12.5 Hz grid as the Mimi codes

The stream length is pinned to the matching precomputed Mimi-code length
(`datasets/mead/mimi_codes/<key>.npy`, shape [K, T_mimi]) so audio and text
conditioning stay frame-aligned inside the model.

Saved: datasets/mead/moshi_text_codes/<key>.npy   int32   shape [1, T_mimi]
    values:  0        = end_of_text_padding (EPAD)
             3        = text_padding (PAD) / in-word padding
             1..31999 = SentencePiece token ids
             -1       = zero_padding (only if the transcript is entirely missing)

Transcription source (pick one):
  * --transcripts_dir DIR : if DIR/<key>.json exists it is used directly.  Format
      is the openai-whisper / WhisperX JSON:
      {"segments": [{"words": [{"word": "..", "start": s, "end": e}, ...]}, ...]}
      Any inline Whisper run also writes its JSON here (so a second pass is free).
  * otherwise Whisper is run inline (needs `openai-whisper`: `pip install openai-whisper`).

Run in the `miburi` conda env (needs `sentencepiece`, `torch`; `openai-whisper`
only if no transcripts are supplied).
"""
import argparse
import glob
import json
import math
import os
import time
from pathlib import Path

import sys

import librosa
import numpy as np
import sentencepiece
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))   # so `interleaver` resolves
from interleaver import Interleaver

MIMI_FRAME_RATE = 12.5          # kyutai/mimi config.frame_rate
MAIN_SPEAKER = "SPEAKER_MAIN"


def load_tokenizer(spm_path: str) -> sentencepiece.SentencePieceProcessor:
    sp = sentencepiece.SentencePieceProcessor()
    sp.load(spm_path)
    return sp


def whisper_json_to_alignments(data: dict) -> list:
    """openai-whisper / WhisperX JSON -> [(word, (start, end), MAIN_SPEAKER), ...]."""
    alignments = []
    prev_start = prev_end = 0.0
    for seg in data.get("segments", []):
        for w in seg.get("words", []):
            word = (w.get("word") or "").strip()
            start = w.get("start")
            end = w.get("end")
            if start is None:
                start = prev_start
            if end is None:
                end = prev_end
            start, end = float(start), float(end)
            if word:
                alignments.append((word, (start, end), MAIN_SPEAKER))
            prev_start, prev_end = start, end
    return alignments


def build_text_stream(interleaver: Interleaver, alignments, t_mimi: int) -> np.ndarray:
    """alignments -> int32 [1, t_mimi] on the 12.5 Hz grid (Moshi Inner Monologue)."""
    # ceil(segment_duration * frame_rate) must equal t_mimi -> use the exact
    # fractional duration of the precomputed Mimi stream.
    segment_duration = t_mimi / interleaver.audio_frame_rate
    if alignments is None:
        toks = interleaver.build_token_stream(None, segment_duration)
    else:
        toks = interleaver.prepare_item(alignments, segment_duration,
                                        main_speaker=MAIN_SPEAKER)
    toks = toks.view(-1).cpu().numpy().astype(np.int32)         # [T]
    if toks.shape[0] < t_mimi:                                  # numeric guard
        toks = np.concatenate(
            [toks, np.full(t_mimi - toks.shape[0], interleaver.text_padding, np.int32)])
    toks = toks[:t_mimi]
    return toks[None, :]                                        # [1, T]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_dir", default="datasets/mead/wav")
    ap.add_argument("--motion_dir", default="datasets/mead/param")
    ap.add_argument("--mimi_codes_dir", default="datasets/mead/mimi_codes",
                    help="Precomputed Mimi codes; their length pins the text-stream length.")
    ap.add_argument("--out_dir", default="datasets/mead/moshi_text_codes")
    ap.add_argument("--spm", default=(
        "assets_dep/kyutai_cache/models--kyutai--moshiko-pytorch-bf16/snapshots/"
        "2bfc9ae6e89079a5cc7ed2a68436010d91a3d289/tokenizer_spm_32k_3.model"))
    ap.add_argument("--transcripts_dir", default="datasets/mead/whisper_transcription",
                    help="Reused if <key>.json exists; inline Whisper writes here.")
    ap.add_argument("--whisper_model", default="small.en",
                    help="openai-whisper model for the inline fallback (tiny.en/base.en/small.en/medium.en).")
    ap.add_argument("--language", default="en")
    ap.add_argument("--text_padding", type=int, default=3)
    ap.add_argument("--end_of_text_padding", type=int, default=0)
    ap.add_argument("--zero_padding", type=int, default=-1)
    ap.add_argument("--audio_delay", type=float, default=0.0,
                    help="Text-vs-audio delay in seconds (Moshi default 0 for stored conditions).")
    ap.add_argument("--src_sr", type=int, default=16000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    sp = load_tokenizer(args.spm)
    print(f"SentencePiece: vocab={sp.vocab_size()} bos={sp.bos_id()} eos={sp.eos_id()} "
          f"pad={sp.pad_id()}  (text_padding={args.text_padding})")

    interleaver = Interleaver(
        sp,
        audio_frame_rate=MIMI_FRAME_RATE,
        text_padding=args.text_padding,
        end_of_text_padding=args.end_of_text_padding,
        zero_padding=args.zero_padding,
        keep_main_only=True,
        main_speaker_label=MAIN_SPEAKER,
        audio_delay=args.audio_delay,
        device="cpu",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tr_dir = Path(args.transcripts_dir) if args.transcripts_dir else None
    if tr_dir:
        tr_dir.mkdir(parents=True, exist_ok=True)

    stems = sorted(Path(p).stem for p in glob.glob(os.path.join(args.motion_dir, "*.npy")))
    if args.limit:
        stems = stems[:args.limit]
    print(f"{len(stems)} clips -> {out_dir}")

    whisper_model = None                     # lazily loaded only if needed

    n_done = n_skip = n_miss = n_empty = n_json = n_asr = 0
    t0 = time.time()
    for i, stem in enumerate(stems):
        dst = out_dir / f"{stem}.npy"
        if dst.exists() and not args.overwrite:
            n_skip += 1
            continue

        mimi_path = os.path.join(args.mimi_codes_dir, f"{stem}.npy")
        if not os.path.exists(mimi_path):
            n_miss += 1
            continue
        t_mimi = int(np.load(mimi_path).shape[1])

        # --- word-level timestamps: cached JSON first, else inline Whisper ---
        alignments = None
        tr_path = (tr_dir / f"{stem}.json") if tr_dir else None
        if tr_path is not None and tr_path.exists():
            with open(tr_path) as f:
                alignments = whisper_json_to_alignments(json.load(f))
            n_json += 1
        else:
            wav_path = os.path.join(args.wav_dir, f"{stem}.wav")
            if not os.path.exists(wav_path):
                n_miss += 1
                continue
            if whisper_model is None:
                import whisper
                dev = args.device if torch.cuda.is_available() else "cpu"
                print(f"Loading Whisper '{args.whisper_model}' on {dev} ...")
                whisper_model = whisper.load_model(args.whisper_model, device=dev)
            w, _ = librosa.load(wav_path, sr=args.src_sr)
            result = whisper_model.transcribe(
                w.astype(np.float32), language=args.language,
                word_timestamps=True, verbose=False)
            if tr_path is not None:
                with open(tr_path, "w") as f:
                    json.dump({"segments": [
                        {"words": [
                            {"word": ww.get("word", ""), "start": ww.get("start"),
                             "end": ww.get("end")}
                            for ww in seg.get("words", [])]}
                        for seg in result.get("segments", [])]}, f)
            alignments = whisper_json_to_alignments(result)
            n_asr += 1

        alignments = [a for a in alignments if a[1][1] > a[1][0]]
        if not alignments:
            n_empty += 1                      # speech but no transcript -> all-PAD stream

        codes = build_text_stream(interleaver, alignments, t_mimi)
        np.save(dst, codes)
        n_done += 1

        if (i + 1) % 250 == 0:
            el = time.time() - t0
            print(f"  [{i + 1}/{len(stems)}] done={n_done} skip={n_skip} miss={n_miss} "
                  f"empty={n_empty} json={n_json} asr={n_asr} | {el:.0f}s")

    print(f"\nDONE  written={n_done} skipped={n_skip} missing={n_miss} "
          f"empty_transcript={n_empty}  (from json={n_json}, asr={n_asr})  "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
