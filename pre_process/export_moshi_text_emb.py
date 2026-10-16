"""Export Moshi's text embedding table (`lm.text_emb.weight`) to a single .pt file.

MIBURI-style conditioning initialises the text-conditioning embedding from Moshi's
`text_emb` table and (optionally) lets it adapt:

    Reference: miburi/scripts/trainers/uflgtdm3_trainer.py
        text_procemb = copy.deepcopy(lm.text_emb.weight.data)
    Reference: miburi/models/gesture_lm.py
        self.text_procemb = ScaledEmbedding(num_textemb, dim_textemb,
                                            norm=False, zero_idx=-1,
                                            _weight=text_procemb,
                                            _freeze=textaudio_emb_freeze)

Only `text_emb.weight` is needed -- not the transformer -- so this reads it
straight out of the safetensors without instantiating the LM.  Companion of
``pre_process/export_moshi_audio_emb.py``.

Run in the `miburi` conda env (needs `safetensors`, `torch`).  One-time.

Output dict (torch.save):
    table:                FloatTensor [rows, dim]   (rows=32001, dim=4096)
    vocab:                32000       # SentencePiece vocab (valid ids 0..31999)
    rows:                 32001
    zero_idx:             -1          # ScaledEmbedding convention: id==-1 -> zero
    text_padding:         3           # Moshi/Interleaver PAD id  (== spm pad_id)
    end_of_text_padding:  0           # Moshi/Interleaver EPAD id
    dim:                  4096
    source:               "kyutai/moshiko-pytorch-bf16"
"""
import argparse
import os
from pathlib import Path

import torch
from safetensors import safe_open

DEFAULT_SAFETENSORS = (
    "assets_dep/kyutai_cache/models--kyutai--moshiko-pytorch-bf16/snapshots/"
    "2bfc9ae6e89079a5cc7ed2a68436010d91a3d289/model.safetensors"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--safetensors", default=DEFAULT_SAFETENSORS,
                    help="Path to the Moshi LM model.safetensors")
    ap.add_argument("--repo", default="kyutai/moshiko-pytorch-bf16",
                    help="Only recorded as metadata")
    ap.add_argument("--key", default="text_emb.weight",
                    help="Tensor name of the temporal-transformer text embedding")
    ap.add_argument("--vocab", type=int, default=32000,
                    help="SentencePiece vocab size (valid ids 0..vocab-1)")
    ap.add_argument("--zero_idx", type=int, default=-1)
    ap.add_argument("--text_padding", type=int, default=3)
    ap.add_argument("--end_of_text_padding", type=int, default=0)
    ap.add_argument("--out", default="datasets/moshi_text_emb.pt")
    args = ap.parse_args()

    if not os.path.exists(args.safetensors):
        raise FileNotFoundError(
            f"{args.safetensors}\nPass --safetensors <path to Moshi model.safetensors>"
        )

    with safe_open(args.safetensors, framework="pt") as f:
        keys = set(f.keys())
        if args.key not in keys:
            raise KeyError(
                f"{args.key} not in checkpoint "
                f"(text-ish keys: {sorted(k for k in keys if 'text' in k.lower())})"
            )
        w = f.get_tensor(args.key).float()             # [rows, dim], bf16 -> fp32

    rows, dim = w.shape
    if rows < args.vocab:
        raise ValueError(f"table has {rows} rows but vocab={args.vocab}")
    print(f"  {args.key:24s} {tuple(w.shape)}  mean={w.mean():.4e} std={w.std():.4e}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "table": w.contiguous(),
            "vocab": int(args.vocab),
            "rows": int(rows),
            "zero_idx": int(args.zero_idx),
            "text_padding": int(args.text_padding),
            "end_of_text_padding": int(args.end_of_text_padding),
            "dim": int(dim),
            "source": args.repo,
        },
        out,
    )
    print(f"\nWrote {out}  table={tuple(w.shape)}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
