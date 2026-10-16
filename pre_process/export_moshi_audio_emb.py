"""Export Moshi's 8 audio RVQ embedding tables (emb[:8]) to a single .pt file.

MIBURI-style conditioning initialises a small set of learnable embedding tables
from Moshi's audio embeddings and (optionally) lets them adapt.  We only need
`emb.0 .. emb.7` from the Moshi 7B checkpoint -- not the transformer -- so this
reads them straight out of the safetensors without instantiating the LM.

Reference: miburi/scripts/trainers/uflgtdm3_trainer.py
    audio_procemb = [copy.deepcopy(a_emb.weight.data) for a_emb in lm.emb[:8]]

Run in the `miburi` conda env (needs `safetensors`, `torch`).  One-time.

Output dict (torch.save):
    tables:         FloatTensor [K, card+1, dim]   (K=8, card+1=2049, dim=4096)
    num_codebooks:  8
    card:           2048        # Mimi codebook size (valid ids 0..2047)
    zero_idx:       -1          # ScaledEmbedding convention: id==-1 -> zero vector
    dim:            4096
    source:         "kyutai/moshiko-pytorch-bf16"
"""
import argparse
import os
from pathlib import Path

import torch
from safetensors import safe_open

DEFAULT_SAFETENSORS = (
    "/home/mpate169/Mayur/miburi/assets_dep/kyutai_cache/"
    "models--kyutai--moshiko-pytorch-bf16/snapshots/"
    "2bfc9ae6e89079a5cc7ed2a68436010d91a3d289/model.safetensors"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--safetensors", default=DEFAULT_SAFETENSORS,
                    help="Path to the Moshi LM model.safetensors")
    ap.add_argument("--repo", default="kyutai/moshiko-pytorch-bf16",
                    help="Only recorded as metadata")
    ap.add_argument("--num_codebooks", type=int, default=8)
    ap.add_argument("--card", type=int, default=2048)
    ap.add_argument("--zero_idx", type=int, default=-1)
    ap.add_argument("--out", default="datasets/moshi_audio_emb.pt")
    args = ap.parse_args()

    if not os.path.exists(args.safetensors):
        raise FileNotFoundError(
            f"{args.safetensors}\nPass --safetensors <path to Moshi model.safetensors>"
        )

    tables = []
    with safe_open(args.safetensors, framework="pt") as f:
        keys = set(f.keys())
        for k in range(args.num_codebooks):
            key = f"emb.{k}.weight"
            if key not in keys:
                raise KeyError(f"{key} not in checkpoint (have {sorted(x for x in keys if x.startswith('emb.'))})")
            w = f.get_tensor(key).float()          # [card+1, dim], bf16 -> fp32
            tables.append(w)
            print(f"  {key:20s} {tuple(w.shape)}  mean={w.mean():.4e} std={w.std():.4e}")

    tables = torch.stack(tables, dim=0).contiguous()   # [K, card+1, dim]
    assert tables.shape[0] == args.num_codebooks
    assert tables.shape[1] == args.card + 1, (
        f"expected card+1={args.card + 1} rows, got {tables.shape[1]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "tables": tables,
            "num_codebooks": args.num_codebooks,
            "card": args.card,
            "zero_idx": args.zero_idx,
            "dim": int(tables.shape[-1]),
            "source": args.repo,
        },
        out,
    )
    print(f"\nWrote {out}  tables={tuple(tables.shape)}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
