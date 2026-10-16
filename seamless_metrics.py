"""UniLS-protocol metrics on the Seamless Interaction test split (UniLS-Talk), ours or a baseline's.
--split val scores the val split instead (development only; test stays the default).

The per-pair numbers are UniLS's own code (deps/UniLS = github.com/xg-chu/UniLS @ 9cad596,
core/trainer/inferencer.py::_calc_metrics), imported, not re-implemented:
    test pairs   seamless_interaction_pairs.json["test"] with length > 101 frames
                 (sitalk_data.py, CLIP_LENGTH 100); speaker A = first track of the pair
                 (--split val: the same rule on ["val"]; --pairs keeps a listed subset)
    motion       108-D = exp 100 | head rot 3 | jaw 1 | gaze 4  (LMDB motioncode[:, :104],
                 [106:108], [109:111]); head rotation zero-centred per clip
    vertices     FLAME 2020 (300 shape = 0, 100 exp), no head pose, mm
    masks        speaker A's own speech_mask; listening = not speaking
    S_LVE S_MHD S_FDD S_PDD S_JDD   on speaking frames
    L_FDD L_PDD L_JDD               on listening frames
    mean = nanmean over pairs (as UniLS eval.py); std over pairs (FacePlex reports mean ± std)

Added (FacePlex Table 1, our reading of "recomputed from pooled raw FLAME listening frames"):
    L_FID   Frechet distance between ALL generated and ALL ground-truth listening frames of
            the split, on the 108-D raw motion above

    python seamless_metrics.py --pred results/seamless/<exp>/<id>/test
    python seamless_metrics.py --pred gt          # sanity check: every metric must be 0
    python seamless_metrics.py --split val --pairs autoresearch/val300.json --pred <dir>

Prediction files: <trackA>.npy, [T, 108] raw (not normalised), same frame count as the GT,
<trackA> = basename of the pair's first track, e.g. V00_S1320_I00000211_P0252_0080_0096.
"""
import argparse
import importlib.util
import io
import json
import sys
import types
from pathlib import Path

import lmdb
import numpy as np
import torch
from scipy import linalg

ROOT = Path(__file__).resolve().parent
UNILS = ROOT / "deps/UniLS"
DATA = ROOT / "datasets/unilstalk/SeamlessInteractionTalk"
MIN_LEN = max(100, 100) + 1          # sitalk_data.py: i[2] > max(100, CLIP_LENGTH) + 1
METRICS = ["S_LVE", "S_MHD", "S_FDD", "S_PDD", "S_JDD", "L_FDD", "L_PDD", "L_JDD"]


def _load_unils():
    """FLAME.py / lbs.py / metrics.py from deps/UniLS without their package __init__s
    (those pull in pytorch3d, transformers)."""
    pkg = types.ModuleType("unils_flame")
    pkg.__path__ = [str(UNILS / "core/libs/flame_model")]
    sys.modules["unils_flame"] = pkg
    spec = importlib.util.spec_from_file_location("unils_flame.FLAME", UNILS / "core/libs/flame_model/FLAME.py")
    flame = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = flame
    spec.loader.exec_module(flame)
    spec = importlib.util.spec_from_file_location("unils_metrics", UNILS / "core/models/modules/metrics.py")
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    return flame.FLAMEModel, metrics.calc_val_metrics


def split_pairs(split="test"):
    meta = json.load(open(DATA / "seamless_interaction_pairs.json"))[split]
    return [(Path(a).name, Path(b).name, n) for a, b, n in meta.values() if n > MIN_LEN]


def load_track(txn, key):
    rec = np.load(io.BytesIO(txn.get(key.encode())), allow_pickle=False)
    m = rec["motioncode"]
    m = np.concatenate([m[:, :104], m[:, 106:108], m[:, 109:111]], axis=1).astype(np.float32)
    m[:, 100:103] -= m[:, 100:103].mean(axis=0, keepdims=True)
    return m, rec["speech_mask"].astype(bool)


def dispersion_delta(pred, gt):
    # UniLS _calc_code_dispersion_delta: FDD formula on motion-code channels
    return (pred.std(dim=1) - gt.std(dim=1)).abs().mean() * 100


def frechet(a, b):
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + off).dot(s2 + off))
    covmean = covmean.real
    d = mu1 - mu2
    return float(d.dot(d) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="folder of <trackA>.npy, or 'gt'")
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--pairs", default=None,
                    help='JSON {"pairs": [[trackA, trackB, n], ...]}: score only these pairs of --split')
    ap.add_argument("--max_pairs", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    FLAMEModel, calc_val_metrics = _load_unils()
    flame = FLAMEModel(n_shape=300, n_exp=100).to(args.device).eval()
    verts = lambda m: flame.get_flame_verts(m, with_headpose=False)

    txn = lmdb.open(str(DATA / "data_lmdb"), readonly=True, lock=False, readahead=False).begin()
    pairs = split_pairs(args.split)
    if args.pairs is not None:
        keep = {p[0] for p in json.load(open(args.pairs))["pairs"]}
        unknown = keep - {a for a, _, _ in pairs}
        if unknown:
            sys.exit(f"{len(unknown)} pairs in {args.pairs} are not {args.split} pairs (first: {sorted(unknown)[:3]})")
        pairs = [p for p in pairs if p[0] in keep]
    pairs = pairs[: args.max_pairs]
    pred_dir = None if args.pred == "gt" else Path(args.pred)
    if pred_dir is not None:
        missing = [a for a, _, _ in pairs if not (pred_dir / f"{a}.npy").exists()]
        if missing:
            sys.exit(f"{len(missing)}/{len(pairs)} predictions missing under {pred_dir} (first: {missing[:3]})")

    rows, listen_gt, listen_pred = [], [], []
    for i, (key_a, _, _) in enumerate(pairs):
        gt, speaking = load_track(txn, key_a)
        T = gt.shape[0]
        pred = gt if pred_dir is None else np.load(pred_dir / f"{key_a}.npy").astype(np.float32)
        if pred.shape[0] < T or pred.shape[1] != 108:
            sys.exit(f"{key_a}: prediction {pred.shape}, ground truth {gt.shape}")
        pred, speaking = pred[:T], speaking[:T]
        listening = ~speaking

        g = torch.from_numpy(gt)[None].to(args.device)
        p = torch.from_numpy(np.ascontiguousarray(pred))[None].to(args.device)
        gv, pv = verts(g), verts(p)
        g_pose, g_jaw = g[..., 100:103], g[..., 103:104]
        p_pose, p_jaw = p[..., 100:103], p[..., 103:104]
        r = dict.fromkeys(METRICS, float("nan"))
        s, l = torch.from_numpy(speaking).to(args.device), torch.from_numpy(listening).to(args.device)
        if speaking.sum() > 1:
            lve, mhd, fdd = calc_val_metrics(pv[:, s], gv[:, s])
            r.update(S_LVE=lve.item(), S_MHD=mhd.item(), S_FDD=fdd.item(),
                     S_PDD=dispersion_delta(p_pose[:, s], g_pose[:, s]).item(),
                     S_JDD=dispersion_delta(p_jaw[:, s], g_jaw[:, s]).item())
        if listening.sum() > 1:
            _, _, fdd = calc_val_metrics(pv[:, l], gv[:, l])
            r.update(L_FDD=fdd.item(),
                     L_PDD=dispersion_delta(p_pose[:, l], g_pose[:, l]).item(),
                     L_JDD=dispersion_delta(p_jaw[:, l], g_jaw[:, l]).item())
        rows.append(r)
        listen_gt.append(gt[listening])
        listen_pred.append(pred[listening])
        if (i + 1) % 500 == 0:
            print(f"[{i + 1}/{len(pairs)}]", flush=True)

    out = {"pred": str(args.pred), "pairs": len(pairs)}
    for k in METRICS:
        v = np.array([r[k] for r in rows])
        out[k] = float(np.nanmean(v))
        out[f"{k}_std"] = float(np.nanstd(v))
        out[f"{k}_n"] = int(np.isfinite(v).sum())
    out["L_FID"] = frechet(np.concatenate(listen_pred), np.concatenate(listen_gt))

    print(f"{len(pairs)} {args.split} pairs, pred = {args.pred}")
    for k in METRICS + ["L_FID"]:
        sd = f" ± {out[k + '_std']:.3f}" if k + "_std" in out else ""
        print(f"  {k:6s} {out[k]:9.3f}{sd}")
    if pred_dir is not None:
        json.dump(out, open(pred_dir / "seamless_metrics.json", "w"), indent=1)
        print(f"-> {pred_dir / 'seamless_metrics.json'}")


if __name__ == "__main__":
    main()
