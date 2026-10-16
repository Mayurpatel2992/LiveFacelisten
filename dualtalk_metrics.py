"""DualTalk metrics on generated tracks (ours or a baseline's), EXP and JAW only.

Port of DualTalk's metric/metric.py (github.com/ZiqiaoPeng/DualTalk): per file
FD, paired FD (with the partner's ground truth), MSE, SID (k-means entropy,
k=40 exp / 10 jaw), STS, diversity, variance and rPCC, then the mean over
files. POSE is dropped (our models do not generate head pose).

Added, using the own-track VAD (datasets/dualtalk/vad):
    *_speak / *_listen     MSE on the agent's speaking / listening frames
    fd_listen_pooled_*     FD between ALL generated and ALL ground-truth
                           listening frames of the split (UniLS-style
                           distributional listening score; per-file FD is
                           unstable on short listening stretches)

    python dualtalk_metrics.py --pred results/dualtalk/<exp>/<id>/test --split test
    python dualtalk_metrics.py --pred <DualTalk result_DualTalk dir> --split test

Prediction files: <conv>_speakerX.npy, [T, >=53] raw FLAME (exp 0:50, jaw 50:53).
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import linalg
from sklearn.cluster import KMeans


# --- from DualTalk metric/metric.py (sid / sts vectorised, same values) ------- #
def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]
    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()


def calculate_activation_statistics(activations):
    return np.mean(activations, axis=0), np.cov(activations, rowvar=False)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError('Imaginary component {}'.format(np.max(np.abs(covmean.imag))))
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


def calculate_variance(activations):
    return np.mean(np.var(activations, axis=0))


def calcuate_sid(gt, pred, k):
    # n_init=1 == DualTalk's n_init='auto' with k-means++ (older sklearn lacks 'auto')
    kmeans_gt = KMeans(n_clusters=k, random_state=0, n_init=1).fit(gt)
    kmeans_pred = kmeans_gt.predict(pred)
    hist_cnt = np.bincount(kmeans_pred, minlength=k) / len(kmeans_pred)
    return -sum(h * np.log2(h + 1e-6) for h in hist_cnt)


def sts(x, y, timestep=0.1):
    dx, dy = np.diff(x, axis=0), np.diff(y, axis=0)
    return np.sqrt((((dx - dy) ** 2) / timestep).sum())
# ---------------------------------------------------------------------------- #


def fd(a, b):
    return calculate_frechet_distance(*calculate_activation_statistics(a), *calculate_activation_statistics(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--split", required=True, help="test | ood | val (selects GT folder + list)")
    ap.add_argument("--root", default="datasets/dualtalk")
    ap.add_argument("--out", default=None, help="default: <pred>/metrics.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subset", action="store_true",
                    help="score only the tracks present in --pred (diagnostics on a few tracks)")
    args = ap.parse_args()
    np.random.seed(args.seed)

    src = json.load(open(os.path.join(args.root, "splits", "source_dirs.json")))[args.split]
    gt_dir = os.path.join(args.root, "raw", src)
    convs = open(os.path.join(args.root, "splits", args.split)).read().split()
    keys = [f"{c}_{s}" for c in convs for s in ("speaker1", "speaker2")]
    if args.subset:
        keys = [k for k in keys if os.path.exists(os.path.join(args.pred, f"{k}.npy"))]
    missing = [k for k in keys if not os.path.exists(os.path.join(args.pred, f"{k}.npy"))]
    if missing:
        raise FileNotFoundError(f"{len(missing)}/{len(keys)} predictions missing, e.g. {missing[:3]}")

    per = {}                                  # metric -> list over files
    pooled = {"pred_exp": [], "gt_exp": [], "pred_jaw": [], "gt_jaw": []}

    def add(name, val):
        per.setdefault(name, []).append(float(val))

    for key in keys:
        pre = np.load(os.path.join(args.pred, f"{key}.npy"))
        gt = np.load(os.path.join(gt_dir, f"{key}.npz"))
        other = key[:-1] + ("2" if key.endswith("1") else "1")
        ogt = np.load(os.path.join(gt_dir, f"{other}.npz"))
        vad = np.load(os.path.join(args.root, "vad", f"{key}.npy")).astype(bool)
        T = min(pre.shape[0], gt["exp"].shape[0], ogt["exp"].shape[0], len(vad))
        parts = {
            "exp": (pre[:T, :50], gt["exp"][:T, :50], ogt["exp"][:T, :50], 40),
            "jaw": (pre[:T, 50:53], gt["pose"][:T, 3:6], ogt["pose"][:T, 3:6], 10),
        }
        vad = vad[:T]
        for name, (p, g, o, k) in parts.items():
            p, g, o = (x.astype(np.float64) for x in (p, g, o))
            add(f"fid_{name}", fd(p, g))
            add(f"p_fid_{name}", fd(np.concatenate([o, p], -1), np.concatenate([o, g], -1)))
            add(f"mse_{name}", np.mean((p - g) ** 2))
            add(f"sid_{name}", calcuate_sid(p, g, k))
            add(f"sts_{name}", sts(p, g))
            add(f"diversity_pre_{name}", calculate_diversity(p, 100))
            add(f"diversity_gt_{name}", calculate_diversity(g, 100))
            add(f"variance_pre_{name}", calculate_variance(p))
            add(f"variance_gt_{name}", calculate_variance(g))
            pcc_gt = np.corrcoef(g.reshape(-1), o.reshape(-1))[0, 1]
            pcc_pred = np.corrcoef(p.reshape(-1), o.reshape(-1))[0, 1]
            add(f"rpcc_{name}", abs(pcc_gt - pcc_pred))
            if vad.any():
                add(f"mse_{name}_speak", np.mean((p[vad] - g[vad]) ** 2))
            if (~vad).any():
                add(f"mse_{name}_listen", np.mean((p[~vad] - g[~vad]) ** 2))
                pooled[f"pred_{name}"].append(p[~vad])
                pooled[f"gt_{name}"].append(g[~vad])

    result = {name: float(np.nanmean(v)) for name, v in per.items()}
    for name in ("exp", "jaw"):
        result[f"fd_listen_pooled_{name}"] = float(fd(np.concatenate(pooled[f"pred_{name}"]),
                                                     np.concatenate(pooled[f"gt_{name}"])))
    result["num_tracks"] = len(keys)
    out = args.out or os.path.join(args.pred, "metrics.json")
    json.dump(result, open(out, "w"), indent=1)
    for k, v in result.items():
        print(f"{k:28s} {v:.6g}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
