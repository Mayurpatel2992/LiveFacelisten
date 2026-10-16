"""Generate Seamless test motion with a trained listen+speak run (UniLS / FacePlex protocol).

Speaker A of every test pair kept by the protocol (> 101 frames, 3457 pairs) is generated
free-running with the cached AR loop from its own (+ partner, Option A / B) Moshi streams,
mapped back to raw 108-D FLAME (exp 100 | head rot 3 | jaw 1 | gaze 4) and saved as
    results/seamless/<experiment>/<run_id>/<split>[_gate-<g>]<tag>/<trackA>.npy   [T, 108]
Score with seamless_metrics.py (UniLS's own metric code).

    python3.9 evaluate_seamless.py folder=outputs/Seamless/seamless_controlnet/<id>
    python3.9 evaluate_seamless.py folder=... device=3 shard=3 num_shards=6   # one per GPU
"""
import json
import logging
import time
from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import framework.launch.prepare  # noqa
from framework.data.tools.collate import collate_motion_and_audio
from framework.model.utils.tools import load_checkpoint

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="evaluation_seamless")
def main(newcfg: DictConfig) -> None:
    folder = Path(newcfg.folder).resolve()
    prevcfg = OmegaConf.load(folder / ".hydra/config.yaml")
    cfg = OmegaConf.merge(prevcfg, newcfg)
    pl.seed_everything(cfg.seed)
    duplex = "duplex" in cfg.model
    cfg.data.progress_bar = False
    cfg.data.load_partner = bool(duplex)

    if cfg.ckpt in ("best", "latest"):
        ckpts = folder.glob(f"tensorboard/*/checkpoints/{cfg.ckpt}-epoch=*.ckpt")
        ckpt = max(ckpts, key=lambda p: int(p.stem.split("=")[-1]))
    else:
        ckpt = Path(cfg.ckpt)
    gate_tag = ""
    if duplex and cfg.gate is not None:
        cfg.model.duplex.gate_infer_source = cfg.gate
        gate_tag = f"_gate-{cfg.gate}"
    out_dir = Path(cfg.path.code_dir) / "results/seamless" / cfg.experiment / folder.name / f"{cfg.split}{gate_tag}{cfg.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"checkpoint: {ckpt}\noutputs:    {out_dir}")

    data_module = instantiate(cfg.data)
    dataset = getattr(data_module, f"{cfg.split}_dataset")
    model = instantiate(cfg.model, nfeats=data_module.nfeats, split_path=data_module.split_path,
                        one_hot_dim=data_module.one_hot_dim, resumed_training=False,
                        logger_name="none", _recursive_=False)
    model.to(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
    load_checkpoint(model, str(ckpt), eval_mode=True, device=model.device)
    model.eval()
    if cfg.sampling_steps is not None:
        model.sampling_steps = int(cfg.sampling_steps)
    if cfg.token_temperature is not None:
        model.infer_token_temperature = float(cfg.token_temperature)

    norm = json.load(open(Path(cfg.data.root) / "norm.json"))
    lo = np.asarray(norm["lo"], dtype=np.float32)
    scale = np.maximum(np.asarray(norm["hi"], dtype=np.float32) - lo, 1e-6)

    # speaker A of each protocol pair (the dataset already keeps only pairs > 101 frames)
    todo = [i for i, k in enumerate(dataset.keyids) if dataset.index[k][3] == 1]
    if cfg.pairs is not None:
        keep = {p[0] for p in json.load(open(cfg.pairs))["pairs"]}
        todo = [i for i in todo if dataset.keyids[i] in keep]
        if len(todo) != len(keep):
            raise ValueError(f"{len(keep) - len(todo)} of the {len(keep)} pairs in {cfg.pairs} "
                             f"are not {cfg.split} pairs")
    todo = todo[cfg.shard::cfg.num_shards]
    if cfg.max_tracks is not None:
        todo = todo[:int(cfg.max_tracks)]
    t0 = time.time()
    for n, i in enumerate(todo):
        item = dataset[i]
        key = item["keyid"]
        if (out_dir / f"{key}.npy").exists():
            continue
        batch = collate_motion_and_audio([item])
        with torch.no_grad():
            preds = model.forward_multi_sample(batch, num_samples=int(cfg.number_of_samples), generation=False)
        T = item["motion"].shape[1]
        for k, p in enumerate(preds):
            raw = p[0].float().cpu().numpy() * scale + lo                 # [T,108] raw FLAME
            if raw.shape[0] != T:
                raise RuntimeError(f"{key}: generated {raw.shape[0]} frames, ground truth has {T}")
            np.save(out_dir / (f"{key}.npy" if k == 0 else f"{key}_s{k}.npy"), raw)
        if (n + 1) % 50 == 0:
            el = time.time() - t0
            logger.info(f"[{n + 1}/{len(todo)}] {el:.0f}s, ~{el / (n + 1) * (len(todo) - n - 1):.0f}s left")
    logger.info(f"done: {len(todo)} tracks -> {out_dir}")


if __name__ == "__main__":
    main()
