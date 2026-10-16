"""Stage-1 upper bound: ground-truth DualTalk motion -> VQ-VAE encode / quantize /
decode, saved like evaluate_dualtalk.py so dualtalk_metrics.py can score it.
If this keeps the GT variance / FD, Stage-1 is not what flattens the generated motion.

    python3.9 eval_stage1_recon.py folder=outputs/DualTalk/dualtalk_vqvae_prior/<id> \
        ckpt=<path.ckpt> split=test tag=_ep69
Output: results/dualtalk/stage1_recon/<run_id>/<split><tag>/<keyid>.npy   [T, 53] raw FLAME
"""
import logging
from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import framework.launch.prepare  # noqa
from framework.model.utils.tools import load_checkpoint

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="evaluation_dualtalk")
def main(newcfg: DictConfig) -> None:
    folder = Path(newcfg.folder).resolve()
    cfg = OmegaConf.merge(OmegaConf.load(folder / ".hydra/config.yaml"), newcfg)
    pl.seed_everything(cfg.seed)
    cfg.data.progress_bar = False
    ckpt = (sorted(folder.glob("tensorboard/*/checkpoints/best-epoch=*.ckpt"))[-1]
            if cfg.ckpt == "best" else Path(cfg.ckpt))
    out_dir = Path(cfg.path.code_dir) / "results/dualtalk/stage1_recon" / folder.name / f"{cfg.split}{cfg.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"checkpoint: {ckpt}\noutputs:    {out_dir}")

    data_module = instantiate(cfg.data)
    dataset = getattr(data_module, f"{cfg.split}_dataset")
    model = instantiate(cfg.model, nfeats=data_module.nfeats, resumed_training=False, _recursive_=False)
    model.to(f"cuda:{cfg.device}")
    load_checkpoint(model, str(ckpt), eval_mode=True, device=model.device)
    model.eval()

    exp_lo, exp_hi = cfg.data.exp_range
    jaw_lo, jaw_hi = cfg.data.jaw_range
    scale = np.array([exp_hi - exp_lo] * 50 + [jaw_hi - jaw_lo] * 3, dtype=np.float32)
    offset = np.array([exp_lo] * 50 + [jaw_lo] * 3, dtype=np.float32)

    n = len(dataset) if cfg.max_tracks is None else min(len(dataset), int(cfg.max_tracks))
    for i in range(n):
        item = dataset[i]
        motion = item["motion"].float()                    # [1, T, 53] normalised
        T = motion.shape[1]
        with torch.no_grad():
            rec = model({"motion": motion})[0, :T]
        raw = rec.float().cpu().numpy() * scale + offset
        np.save(out_dir / f"{item['keyid']}.npy", raw)
    logger.info(f"done: {n} tracks -> {out_dir}")


if __name__ == "__main__":
    main()
