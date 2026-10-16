"""Generate DualTalk test / OOD motion with a trained listen+speak run.

Works for the speaking model (own track only) and for Option A / B (own +
partner). Every track of the split is generated free-running with the cached
AR loop, mapped back to raw FLAME (exp 50 + jaw 3) and saved as
    results/dualtalk/<experiment>/<run_id>/<split>[_gate-<g>]/<keyid>.npy   [T, 53]
Score with dualtalk_metrics.py.

    python3.9 evaluate_dualtalk.py folder=outputs/DualTalk/dualtalk_controlnet/<id> split=test
    python3.9 evaluate_dualtalk.py folder=... split=ood gate=vad      # oracle switch
"""
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


@torch.no_grad()
def teacher_forced(model, batch):
    """Diagnostic, speaking model only: backbone sees the GROUND-TRUTH code history
    (no AR feedback); codes sampled from the predicted logits; full flow ODE per
    frame (frames in parallel, flow history off); frozen Stage-1 decode."""
    style = model._style_one_hot(batch).to(model.device)
    audio, prosody = model._assemble_audio_eval(batch, False)
    atok = model.encode_audio_tokens(audio, style, prosody)               # [1,T,d]
    T = atok.shape[1]
    motion = batch["motion"][0] if isinstance(batch["motion"], list) else batch["motion"]
    motion = motion.to(model.device).float().reshape(1, -1, motion.shape[-1])[:, :T]
    z0, _, _ = model.motion_prior.quantize(model.motion_prior.motion_encoder(motion))
    T = min(T, z0.shape[1]); atok, z0 = atok[:, :T], z0[:, :T]
    h = model.run_backbone(model._shift_motion(z0), atok)
    z_hat, _ = model._logits_to_latent(model._code_logits(h), stochastic=True)
    z = torch.randn_like(z_hat)
    steps = model.sampling_steps
    for i in range(steps):
        t = torch.full((1,), 1.0 - i / steps, device=z.device, dtype=z.dtype)
        z = z - model.flow_head(z, t, z_hat, audio_tokens=atok,
                                hidden=h if model.flow_use_hidden else None, history=None) / steps
    zq, _, _ = model.motion_prior.quantize(z)
    return model.motion_prior.motion_decoder(zq, use_chunked_attn=True)


@hydra.main(version_base=None, config_path="configs", config_name="evaluation_dualtalk")
def main(newcfg: DictConfig) -> None:
    folder = Path(newcfg.folder).resolve()
    prevcfg = OmegaConf.load(folder / ".hydra/config.yaml")
    cfg = OmegaConf.merge(prevcfg, newcfg)
    pl.seed_everything(cfg.seed)
    duplex = "duplex" in cfg.model
    cfg.data.progress_bar = False
    cfg.data.load_partner = bool(duplex)

    ckpt = (sorted(folder.glob("tensorboard/*/checkpoints/best-epoch=*.ckpt"))[-1]
            if cfg.ckpt == "best" else Path(cfg.ckpt))
    gate_tag = ""
    if duplex and cfg.gate is not None:
        cfg.model.duplex.gate_infer_source = cfg.gate
        gate_tag = f"_gate-{cfg.gate}"
    out_dir = Path(cfg.path.code_dir) / "results/dualtalk" / cfg.experiment / folder.name / f"{cfg.split}{gate_tag}{cfg.tag}"
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

    exp_lo, exp_hi = cfg.data.exp_range
    jaw_lo, jaw_hi = cfg.data.jaw_range
    scale = np.array([exp_hi - exp_lo] * 50 + [jaw_hi - jaw_lo] * 3, dtype=np.float32)
    offset = np.array([exp_lo] * 50 + [jaw_lo] * 3, dtype=np.float32)

    n = len(dataset) if cfg.max_tracks is None else min(len(dataset), int(cfg.max_tracks))
    t0 = time.time()
    for i in range(n):
        item = dataset[i]
        key = item["keyid"]
        if (out_dir / f"{key}.npy").exists():
            continue
        batch = collate_motion_and_audio([item])
        with torch.no_grad():
            if cfg.teacher_forced:
                preds = [teacher_forced(model, batch)]
            else:
                preds = model.forward_multi_sample(batch, num_samples=int(cfg.number_of_samples),
                                                   generation=False)
        for k, p in enumerate(preds):
            raw = p[0].float().cpu().numpy() * scale + offset           # [T,53] raw FLAME
            np.save(out_dir / (f"{key}.npy" if k == 0 else f"{key}_s{k}.npy"), raw)
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            logger.info(f"[{i + 1}/{n}] {el:.0f}s, ~{el / (i + 1) * (n - i - 1):.0f}s left")
    logger.info(f"done: {n} tracks -> {out_dir}")


if __name__ == "__main__":
    main()
