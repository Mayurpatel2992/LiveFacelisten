"""Pre-flight verification for the Moshi/Mimi Stage-2 ablation experiments.

Runs entirely in the `probtalk3d_hubert` env (no moshi / MimiModel needed --
Mimi codes are precomputed, Moshi emb[:8] is a cached tensor).

For the chosen experiment it:
  1. builds the data module + model exactly like train_all.py (no training),
  2. pulls one real training batch and prints the full shape trace (spec S16),
  3. prints the numeric temporal-alignment check for one clip (spec S19),
  4. runs one forward + backward, prints the param / grad / freeze table
     (spec S17) and asserts frozen modules receive no gradient (spec S18),
  5. runs the built-in causal leakage_test() (spec S20),
  6. overfits 2 samples for ~80 steps and prints the loss curve (spec S32).

Usage:
    python pre_process/verify_moshi_experiment.py --experiment moshi_frozen
    python pre_process/verify_moshi_experiment.py --experiment hubert_moshi
    python pre_process/verify_moshi_experiment.py --experiment moshi_trainable
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import framework.launch.prepare  # noqa  (registers OmegaConf resolvers)
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

REPO = Path(__file__).resolve().parents[1]
STAGE1_DIR = REPO / "outputs/MEAD/vqvae_prior/10etdx0z"
STAGE1_CKPT = STAGE1_DIR / "tensorboard/10etdx0z/checkpoints/best-epoch=200.ckpt"

EXPERIMENTS = {
    "moshi_frozen":         ("mead_pred_moshi",      "model_vqvae_arflow_pred_moshi_frozen",         "moshi"),
    "hubert_moshi":         ("mead_pred_moshi",      "model_vqvae_arflow_pred_hubert_moshi",         "hubert_moshi"),
    "moshi_trainable":      ("mead_pred_moshi",      "model_vqvae_arflow_pred_moshi_trainable",      "moshi"),
    "moshi_text":           ("mead_pred_moshi_text", "model_vqvae_arflow_pred_moshi_text",           "moshi"),
    "moshi_text_trainable": ("mead_pred_moshi_text", "model_vqvae_arflow_pred_moshi_text_trainable", "moshi"),
}


def banner(t):
    print("\n" + "=" * 78 + f"\n {t}\n" + "=" * 78)


def build(data_name, model_name):
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base=None,
                               job_name="verify_moshi"):
        cfg = compose(
            config_name="train",
            return_hydra_config=True,
            overrides=[
                "experiment=verify", "state=new",
                f"data={data_name}", f"model={model_name}",
                "machine.batch_size=2", "data.debug=true",
                f"model.folder_prior={STAGE1_DIR}",
                "model.version_prior=0",
                f"model.ckpt_path_prior='{STAGE1_CKPT}'",
            ],
        )
        HydraConfig.instance().set_config(cfg)   # so code_path/get_original_cwd() resolves
    import pytorch_lightning as pl
    pl.seed_everything(1234)
    dm = instantiate(cfg.data)
    model = instantiate(cfg.model, nfeats=dm.nfeats, split_path=dm.split_path,
                        one_hot_dim=dm.one_hot_dim, resumed_training=False,
                        _recursive_=False)
    # allsplit_step() touches self.trainer / self.log; outside Trainer.fit those
    # raise. Attach a minimal stub + silence logging so we can drive it directly.
    class _StubTrainer:
        current_epoch = 0        # keep scheduled-sampling / uncertainty-masking OFF
        global_step = 10_000     # past flow_warmup_steps -> full flow loss is active
        optimizers = []
    model._trainer = _StubTrainer()
    model.log = lambda *a, **k: None
    model.log_dict = lambda *a, **k: None
    return cfg, dm, model


def one_batch(dm):
    dm.setup("fit")
    dl = dm.train_dataloader()
    return next(iter(dl))


def shape_trace(model, batch, expect_cond):
    banner("1-2.  SHAPE TRACE  (one training batch)")
    dev = model.device
    print(f"audio_conditioning     : {model.audio_conditioning}  (expected {expect_cond})")
    assert model.audio_conditioning == expect_cond
    print(f"model.audio_dim        : {model.audio_dim}")
    b0_motion = batch['motion'][0]
    print(f"raw waveform[0]        : len={len(batch['audio'][0])} samples @16kHz "
          f"= {len(batch['audio'][0]) / 16000:.2f}s")
    print(f"motion GT[0]           : {tuple(b0_motion.shape)}  (=> {b0_motion.shape[1]} frames @25fps)")
    if 'mimi_codes' in batch:
        c0 = batch['mimi_codes'][0]
        print(f"Mimi codes[0]          : {tuple(c0.shape)}  dtype={c0.dtype}  "
              f"min={int(c0.min())} max={int(c0.max())}")
        tc0 = batch['moshi_text_codes'][0] if 'moshi_text_codes' in batch else None
        if tc0 is not None:
            print(f"Moshi text codes[0]   : {tuple(tc0.shape)}  dtype={tc0.dtype}  "
                  f"min={int(tc0.min())} max={int(tc0.max())}  "
                  f"(PAD=3 frac={float((tc0 == 3).float().mean()):.2f})")
        feat0 = model._moshi_feature(c0, tc0)
        print(f"summed Moshi emb[0]    : {tuple(feat0.shape)}  "
              f"(K=8 audio lookups summed -> 4096{' , concat text -> 8192' if model.moshi_text_emb is not None and model.moshi_text_fuse == 'concat' else ''})")
    # full alignment path
    ra, rm, pros = model._align_audio_motion(batch)
    print(f"aligned audio feat[0]  : {tuple(ra[0].shape)}   (audio_dim last)")
    print(f"aligned motion[0]      : {tuple(rm[0].shape)}")
    assert ra[0].shape[1] == rm[0].shape[1], "audio/motion length mismatch after alignment"
    assert ra[0].shape[-1] == model.audio_dim, f"{ra[0].shape[-1]} != audio_dim {model.audio_dim}"
    # downstream projections on a cropped/stacked tensor
    T = min(model.crop_length, ra[0].shape[1])
    audio = torch.cat([a[:, :T] for a in ra], 0).to(dev, torch.float32)
    style = model.all_identity_onehot.new_zeros(len(batch['motion']), model.one_hot_total)
    style[:, 0] = 1.0
    style = style.to(dev)
    prosody = torch.stack(pros, 0).to(dev, torch.float32)
    ptok = model.prosody_feature_map(prosody)
    print(f"prosody_feature_map out: {tuple(ptok.shape)}   (Linear(103 -> {model.audio_dim}))")
    afeat = model.audio_feature_map(audio * ptok)
    print(f"audio projection out   : {tuple(afeat.shape)}   (Linear({model.audio_dim} -> {model.hparams.arflow.d_model}))")
    atok = model.encode_audio_tokens(audio, style, prosody)
    print(f"causal audio tokens    : {tuple(atok.shape)}")
    h = model.run_backbone(model._shift_motion(
        torch.randn(len(batch['motion']), T, model.latent_dim, device=dev)), atok)
    print(f"AR backbone hidden     : {tuple(h.shape)}")
    logits = model._code_logits(h)
    print(f"AR code logits         : {tuple(logits.shape)}  (B,T,subcodes=2,n_embed={model.n_embed})")
    z_hat, _ = model._logits_to_latent(logits)
    print(f"quantized AR latent    : {tuple(z_hat.shape)}")
    tt = torch.rand(len(batch['motion']), device=dev)
    v = model.flow_head(torch.randn_like(z_hat), tt, z_hat, audio_tokens=atok,
                        hidden=h if model.flow_use_hidden else None)
    print(f"flow head velocity out : {tuple(v.shape)}  (== [B,T,{model.latent_dim}])")
    dec = model.motion_prior.motion_decoder(
        model.motion_prior.quantize(z_hat)[0], use_chunked_attn=True)
    print(f"Stage-1 decoded motion : {tuple(dec.shape)}  (== [B,T,53] = 50 exp + 3 jaw)")


def temporal_check(model, batch):
    banner("3.  TEMPORAL ALIGNMENT  (numeric, clip 0)")
    n = len(batch['audio'][0])
    dur = n / 16000.0
    mframes = batch['motion'][0].shape[1]
    print(f"audio duration            : {dur:.3f} s   ({n} samples @ 16 kHz)")
    print(f"motion frame count        : {mframes}   -> {mframes / 25:.3f} s @ 25 fps")
    if 'mimi_codes' in batch:
        tm = batch['mimi_codes'][0].shape[1]
        print(f"Mimi token count          : {tm}   -> {tm / 12.5:.3f} s @ 12.5 Hz")
        mode = getattr(model, "moshi_align_mode", "interp")
        print(f"align mode                : {mode}   (2*T_mimi = {2 * tm})")
        ra, rm, _ = model._align_audio_motion(batch)
        L = ra[0].shape[1]
        print(f"aligned audio == motion   : {L} == {rm[0].shape[1]}   (2*T_mimi - motion = {2 * tm - mframes:+d})")
        assert L == rm[0].shape[1], "audio/motion length mismatch"
        if mode == "interp":
            # linear resample -> exactly the motion grid (HuBERT path may cap it in fusion)
            assert L == rm[0].shape[1] and L <= mframes
        else:  # hold
            assert L <= min(2 * tm, mframes)


def grad_freeze_audit(model, batch):
    banner("4.  FORWARD + BACKWARD  ->  param / grad / freeze audit")
    model.train()
    model.zero_grad(set_to_none=True)
    loss = model.allsplit_step("train", {k: (list(v) if isinstance(v, list) else v)
                                         for k, v in batch.items()}, 0)
    print(f"train loss (1 batch)   : {loss.item():.4f}   finite={torch.isfinite(loss).item()}")
    assert torch.isfinite(loss), "non-finite loss"
    loss.backward()

    print(f"\n{'Module':40s} {'State':10s} {'Params':>13s} {'Grad?':>6s}")
    print("-" * 74)
    for label, state, n_params, n_grad in model.param_groups_report():
        got_grad = "?"
        if label.startswith("Stage1"):
            got_grad = "YES(!!)" if _any_grad(model.motion_prior) else "NO"
        print(f"{label:40s} {state:10s} {n_params:13,d} {'YES' if n_grad else 'NO':>6s}")
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("-" * 74)
    print(f"total     {total:,}\ntrainable {train:,}\nfrozen    {total - train:,}")

    banner("5.  FROZEN MODULES RECEIVED NO GRADIENT")
    s1 = all(p.grad is None for p in model.motion_prior.parameters())
    print(f"all(p.grad is None for p in stage1.parameters())      : {s1}")
    assert s1, "Stage-1 received gradients!"
    if model.moshi_emb is not None:
        mg = all(p.grad is None for p in model.moshi_emb.parameters())
        should_be_none = not model.moshi_emb.trainable
        print(f"all(p.grad is None for p in moshi_emb.parameters())   : {mg}   "
              f"(expected {should_be_none}: trainable={model.moshi_emb.trainable})")
        assert mg == should_be_none, "Moshi emb grad state does not match config!"
        if model.moshi_emb.trainable:
            gnorm = sum((p.grad.norm().item() ** 2) for p in model.moshi_emb.parameters()
                        if p.grad is not None) ** 0.5
            print(f"Moshi emb grad L2 norm (Exp3)                         : {gnorm:.4e}  (>0 expected)")
            assert gnorm > 0
    for name in ("audio_feature_map", "prosody_feature_map", "backbone", "flow_head", "ar_code_heads"):
        mod = getattr(model, name)
        has = any(p.grad is not None and p.grad.abs().sum() > 0 for p in mod.parameters())
        print(f"{name:20s} has non-zero grad                    : {has}")
        assert has, f"{name} got no gradient"


def _any_grad(module):
    return any(p.grad is not None for p in module.parameters())


def leakage(model):
    banner("6.  CAUSAL LEAKAGE TEST  (perturb t>t0 must not change outputs <=t0)")
    ok = model.leakage_test(T=32, t0=10, device=model.device)
    print(f"leakage_test passed: {ok}")
    assert ok


def overfit(model, batch):
    banner("7.  TINY OVERFIT  (2 samples, 80 steps)")
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    model.train()
    losses = []
    for step in range(80):
        opt.zero_grad(set_to_none=True)
        loss = model.allsplit_step("train", {k: (list(v) if isinstance(v, list) else v)
                                             for k, v in batch.items()}, 0)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 10 == 0 or step == 79:
            print(f"  step {step:3d}   loss {loss.item():.4f}")
    drop = (losses[0] - min(losses[-10:])) / max(abs(losses[0]), 1e-6)
    print(f"\nloss[0]={losses[0]:.4f}  ->  min(last10)={min(losses[-10:]):.4f}   "
          f"relative drop {drop * 100:.1f}%")
    assert min(losses[-10:]) < losses[0], "loss did not decrease -- conditioning may not reach Stage-2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, choices=list(EXPERIMENTS))
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--skip-overfit", action="store_true")
    args = ap.parse_args()

    if not STAGE1_CKPT.exists():
        sys.exit(f"Stage-1 checkpoint not found: {STAGE1_CKPT}")

    data_name, model_name, expect_cond = EXPERIMENTS[args.experiment]
    banner(f"EXPERIMENT: {args.experiment}   data={data_name}  model={model_name}")
    cfg, dm, model = build(data_name, model_name)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    batch = one_batch(dm)

    shape_trace(model, batch, expect_cond)
    temporal_check(model, batch)
    grad_freeze_audit(model, batch)
    leakage(model)
    if not args.skip_overfit:
        # fresh model so overfit isn't polluted by the audit's backward
        _, _, model2 = build(data_name, model_name)
        model2 = model2.to(dev)
        overfit(model2, batch)

    banner(f"ALL CHECKS PASSED for {args.experiment}")


if __name__ == "__main__":
    main()
