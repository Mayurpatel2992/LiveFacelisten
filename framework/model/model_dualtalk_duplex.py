"""Shared base for the listen+speak (duplex) models on DualTalk.

The agent's motion is generated from two separate audio streams:
    own     : the agent's own voice (Moshi's output stream in deployment)
    partner : the other person's voice (the user's microphone stream)

Both are the MIBURI/Exp-5 Moshi features (Mimi codes + Inner-Monologue text),
each passed through a causal audio front-end:
    own     -> the trained speaking model's front-end (frozen)
    partner -> a trainable copy of it, initialised from the same weights

Everything starts from a trained DualTalk speaking model
(configs/model/model_dualtalk_speak.yaml, own track only), loaded from
``duplex.speak_ckpt_path`` before any new module is added, then frozen.
Subclasses add the listening path through four hooks:

    build_duplex(dcfg)                                   new modules + requires_grad
    duplex_hidden(x, own_tok, par_tok, gate, cache)      backbone -> h  (cache=None: parallel)
    new_duplex_cache()                                   incremental state for streaming
    duplex_gate(own_tok, vad, split)                     [B,T,1] gate or None

Option A (UniLS-style):   framework/model/model_dualtalk_unils.py
Option B (ControlNet):    framework/model/model_dualtalk_controlnet.py

Causality: every new path is causal (frame t sees audio <= t and motion < t);
``duplex_leakage_test`` checks it bit-exactly like the base ``leakage_test``.
"""
import logging
import random
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from torch import Tensor

from framework.model.model_vqvae_arflow_pred import RotaryEmbedding, VqvaeARFlowPredict, align_moshi

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- #
#  Building blocks                                                              #
# ----------------------------------------------------------------------------- #
class LoRALinear(nn.Module):
    """Frozen nn.Linear + trainable low-rank update (B zero-init -> starts as base)."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_b.weight)
        self.scale = alpha / rank

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scale


class CausalCrossAttention(nn.Module):
    """Motion frame t attends to audio tokens 0..t (optionally the last `window`).

    Memory tokens are aligned one-to-one with the query frames (audio token t
    belongs to motion frame t), so the causal mask is the same lower triangle
    as self-attention, and in streaming each step appends one memory K/V to
    the cache. RoPE on q (query frame) and k (memory frame) gives relative
    recency. The output projection is zero-initialised: a freshly added block
    leaves the residual stream unchanged.
    """

    def __init__(self, dim: int, num_heads: int, window: int = 0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window = int(window)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: Tensor, mem: Tensor, kv_cache: Optional[dict] = None) -> Tensor:
        B, S, D = x.shape
        H, hd = self.num_heads, self.head_dim
        start = 0
        if kv_cache is not None and kv_cache.get("k") is not None:
            start = kv_cache["pos"]
        q = self.rope(self.q(x).reshape(B, S, H, hd).transpose(1, 2), start_pos=start)
        kv = self.kv(mem).reshape(B, S, 2, H, hd).permute(2, 0, 3, 1, 4)
        k, v = self.rope(kv[0], start_pos=start), kv[1]
        if kv_cache is not None:
            if kv_cache.get("k") is not None:
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            if self.window > 0:                       # keep only what can still be attended
                k, v = k[:, :, -self.window:], v[:, :, -self.window:]
            kv_cache.update(k=k, v=v, pos=start + S)
        K = k.shape[2]
        q_pos = torch.arange(start, start + S, device=x.device).unsqueeze(1)
        k_pos = torch.arange(start + S - K, start + S, device=x.device).unsqueeze(0)
        keep = k_pos <= q_pos
        if self.window > 0:
            keep = keep & (k_pos > q_pos - self.window)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)
        return self.proj(out.transpose(1, 2).reshape(B, S, D))


# ----------------------------------------------------------------------------- #
#  Duplex base model                                                            #
# ----------------------------------------------------------------------------- #
class DualTalkDuplexBase(VqvaeARFlowPredict):
    def __init__(self, nfeats: int, split_path: str, one_hot_dim: List,
                 resumed_training: bool, **kwargs):
        super().__init__(nfeats=nfeats, split_path=split_path, one_hot_dim=one_hot_dim,
                         resumed_training=resumed_training, **kwargs)
        if self.audio_conditioning != 'moshi':
            raise ValueError("the duplex models expect arflow.audio_conditioning='moshi'")
        dcfg = self.hparams.duplex

        # 1) the trained speaking model, then freeze all of it
        self._load_speak_checkpoint(dcfg.speak_ckpt_path)
        for p in self.parameters():
            p.requires_grad = False
        self.moshi_emb.trainable = False
        if self.moshi_text_emb is not None:
            self.moshi_text_emb.trainable = False

        # 2) partner audio front-end: trainable copy of the trained own one
        d = self.hparams.arflow.d_model
        self.partner_audio_feature_map = nn.Linear(self.audio_dim, d)
        self.partner_audio_feature_map.load_state_dict(self.audio_feature_map.state_dict())
        self.partner_audio_encoder = type(self.audio_encoder)(
            in_dim=d, dim=d, num_layers=self.hparams.arflow.audio_layers,
            num_heads=self.hparams.arflow.num_heads, ffn_dim=self.hparams.arflow.ffn_dim,
            kernel_size=self.hparams.arflow.audio_kernel, dropout=self.hparams.arflow.dropout)
        self.partner_audio_encoder.load_state_dict(self.audio_encoder.state_dict())

        # 3) option-specific modules
        self.build_duplex(dcfg)

        # re-create the optimizer over what is now trainable
        self.optimizer = instantiate(self.hparams.optim,
                                     params=[p for p in self.parameters() if p.requires_grad])
        self.log_param_table()

    # ---- hooks --------------------------------------------------------- #
    def build_duplex(self, dcfg):
        raise NotImplementedError

    def duplex_hidden(self, x: Tensor, own_tok: Tensor, par_tok: Tensor,
                      gate: Optional[Tensor], cache=None) -> Tensor:
        raise NotImplementedError

    def new_duplex_cache(self):
        raise NotImplementedError

    def duplex_gate(self, own_tok: Tensor, vad: Optional[Tensor], split: str) -> Optional[Tensor]:
        return None

    def extra_losses(self, own_tok: Tensor, vad: Tensor, mask: Tensor, split: str) -> dict:
        return {}

    # ---- setup --------------------------------------------------------- #
    def _load_speak_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"speaking checkpoint does not match the speaking model: "
                               f"missing={missing[:5]} unexpected={unexpected[:5]}")
        logger.info(f"Loaded speaking model (frozen base):\n    {path}")

    def param_groups_report(self):
        rows = super().param_groups_report()
        extra = {
            "Partner audio front-end": ("partner_audio_feature_map", "partner_audio_encoder"),
            "Duplex modules (option)": self.duplex_param_prefixes(),
        }
        for label, prefixes in extra.items():
            ps = [p for n, p in self.named_parameters()
                  if any(n == q or n.startswith(q + ".") for q in prefixes)]
            if ps:
                n_params = sum(p.numel() for p in ps)
                n_grad = sum(p.numel() for p in ps if p.requires_grad)
                state = "TRAINABLE" if n_grad == n_params else ("FROZEN" if n_grad == 0 else "PARTIAL")
                rows.append((label, state, n_params, n_grad))
        return rows

    def duplex_param_prefixes(self):
        return ()

    # ---- audio ----------------------------------------------------------- #
    def encode_partner_tokens(self, audio: Tensor, style_one_hot: Tensor) -> Tensor:
        feat = self.partner_audio_feature_map(audio)
        feat = feat * self.style_embedding(style_one_hot).unsqueeze(1)
        return self.partner_audio_encoder(feat)

    def _align_duplex(self, batch):
        """Per item: own / partner Moshi features, motion, own VAD on the 25 fps grid."""
        own_l, par_l, mot_l, vad_l = [], [], [], []
        for i in range(len(batch['motion'])):
            m = batch['motion'][i]
            T = m.shape[1]
            fo = self._moshi_feature(batch['mimi_codes'][i], self._moshi_text_codes(batch, i))
            text_p = batch['partner_moshi_text_codes'][i] if self.moshi_text_emb is not None else None
            fp = self._moshi_feature(batch['partner_mimi_codes'][i], text_p)
            ao = align_moshi(fo, T, self.moshi_align_mode)
            ap = align_moshi(fp, T, self.moshi_align_mode)
            T = min(T, ao.shape[1], ap.shape[1])           # 'hold' can be short
            v = batch['vad'][i][:T].to(ao.device)
            if v.shape[0] < T:
                v = F.pad(v, (0, T - v.shape[0]))
            own_l.append(ao[:, :T]); par_l.append(ap[:, :T])
            mot_l.append(m[:, :T]); vad_l.append(v[None])
        return own_l, par_l, mot_l, vad_l

    @staticmethod
    def _crop_or_pad(x: Tensor, start: int, crop_len: int) -> Tensor:
        x = x[:, start:start + crop_len]
        if x.shape[1] < crop_len:
            pad = [0, 0] * (x.dim() - 2) + [0, crop_len - x.shape[1]]
            x = F.pad(x, pad)
        return x

    # ---- backbone -------------------------------------------------------- #
    def _duplex_run(self, motion_shifted: Tensor, own_tok: Tensor, par_tok: Tensor,
                    gate: Optional[Tensor]) -> Tensor:
        x = self._build_backbone_input(motion_shifted, own_tok)
        return self.duplex_hidden(x, own_tok, par_tok, gate, cache=None)

    # ---- training / validation ----------------------------------------- #
    def allsplit_step(self, split: str, batch, batch_idx):
        style = self._style_one_hot(batch).to(self.device, dtype=torch.float32)
        own_l, par_l, mot_l, vad_l = self._align_duplex(batch)

        if split == "train":
            crop_len = self.crop_length
            starts = [random.randint(0, max(0, m.shape[1] - crop_len)) for m in mot_l]
            seq_lengths = [min(m.shape[1], crop_len) for m in mot_l]
            own = torch.cat([self._crop_or_pad(a, s, crop_len) for a, s in zip(own_l, starts)])
            par = torch.cat([self._crop_or_pad(a, s, crop_len) for a, s in zip(par_l, starts)])
            motion = torch.cat([self._crop_or_pad(m, s, crop_len) for m, s in zip(mot_l, starts)])
            vad = torch.cat([self._crop_or_pad(v, s, crop_len) for v, s in zip(vad_l, starts)])
        else:
            seq_lengths = [m.shape[1] for m in mot_l]
            own, par, motion, vad = (torch.cat(x) for x in (own_l, par_l, mot_l, vad_l))

        own = own.to(self.device, dtype=torch.float32)
        par = par.to(self.device, dtype=torch.float32)
        motion = motion.to(self.device, dtype=torch.float32)
        vad = vad.to(self.device, dtype=torch.float32)                     # [B,T]
        assert own.shape[-1] == self.audio_dim and par.shape[-1] == self.audio_dim
        assert motion.shape[-1] == self.nfeats, f"Motion dim mismatch: {motion.shape[-1]}"
        B, T, _ = motion.shape

        mask = torch.zeros(B, T, device=motion.device, dtype=motion.dtype)
        for i, length in enumerate(seq_lengths):
            mask[i, :min(length, T)] = 1.0
        valid_mask = mask.bool()

        self.motion_prior.eval()
        with torch.no_grad():
            z_m = self.motion_prior.motion_encoder(motion)
            z0, _, quant_info = self.motion_prior.quantize(z_m)
            z0 = z0.detach()
            target_codes = quant_info[2].detach().long()

        own_tok = self.encode_audio_tokens(own, style, None)               # [B,T,d]
        par_tok = self.encode_partner_tokens(par, style)                   # [B,T,d]
        gate = self.duplex_gate(own_tok, vad, split)                       # [B,T,1] | None

        # --- history corruption: identical to the speaking model ---
        motion_input = z0
        if split == "train" and self.ss_enable:
            p = self._scheduled_sampling_prob()
            if p > 0.0:
                with torch.no_grad():
                    h0 = self._duplex_run(self._shift_motion(motion_input), own_tok, par_tok, gate)
                    pred_latent0, _ = self._logits_to_latent(self._code_logits(h0))
                replace = (torch.rand(B, T, device=motion.device) < p) & valid_mask
                motion_input = torch.where(replace.unsqueeze(-1), pred_latent0.detach(), z0)
        if split == "train" and self.history_dropout > 0.0:
            drop = (torch.rand(B, T, device=motion.device) < self.history_dropout) & valid_mask
            motion_input = self._apply_uncertainty_mask(motion_input, drop)
        use_um = (split == "train" and self.um_enable and self.trainer is not None
                  and self.trainer.current_epoch >= self.um_start_epoch)
        if use_um:
            with torch.no_grad():
                h_probe = self._duplex_run(self._shift_motion(motion_input), own_tok, par_tok, gate)
                unc = self._token_uncertainty(self._code_logits(h_probe))
                uncertain_mask = self._find_uncertain_tokens(unc, valid_mask=valid_mask)
            motion_input = self._apply_uncertainty_mask(motion_input, uncertain_mask)

        h = self._duplex_run(self._shift_motion(motion_input), own_tok, par_tok, gate)

        code_logits = self._code_logits(h)
        z_hat, _ = self._logits_to_latent(code_logits, differentiable=(split == "train"))

        t = torch.rand(B, device=self.device)
        z1 = torch.randn_like(z0)
        z_t = self.scheduler.interpolate(z0, z1, t)
        v_target = self.scheduler.target_velocity(z0, z1)
        hidden = h if self.flow_use_hidden else None
        flow_history = self._teacher_flow_history(z0, valid_mask, split)
        v_hat = self.flow_head(z_t, t, z_hat, audio_tokens=own_tok, hidden=hidden,
                               history=flow_history)

        def masked(per_frame, m=mask):
            return (per_frame * m).sum() / m.sum().clamp(min=1.0)

        ar_per = F.cross_entropy(code_logits.reshape(-1, self.n_embed), target_codes.reshape(-1),
                                 reduction='none').reshape(B, T, self.num_subcodes).mean(dim=-1)
        loss_ar = masked(ar_per)
        loss_flow = masked(F.mse_loss(v_hat, v_target, reduction='none').mean(dim=-1))
        z0_hat = z_t - t.view(B, 1, 1) * v_hat
        loss_lat = masked(F.l1_loss(z0_hat, z0, reduction='none').mean(dim=-1))
        z0_hat_q, _, _ = self.motion_prior.quantize(z0_hat)
        motion_pred = self.motion_prior.motion_decoder(z0_hat_q, use_chunked_attn=(split != "train"))
        e = self.exp_dims
        exp_per = F.l1_loss(motion_pred[..., :e], motion[..., :e], reduction='none').mean(dim=-1)
        jaw_per = F.l1_loss(motion_pred[..., e:], motion[..., e:], reduction='none').mean(dim=-1)
        loss_exp, loss_jaw = masked(exp_per), masked(jaw_per)

        flow_scale = self._flow_warmup_scale() if split == "train" else 1.0
        loss = (self.lambda_ar * loss_ar
                + flow_scale * (self.lambda_flow * loss_flow + self.lambda_lat * loss_lat
                                + self.lambda_exp * loss_exp + self.lambda_jaw * loss_jaw))
        extra = self.extra_losses(own_tok, vad, mask, split)
        for name, (val, weight) in extra.items():
            loss = loss + weight * val

        with torch.no_grad():
            code_correct = (code_logits.argmax(dim=-1) == target_codes).float().mean(dim=-1)
            ar_acc = masked(code_correct)
            speak_m, listen_m = mask * vad, mask * (1.0 - vad)
            split_logs = [("exp_speak", masked(exp_per, speak_m)), ("exp_listen", masked(exp_per, listen_m)),
                          ("jaw_speak", masked(jaw_per, speak_m)), ("jaw_listen", masked(jaw_per, listen_m))]

        logs = [("ar", loss_ar), ("flow", loss_flow), ("lat", loss_lat), ("exp", loss_exp),
                ("jaw", loss_jaw), ("total", loss), ("ar_acc", ar_acc)] + split_logs
        logs += [(name, val) for name, (val, _) in extra.items()]
        for name, val in logs:
            self.log(f"{name}/{split}", val.detach(), on_step=False, on_epoch=True,
                     prog_bar=False, sync_dist=True)

        if split == "val":
            self.metrics.update(motion_pred.detach(), motion.detach(), seq_lengths)
            if batch_idx < self.free_run_val_batches:
                free_t = min(T, int(self.free_run_val_max_frames), int(seq_lengths[0]))
                if free_t > 0:
                    cuda_devices = ([motion.device.index]
                                    if motion.is_cuda and motion.device.index is not None else [])
                    with torch.random.fork_rng(devices=cuda_devices):
                        torch.manual_seed(self.free_run_val_seed + batch_idx)
                        g_inf = self.infer_gate(own_tok[:1, :free_t], vad[:1, :free_t])
                        free_motion = self.generate_duplex_batched(
                            own_tok[:1, :free_t], par_tok[:1, :free_t], g_inf, num_samples=1)[0]
                    free_target = motion[:1, :free_t]
                    self.log("free_run_l1/val", F.l1_loss(free_motion, free_target), on_step=False,
                             on_epoch=True, batch_size=1, sync_dist=True)
                    if free_t > 1:
                        self.log("free_run_vel_l1/val",
                                 F.l1_loss(free_motion[:, 1:] - free_motion[:, :-1],
                                           free_target[:, 1:] - free_target[:, :-1]),
                                 on_step=False, on_epoch=True, batch_size=1, sync_dist=True)

        if split == "train":
            self.log("loss_train", loss.detach(), prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
            self.log("flow_scale", flow_scale, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        if split == "val":
            self.log("loss_val", loss.detach(), prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    # ---- inference ------------------------------------------------------ #
    def infer_gate(self, own_tok: Tensor, vad: Optional[Tensor]) -> Optional[Tensor]:
        """Gate used at inference (subclasses: classifier / oracle VAD / none)."""
        return None

    @torch.no_grad()
    def _duplex_autoregressive_latents(self, own_tok: Tensor, par_tok: Tensor,
                                       gate: Optional[Tensor], sampling_steps: int) -> Tensor:
        """Cached free-running loop: frame t sees own/partner audio <= t, motion < t."""
        device = own_tok.device
        B, T, _ = own_tok.shape
        cache = self.new_duplex_cache()
        prev_motion = self.bos_motion.expand(B, 1, self.latent_dim).to(device=device, dtype=own_tok.dtype)
        flow_hist = self._new_flow_history(B, device, own_tok.dtype)
        latents = []
        for t in range(T):
            a_t, p_t = own_tok[:, t:t + 1], par_tok[:, t:t + 1]
            g_t = gate[:, t:t + 1] if gate is not None else None
            x_t = self._build_backbone_input(prev_motion, a_t)
            h_t = self.duplex_hidden(x_t, a_t, p_t, g_t, cache=cache)
            z_hat_t, _, _ = self._predict_latent(h_t)
            z0_hat_t = self._flow_ode_step(z_hat_t, a_t, sampling_steps,
                                           hidden_t=h_t if self.flow_use_hidden else None,
                                           history_t=self._flow_history_feature(flow_hist))
            flow_hist = self._push_flow_history(flow_hist, z0_hat_t)
            latents.append(z0_hat_t)
            prev_motion = z_hat_t
        return torch.cat(latents, dim=1)

    @torch.no_grad()
    def generate_duplex_batched(self, own_tok: Tensor, par_tok: Tensor, gate: Optional[Tensor],
                                num_samples: int, sampling_steps: Optional[int] = None) -> List[Tensor]:
        """N parallel free-running samples for one clip. Returns N x [1,T,53]."""
        self.eval()
        steps = sampling_steps or self.sampling_steps
        N = num_samples
        g = gate.expand(N, -1, -1) if gate is not None else None
        z0_hat = self._duplex_autoregressive_latents(own_tok.expand(N, -1, -1),
                                                     par_tok.expand(N, -1, -1), g, steps)
        z0_hat_q, _, _ = self.motion_prior.quantize(z0_hat)
        motion_pred = self.motion_prior.motion_decoder(z0_hat_q, use_chunked_attn=True)
        return [motion_pred[i:i + 1] for i in range(N)]

    @torch.no_grad()
    def forward_multi_sample(self, batch, num_samples=10, generation=False):
        """Evaluation entry: first item of the batch, own + partner streams."""
        self.to(self.device)
        style = self._style_one_hot(batch).to(self.device)[:1]
        own_l, par_l, _, vad_l = self._align_duplex(batch)
        own_tok = self.encode_audio_tokens(own_l[0].to(self.device), style, None)
        par_tok = self.encode_partner_tokens(par_l[0].to(self.device), style)
        gate = self.infer_gate(own_tok, vad_l[0].to(self.device))
        return self.generate_duplex_batched(own_tok, par_tok, gate, num_samples)

    def forward(self, batch, sample=False, generation=True) -> Tensor:
        return self.forward_multi_sample(batch, num_samples=1)[0]

    # ---- causality check ------------------------------------------------ #
    @torch.no_grad()
    def duplex_leakage_test(self, T: int = 32, t0: int = 10, device=None) -> bool:
        """Perturbing own/partner audio or motion after t0 must leave every output
        at <= t0 bit-exact; the incremental loop must match the parallel pass."""
        self.eval()
        device = device or self.device
        B = 2
        style = torch.ones(B, self.one_hot_total, device=device)
        own = torch.randn(B, T, self.audio_dim, device=device)
        par = torch.randn(B, T, self.audio_dim, device=device)
        mot = torch.randn(B, T, self.latent_dim, device=device)
        vad = (torch.rand(B, T, device=device) > 0.5).float()

        def run(o, p, m):
            ot = self.encode_audio_tokens(o, style, None)
            pt = self.encode_partner_tokens(p, style)
            g = self.duplex_gate(ot, vad, "val")
            return ot, pt, g, self._duplex_run(self._shift_motion(m), ot, pt, g)

        ot1, pt1, g1, h1 = run(own, par, mot)
        o2, p2, m2 = own.clone(), par.clone(), mot.clone()
        for x in (o2, p2, m2):
            x[:, t0 + 1:] += 5.0 * torch.randn_like(x[:, t0 + 1:])
        _, _, g2, h2 = run(o2, p2, m2)
        ok = True
        diffs = {"hidden": (h1 - h2)[:, :t0 + 1].abs().max().item()}
        if g1 is not None:
            diffs["gate"] = (g1 - g2)[:, :t0 + 1].abs().max().item()
        # incremental (streaming) pass == parallel pass, same teacher-forced motion
        cache = self.new_duplex_cache()
        shifted = self._shift_motion(mot)
        h_inc = []
        for t in range(T):
            x_t = self._build_backbone_input(shifted[:, t:t + 1], ot1[:, t:t + 1])
            h_inc.append(self.duplex_hidden(x_t, ot1[:, t:t + 1], pt1[:, t:t + 1],
                                            g1[:, t:t + 1] if g1 is not None else None, cache=cache))
        diffs["incremental_vs_parallel"] = (torch.cat(h_inc, 1) - h1).abs().max().item()
        for name, dmax in diffs.items():
            tol = 1e-4 if name == "incremental_vs_parallel" else 0.0
            passed = dmax <= tol
            ok = ok and passed
            logger.info(f"[duplex leakage] {name:24s} max|d|={dmax:.3e} {'OK' if passed else 'FAIL'}")
            print(f"[duplex leakage] {name:24s} max|d|={dmax:.3e} {'OK' if passed else 'FAIL'}")
        return ok
