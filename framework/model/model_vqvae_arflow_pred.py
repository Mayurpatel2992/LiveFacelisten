"""Stage-2 variant: genuinely autoregressive + streamable talking-head prediction.

Design (see project spec):
  - A causal (decoder-style) transformer backbone consumes, per frame t,
    [ embed(motion tokens z'_{<t})  ⊕  aligned audio a_{≤t} ]  (concat + proj, default)
    and produces a hidden state h_t.
  - Two heads on h_t:
        (1) Discrete AR head : score h_t against the *frozen* Stage-1 codebook →
            logits over the 2 sub-codes/frame → cross-entropy vs GT indices.
            A low-temperature Gumbel-Softmax gives a differentiable sampled token
            whose (frozen) codebook embedding ẑ_t conditions the flow head.
        (2) Flow head (per-token, streamable) : condition = concat[ẑ_t, h_t];
            inputs = timestep t_d, aligned audio token a_t, noisy motion latent
            z_t = (1-t_d) z0 + t_d z1  →  predicts velocity  →  flow-matching loss
            (+ latent / exp / jaw L1 through the frozen decoder).

  - AR is *real*, not just masked:
        Training  : teacher-forced, parallel, causal mask + one-step input shift
                    (frame t predicted from motion <t, audio ≤t). Optional
                    scheduled sampling to cut exposure bias.
        Inference : sequential feedback loop (KV-cached) — the token predicted at t
                    is appended and fed back as input for t+1.

  - Causality / no-leakage (hard requirements):
        * backbone self-attention is causal-masked, input shifted by one;
        * audio is made causal via a dedicated CausalAudioEncoder (+ optional
          chunked/streaming HuBERT extraction);
        * flow head is strictly per-token (no temporal mixing);
        * RoPE positions for streaming stability;
        * a leakage unit test asserts outputs at ≤t are bit-exact unchanged when
          any input at time >t is perturbed.

Stage-1 (LiveVQFormer / VQVAE) stays frozen; only the modules here are trained.
"""

import os
import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional
from hydra.utils import instantiate
import logging

from framework.model.utils.tools import load_checkpoint, create_one_hot, resample_input
from framework.model.utils.diffusion_mlp import SinusoidalTimeEmbedding, FlowMatchingScheduler
from framework.model.metrics.compute import ComputeMetrics
from framework.model.base import BaseModel
from framework.data.utils import get_split_keyids

logger = logging.getLogger(__name__)


def align_hold_x2(feat: Tensor, target_len: int) -> Tensor:
    """Causal 12.5 Hz -> 25 fps hold: M_t -> motion frames 2t and 2t+1.

    Frame 2t / 2t+1 depend on Mimi steps <= t only -> strictly causal. Used for
    true frame-by-frame streaming inference. Right-crops to <= target_len; a
    caller that also has motion trims it to match.

    feat: [B, T_mimi, D]  ->  [B, min(2*T_mimi, target_len), D]
    """
    held = feat.repeat_interleave(2, dim=1)                  # [B, 2*T_mimi, D]
    if held.shape[1] > target_len:
        held = held[:, :target_len]
    return held


def align_moshi(feat: Tensor, target_len: int, mode: str = "interp") -> Tensor:
    """12.5 Hz Mimi feature -> 25 fps motion grid.

    mode='interp'  (offline training / offline evaluation): linear resample to
        exactly ``target_len``. Smooth and non-causal -- sanctioned for offline
        use and avoids the 12.5 Hz "double-step" a pure hold produces (both
        frames of a pair would get byte-identical conditioning).
    mode='hold'    (streaming): causal repeat-interleave x2, cropped to
        <= target_len; caller trims motion to match.

    feat: [B, T_mimi, D]  ->  interp: [B, target_len, D] | hold: [B, <=target_len, D]
    """
    if mode == "interp":
        x = feat.transpose(1, 2)                             # [B, D, T_mimi]
        x = F.interpolate(x, size=int(target_len), mode="linear", align_corners=False)
        return x.transpose(1, 2)                             # [B, target_len, D]
    if mode == "hold":
        return align_hold_x2(feat, target_len)
    raise ValueError(f"unknown moshi align mode {mode!r}")


# ----------------------------------------------------------------------------- #
#  Rotary positional embedding (RoPE) — relative positions for streaming        #
# ----------------------------------------------------------------------------- #
class RotaryEmbedding(nn.Module):
    """Standard rotary positional embedding applied to Q/K per head-dim."""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even, got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _cos_sin(self, positions: Tensor, dtype):
        # positions: [S] -> angles [S, head_dim/2]
        angles = positions.float().unsqueeze(-1) * self.inv_freq.unsqueeze(0)
        cos = torch.cos(angles).repeat_interleave(2, dim=-1)  # [S, head_dim]
        sin = torch.sin(angles).repeat_interleave(2, dim=-1)  # [S, head_dim]
        return cos.to(dtype), sin.to(dtype)

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        # (..., 2k, 2k+1) -> (..., -x_{2k+1}, x_{2k})
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def forward(self, x: Tensor, start_pos: int = 0) -> Tensor:
        # x: [B, H, S, head_dim]
        S = x.shape[-2]
        positions = torch.arange(start_pos, start_pos + S, device=x.device)
        cos, sin = self._cos_sin(positions, x.dtype)          # [S, head_dim]
        cos = cos.unsqueeze(0).unsqueeze(0)                    # [1,1,S,head_dim]
        sin = sin.unsqueeze(0).unsqueeze(0)
        return x * cos + self._rotate_half(x) * sin


# ----------------------------------------------------------------------------- #
#  Causal self-attention with RoPE + optional KV cache (for streaming)          #
# ----------------------------------------------------------------------------- #
class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: Tensor, kv_cache: Optional[dict] = None):
        """x: [B, S, D].
        If kv_cache is provided (dict with 'k','v' or empty), runs incremental
        decoding: past K/V are prepended and the cache is updated in place.
        Returns out [B, S, D]. Causality is guaranteed by the triangular mask
        (full pass) or by construction (incremental, S==1 queries only attend
        to cached past + themselves).
        """
        B, S, D = x.shape
        H, hd = self.num_heads, self.head_dim

        start_pos = 0
        if kv_cache is not None and kv_cache.get("k") is not None:
            start_pos = kv_cache["k"].shape[2]  # number of cached time steps

        qkv = self.qkv(x).reshape(B, S, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]         # each [B, H, S, hd]

        # RoPE on absolute positions (start_pos accounts for cached prefix)
        q = self.rope(q, start_pos=start_pos)
        k = self.rope(k, start_pos=start_pos)

        # Update / read KV cache for incremental (streaming) decoding.
        if kv_cache is not None and kv_cache.get("k") is not None:
            k = torch.cat([kv_cache["k"], k], dim=2)
            v = torch.cat([kv_cache["v"], v], dim=2)
        if kv_cache is not None:
            kv_cache["k"] = k
            kv_cache["v"] = v

        K = k.shape[2]                            # total key length (= start_pos + S)
        attn_mask, is_causal = None, False
        if start_pos == 0 and S > 1:
            # Full parallel pass (training): standard square causal mask.
            is_causal = True
        elif S > 1:
            # Priming a chunk on top of an existing cache: query i (abs pos
            # start_pos+i) may attend to keys with abs pos ≤ start_pos+i only.
            q_pos = torch.arange(start_pos, start_pos + S, device=x.device).unsqueeze(1)
            k_pos = torch.arange(K, device=x.device).unsqueeze(0)
            attn_mask = (k_pos <= q_pos)          # [S, K] bool, True = keep
        # else S == 1: a single query legitimately attends to the whole past +
        # itself -> no future leakage, no mask needed.

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).contiguous().reshape(B, S, D)
        return self.proj(out)


class CausalTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, x: Tensor, kv_cache: Optional[dict] = None):
        x = x + self.attn(self.norm1(x), kv_cache=kv_cache)
        x = x + self.ffn(self.norm2(x))
        return x


class CausalTransformerBackbone(nn.Module):
    """Decoder-style causal transformer over fused (motion ⊕ audio) tokens."""

    def __init__(self, dim: int, num_layers: int, num_heads: int,
                 ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            CausalTransformerBlock(dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, kv_caches: Optional[List[dict]] = None):
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            x = layer(x, kv_cache=cache)
        return self.norm(x)

    def new_kv_caches(self):
        return [dict() for _ in self.layers]


# ----------------------------------------------------------------------------- #
#  Causal audio encoder (makes audio a_t depend on audio ≤ t only)              #
# ----------------------------------------------------------------------------- #
class CausalConv1d(nn.Conv1d):
    """Left-padded (causal) 1D convolution."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.left_pad = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x):  # x: [B, C, T]
        x = F.pad(x, (self.left_pad, 0))
        return super().forward(x)


class CausalAudioEncoder(nn.Module):
    """HuBERT (frame-paired) features -> causal per-frame audio tokens a_t.

    Causality: a causal depthwise conv stack (left padding) followed by a
    causal transformer.  a_t depends only on audio features ≤ t.

    Supports both a parallel `forward` (training) and a truly incremental
    `step` (online streaming, one frame at a time) that reproduces `forward`
    bit-exactly via a conv ring-buffer + transformer KV cache.
    """

    def __init__(self, in_dim: int, dim: int, num_layers: int, num_heads: int,
                 ffn_dim: int, kernel_size: int = 5, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.in_proj = nn.Linear(in_dim, dim)
        self.conv = nn.Sequential(
            CausalConv1d(dim, dim, kernel_size, groups=dim),
            nn.GELU(),
        )
        self.conv_norm = nn.LayerNorm(dim)
        self.transformer = CausalTransformerBackbone(dim, num_layers, num_heads, ffn_dim, dropout)

    def forward(self, audio_feat: Tensor) -> Tensor:
        # audio_feat: [B, T, in_dim]  ->  a: [B, T, dim]
        x = self.in_proj(audio_feat)
        x = x + self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.conv_norm(x)
        a = self.transformer(x)
        return a

    # ---- incremental streaming (one frame at a time) ---- #
    def init_state(self, batch_size: int, device) -> dict:
        k = self.kernel_size
        return {
            # ring buffer of the last (k-1) in_proj frames (zero = causal left pad)
            "conv_buf": torch.zeros(batch_size, self.dim, k - 1, device=device),
            "kv": self.transformer.new_kv_caches(),
        }

    def step(self, audio_feat_t: Tensor, state: dict) -> Tensor:
        """audio_feat_t: [B, 1, in_dim] -> a_t: [B, 1, dim]. Updates `state`."""
        conv0 = self.conv[0]                          # CausalConv1d (depthwise)
        x = self.in_proj(audio_feat_t)               # [B,1,dim]
        xt = x.transpose(1, 2)                        # [B,dim,1]
        conv_in = torch.cat([state["conv_buf"], xt], dim=2)   # [B,dim,k]
        state["conv_buf"] = conv_in[:, :, 1:]        # keep last k-1
        c = F.conv1d(conv_in, conv0.weight, conv0.bias,
                     stride=1, padding=0, dilation=conv0.dilation, groups=conv0.groups)  # [B,dim,1]
        c = F.gelu(c)
        x = x + c.transpose(1, 2)                     # residual, [B,1,dim]
        x = self.conv_norm(x)
        for i, layer in enumerate(self.transformer.layers):
            x = layer(x, kv_cache=state["kv"][i])
        return self.transformer.norm(x)              # [B,1,dim]


# ----------------------------------------------------------------------------- #
#  Per-token flow-matching head (strictly per-frame -> streamable, no leakage)  #
# ----------------------------------------------------------------------------- #
class ARFlowMatchingHead(nn.Module):
    """Predicts flow-matching velocity for a single frame's 256-d motion latent.

    Operates independently per frame (no temporal mixing) so it is trivially
    causal / streamable.  The four inputs (per the AR-flow spec) are concatenated:
        (1) audio token          a_t         [B, T, audio_dim]
        (2) noisy motion token   z_t         [B, T, latent_dim]
        (3) quantized AR token   ẑ_t         [B, T, latent_dim]   (from discrete head)
        (4) timestamp            t_d         [B] or [B, T]  -> sinusoidal embedding
    Optionally, the AR backbone hidden h_t may be appended (config-gated).

    Optionally (history_frames K > 0), the head also sees the previous K clean
    continuous latents z0_{t-1..t-K} plus a validity flag per frame, as a
    [B, T, K*latent_dim + K] feature. history_mode selects where it enters:
        'cond'  : concatenated with the conditions (t, audio, ẑ_t, h_t)
        'input' : linearly projected and added to the noisy input z_t
    """

    def __init__(self, latent_dim: int, audio_dim: int,
                 time_dim: int = 256, hidden_dim: int = 512,
                 audio_proj_dim: int = 256, extra_cond_dim: int = 0,
                 history_frames: int = 0, history_mode: str = 'cond'):
        super().__init__()
        self.audio_proj = nn.Sequential(nn.Linear(audio_dim, audio_proj_dim), nn.SiLU())
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        if history_mode not in ('cond', 'input'):
            raise ValueError(f"history_mode must be 'cond' or 'input', got {history_mode!r}")
        self.history_frames = history_frames
        self.history_mode = history_mode
        history_dim = history_frames * (latent_dim + 1)      # K latents + K validity flags
        self.history_proj = None
        if history_frames > 0 and history_mode == 'input':
            self.history_proj = nn.Linear(history_dim, latent_dim)
        # noisy z_t + quantized ẑ_t + time-emb + audio (+ optional hidden h_t)
        # (+ history when history_mode == 'cond')
        in_dim = latent_dim + latent_dim + time_dim + audio_proj_dim + extra_cond_dim
        if history_frames > 0 and history_mode == 'cond':
            in_dim += history_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_t: Tensor, t: Tensor, z_hat: Tensor, audio_tokens: Tensor,
                hidden: Optional[Tensor] = None, history: Optional[Tensor] = None) -> Tensor:
        # z_t/z_hat/audio/hidden/history: [B, T, *]   t: [B] or [B, T]
        if (history is None) != (self.history_frames == 0):
            raise ValueError(f"flow head built with history_frames={self.history_frames} "
                             f"but history={'missing' if history is None else 'given'}")
        if t.dim() == 1:
            t_emb = self.time_mlp(t).unsqueeze(1).expand(-1, z_t.shape[1], -1)
        else:
            B, T = t.shape
            t_emb = self.time_mlp(t.reshape(-1)).reshape(B, T, -1)
        if self.history_proj is not None:
            z_t = z_t + self.history_proj(history)
        parts = [z_t, z_hat, t_emb, self.audio_proj(audio_tokens)]
        if hidden is not None:
            parts.append(hidden)
        if history is not None and self.history_mode == 'cond':
            parts.append(history)
        x = torch.cat(parts, dim=-1)
        return self.net(x)


# ----------------------------------------------------------------------------- #
#  Main model                                                                    #
# ----------------------------------------------------------------------------- #
class VqvaeARFlowPredict(BaseModel):
    def __init__(self,
                 nfeats: int,
                 split_path: str,
                 one_hot_dim: List,
                 resumed_training: bool,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.nfeats = nfeats
        self.resumed_training = resumed_training
        self.working_dir = self.hparams.working_dir

        # --- Audio conditioning selection (config-driven) ---------------- #
        # 'hubert'        : baseline -- HuBERT 50 Hz, frame-paired -> 1536-D
        # 'moshi'         : MIBURI-style -- Mimi codes -> sum(Moshi emb[:8]) -> 4096-D
        # 'hubert_moshi'  : concat of the two -> 5632-D
        # The HuBERT branch below is byte-identical to the original baseline.
        cfg_arflow = self.hparams.arflow
        self.audio_conditioning = cfg_arflow.get('audio_conditioning', 'hubert')
        if self.audio_conditioning not in ('hubert', 'moshi', 'hubert_moshi'):
            raise ValueError(f"unknown arflow.audio_conditioning={self.audio_conditioning!r}")

        self.feature_extractor = None
        self.moshi_emb = None
        self.moshi_text_emb = None
        hubert_dim, moshi_dim = 0, 0

        if self.audio_conditioning in ('hubert', 'hubert_moshi'):
            # --- Audio feature extractor (HuBERT) ---
            self.feature_extractor = instantiate(self.hparams.feature_extractor)
            logger.info(f"1. Audio feature extractor '{self.feature_extractor.hparams.name}' loaded")
            audio_encoded_dim = self.feature_extractor.audio_encoded_dim   # 768
            hubert_dim = audio_encoded_dim * 2                             # frame-paired -> 1536

        if self.audio_conditioning in ('moshi', 'hubert_moshi'):
            from framework.model.feature_extractor.moshi_mimi import MoshiMimiEmb
            self.moshi_emb = MoshiMimiEmb(
                emb_path=cfg_arflow.moshi_emb_path,
                num_codebooks=cfg_arflow.get('moshi_num_codebooks', 8),
                trainable=cfg_arflow.get('moshi_embedding_trainable', False),
                feature_norm=cfg_arflow.get('moshi_feature_norm', 'layernorm'),
            )
            self.moshi_framerate = float(self.moshi_emb.hparams.output_framerate)   # 12.5
            # 'interp' for offline training/eval (smooth, no 12.5 Hz double-step);
            # 'hold' is the causal streaming rule.
            self.moshi_align_mode = cfg_arflow.get('moshi_align_train', 'interp')
            moshi_dim = self.moshi_emb.audio_encoded_dim                            # 4096
            logger.info(f"1. Moshi/Mimi conditioning loaded "
                        f"(trainable={self.moshi_emb.trainable}, dim={moshi_dim}, "
                        f"norm={self.moshi_emb.feature_norm}, align={self.moshi_align_mode})")

            # --- Optional MIBURI text conditioning (Inner-Monologue stream) --- #
            # Precomputed 12.5 Hz interleaved Moshi text tokens -> Moshi text_emb
            # lookup -> 4096-D, fused with the Mimi-audio 4096-D feature. MIBURI
            # keeps audio/text as two separate cross-attention memories; the
            # LiveFace AR backbone fuses conditions additively, so here the two
            # 4096-D streams are concatenated (default) or summed before the
            # existing 12.5->25 Hz alignment (both ops commute with align_moshi).
            self.moshi_use_text = bool(cfg_arflow.get('moshi_use_text', False))
            self.moshi_text_fuse = cfg_arflow.get('moshi_text_fuse', 'concat')
            if self.moshi_use_text:
                if self.moshi_text_fuse not in ('concat', 'sum'):
                    raise ValueError(f"arflow.moshi_text_fuse must be 'concat' or 'sum', "
                                     f"got {self.moshi_text_fuse!r}")
                from framework.model.feature_extractor.moshi_mimi import MoshiTextEmb
                self.moshi_text_emb = MoshiTextEmb(
                    emb_path=cfg_arflow.moshi_text_emb_path,
                    trainable=cfg_arflow.get('moshi_text_emb_trainable', False),
                    feature_norm=cfg_arflow.get('moshi_text_feature_norm',
                                                cfg_arflow.get('moshi_feature_norm', 'layernorm')),
                )
                if self.moshi_text_fuse == 'concat':
                    moshi_dim += self.moshi_text_emb.text_encoded_dim                # 4096 -> 8192
                logger.info(f"1. Moshi text conditioning loaded "
                            f"(trainable={self.moshi_text_emb.trainable}, "
                            f"fuse={self.moshi_text_fuse}, dim={moshi_dim}, "
                            f"norm={self.moshi_text_emb.feature_norm})")

        self.audio_dim = hubert_dim + moshi_dim
        # hubert -> 1536 (baseline)  |  moshi -> 4096  |  hubert_moshi -> 5632

        # --- Style one-hot (identity list only for style_mode='onehot'; with 'constant'
        # it is unused, and Seamless's 142k train tracks would make a 142k x 142k eye) ---
        if self.hparams.arflow.get('style_mode', 'onehot') == 'onehot':
            self.all_identity_list = get_split_keyids(path=split_path, split="train")
            self.all_identity_onehot = torch.eye(len(self.all_identity_list))
        else:
            self.all_identity_list, self.all_identity_onehot = [], None
        self.one_hot_total = sum(one_hot_dim[:])

        # --- Frozen Stage-1 motion prior (LiveVQFormer / VQVAE) ---
        self.motion_prior = instantiate(self.hparams.motion_prior,
                                        nfeats=self.hparams.nfeats,
                                        logger_name="none",
                                        resumed_training=False,
                                        _recursive_=False)
        logger.info(f"2. '{self.motion_prior.hparams.modelname}' loaded (frozen)")
        if os.path.exists(self.hparams.ckpt_path_prior):
            load_checkpoint(model=self.motion_prior, ckpt_path=self.hparams.ckpt_path_prior,
                            eval_mode=True, device=self.device)
        else:
            raise ValueError(f"Motion Autoencoder path not found: {self.hparams.ckpt_path_prior}")
        for p in self.motion_prior.parameters():
            p.requires_grad = False
        self.motion_prior.eval()
        _sp_total = sum(p.numel() for p in self.motion_prior.parameters())
        _sp_train = sum(p.numel() for p in self.motion_prior.parameters() if p.requires_grad)
        logger.info(f"Loaded Stage-1 checkpoint:\n    {self.hparams.ckpt_path_prior}")
        logger.info(f"Stage-1 trainable parameters: {_sp_train}")
        logger.info(f"Stage-1 frozen parameters: {_sp_total}")

        # Frozen Stage-1 geometry.  Quantizer layout v2 assigns independent
        # codebook entries to e_dim-sized groups *within each frame*.
        self.n_embed = self.motion_prior.quantize.n_e           # codebook entries
        self.code_dim = self.motion_prior.quantize.e_dim        # 128
        self.latent_dim = self.code_dim * 2                     # 256 per frame
        self.num_subcodes = self.latent_dim // self.code_dim

        cfg = self.hparams.arflow
        d = cfg.d_model

        # --- Prosody / style / audio front-end (mirrors Stage-2 diffusion) ---
        # use_prosody=False (DualTalk): whole-clip DisVoice prosody is not
        # causal, so the audio feature is used unscaled.
        # style_mode='constant' (DualTalk): no identity/emotion labels, a single
        # learned style vector replaces the MEAD one-hot.
        self.use_prosody = bool(cfg.get('use_prosody', True))
        self.style_mode = cfg.get('style_mode', 'onehot')
        if self.style_mode not in ('onehot', 'constant'):
            raise ValueError(f"unknown arflow.style_mode={self.style_mode!r}")
        self.prosody_feature_map = nn.Linear(103, self.audio_dim) if self.use_prosody else None
        self.audio_feature_map = nn.Linear(self.audio_dim, d)
        self.style_embedding = nn.Linear(self.one_hot_total, d, bias=False)

        # --- Causal audio encoder ---
        self.audio_encoder = CausalAudioEncoder(
            in_dim=d, dim=d,
            num_layers=cfg.audio_layers, num_heads=cfg.num_heads,
            ffn_dim=cfg.ffn_dim, kernel_size=cfg.audio_kernel, dropout=cfg.dropout,
        )

        # --- Motion-token embedding front-end + BOS ---
        self.motion_token_proj = nn.Linear(self.latent_dim, d)
        self.bos_motion = nn.Parameter(torch.zeros(1, 1, self.latent_dim))
        self.mask_motion = nn.Parameter(torch.zeros(1, 1, self.latent_dim))
        nn.init.normal_(self.bos_motion, std=0.02)
        nn.init.normal_(self.mask_motion, std=0.02)

        # --- Fusion of (motion ⊕ audio) -> backbone input ---
        self.fuse_proj = nn.Linear(2 * d, d)

        # --- Causal AR backbone ---
        self.backbone = CausalTransformerBackbone(
            dim=d, num_layers=cfg.num_layers, num_heads=cfg.num_heads,
            ffn_dim=cfg.ffn_dim, dropout=cfg.dropout,
        )

        # --- Discrete AR heads: one categorical prediction per sub-code ---
        # Cross-entropy trains the actual symbols used by the frozen decoder;
        # continuous L1 regression between code embeddings can average modes.
        self.ar_code_heads = nn.ModuleList([
            nn.Linear(d, self.n_embed) for _ in range(self.num_subcodes)
        ])

        # --- Flow head (per-token) ---
        # Inputs (spec): audio token a_t, noisy motion token z_t, quantized AR
        # token ẑ_t, timestamp t_d. Optionally also the backbone hidden h_t.
        self.flow_use_hidden = cfg.get('flow_use_hidden', False)
        # Explicit motion history for the flow head: the previous K clean latents
        # (ground truth + noise/dropout in training, the head's own z0_hat at
        # inference). 0 = off (baseline).
        self.flow_history_frames = int(cfg.get('flow_history_frames', 0))
        self.flow_history_mode = cfg.get('flow_history_mode', 'cond')
        # Training-only exposure-bias guards on the GT history: per-sequence noise
        # std ~ U(0, flow_history_noise_std), and per-frame drop -> invalid.
        self.flow_history_noise_std = float(cfg.get('flow_history_noise_std', 0.0))
        self.flow_history_dropout = float(cfg.get('flow_history_dropout', 0.0))
        self.flow_head = ARFlowMatchingHead(
            latent_dim=self.latent_dim,
            audio_dim=d,
            time_dim=cfg.time_dim, hidden_dim=cfg.flow_hidden_dim,
            audio_proj_dim=cfg.get('audio_proj_dim', 256),
            extra_cond_dim=(d if self.flow_use_hidden else 0),
            history_frames=self.flow_history_frames,
            history_mode=self.flow_history_mode,
        )
        self.scheduler = FlowMatchingScheduler()

        # --- Loss weights ---
        loss_cfg = self.hparams.loss
        # lambda_ar: weight on categorical codebook cross-entropy.
        self.lambda_ar = loss_cfg.get('lambda_ar', loss_cfg.get('lambda_ce', 1.0))
        self.lambda_flow = loss_cfg.lambda_flow
        self.lambda_lat = loss_cfg.lambda_lat
        self.lambda_exp = loss_cfg.lambda_exp
        self.lambda_jaw = loss_cfg.lambda_jaw
        # motion[..., :exp_dims] -> 'exp' loss, motion[..., exp_dims:] -> 'jaw' loss
        # (MEAD / DualTalk 53-D: exp 50 | jaw 3; Seamless 108-D: exp 100 | head, jaw, gaze 8)
        self.exp_dims = int(loss_cfg.get('exp_dims', 50))
        self.flow_warmup_steps = loss_cfg.get('flow_warmup_steps', 0)
        # Training-only frame weight on the agent's own speaking frames (batch['vad']);
        # 1.0 = off (MEAD default, every frame equal).
        self.speak_weight = float(loss_cfg.get('speak_weight', 1.0))

        # --- Sampling / scheduled-sampling config ---
        self.sampling_steps = cfg.sampling_steps
        self.train_token_temperature = cfg.get('train_token_temperature', 1.0)
        if self.train_token_temperature <= 0:
            raise ValueError("arflow.train_token_temperature must be > 0")
        # >0 => categorical sampling at inference; 0 => argmax.
        self.infer_token_temperature = cfg.get('infer_token_temperature', 0.0)
        self.stream_decode_window = cfg.get('stream_decode_window', 64)
        # Finding 2: mild random history dropout (LiveGesture-style), applied only
        # in training, in ADDITION to (much milder) uncertainty masking.
        self.history_dropout = cfg.get('history_dropout', 0.0)
        ss = self.hparams.get('scheduled_sampling', {})
        self.ss_enable = ss.get('enable', False)
        self.ss_start_epoch = ss.get('start_epoch', 0)
        self.ss_end_epoch = ss.get('end_epoch', 100)
        self.ss_max_prob = ss.get('max_prob', 0.5)

        # --- Uncertainty masking (training-only two-pass; keep it mild) ---
        um = self.hparams.get('uncertainty_masking', {})
        self.um_enable = um.get('enable', False)
        self.um_mask_ratio = um.get('mask_ratio', 0.15)
        self.um_start_epoch = um.get('start_epoch', 40)
        self.infer_mask_ratio = um.get('infer_mask_ratio', 0.0)

        self.crop_length = self.hparams.crop_length
        val_cfg = self.hparams.get('free_run_validation', {})
        self.free_run_val_batches = val_cfg.get('batches', 0)
        self.free_run_val_max_frames = val_cfg.get('max_frames', self.crop_length)
        self.free_run_val_seed = val_cfg.get('seed', 1234)

        self.optimizer = instantiate(self.hparams.optim,
                                     params=filter(lambda p: p.requires_grad, self.parameters()))
        self.metrics = ComputeMetrics()

        self.__post_init__()
        self.log_param_table()

    # ------------------------------------------------------------------ #
    #  Parameter / freeze audit                                          #
    # ------------------------------------------------------------------ #
    def param_groups_report(self):
        """[(group_name, state, n_params, any_requires_grad), ...] for logging/tests."""
        groups = {
            "Stage1 motion_encoder": "motion_prior.motion_encoder",
            "Stage1 quantizer/codebook": "motion_prior.quantize",
            "Stage1 motion_decoder": "motion_prior.motion_decoder",
            "HuBERT feature_extractor": "feature_extractor",
            "Moshi audio embedding tables": "moshi_emb",
            "Moshi text embedding table": "moshi_text_emb",
            "audio_feature_map (audio projection)": "audio_feature_map",
            "prosody_feature_map": "prosody_feature_map",
            "style_embedding": "style_embedding",
            "CausalAudioEncoder": "audio_encoder",
            "AR backbone (transformer)": "backbone",
            "AR code heads": "ar_code_heads",
            "Flow-matching head": "flow_head",
        }
        rows = []
        for label, prefix in groups.items():
            ps = [p for n, p in self.named_parameters() if n == prefix or n.startswith(prefix + ".")]
            if not ps:
                continue
            n_params = sum(p.numel() for p in ps)
            n_grad = sum(p.numel() for p in ps if p.requires_grad)
            state = "TRAINABLE" if n_grad == n_params and n_params else (
                "FROZEN" if n_grad == 0 else "PARTIAL")
            rows.append((label, state, n_params, n_grad))
        return rows

    def log_param_table(self):
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"[arflow] audio_conditioning = {self.audio_conditioning} | audio_dim = {self.audio_dim}")
        logger.info(f"{'Module':40s} {'State':10s} {'Params':>14s} {'Grad?':>6s}")
        logger.info("-" * 74)
        for label, state, n_params, n_grad in self.param_groups_report():
            logger.info(f"{label:40s} {state:10s} {n_params:14,d} {'YES' if n_grad else 'NO':>6s}")
        logger.info("-" * 74)
        logger.info(f"{'TOTAL':40s} {'':10s} {total:14,d}")
        logger.info(f"{'trainable':40s} {'':10s} {train:14,d}")
        logger.info(f"{'frozen':40s} {'':10s} {total - train:14,d}")

    # ------------------------------------------------------------------ #
    #  Frozen codebook helpers                                             #
    # ------------------------------------------------------------------ #
    def _code_logits(self, hidden: Tensor) -> Tensor:
        """[B,T,d] -> [B,T,num_subcodes,n_embed]."""
        return torch.stack([head(hidden) for head in self.ar_code_heads], dim=2)

    def _indices_to_latent(self, indices: Tensor) -> Tensor:
        """Decode [B,T,num_subcodes] code indices into [B,T,latent_dim]."""
        if indices.shape[-1] != self.num_subcodes:
            raise ValueError(
                f"Expected {self.num_subcodes} subcodes, got shape {tuple(indices.shape)}"
            )
        weight = self.motion_prior.quantize.embedding.weight
        embedded = F.embedding(indices, weight)
        return embedded.reshape(*indices.shape[:2], self.latent_dim)

    def _logits_to_latent(self, logits: Tensor, differentiable: bool = False,
                          stochastic: bool = False):
        """Convert categorical code logits to decoder-compatible embeddings.

        Training uses straight-through Gumbel-Softmax so flow losses also update
        the AR heads. Validation/inference use exactly discrete indices.
        """
        if differentiable:
            probs = F.gumbel_softmax(
                logits, tau=self.train_token_temperature, hard=True, dim=-1
            )
            weight = self.motion_prior.quantize.embedding.weight
            embedded = torch.einsum('btgv,ve->btge', probs, weight)
            indices = probs.argmax(dim=-1)
            return embedded.reshape(*logits.shape[:2], self.latent_dim), indices

        if stochastic and self.infer_token_temperature > 0:
            probs = F.softmax(logits / self.infer_token_temperature, dim=-1)
            flat = probs.reshape(-1, self.n_embed)
            indices = torch.multinomial(flat, num_samples=1).reshape(*logits.shape[:-1])
        else:
            indices = logits.argmax(dim=-1)
        return self._indices_to_latent(indices), indices

    # ------------------------------------------------------------------ #
    #  Uncertainty masking helpers (categorical entropy)                  #
    # ------------------------------------------------------------------ #
    def _token_uncertainty(self, logits: Tensor) -> Tensor:
        """Normalized code entropy: [B,T,G,V] -> [B,T,G]."""
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        return entropy / math.log(self.n_embed)

    def _find_uncertain_tokens(self, uncertainty: Tensor,
                               mask_ratio: Optional[float] = None,
                               valid_mask: Optional[Tensor] = None) -> Tensor:
        """Return [B,T,G] mask for the most uncertain tokens per sub-code."""
        if uncertainty.ndim != 3 or uncertainty.shape[-1] != self.num_subcodes:
            raise ValueError(
                "uncertainty must have shape [B,T,num_subcodes], got "
                f"{tuple(uncertainty.shape)}"
            )
        if mask_ratio is None:
            mask_ratio = self.um_mask_ratio
        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError(f"mask_ratio must be in [0,1], got {mask_ratio}")
        if valid_mask is None:
            valid_mask = torch.ones(
                uncertainty.shape[:2], device=uncertainty.device, dtype=torch.bool
            )
        else:
            valid_mask = valid_mask.bool()

        # Rank time independently for each codebook, matching LiveGesture's
        # codebook-specific uncertainty masking without touching padding.
        scores = uncertainty.permute(0, 2, 1)                 # [B,G,T]
        eligible = valid_mask.unsqueeze(1).expand_as(scores)  # [B,G,T]
        valid_count = eligible.sum(dim=-1)                    # [B,G]
        select_count = torch.round(valid_count.float() * mask_ratio).long()
        select_count = torch.minimum(select_count, valid_count)
        scores = scores.masked_fill(~eligible, float('-inf'))
        order = scores.argsort(dim=-1, descending=True)
        rank = torch.empty_like(order)
        positions = torch.arange(order.shape[-1], device=order.device).expand_as(order)
        rank.scatter_(-1, order, positions)
        selected = (rank < select_count.unsqueeze(-1)) & eligible
        return selected.permute(0, 2, 1)                      # [B,T,G]

    def _apply_uncertainty_mask(self, motion_latent: Tensor, token_mask: Tensor) -> Tensor:
        """Replace selected frame or sub-code histories with learned embeddings."""
        B, T, _ = motion_latent.shape
        if token_mask.ndim == 2:
            token_mask = token_mask.unsqueeze(-1).expand(B, T, self.num_subcodes)
        if token_mask.shape != (B, T, self.num_subcodes):
            raise ValueError(
                "token_mask must have shape [B,T] or [B,T,num_subcodes], got "
                f"{tuple(token_mask.shape)}"
            )
        grouped = motion_latent.reshape(B, T, self.num_subcodes, self.code_dim)
        mask_token = self.mask_motion.to(
            device=motion_latent.device, dtype=motion_latent.dtype
        ).reshape(1, 1, self.num_subcodes, self.code_dim)
        masked = torch.where(token_mask.unsqueeze(-1), mask_token, grouped)
        return masked.reshape(B, T, self.latent_dim)

    # ------------------------------------------------------------------ #
    #  Audio front-end (prosody * audio, project, style-modulate)        #
    # ------------------------------------------------------------------ #
    def _style_one_hot(self, batch) -> Tensor:
        if self.style_mode == 'constant':
            return torch.ones(len(batch["keyid"]), self.one_hot_total)
        return create_one_hot(keyids=batch["keyid"],
                              IDs_list=self.all_identity_list,
                              IDs_labels=self.all_identity_onehot,
                              one_hot_dim=self.hparams.one_hot_dim)

    def _stack_prosody(self, prosody_list) -> Optional[Tensor]:
        if not self.use_prosody:
            return None
        return torch.stack(list(prosody_list), dim=0)

    def encode_audio_tokens(self, audio: Tensor, style_one_hot: Tensor,
                            prosody: Optional[Tensor]) -> Tensor:
        if self.prosody_feature_map is not None:
            audio = audio * self.prosody_feature_map(prosody)  # [B,1,audio_dim]
        audio_feat = self.audio_feature_map(audio)           # [B,T,d]
        style_emb = self.style_embedding(style_one_hot).unsqueeze(1)  # [B,1,d]
        audio_feat = audio_feat * style_emb
        return self.audio_encoder(audio_feat)                # causal -> [B,T,d]

    # ------------------------------------------------------------------ #
    #  Backbone: fuse shifted motion tokens with aligned audio           #
    # ------------------------------------------------------------------ #
    def _build_backbone_input(self, motion_latent_input: Tensor, audio_tokens: Tensor) -> Tensor:
        """motion_latent_input: [B, T, latent_dim] already shifted (BOS at t=0),
        audio_tokens: [B, T, d].  Returns fused [B, T, d]."""
        m = self.motion_token_proj(motion_latent_input)      # [B,T,d]
        fused = torch.cat([m, audio_tokens], dim=-1)         # concat + proj (default)
        return self.fuse_proj(fused)

    def _shift_motion(self, motion_latent: Tensor) -> Tensor:
        """Prepend BOS, drop last -> input at t is z'_{t-1}."""
        B = motion_latent.shape[0]
        bos = self.bos_motion.expand(B, 1, self.latent_dim).to(motion_latent.dtype)
        return torch.cat([bos, motion_latent[:, :-1]], dim=1)

    def run_backbone(self, motion_latent_input_shifted: Tensor, audio_tokens: Tensor) -> Tensor:
        x = self._build_backbone_input(motion_latent_input_shifted, audio_tokens)
        return self.backbone(x)                              # [B,T,d]

    # ------------------------------------------------------------------ #
    #  Flow-head motion history                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _pack_flow_history(prev: Tensor, valid: Tensor) -> Tensor:
        """prev [B,T,K,D] (index 0 = t-1), valid [B,T,K] -> [B,T,K*D+K]."""
        prev = prev * valid.unsqueeze(-1)                    # invalid frames -> zeros
        return torch.cat([prev.flatten(-2), valid], dim=-1)

    def _teacher_flow_history(self, z0: Tensor, valid_mask: Tensor, split: str) -> Optional[Tensor]:
        """Shifted ground-truth latents z0_{t-1..t-K} as flow-head history.

        Frames before the crop start or on padding are flagged invalid. In
        training, noise and dropout keep the head from relying on perfect
        history it will not have at inference (exposure bias)."""
        K = self.flow_history_frames
        if K == 0:
            return None
        B, T, D = z0.shape
        prev = z0.new_zeros(B, T, K, D)
        valid = z0.new_zeros(B, T, K)
        vm = valid_mask.to(z0.dtype)
        for k in range(1, min(K, T - 1) + 1):
            prev[:, k:, k - 1] = z0[:, :-k]
            valid[:, k:, k - 1] = vm[:, :-k]
        if split == "train":
            if self.flow_history_noise_std > 0.0:
                sigma = torch.rand(B, 1, 1, 1, device=z0.device, dtype=z0.dtype)
                prev = prev + sigma * self.flow_history_noise_std * torch.randn_like(prev)
            if self.flow_history_dropout > 0.0:
                keep = torch.rand(B, T, K, device=z0.device) >= self.flow_history_dropout
                valid = valid * keep.to(valid.dtype)
        return self._pack_flow_history(prev, valid)

    def _new_flow_history(self, B: int, device, dtype):
        """Empty inference-time history buffer: (prev [B,K,D], valid [B,K])."""
        K = self.flow_history_frames
        return (torch.zeros(B, K, self.latent_dim, device=device, dtype=dtype),
                torch.zeros(B, K, device=device, dtype=dtype))

    def _flow_history_feature(self, buf) -> Optional[Tensor]:
        """Inference buffer -> [B,1,K*D+K] flow-head history feature."""
        if self.flow_history_frames == 0:
            return None
        prev, valid = buf
        return self._pack_flow_history(prev.unsqueeze(1), valid.unsqueeze(1))

    def _push_flow_history(self, buf, z0_hat_t: Tensor):
        """Prepend the newest clean output z0_hat_t [B,1,D]; drop the oldest."""
        if self.flow_history_frames == 0:
            return buf
        prev, valid = buf
        prev = torch.cat([z0_hat_t, prev[:, :-1]], dim=1)
        valid = torch.cat([torch.ones_like(valid[:, :1]), valid[:, :-1]], dim=1)
        return prev, valid

    # ------------------------------------------------------------------ #
    #  Audio-conditioning front-end (config-branched)                    #
    # ------------------------------------------------------------------ #
    def _moshi_text_codes(self, batch, idx: int) -> Optional[Tensor]:
        """Per-item interleaved Moshi text-token stream, or None when text is off."""
        if self.moshi_text_emb is None:
            return None
        if 'moshi_text_codes' not in batch:
            raise ValueError("moshi_use_text=True but 'moshi_text_codes' is missing from the "
                             "batch -- use configs/data/mead_pred_moshi_text.yaml.")
        return batch['moshi_text_codes'][idx]

    def _moshi_feature(self, mimi_codes_i: Tensor, text_codes_i: Optional[Tensor] = None) -> Tensor:
        """Cached codes -> [1, T_mimi, moshi_dim] Moshi-embedded 12.5 Hz feature.

        mimi_codes_i: [K, T_mimi]      -> sum(Moshi emb[:8])            -> [1,T,4096]
        text_codes_i: [1, T_mimi]/[T]  -> Moshi text_emb lookup         -> [1,T,4096]
        When text conditioning is on, the two 4096-D streams are concatenated
        (default) or summed here, at 12.5 Hz, before align_moshi.
        """
        codes = mimi_codes_i.unsqueeze(0).to(self.device)          # [1, K, T_mimi]
        feat = self.moshi_emb(codes)                               # [1, T_mimi, 4096]
        if self.moshi_text_emb is not None:
            if text_codes_i is None:
                raise ValueError("moshi_use_text=True but no moshi_text_codes in the batch. "
                                 "Run pre_process/precompute_moshi_text_tokens.py and use "
                                 "configs/data/mead_pred_moshi_text.yaml.")
            tcodes = text_codes_i.to(self.device)
            if tcodes.dim() == 1:
                tcodes = tcodes.unsqueeze(0)                       # [1, T_mimi]
            tfeat = self.moshi_text_emb(tcodes)                    # [1, T_mimi, 4096]
            if tfeat.shape[1] != feat.shape[1]:                    # numeric guard
                T = min(tfeat.shape[1], feat.shape[1])
                feat, tfeat = feat[:, :T], tfeat[:, :T]
            if self.moshi_text_fuse == 'concat':
                feat = torch.cat([feat, tfeat], dim=-1)            # [1, T, 8192]
            else:
                feat = feat + tfeat                                # [1, T, 4096]
        return feat

    def _align_audio_motion(self, batch):
        """Per-item audio feature + motion aligned to the 25 fps motion grid.

        Returns 3 lists: audio [1,T,audio_dim], motion [1,T,53], prosody [1,1,103].
        'hubert' reproduces the original baseline path exactly; the Moshi paths
        use ``align_moshi`` (linear-interp offline / causal hold for streaming).
        """
        ra_list, rm_list, pros_list = [], [], []
        hub_feats = None
        if self.audio_conditioning in ('hubert', 'hubert_moshi'):
            hub_feats = self.feature_extractor(batch['audio'], False)   # list [1,Ts,768]

        for idx in range(len(batch['motion'])):
            motion_i = batch['motion'][idx]
            if self.audio_conditioning == 'hubert':
                ra, rm = resample_input(hub_feats[idx], motion_i,
                                        self.feature_extractor.hparams.output_framerate,
                                        self.hparams.video_framerate)
            elif self.audio_conditioning == 'moshi':
                feat = self._moshi_feature(batch['mimi_codes'][idx],
                                           self._moshi_text_codes(batch, idx))  # [1,T_mimi,moshi_dim]
                rm = motion_i
                ra = align_moshi(feat, rm.shape[1], self.moshi_align_mode)
                if ra.shape[1] < rm.shape[1]:                           # 'hold' can be short
                    rm = rm[:, :ra.shape[1]]
            else:  # 'hubert_moshi'
                ra_h, rm = resample_input(hub_feats[idx], motion_i,
                                          self.feature_extractor.hparams.output_framerate,
                                          self.hparams.video_framerate)
                feat = self._moshi_feature(batch['mimi_codes'][idx],
                                           self._moshi_text_codes(batch, idx))
                ra_m = align_moshi(feat, rm.shape[1], self.moshi_align_mode)
                if ra_m.shape[1] < rm.shape[1]:
                    rm = rm[:, :ra_m.shape[1]]
                    ra_h = ra_h[:, :rm.shape[1]]
                ra = torch.cat([ra_h.to(ra_m.device), ra_m], dim=-1)   # [1,T,2560+4096]
            ra_list.append(ra)
            rm_list.append(rm)
            pros_list.append(batch['prosody'][idx])
        return ra_list, rm_list, pros_list

    def _assemble_audio_eval(self, batch, generation: bool):
        """Eval-time audio feature for the first (only) item -> [1, T, audio_dim].

        Mirrors the original inline block for 'hubert'; adds the Moshi paths.
        With generation=False (evaluation on MEAD) the feature is aligned to the
        GT motion length; with generation=True (arbitrary audio) it spans the
        whole clip. The Moshi paths need cached Mimi codes in the batch.
        """
        prosody = self._stack_prosody(batch['prosody'])
        if prosody is not None:
            prosody = prosody.to(self.device)

        def _hubert_af():
            af = self.feature_extractor(batch['audio'], False)[0]
            if af.shape[1] % 2 != 0:
                af = af[:, :af.shape[1] - 1, :]
            if not generation and af.shape[1] > batch['motion'][0].shape[1] * 2:
                af = af[:, :batch['motion'][0].shape[1] * 2, :]
            return torch.reshape(af, (1, af.shape[1] // 2, af.shape[2] * 2))   # [1,Th,1536]

        if self.audio_conditioning == 'hubert':
            audio = _hubert_af()
        else:
            if 'mimi_codes' not in batch:
                raise NotImplementedError(
                    "audio_conditioning='%s' needs precomputed Mimi codes in the batch. "
                    "Evaluation on MEAD provides them; live generation.py does not "
                    "(would require running Mimi on the fly)." % self.audio_conditioning)
            feat = self._moshi_feature(batch['mimi_codes'][0],
                                       self._moshi_text_codes(batch, 0))       # [1,T_mimi,moshi_dim]
            # offline eval (generation=False) matches training -> align_moshi;
            # generation=True is the true-streaming path -> causal hold.
            if self.audio_conditioning == 'moshi':
                if generation:
                    audio = align_hold_x2(feat, feat.shape[1] * 2)
                else:
                    audio = align_moshi(feat, batch['motion'][0].shape[1], self.moshi_align_mode)
            else:  # 'hubert_moshi'
                af_h = _hubert_af()                                           # [1,Th,2560]
                if generation:
                    af_m = align_hold_x2(feat, af_h.shape[1])
                    T = min(af_h.shape[1], af_m.shape[1])
                    af_h, af_m = af_h[:, :T], af_m[:, :T]
                else:
                    af_m = align_moshi(feat, af_h.shape[1], self.moshi_align_mode)
                audio = torch.cat([af_h.to(af_m.device), af_m], dim=-1)       # [1,T,6656]
        return audio.to(self.device), prosody

    # ------------------------------------------------------------------ #
    #  Training / validation step                                        #
    # ------------------------------------------------------------------ #
    def allsplit_step(self, split: str, batch, batch_idx):
        style_one_hot = self._style_one_hot(batch)

        resample_audio_feature, resample_motion_feature, prosody_feature = \
            self._align_audio_motion(batch)

        if split == "train":
            crop_len = self.crop_length
            cropped_audio, cropped_motion, cropped_vad = [], [], []
            use_vad = self.speak_weight != 1.0 and batch.get('vad') is not None
            for i, (a, m) in enumerate(zip(resample_audio_feature, resample_motion_feature)):
                T = a.shape[1]
                start = 0
                if T > crop_len:
                    start = random.randint(0, T - crop_len)
                    a = a[:, start:start + crop_len, :]
                    m = m[:, start:start + crop_len, :]
                elif T < crop_len:
                    a = torch.cat([a, torch.zeros(1, crop_len - T, a.shape[2], dtype=a.dtype, device=a.device)], dim=1)
                    m = torch.cat([m, torch.zeros(1, crop_len - T, m.shape[2], dtype=m.dtype, device=m.device)], dim=1)
                if use_vad:
                    v = torch.zeros(crop_len)
                    seg = batch['vad'][i].float().reshape(-1)[start:start + min(T, crop_len)]
                    v[:seg.numel()] = seg
                    cropped_vad.append(v)
                cropped_audio.append(a)
                cropped_motion.append(m)
            batch['audio'] = torch.cat(cropped_audio, dim=0)
            batch['motion'] = torch.cat(cropped_motion, dim=0)
            batch['prosody'] = self._stack_prosody(prosody_feature)
            seq_lengths = [min(a.shape[1], crop_len) for a in resample_audio_feature]
        else:
            cropped_vad = []
            batch['audio'] = torch.cat(resample_audio_feature, dim=0)
            batch['motion'] = torch.cat(resample_motion_feature, dim=0)
            batch['prosody'] = self._stack_prosody(prosody_feature)
            seq_lengths = [a.shape[1] for a in resample_audio_feature]

        audio = batch['audio'].to(self.device, dtype=torch.float32)       # [B,T,1536]
        motion = batch['motion'].to(self.device, dtype=torch.float32)     # [B,T,53]
        style = style_one_hot.to(self.device, dtype=torch.float32)
        prosody = (batch['prosody'].to(self.device, dtype=torch.float32)
                   if batch['prosody'] is not None else None)

        assert audio.shape[-1] == self.audio_dim, f"Audio dim {audio.shape[-1]} != {self.audio_dim}"
        assert motion.shape[-1] == self.nfeats, f"Motion dim mismatch: {motion.shape[-1]}"
        assert style.shape[-1] == self.one_hot_total, f"Style dim mismatch: {style.shape[-1]}"

        B, T, _ = motion.shape

        # Valid-frame mask is built before any history corruption so padding is
        # never selected for scheduled sampling or uncertainty masking.
        mask = torch.zeros(B, T, device=motion.device, dtype=motion.dtype)
        for i, length in enumerate(seq_lengths):
            mask[i, :min(length, T)] = 1.0
        valid_mask = mask.bool()
        # Loss weights per frame: speaking frames x speak_weight (training only).
        loss_w = mask
        if split == "train" and self.speak_weight != 1.0 and cropped_vad:
            vad = torch.stack(cropped_vad).to(motion.device, dtype=motion.dtype)[:, :T]
            loss_w = mask * (1.0 + (self.speak_weight - 1.0) * vad)

        # Frozen Stage-1 must stay in eval mode even while the parent is training
        # (its decoder has dropout that would otherwise corrupt the frozen
        # teacher targets z0 and the decoded motion_pred).
        self.motion_prior.eval()

        # --- Frozen per-frame target latent and categorical code indices ---
        with torch.no_grad():
            z_m = self.motion_prior.motion_encoder(motion)   # [B,T,256]
            z0, _, quant_info = self.motion_prior.quantize(z_m)
            z0 = z0.detach()
            target_codes = quant_info[2].detach().long()     # [B,T,num_subcodes]
        if target_codes.shape != (B, T, self.num_subcodes):
            raise RuntimeError(
                "Stage-1 quantizer returned misaligned code indices: "
                f"expected {(B, T, self.num_subcodes)}, got {tuple(target_codes.shape)}"
            )

        # --- Aligned causal audio tokens ---
        audio_tokens = self.encode_audio_tokens(audio, style, prosody)   # [B,T,d]

        # --- Teacher-forced backbone input = the TRUE quantized latent ---
        motion_input = z0
        if split == "train" and self.ss_enable:
            p = self._scheduled_sampling_prob()
            if p > 0.0:
                with torch.no_grad():
                    h0 = self.run_backbone(self._shift_motion(motion_input), audio_tokens)
                    pred_latent0, _ = self._logits_to_latent(self._code_logits(h0))
                    pred_latent0 = pred_latent0.detach()
                replace = (torch.rand(B, T, device=motion.device) < p) & valid_mask
                motion_input = torch.where(replace.unsqueeze(-1), pred_latent0, z0)

        # --- Finding 2: mild random history dropout (training only) ---
        if split == "train" and self.history_dropout > 0.0:
            drop = ((torch.rand(B, T, device=motion.device) < self.history_dropout)
                    & valid_mask)
            motion_input = self._apply_uncertainty_mask(motion_input, drop)

        # ------------------------------------------------------------------ #
        #  Uncertainty masking: mild training-only two-pass scheme.          #
        #  Pass 1 (probe, no-grad): score each sub-code with normalized       #
        #    categorical entropy.                                            #
        #  Pass 2 (main): replace uncertain valid history sub-codes with     #
        #    learned MASK embeddings and re-run; the loss from this pass      #
        #    drives updates. Off at validation / inference.                  #
        # ------------------------------------------------------------------ #
        use_um = (
            split == "train"
            and self.um_enable
            and self.trainer is not None
            and self.trainer.current_epoch >= self.um_start_epoch
        )

        if use_um:
            with torch.no_grad():
                h_probe = self.run_backbone(self._shift_motion(motion_input), audio_tokens)
                code_logits_probe = self._code_logits(h_probe)            # [B,T,G,V]
                unc = self._token_uncertainty(code_logits_probe)          # [B,T,G]
                uncertain_mask = self._find_uncertain_tokens(
                    unc, valid_mask=valid_mask
                )
            motion_input = self._apply_uncertainty_mask(motion_input, uncertain_mask)

        # --- Main forward (pass 2 when UM is active, sole pass otherwise) ---
        h = self.run_backbone(self._shift_motion(motion_input), audio_tokens)  # [B,T,d]

        # --- AR categorical heads: predict the current frame's sub-codes ---
        code_logits = self._code_logits(h)                   # [B,T,G,V]
        z_hat, _ = self._logits_to_latent(
            code_logits, differentiable=(split == "train")
        )                                                    # [B,T,256]

        # --- Flow-matching forward process ---
        t = torch.rand(B, device=self.device)                # t ~ U(0,1)
        z1 = torch.randn_like(z0)
        z_t = self.scheduler.interpolate(z0, z1, t)
        v_target = self.scheduler.target_velocity(z0, z1)

        # Flow head inputs: audio token a_t, noisy motion token z_t, quantized AR
        # token ẑ_t, timestamp t, + backbone hidden h_t (Finding 3: h_t carries
        # the only causal temporal context the per-frame flow head can see).
        hidden = h if self.flow_use_hidden else None
        flow_history = self._teacher_flow_history(z0, valid_mask, split)
        v_hat = self.flow_head(z_t, t, z_hat, audio_tokens=audio_tokens, hidden=hidden,
                               history=flow_history)                                     # [B,T,256]

        # --- Losses ---
        # Cross-entropy is aligned frame-for-frame: shifted input at position t
        # contains z_{t-1}, while this target is code(z_t).
        ar_per = F.cross_entropy(
            code_logits.reshape(-1, self.n_embed),
            target_codes.reshape(-1),
            reduction='none',
        ).reshape(B, T, self.num_subcodes).mean(dim=-1)
        loss_ar = (ar_per * loss_w).sum() / loss_w.sum().clamp(min=1.0)

        # Every loss is first averaged over features, then over valid frames.
        # This prevents the 256-D flow/latent terms from receiving an accidental
        # 256x multiplier relative to the scalar AR loss.
        flow_per = F.mse_loss(v_hat, v_target, reduction='none').mean(dim=-1)
        loss_flow = (flow_per * loss_w).sum() / loss_w.sum().clamp(min=1.0)

        # Recover z0 and latent L1
        t_expand = t.view(B, 1, 1)
        z0_hat = z_t - t_expand * v_hat
        lat_per = F.l1_loss(z0_hat, z0, reduction='none').mean(dim=-1)
        loss_lat = (lat_per * loss_w).sum() / loss_w.sum().clamp(min=1.0)

        # Motion-space losses through frozen quantizer + decoder
        z0_hat_q, _, _ = self.motion_prior.quantize(z0_hat)
        motion_pred = self.motion_prior.motion_decoder(z0_hat_q, use_chunked_attn=(split != "train"))
        exp_per = F.l1_loss(
            motion_pred[..., :self.exp_dims], motion[..., :self.exp_dims], reduction='none'
        ).mean(dim=-1)
        loss_exp = (exp_per * loss_w).sum() / loss_w.sum().clamp(min=1.0)
        jaw_per = F.l1_loss(
            motion_pred[..., self.exp_dims:], motion[..., self.exp_dims:], reduction='none'
        ).mean(dim=-1)
        loss_jaw = (jaw_per * loss_w).sum() / loss_w.sum().clamp(min=1.0)

        # Use warmup scaling only for training; validation always uses full weight
        # so the checkpoint monitor compares epochs fairly.
        flow_scale = self._flow_warmup_scale() if split == "train" else 1.0
        loss = (self.lambda_ar * loss_ar
                + flow_scale * (self.lambda_flow * loss_flow
                                + self.lambda_lat * loss_lat
                                + self.lambda_exp * loss_exp
                                + self.lambda_jaw * loss_jaw))

        # --- AR token accuracy (monitoring only) ---
        with torch.no_grad():
            code_correct = (code_logits.argmax(dim=-1) == target_codes).float().mean(dim=-1)
            ar_acc = (code_correct * mask).sum() / mask.sum().clamp(min=1.0)

        # --- Logging (epoch-level for monitors) ---
        for name, val in [("ar", loss_ar), ("flow", loss_flow), ("lat", loss_lat),
                          ("exp", loss_exp), ("jaw", loss_jaw), ("total", loss),
                          ("ar_acc", ar_acc)]:
            self.log(f"{name}/{split}", val.detach(), on_step=False, on_epoch=True,
                     prog_bar=False, sync_dist=True)

        if split == "val":
            self.metrics.update(motion_pred.detach(), motion.detach(), seq_lengths)
            if batch_idx < self.free_run_val_batches:
                free_t = min(T, int(self.free_run_val_max_frames), int(seq_lengths[0]))
                if free_t > 0:
                    cuda_devices = ([motion.device.index]
                                    if motion.is_cuda and motion.device.index is not None
                                    else [])
                    with torch.random.fork_rng(devices=cuda_devices):
                        torch.manual_seed(self.free_run_val_seed + batch_idx)
                        free_motion = self.generate_streaming_batched(
                            audio_tokens[:1, :free_t], num_samples=1,
                            sampling_steps=self.sampling_steps, infer_mask_ratio=0.0,
                        )[0]
                    free_target = motion[:1, :free_t]
                    free_l1 = F.l1_loss(free_motion, free_target)
                    self.log("free_run_l1/val", free_l1, on_step=False, on_epoch=True,
                             batch_size=1, sync_dist=True)
                    if free_t > 1:
                        free_vel_l1 = F.l1_loss(
                            free_motion[:, 1:] - free_motion[:, :-1],
                            free_target[:, 1:] - free_target[:, :-1],
                        )
                        self.log("free_run_vel_l1/val", free_vel_l1,
                                 on_step=False, on_epoch=True, batch_size=1,
                                 sync_dist=True)

        if split == "train":
            self.log("loss_train", loss.detach(), prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
            for name, val in [("ar", loss_ar), ("flow", loss_flow), ("lat", loss_lat),
                              ("exp", loss_exp), ("jaw", loss_jaw)]:
                self.log(f"{name}/train_step", val.detach(), on_step=True, on_epoch=False,
                         prog_bar=False, sync_dist=True)
            self.log("flow_scale", flow_scale, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
            if use_um:
                self.log("um_active", 1.0, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        if split == "val":
            self.log("loss_val", loss.detach(), prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)

        return loss

    def _scheduled_sampling_prob(self) -> float:
        if not self.ss_enable:
            return 0.0
        e = float(self.trainer.current_epoch) if self.trainer is not None else 0.0
        if e <= self.ss_start_epoch:
            return 0.0
        if e >= self.ss_end_epoch:
            return self.ss_max_prob
        frac = (e - self.ss_start_epoch) / max(1.0, (self.ss_end_epoch - self.ss_start_epoch))
        return self.ss_max_prob * frac

    def _flow_warmup_scale(self) -> float:
        if self.flow_warmup_steps <= 0 or self.trainer is None:
            return 1.0
        step = float(self.trainer.global_step)
        return min(1.0, step / float(self.flow_warmup_steps))

    def allsplit_batch_end(self, split: str, batch_idx: int):
        if split == "val":
            metrics_dict = self.metrics.compute()
            dico = {f"Metrics/{metric}": value for metric, value in metrics_dict.items()}
            dico.update({"epoch": float(self.trainer.current_epoch),
                         "step": float(self.trainer.current_epoch)})
            self.log_dict(dico, sync_dist=True)

    def training_step(self, batch, batch_idx):
        loss = self.allsplit_step("train", batch, batch_idx)
        self.allsplit_batch_end("train", batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.allsplit_step("val", batch, batch_idx)
        self.allsplit_batch_end("val", batch_idx)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.allsplit_step("test", batch, batch_idx)
        self.allsplit_batch_end("test", batch_idx)
        return loss

    def configure_optimizers(self):
        return {"optimizer": self.optimizer}

    def on_train_start(self):
        if self.resumed_training:
            self.resumed_training = False
        for pg in self.trainer.optimizers[0].param_groups:
            if pg['lr'] != self.hparams.optim.lr:
                pg['lr'] = self.hparams.optim.lr
                print("New learning rate:", self.hparams.optim.lr)
        print("Current learning rate:", self.optimizer.param_groups[0]['lr'])

    # ------------------------------------------------------------------ #
    #  Per-frame flow ODE (for one frame's latent)                       #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _flow_ode_step(self, z_hat_t: Tensor, audio_t: Tensor, sampling_steps: int,
                       hidden_t: Optional[Tensor] = None,
                       z_init: Optional[Tensor] = None,
                       history_t: Optional[Tensor] = None) -> Tensor:
        """Integrate the per-frame flow ODE from t=1 (noise) to t=0.
        Inputs (spec): quantized AR token ẑ_t, audio token a_t, timestamp, and
        the noisy motion token z (integrated). z_hat_t/audio_t: [B, 1, *].

        z_init: optional initial-noise tensor. Streaming supplies a slowly
        drifting (AR(1)-correlated) buffer instead of an independent draw so the
        two motion frames emitted per Mimi step, and successive steps, don't get
        an uncorrelated random offset each time — that offset is the 12.5 Hz
        buzz the mesh shows. This changes only where integration starts, not how
        far it runs, so it adds no latency."""
        B = z_hat_t.shape[0]
        dev = z_hat_t.device
        if sampling_steps <= 0:
            raise ValueError(f"sampling_steps must be positive, got {sampling_steps}")
        if z_init is not None:
            z = z_init.to(device=dev, dtype=z_hat_t.dtype).expand(B, 1, self.latent_dim).clone()
        else:
            z = torch.randn(B, 1, self.latent_dim, device=dev, dtype=z_hat_t.dtype)
        dt = 1.0 / sampling_steps
        # Pre-allocate timestamp tensor and reuse
        t_buf = torch.empty(B, device=dev, dtype=z_hat_t.dtype)
        for i in range(sampling_steps):
            t_buf.fill_(1.0 - i * dt)
            v = self.flow_head(z, t_buf, z_hat_t, audio_tokens=audio_t, hidden=hidden_t,
                               history=history_t)
            z = z.sub_(v, alpha=dt)  # in-place: z -= dt * v
        return z                                             # [B,1,256]

    @torch.no_grad()
    def _predict_latent(self, h_t: Tensor):
        """h_t -> discrete codebook latent, logits, and selected indices."""
        logits = self._code_logits(h_t)                       # [B,1,G,V]
        z_hat_t, indices = self._logits_to_latent(logits, stochastic=True)
        return z_hat_t, logits, indices

    @torch.no_grad()
    def _autoregressive_latents(self, audio_tokens: Tensor, sampling_steps: int,
                                history_mask: Optional[Tensor] = None,
                                report_latency: bool = False):
        """Run the same cached, free-running loop used at deployment.

        ``history_mask[:, t]`` (whole frame) or ``[:,t,g]`` (sub-code) suppresses
        predicted token t when it is fed to position t+1. This explicit
        convention avoids the previous refinement off-by-one, which masked
        token t and then tried to regenerate t itself.
        """
        device = audio_tokens.device
        B, T, _ = audio_tokens.shape
        if history_mask is not None:
            valid_shapes = {(B, T), (B, T, self.num_subcodes)}
            if tuple(history_mask.shape) not in valid_shapes:
                raise ValueError(
                    "history_mask must have shape [B,T] or [B,T,num_subcodes], "
                    f"got {tuple(history_mask.shape)}"
                )

        kv_caches = self.backbone.new_kv_caches()
        prev_motion = self.bos_motion.expand(B, 1, self.latent_dim).to(
            device=device, dtype=audio_tokens.dtype
        )
        flow_hist = self._new_flow_history(B, device, audio_tokens.dtype)
        motion_latents, token_latents, all_logits, per_frame_ms = [], [], [], []

        for t_idx in range(T):
            if report_latency and device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            a_t = audio_tokens[:, t_idx:t_idx + 1]
            x_t = self._build_backbone_input(prev_motion, a_t)
            for layer_idx, layer in enumerate(self.backbone.layers):
                x_t = layer(x_t, kv_cache=kv_caches[layer_idx])
            h_t = self.backbone.norm(x_t)

            z_hat_t, logits_t, _ = self._predict_latent(h_t)
            hidden_t = h_t if self.flow_use_hidden else None
            z0_hat_t = self._flow_ode_step(
                z_hat_t, a_t, sampling_steps, hidden_t=hidden_t,
                history_t=self._flow_history_feature(flow_hist),
            )
            flow_hist = self._push_flow_history(flow_hist, z0_hat_t)
            motion_latents.append(z0_hat_t)
            token_latents.append(z_hat_t)
            all_logits.append(logits_t)

            if history_mask is None:
                prev_motion = z_hat_t
            else:
                mask_t = history_mask[:, t_idx:t_idx + 1]
                prev_motion = self._apply_uncertainty_mask(z_hat_t, mask_t)

            if report_latency and device.type == "cuda":
                torch.cuda.synchronize()
            per_frame_ms.append((time.perf_counter() - start_time) * 1000.0)

        return (
            torch.cat(motion_latents, dim=1),
            torch.cat(token_latents, dim=1),
            torch.cat(all_logits, dim=1),
            per_frame_ms,
        )

    # ------------------------------------------------------------------ #
    #  Streaming autoregressive inference (KV-cached feedback loop)       #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate_streaming(self, audio: Tensor, style_one_hot: Tensor, prosody: Tensor,
                           sampling_steps: Optional[int] = None, report_latency: bool = True,
                           infer_mask_ratio: Optional[float] = None):
        """Real autoregression: token predicted at t is fed back as input for t+1.

        audio: [1, T, audio_dim]  (already frame-paired + causal-ready)
        Returns motion_pred [1, T, 53].
        """
        self.eval()
        if sampling_steps is None:
            sampling_steps = self.sampling_steps
        device = audio.device
        B, T, _ = audio.shape
        assert B == 1, "streaming inference expects batch size 1"

        audio_tokens = self.encode_audio_tokens(audio, style_one_hot, prosody)   # [1,T,d]
        z0_hat, _, code_logits, per_frame_ms = self._autoregressive_latents(
            audio_tokens, sampling_steps, report_latency=report_latency
        )

        # ----- Inference uncertainty-masking refinement pass --------------- #
        # If infer_mask_ratio > 0, suppress uncertain predicted history tokens
        # and rerun the complete free-running loop. A token selected at t is
        # masked only when it becomes history for t+1; it is not incorrectly
        # treated as an input to its own prediction.
        imr = infer_mask_ratio if infer_mask_ratio is not None else self.infer_mask_ratio
        if imr > 0.0:
            uncertainty = self._token_uncertainty(code_logits)
            uncertain_mask = self._find_uncertain_tokens(uncertainty, mask_ratio=imr)
            z0_hat, _, _, _ = self._autoregressive_latents(
                audio_tokens, sampling_steps, history_mask=uncertain_mask
            )
            logger.info(f"[stream] uncertainty refinement: mask_ratio={imr}, "
                        f"suppressed {int(uncertain_mask.sum())}/"
                        f"{T * self.num_subcodes} history tokens")
        # ------------------------------------------------------------------ #

        z0_hat_q, _, _ = self.motion_prior.quantize(z0_hat)
        motion_pred = self.motion_prior.motion_decoder(z0_hat_q, use_chunked_attn=True)

        if report_latency:
            mean_ms = sum(per_frame_ms) / max(1, len(per_frame_ms))
            logger.info(f"[stream] per-frame latency: mean={mean_ms:.2f} ms "
                        f"(excl. flow-decoder), frames={T}, ode_steps={sampling_steps}")
        return motion_pred

    # ------------------------------------------------------------------ #
    #  TRUE online streaming API: feed one audio frame, emit one motion   #
    #  frame.  Everything trainable is causal & incremental:              #
    #    audio-encoder (conv ring-buffer + KV cache) -> backbone (KV      #
    #    cache) -> discrete head -> flow ODE -> sliding-window decode.     #
    #                                                                     #
    #  Stage-1 layout v2 uses causal attention, causal convolution, and     #
    #  per-time-step normalization. A trailing window can therefore be     #
    #  decoded without future-frame statistics leaking into the output.    #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def stream_reset(self, style_one_hot: Tensor, prosody: Tensor,
                     sampling_steps: Optional[int] = None, device=None):
        """Initialise a streaming session. style/prosody are per-clip constants."""
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        # Flow-ODE noise correlation for streaming (demo only; offline eval and
        # generation never call stream_*). The ODE otherwise draws independent
        # noise per frame, which the mesh shows as a constant tremor. We walk it
        # as an AR(1) process  n <- rho*n + sqrt(1-rho^2)*eps  (unit variance
        # preserved, so no loss of expression range). rho -> 1 is smoothest but
        # can lock onto one unlucky low-energy draw; rho = 0 restores the old
        # per-frame randn. Override with FACE_STREAM_NOISE_RHO.
        try:
            _noise_rho = float(os.environ.get("FACE_STREAM_NOISE_RHO", "0.95"))
        except ValueError:
            _noise_rho = 0.95
        _noise_rho = float(min(max(_noise_rho, 0.0), 1.0))
        self._stream = {
            "steps": sampling_steps if sampling_steps is not None else self.sampling_steps,
            # per-clip constants for the audio front-end
            "prosody_proj": (self.prosody_feature_map(prosody.to(device))          # [1,1,1536]
                             if self.prosody_feature_map is not None else None),
            "style_emb": self.style_embedding(style_one_hot.to(device)).unsqueeze(1),  # [1,1,d]
            # incremental states
            "audio_state": self.audio_encoder.init_state(1, device),
            "bb_kv": self.backbone.new_kv_caches(),
            "prev_motion": self.bos_motion.expand(1, 1, self.latent_dim).to(device),
            "flow_hist": self._new_flow_history(1, device, torch.float32),
            "latent_buf": [],          # emitted latents (for sliding-window decode)
            "device": device,
            "noise_rho": _noise_rho,
            "flow_noise": (
                torch.randn(1, 1, self.latent_dim, device=device)
                if _noise_rho > 0.0 else None
            ),
        }

    @torch.no_grad()
    def stream_step(self, audio_pair_t: Tensor) -> Tensor:
        """Consume one frame-paired audio token, emit one motion frame.
        audio_pair_t: [1, 1, audio_dim] -> motion frame [1, 1, 53]."""
        s = self._stream
        # (1) causal per-frame audio token a_t
        audio_pair_t = audio_pair_t.to(s["device"])
        if s["prosody_proj"] is not None:
            audio_pair_t = audio_pair_t * s["prosody_proj"]
        af = self.audio_feature_map(audio_pair_t)                                      # [1,1,d]
        af = af * s["style_emb"]
        a_t = self.audio_encoder.step(af, s["audio_state"])                            # [1,1,d]

        # (2) backbone step (KV-cached) over fed-back previous token + a_t
        x_t = self._build_backbone_input(s["prev_motion"], a_t)
        for i, layer in enumerate(self.backbone.layers):
            x_t = layer(x_t, kv_cache=s["bb_kv"][i])
        h_t = self.backbone.norm(x_t)                                                  # [1,1,d]

        # (3) AR head -> quantized latent token; (4) flow ODE -> continuous latent
        z_hat_t, _, _ = self._predict_latent(h_t)
        hidden_t = h_t if self.flow_use_hidden else None
        z_init = s.get("flow_noise")
        z0_hat_t = self._flow_ode_step(z_hat_t, a_t, s["steps"], hidden_t=hidden_t,
                                       z_init=z_init,
                                       history_t=self._flow_history_feature(s["flow_hist"]))  # [1,1,256]
        s["flow_hist"] = self._push_flow_history(s["flow_hist"], z0_hat_t)
        # advance the AR(1) noise walk for the next frame
        rho = s.get("noise_rho", 0.0)
        if z_init is not None and 0.0 < rho < 1.0:
            eps = torch.randn_like(z_init)
            s["flow_noise"] = z_init.mul(rho).add_(eps, alpha=(1.0 - rho * rho) ** 0.5)
        s["prev_motion"] = z_hat_t                                                     # feed back

        # (5) sliding-window decode through the frozen decoder, emit newest frame
        s["latent_buf"].append(z0_hat_t)
        w = getattr(self, "stream_decode_window", 0)
        window = s["latent_buf"] if (not w or w <= 0) else s["latent_buf"][-w:]
        z_win = torch.cat(window, dim=1)                                               # [1,w,256]
        z_win_q, _, _ = self.motion_prior.quantize(z_win)
        position_offset = len(s["latent_buf"]) - len(window)
        motion_win = self.motion_prior.motion_decoder(
            z_win_q, use_chunked_attn=True, position_offset=position_offset
        )                                                                              # [1,w,53]
        return motion_win[:, -1:, :]                                                   # [1,1,53]

    @torch.no_grad()
    def generate_streaming_online(self, audio: Tensor, style_one_hot: Tensor, prosody: Tensor,
                                  sampling_steps: Optional[int] = None, report_latency: bool = True):
        """Fully online variant: drives stream_reset/stream_step frame-by-frame
        (truly emits a frame per input frame). audio: [1, T, audio_dim]."""
        device = audio.device
        B, T, _ = audio.shape
        assert B == 1, "streaming inference expects batch size 1"
        self.stream_reset(style_one_hot, prosody, sampling_steps, device=device)
        frames, per_frame_ms = [], []
        for t in range(T):
            if report_latency and device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            frames.append(self.stream_step(audio[:, t:t + 1]))
            if report_latency and device.type == "cuda":
                torch.cuda.synchronize()
            per_frame_ms.append((time.perf_counter() - t0) * 1000.0)
        if report_latency:
            mean_ms = sum(per_frame_ms) / max(1, len(per_frame_ms))
            logger.info(f"[stream-online] per-frame latency (incl. decode): "
                        f"mean={mean_ms:.2f} ms, frames={T}")
        return torch.cat(frames, dim=1)                                                # [1,T,53]

    # ------------------------------------------------------------------ #
    #  Forward (generation entry point, mirrors diffusion model)         #
    # ------------------------------------------------------------------ #
    def forward(self, batch, sample=False, generation=True) -> Tensor:
        self.to(self.device)
        style_one_hot = self._style_one_hot(batch).to(self.device)

        audio, prosody = self._assemble_audio_eval(batch, generation)
        # Finding 6: run inference in fp32 (training is fp32). A short Euler flow
        # ODE + RoPE + KV-cached backbone accumulate visible error under fp16.
        return self.generate_streaming(audio, style_one_hot, prosody,
                                       infer_mask_ratio=self.infer_mask_ratio)

    # ------------------------------------------------------------------ #
    #  Batched streaming: run N samples in parallel (shared audio)        #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate_streaming_batched(self, audio_tokens: Tensor, num_samples: int,
                                   sampling_steps: Optional[int] = None,
                                   infer_mask_ratio: Optional[float] = None) -> List[Tensor]:
        """Run N AR loops in parallel by batching along dim-0.

        audio_tokens: [1, T, d] — pre-computed causal audio tokens (shared).
        Returns list of N motion_pred tensors, each [1, T, 53].
        """
        self.eval()
        if sampling_steps is None:
            sampling_steps = self.sampling_steps
        device = audio_tokens.device
        T = audio_tokens.shape[1]
        N = num_samples

        # Replicate audio tokens along batch dim for parallel samples
        audio_tok = audio_tokens.expand(N, -1, -1)            # [N, T, d]

        z0_hat, _, code_logits, _ = self._autoregressive_latents(
            audio_tok, sampling_steps
        )

        # --- Optional uncertainty-masking refinement (batched) ---
        imr = infer_mask_ratio if infer_mask_ratio is not None else self.infer_mask_ratio
        if imr > 0.0:
            uncertainty = self._token_uncertainty(code_logits)
            uncertain_mask = self._find_uncertain_tokens(uncertainty, mask_ratio=imr)
            z0_hat, _, _, _ = self._autoregressive_latents(
                audio_tok, sampling_steps, history_mask=uncertain_mask
            )

        # Decode through frozen Stage-1 (batched)
        z0_hat_q, _, _ = self.motion_prior.quantize(z0_hat)
        motion_pred = self.motion_prior.motion_decoder(z0_hat_q, use_chunked_attn=True)  # [N, T, 53]

        # Split back into list of [1, T, 53]
        return [motion_pred[i:i+1] for i in range(N)]

    @torch.no_grad()
    def forward_multi_sample(self, batch, num_samples=10, generation=False):
        """Generate multiple samples — audio feature + causal encoding computed
        ONCE, then all N samples run in a single batched AR loop (fp32, Finding 6)."""
        self.to(self.device)
        style_one_hot = self._style_one_hot(batch).to(self.device)
        # Audio feature (HuBERT frame-pair and/or Moshi/Mimi) — done ONCE
        audio, prosody = self._assemble_audio_eval(batch, generation)

        # Causal audio encoding — done ONCE
        audio_tokens = self.encode_audio_tokens(audio, style_one_hot, prosody)  # [1, T, d]

        # Run all N samples in a single batched AR loop
        return self.generate_streaming_batched(
            audio_tokens, num_samples,
            infer_mask_ratio=self.infer_mask_ratio)

    # ------------------------------------------------------------------ #
    #  Leakage unit test: perturbing input at time > t0 must NOT change   #
    #  any output at ≤ t0 (bit-exact).                                     #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def leakage_test(self, T: int = 32, t0: int = 10, device=None, atol: float = 0.0) -> bool:
        self.eval()
        device = device or self.device
        d = self.hparams.arflow.d_model
        B = 2
        audio = torch.randn(B, T, self.audio_dim, device=device)
        style = torch.zeros(B, self.one_hot_total, device=device)
        style[:, 0] = 1.0
        prosody = torch.randn(B, 1, 103, device=device)
        motion_latent = torch.randn(B, T, self.latent_dim, device=device)
        shared_z_t = torch.randn(B, T, self.latent_dim, device=device)

        def run(a, m):
            atok = self.encode_audio_tokens(a, style, prosody)
            h = self.run_backbone(self._shift_motion(m), atok)
            logits = self._code_logits(h)
            t = torch.full((B,), 0.5, device=device)
            z_hat, _ = self._logits_to_latent(logits)
            hidden = h if self.flow_use_hidden else None
            history = self._teacher_flow_history(m, torch.ones(B, T, device=device, dtype=torch.bool), "val")
            v = self.flow_head(shared_z_t, t, z_hat, audio_tokens=atok, hidden=hidden, history=history)
            return atok, h, logits, v

        a1, h1, l1, v1 = run(audio, motion_latent)

        # Perturb every input strictly after t0
        audio2 = audio.clone(); audio2[:, t0 + 1:] += 5.0 * torch.randn_like(audio2[:, t0 + 1:])
        motion2 = motion_latent.clone(); motion2[:, t0 + 1:] += 5.0 * torch.randn_like(motion2[:, t0 + 1:])
        a2, h2, l2, v2 = run(audio2, motion2)

        checks = {
            "audio_tokens": (a1[:, :t0 + 1], a2[:, :t0 + 1]),
            "backbone_h":   (h1[:, :t0 + 1], h2[:, :t0 + 1]),
            "ar_logits":    (l1[:, :t0 + 1], l2[:, :t0 + 1]),
            "flow_v":       (v1[:, :t0 + 1], v2[:, :t0 + 1]),
        }
        ok = True
        for name, (x, y) in checks.items():
            max_diff = (x - y).abs().max().item()
            passed = max_diff <= atol
            ok = ok and passed
            logger.info(f"[leakage] {name:12s} max|Δ|={max_diff:.3e}  {'OK' if passed else 'LEAK'}")
        return ok
