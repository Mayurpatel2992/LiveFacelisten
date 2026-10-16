"""AR-flow model with a small diffusion-transformer (DiT) flow head.

Kept separate from model_vqvae_arflow_pred.py so runs/evals of the MLP-head
models are untouched. VqvaeARFlowDiTPredict is VqvaeARFlowPredict with the
per-frame 4-layer concat MLP flow head swapped for an in-context-conditioned
transformer over the frame's condition tokens:

    [t_emb] [audio a_t] [ẑ_t] [h_t] ([z0_{t-1}] .. [z0_{t-K}]) [noisy z_t]

Each token has its own input projection plus a learned slot embedding; tokens
of one frame attend to each other bidirectionally (all are current or past, so
the head stays causal across frames). The velocity is read from the noisy-z_t
output position. Frames are flattened to [B*T, n_tokens, dim], so training is
fully parallel and inference runs one 6-7 token sequence per ODE step.
Training, losses, AR loop and streaming are inherited unchanged.
"""
import logging
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from hydra.utils import instantiate

from framework.model.utils.diffusion_mlp import SinusoidalTimeEmbedding
from framework.model.model_vqvae_arflow_pred import VqvaeARFlowPredict

logger = logging.getLogger(__name__)


class ARFlowDiTHead(nn.Module):
    """Per-frame in-context DiT flow head; same call signature as ARFlowMatchingHead.

    history (optional) is the packed [B, T, K*latent_dim + K] feature (latents +
    validity flags) built by VqvaeARFlowPredict; invalid history tokens are
    masked out of attention.
    """

    def __init__(self, latent_dim: int, audio_dim: int, hidden_cond_dim: int = 0,
                 history_frames: int = 0, time_dim: int = 256, dim: int = 256,
                 num_layers: int = 3, num_heads: int = 4, ffn_dim: int = 1024):
        super().__init__()
        self.latent_dim = latent_dim
        self.history_frames = history_frames
        self.use_hidden = hidden_cond_dim > 0
        self.time_proj = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, dim), nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.audio_proj = nn.Linear(audio_dim, dim)
        self.zhat_proj = nn.Linear(latent_dim, dim)
        self.hidden_proj = nn.Linear(hidden_cond_dim, dim) if self.use_hidden else None
        self.history_proj = nn.Linear(latent_dim, dim) if history_frames > 0 else None
        self.noisy_proj = nn.Linear(latent_dim, dim)
        self.num_tokens = 3 + int(self.use_hidden) + history_frames + 1
        self.slot_emb = nn.Parameter(torch.zeros(1, self.num_tokens, dim))
        nn.init.normal_(self.slot_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, num_heads, ffn_dim, dropout=0.0,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, latent_dim)

    def forward(self, z_t: Tensor, t: Tensor, z_hat: Tensor, audio_tokens: Tensor,
                hidden: Optional[Tensor] = None, history: Optional[Tensor] = None) -> Tensor:
        # z_t/z_hat/audio/hidden/history: [B, T, *]   t: [B] or [B, T]
        if (history is None) != (self.history_frames == 0):
            raise ValueError(f"DiT head built with history_frames={self.history_frames} "
                             f"but history={'missing' if history is None else 'given'}")
        if (hidden is None) == self.use_hidden:
            raise ValueError(f"DiT head built with use_hidden={self.use_hidden} "
                             f"but hidden={'missing' if hidden is None else 'given'}")
        B, T, _ = z_t.shape
        if t.dim() == 1:
            t_tok = self.time_proj(t).unsqueeze(1).expand(-1, T, -1)
        else:
            t_tok = self.time_proj(t.reshape(-1)).reshape(B, T, -1)
        tokens = [t_tok, self.audio_proj(audio_tokens), self.zhat_proj(z_hat)]
        if self.use_hidden:
            tokens.append(self.hidden_proj(hidden))
        pad_mask = None
        if self.history_frames > 0:
            K, D = self.history_frames, self.latent_dim
            prev = history[..., :K * D].reshape(B, T, K, D)
            valid = history[..., K * D:]                        # [B, T, K]
            first = len(tokens)
            tokens.extend(self.history_proj(prev).unbind(dim=2))
            pad_mask = torch.zeros(B * T, self.num_tokens, dtype=torch.bool, device=z_t.device)
            pad_mask[:, first:first + K] = valid.reshape(B * T, K) < 0.5
        tokens.append(self.noisy_proj(z_t))
        x = torch.stack(tokens, dim=2).reshape(B * T, self.num_tokens, -1) + self.slot_emb
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        v = self.out(self.out_norm(x[:, -1]))                   # noisy-z_t position
        return v.reshape(B, T, self.latent_dim)


class VqvaeARFlowDiTPredict(VqvaeARFlowPredict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.hparams.arflow
        d = cfg.d_model
        # Replace the parent's MLP head (built last in the parent __init__, so
        # every other module is initialised exactly as in the MLP-head model).
        self.flow_head = ARFlowDiTHead(
            latent_dim=self.latent_dim,
            audio_dim=d,
            hidden_cond_dim=(d if self.flow_use_hidden else 0),
            history_frames=self.flow_history_frames,
            time_dim=cfg.time_dim,
            dim=cfg.get('dit_dim', 256),
            num_layers=cfg.get('dit_layers', 3),
            num_heads=cfg.get('dit_heads', 4),
            ffn_dim=cfg.get('dit_ffn_dim', 1024),
        )
        # The parent built its optimizer over the MLP head's parameters.
        self.optimizer = instantiate(self.hparams.optim,
                                     params=filter(lambda p: p.requires_grad, self.parameters()))
        self.__post_init__()
        logger.info(f"[arflow] flow head -> DiT ({self.flow_head.num_tokens} tokens/frame, "
                    f"dim={cfg.get('dit_dim', 256)}, layers={cfg.get('dit_layers', 3)}, "
                    f"heads={cfg.get('dit_heads', 4)})")
        self.log_param_table()
