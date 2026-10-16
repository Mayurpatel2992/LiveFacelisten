import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim, scale=1000.0):
        super().__init__()
        if dim < 4 or dim % 2 != 0:
            raise ValueError(f"dim must be even and >= 4, got {dim}")
        self.dim = dim
        self.scale = scale

    def forward(self, t):
        """
        t: [B] continuous timesteps in [0, 1]
        returns: [B, dim] in same dtype as t
        """
        half_dim = self.dim // 2
        # Compute in FP32 for numerical stability
        freqs = torch.exp(
            torch.arange(half_dim, device=t.device, dtype=torch.float32)
            * (-math.log(10000.0) / (half_dim - 1))
        )
        angles = (t.float().clamp(0.0, 1.0).unsqueeze(1) * self.scale
                  * freqs.unsqueeze(0))  # [B, half_dim]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [B, dim]
        return emb.to(dtype=t.dtype)


class RMSNorm(nn.Module):
    """Root Mean Square Normalization — preserves magnitude signal unlike LayerNorm."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        input_dtype = x.dtype
        x_float = x.float()
        rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_float * rms * self.scale.float()).to(input_dtype)


class MultiScaleTemporalMix(nn.Module):
    """Depthwise 1D convolutions at multiple receptive fields for temporal mixing."""

    def __init__(self, dim, kernel_sizes=(3, 7, 15)):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            self.branches.append(nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=k, padding=k // 2, groups=dim),
                nn.SiLU(),
            ))
        self.proj = nn.Linear(dim * len(kernel_sizes), dim)

    def forward(self, x):
        """
        x: [B, T, D]
        returns: [B, T, D]
        """
        x_t = x.transpose(1, 2)  # [B, D, T]
        outs = [branch(x_t).transpose(1, 2) for branch in self.branches]  # list of [B, T, D]
        return self.proj(torch.cat(outs, dim=-1))  # [B, T, D]


class DiffusionMLPHead(nn.Module):
    """MLP with multi-scale temporal mixing for flow matching velocity prediction.

    Architecture:
        1. Backbone: fuses z_t + cond + time_emb → hidden features
        2. Temporal mixing: multi-scale depthwise 1D convs (k=3,7,15)
        3. Output projection: hidden → latent_dim velocity
    """

    def __init__(self, latent_dim=256, cond_dim=256, time_dim=256, hidden_dim=512, audio_dim=0, audio_proj_dim=256, **kwargs):
        super().__init__()
        self.audio_dim = audio_dim
        self.audio_proj_dim = audio_proj_dim if audio_dim > 0 else 0

        if audio_dim > 0:
            self.audio_proj = nn.Sequential(
                nn.Linear(audio_dim, audio_proj_dim),
                nn.SiLU(),
            )

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        backbone_input_dim = latent_dim + cond_dim + time_dim + self.audio_proj_dim
        self.backbone = nn.Sequential(
            nn.Linear(backbone_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Multi-scale temporal mixing
        self.temporal_mix = MultiScaleTemporalMix(hidden_dim, kernel_sizes=(3, 7, 15))
        self.temporal_norm = RMSNorm(hidden_dim)

        # Output projection
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_t, t, cond, audio_tokens=None):
        """
        z_t:          [B, T, latent_dim]  interpolated latent
        t:            [B]                 continuous timesteps in [0, 1] (float)
        cond:         [B, T, cond_dim]    condition features
        audio_tokens: [B, T, audio_dim]   raw audio features (optional)
        returns: v_hat [B, T, latent_dim]  predicted velocity
        """
        t_emb = self.time_mlp(t)  # [B, time_dim]
        t_emb = t_emb.unsqueeze(1).expand(-1, z_t.shape[1], -1)  # [B, T, time_dim]
        parts = [z_t, cond, t_emb]
        if audio_tokens is not None and self.audio_dim > 0:
            parts.append(self.audio_proj(audio_tokens))  # [B, T, audio_proj_dim]
        x = torch.cat(parts, dim=-1)  # [B, T, latent+cond+time(+audio_proj)]

        h = self.backbone(x)  # [B, T, hidden_dim]

        # Temporal mixing with residual
        h = h + self.temporal_mix(h)
        h = self.temporal_norm(h)

        return self.out(h)  # [B, T, latent_dim]


class FlowMatchingScheduler:
    """Conditional Flow Matching scheduler with linear interpolation.

    Convention: t=0 → clean data, t=1 → noise (Lipman et al., 2023).
    """

    def __init__(self, **kwargs):
        pass

    def to(self, device):
        return self

    def interpolate(self, z0, z1, t):
        """
        Linear interpolation: z_t = (1 - t) * z0 + t * z1
        z0:  [B, T, D]  clean data
        z1:  [B, T, D]  noise sample
        t:   [B]         continuous time in [0, 1]
        returns: z_t [B, T, D]
        """
        t = t.view(-1, 1, 1)  # [B, 1, 1]
        z_t = (1.0 - t) * z0 + t * z1
        return z_t

    def target_velocity(self, z0, z1):
        """
        Target velocity for CFM: v = z1 - z0  (noise - data)
        """
        return z1 - z0
