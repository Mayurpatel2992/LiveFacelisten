import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn, Tensor

from framework.model.utils.transformer_module import Transformer, LinearEmbedding
from framework.model.utils.position_embed import PositionalEncoding


class CausalConv1d(nn.Conv1d):
    """1-D convolution padded only on the past side."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(F.pad(x, (self.causal_padding, 0)))


class TimewiseLayerNorm(nn.Module):
    """Normalize channels independently at each time step.

    InstanceNorm1d computes statistics across the complete time axis and
    therefore leaks future motion into every encoded frame.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels, elementwise_affine=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class TransformerEncoder(pl.LightningModule):
    def __init__(self, latent_dim: int,
                 num_layers: int,
                 num_heads: int,
                 quant_factor: int,
                 intermediate_size: int,
                 nfeats: int,  # passed trough datamodule
                 causal: bool = True,
                 **kwargs) -> None:

        super().__init__()
        self.save_hyperparameters(logger=False)

        self.feature_mapping = nn.Sequential(nn.Linear(nfeats, latent_dim), 
                                             nn.LeakyReLU(0.2, True))

        self.causal = causal
        if causal and quant_factor != 0:
            raise ValueError(
                "Causal motion encoding currently requires quant_factor=0; "
                "the legacy downsampling path is non-causal."
            )

        if quant_factor == 0 and causal:
            layers = [nn.Sequential(
                CausalConv1d(latent_dim, latent_dim, 5, stride=1, padding=0),
                nn.LeakyReLU(0.2, True),
                TimewiseLayerNorm(latent_dim))]
        elif quant_factor == 0:
            layers = [nn.Sequential(
                nn.Conv1d(latent_dim, latent_dim, 5, stride=1, padding=2, padding_mode='replicate'),
                nn.LeakyReLU(0.2, True),
                nn.InstanceNorm1d(latent_dim, affine=False))]
        else:   # this is not used
            layers = [nn.Sequential(
                nn.Conv1d(latent_dim, latent_dim, 5, stride=2, padding=2, padding_mode='zeros'),
                nn.LeakyReLU(0.2, True),
                nn.InstanceNorm1d(latent_dim, affine=False))]
            for _ in range(1, quant_factor):
                layers += [nn.Sequential(
                    nn.Conv1d(latent_dim, latent_dim, 5, stride=1, padding=2, padding_mode='zeros'),
                    nn.LeakyReLU(0.2, True),
                    nn.InstanceNorm1d(latent_dim, affine=False),
                    nn.MaxPool1d(2)
                )]
        self.squasher = nn.Sequential(*layers)
        self.encoder_transformer = Transformer(
            in_size=latent_dim,
            hidden_size=latent_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size)

        self.encoder_pos_embedding = PositionalEncoding(latent_dim, batch_first=True)
        self.encoder_linear_embedding = LinearEmbedding(latent_dim, latent_dim)

    def forward(self, motion: Tensor):
        inputs = self.feature_mapping(motion)
        inputs = self.squasher(inputs.permute(0, 2, 1)).permute(0, 2, 1)
        encoder_features = self.encoder_linear_embedding(inputs)
        encoder_features = self.encoder_pos_embedding(encoder_features)

        if self.causal:
            time = encoder_features.shape[1]
            attention_mask = torch.ones(
                time, time, device=encoder_features.device, dtype=torch.bool
            ).tril()
            mask_info = {
                'max_mask': time,
                'mask_index': -1,
                'mask': attention_mask,
            }
        else:
            mask_info = {'max_mask': None, 'mask_index': -1, 'mask': None}

        encoder_features = self.encoder_transformer((encoder_features, mask_info))

        return encoder_features
    





