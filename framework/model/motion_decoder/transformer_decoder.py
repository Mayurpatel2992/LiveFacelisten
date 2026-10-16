import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn, Tensor

from framework.model.utils.transformer_module import LinearEmbedding
from framework.model.utils.position_embed import PositionalEncoding


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x):
        x = F.pad(x, (self.causal_padding, 0))
        return super().forward(x)


class TimewiseLayerNorm(nn.Module):
    """Normalize channels without using statistics from other frames."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels, elementwise_affine=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class TransformerDecoder(pl.LightningModule):
    def __init__(self, latent_dim: int,
                 num_heads: int,
                 num_layers: int,
                 quant_factor: int,
                 intermediate_size: int,
                 nfeats: int,
                 is_audio=False,
                 causal_norm: bool = True,
                 **kwargs) -> None:

        super().__init__()
        self.save_hyperparameters(logger=False)
        if causal_norm and quant_factor != 0:
            raise ValueError(
                "Causal motion decoding currently requires quant_factor=0; "
                "the legacy transposed-convolution path is non-causal."
            )

        self.expander = nn.ModuleList()
        norm = TimewiseLayerNorm if causal_norm else \
            lambda channels: nn.GroupNorm(1, channels, affine=False)

        if quant_factor == 0:
            self.expander.append(nn.Sequential(
                CausalConv1d(latent_dim, latent_dim, 5, stride=1, padding=0),
                nn.LeakyReLU(0.2, False),
                norm(latent_dim)))
        else:
            self.expander.append(nn.Sequential(
                nn.ConvTranspose1d(
                    latent_dim, latent_dim, 5,
                    stride=2, padding=2, output_padding=1
                ),
                nn.LeakyReLU(0.2, True),
                norm(latent_dim)))
            num_expand_layers = quant_factor + 2 if is_audio else quant_factor
            for _ in range(1, num_expand_layers):
                self.expander.append(nn.Sequential(
                    CausalConv1d(latent_dim, latent_dim, 5, stride=1, padding=0),
                    nn.LeakyReLU(0.2, True),
                    norm(latent_dim)))

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            batch_first=True,
        )
        self.decoder_transformer = nn.TransformerEncoder(
            decoder_layer,
            num_layers=num_layers,
        )

        self.decoder_pos_embedding = PositionalEncoding(latent_dim, batch_first=True)
        self.decoder_linear_embedding = LinearEmbedding(latent_dim, latent_dim)
        self.feature_mapping_reverse = nn.Linear(latent_dim, nfeats, bias=False)

        # Val/inference only: chunk long sequences to avoid OOM
        self.attn_chunk_size = 64

    def _sa_block_chunked(self, layer, x: Tensor, chunk_size: int) -> Tensor:
        """Exact causal self-attn with query chunking."""
        _, T, _ = x.shape
        outs = []
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            q = x[:, start:end]
            kv = x[:, :end]
            query_pos = torch.arange(start, end, device=x.device).unsqueeze(1)
            key_pos = torch.arange(end, device=x.device).unsqueeze(0)
            attn_mask = key_pos > query_pos  # True means blocked
            out, _ = layer.self_attn(
                q, kv, kv,
                attn_mask=attn_mask,
                need_weights=False,
            )
            outs.append(out)
        return layer.dropout1(torch.cat(outs, dim=1))

    def _transformer_causal_full(self, x: Tensor) -> Tensor:
        """Original full-sequence causal attention (training)."""
        # PyTorch Transformer boolean masks use True for blocked positions.
        causal_mask = torch.ones(
            x.size(1), x.size(1), device=x.device, dtype=torch.bool
        ).triu(diagonal=1)
        return self.decoder_transformer(x, mask=causal_mask)

    def _transformer_causal_chunked(self, x: Tensor) -> Tensor:
        """Exact causal attention with lower peak memory (val/inference)."""
        for layer in self.decoder_transformer.layers:
            if layer.norm_first:
                x = x + self._sa_block_chunked(layer, layer.norm1(x), self.attn_chunk_size)
                x = x + layer._ff_block(layer.norm2(x))
            else:
                x = layer.norm1(x + self._sa_block_chunked(layer, x, self.attn_chunk_size))
                x = layer.norm2(x + layer._ff_block(x))

        if self.decoder_transformer.norm is not None:
            x = self.decoder_transformer.norm(x)
        return x

    def forward(self, inputs: Tensor, use_chunked_attn: bool = False,
                position_offset: int = 0):
        # inputs: (B, T, D)
        for i, module in enumerate(self.expander):
            inputs = module(inputs.permute(0, 2, 1)).permute(0, 2, 1)
            if i > 0:   # only relevant for quant_factor > 0
                inputs = inputs.repeat_interleave(2, dim=1)

        decoder_features = self.decoder_linear_embedding(inputs)
        decoder_features = self.decoder_pos_embedding(
            decoder_features, start_pos=position_offset
        )

        # Default (training): original full causal path.
        # Val/inference can opt into chunked attn for long sequences.
        if use_chunked_attn:
            decoder_features = self._transformer_causal_chunked(decoder_features)
        else:
            decoder_features = self._transformer_causal_full(decoder_features)

        pred_recons = self.feature_mapping_reverse(decoder_features)
        return pred_recons
