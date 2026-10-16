import pytorch_lightning as pl
from torch import nn

from framework.model.utils.transformer_module import Transformer, LinearEmbedding
from framework.model.utils.position_embed import PositionalEncoding


class TransformerPredictor(pl.LightningModule):
    def __init__(self, latent_dim: int,
                 num_layers: int,
                 num_heads: int,
                 quant_factor: int,
                 intermediate_size: int,
                 # passed trough datamodule
                 audio_dim: int,
                 one_hot_dim: int,
                 prosody_dim: int,
                 **kwargs) -> None:

        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.audio_feature_map = nn.Linear(audio_dim, latent_dim)               # 768*2->256
        self.prosody_feature_map = nn.Linear(prosody_dim, audio_dim)               # 768*2->256
        self.style_embedding = nn.Linear(one_hot_dim, latent_dim, bias=False)   # 43->256

        if quant_factor == 0:
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

    def forward(self, audio, style_ont_hot, prosody):
        """
        audio: [B, T, 768*2]
        style_ont_hot: [B, 43]
        prosody: [B, 1, prosody_dim]
        """
        dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}

        # make prosody -> audio_dim and fuse with audio here
        prosody = self.prosody_feature_map(prosody) # prosody_dim -> audio_dim
        audio = audio * prosody

        inputs = self.audio_feature_map(audio)                  # [B=1, T, 256]

        # style embedding
        style_ont_hot = self.style_embedding(style_ont_hot).unsqueeze(1)  # [B, 1, 256]
        inputs = inputs * style_ont_hot                                    # [B, T, 256]

        inputs = self.squasher(inputs.permute(0, 2, 1)).permute(0, 2, 1)
        encoder_features = self.encoder_linear_embedding(inputs)
        encoder_features = self.encoder_pos_embedding(encoder_features)
        encoder_features = self.encoder_transformer((encoder_features, dummy_mask))

        return encoder_features

    def encode_condition(self, audio, style_one_hot, prosody, quantizer=None, gumbel_temperature=1.0):
        """
        Encode audio + style into condition features for diffusion head.
        If quantizer is provided, fuse continuous hidden tokens with Gumbel-Softmax
        quantized tokens (element-wise multiply, same logic as prosody * audio).

        audio: [B, T, audio_dim]
        style_one_hot: [B, one_hot_dim]
        prosody: [B, 1, prosody_dim]
        quantizer: VectorQuantizer instance (optional)
        gumbel_temperature: temperature for Gumbel-Softmax sampling
        returns: (cond [B, T, latent_dim], audio_tokens [B, T, latent_dim])
        """
        dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
        prosody = self.prosody_feature_map(prosody)  # prosody_dim -> audio_dim
        audio = audio * prosody
        audio_feat = self.audio_feature_map(audio)  # [B, T, 256]
        style_emb = self.style_embedding(style_one_hot).unsqueeze(1)  # [B, 1, 256]
        audio_tokens = audio_feat * style_emb  # [B, T, 256] — prosody+proj+style processed audio
        x = self.squasher(audio_tokens.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.encoder_linear_embedding(x)
        x = self.encoder_pos_embedding(x)
        hidden_tokens = self.encoder_transformer((x, dummy_mask))  # [B, T, 256]

        if quantizer is not None:
            # classify + differentiable sample from codebook
            quantized_tokens = quantizer.gumbel_softmax_quantize(
                hidden_tokens, temperature=gumbel_temperature, hard=not self.training
            )  # [B, T, 256]
            # fuse: continuous hidden tokens * discrete quantized tokens
            cond = hidden_tokens * quantized_tokens  # [B, T, 256]
        else:
            cond = hidden_tokens

        return cond, audio_tokens







