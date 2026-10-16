"""MIBURI-style Moshi/Mimi audio conditioning for LiveFace Stage-2.

Path (MIBURI, conceptually):

    waveform --frozen Mimi--> 8 RVQ code streams @ 12.5 Hz  (precomputed offline)
             --Moshi emb[:8]--> 8 lookups of dim 4096
             --sum over the 8 codebooks--> one 4096-D vector per Mimi step

Reference: miburi/models/gesture_lm.py::process_conditions
    audio_emb = sum_k  audio_procemb[k]( audio_codes[:, k, :] )        # [B, T, 4096]

Only the 8 embedding tables are needed here -- never the Moshi 7B transformer.
The tables are exported once by ``pre_process/export_moshi_audio_emb.py`` and the
Mimi code streams by ``pre_process/precompute_mimi_codes.py``.

`MoshiMimiEmb` mirrors the small surface the Stage-2 model reads off the HuBERT
feature extractor: ``.audio_encoded_dim`` and ``.hparams.output_framerate`` /
``.hparams.name``.  It does NOT run Mimi -- it consumes cached integer codes.
"""
from typing import Optional

import logging

import torch
import torch.nn as nn
import pytorch_lightning as pl

logger = logging.getLogger(__name__)


class MoshiMimiEmb(pl.LightningModule):
    """Sum of 8 Moshi audio-codebook embeddings.

    Args:
        emb_path:        .pt produced by export_moshi_audio_emb.py
                         {"tables": [K0, card+1, dim], "zero_idx": -1, ...}
        num_codebooks:   how many of the K0 tables to use (MIBURI = 8)
        trainable:       if True the copied tables receive gradients (Exp 3);
                         if False they are frozen (Exp 1 / Exp 2)
        output_framerate: Mimi token rate, 12.5 Hz
    """

    def __init__(self,
                 emb_path: str,
                 num_codebooks: int = 8,
                 trainable: bool = False,
                 feature_norm: str = "layernorm",
                 output_framerate: float = 12.5,
                 name: str = "MoshiMimiEmb",
                 **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        blob = torch.load(emb_path, map_location="cpu")
        tables = blob["tables"]                          # [K0, card+1, dim] fp32
        if tables.dim() != 3:
            raise ValueError(f"expected [K, card+1, dim] tables, got {tuple(tables.shape)}")
        K0, rows, dim = tables.shape
        if num_codebooks > K0:
            raise ValueError(f"asked for {num_codebooks} codebooks, file has {K0}")

        self.num_codebooks = int(num_codebooks)
        self.zero_idx = int(blob.get("zero_idx", -1))
        self.card = int(blob.get("card", rows - 1))
        self.audio_encoded_dim = int(dim)               # 4096  -- read by the model
        self.trainable = bool(trainable)

        # One nn.Embedding per codebook, initialised from Moshi's emb[k].
        # For valid Mimi ids (0..card-1) Moshi's ScaledEmbedding(norm=False,
        # low_rank=None) is exactly nn.Embedding, so a plain lookup reproduces it.
        self.embs = nn.ModuleList([
            nn.Embedding(rows, dim) for _ in range(self.num_codebooks)
        ])
        with torch.no_grad():
            for k in range(self.num_codebooks):
                self.embs[k].weight.copy_(tables[k])

        # The raw summed embedding has per-element std ~0.02 -- ~13x smaller than a
        # HuBERT hidden state (~0.31), and HuBERT states are LayerNorm'd inside the
        # transformer while these are not. Without this the tiny signal is swamped
        # by the Stage-2 multiplicative prosody/style gate. Non-affine LayerNorm
        # rescales to ~unit std (0 params, identical for frozen and trainable
        # variants). Set feature_norm='none' to ablate.
        self.feature_norm = str(feature_norm).lower()
        if self.feature_norm == "layernorm":
            self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        elif self.feature_norm in ("none", "null", "identity"):
            self.norm = nn.Identity()
        else:
            raise ValueError(f"feature_norm must be 'layernorm' or 'none', got {feature_norm!r}")

        for p in self.parameters():
            p.requires_grad = self.trainable

        logger.info(
            f"MoshiMimiEmb: {self.num_codebooks} tables x {tuple(tables[0].shape)} "
            f"from '{emb_path}'  trainable={self.trainable}  dim={self.audio_encoded_dim}  "
            f"feature_norm={self.feature_norm}"
        )

    def train(self, mode: bool = True):
        # Frozen tables: keep in eval so nothing (e.g. future dropout) perturbs them.
        super().train(mode and self.trainable)
        return self

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: LongTensor [B, K, T]  (K >= num_codebooks) -> [B, T, dim].

        Sums the first ``num_codebooks`` code streams through their tables.
        ``zero_idx`` entries map to an exact zero vector (Moshi convention);
        real Mimi ids are always in [0, card-1] so this is a no-op in practice.
        """
        if codes.dim() != 3:
            raise ValueError(f"expected codes [B, K, T], got {tuple(codes.shape)}")
        codes = codes.long()
        feat: Optional[torch.Tensor] = None
        for k in range(self.num_codebooks):
            ck = codes[:, k, :]                          # [B, T]
            is_zero = ck == self.zero_idx
            e = self.embs[k](ck.clamp(min=0))           # [B, T, dim]
            if bool(is_zero.any()):
                e = torch.where(is_zero.unsqueeze(-1), torch.zeros_like(e), e)
            feat = e if feat is None else feat + e
        return self.norm(feat)                           # [B, T, dim], ~unit std

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MoshiTextEmb(pl.LightningModule):
    """Moshi *text* conditioning embedding -- the text half of MIBURI's
    ``process_conditions``.

    MIBURI:
        text_procemb = copy.deepcopy(lm.text_emb.weight.data)          # [V+1, 4096]
        self.text_procemb = ScaledEmbedding(V+1, 4096, norm=False, zero_idx=-1,
                                            _weight=text_procemb,
                                            _freeze=textaudio_emb_freeze)
        ...
        text_emb = self.text_procemb(text_codes)                       # [B, T, 4096]

    The interleaved 12.5 Hz text-token stream is precomputed offline by
    ``pre_process/precompute_moshi_text_tokens.py`` (Moshi Inner-Monologue
    alignment); the table is exported once by
    ``pre_process/export_moshi_text_emb.py``.  The Moshi transformer is never
    instantiated.

    For valid SentencePiece ids Moshi's ``ScaledEmbedding(norm=False,
    low_rank=None)`` is exactly ``nn.Embedding``, so a plain lookup reproduces
    it; ``zero_idx`` (-1, only emitted when a clip has no transcript at all)
    maps to an exact zero vector.  A non-affine LayerNorm mirrors the audio path
    so the two 4096-D signals reach the Stage-2 fusion at comparable scale.
    """

    def __init__(self,
                 emb_path: str,
                 trainable: bool = False,
                 feature_norm: str = "layernorm",
                 output_framerate: float = 12.5,
                 name: str = "MoshiTextEmb",
                 **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        blob = torch.load(emb_path, map_location="cpu")
        table = blob["table"]                            # [rows, dim] fp32
        if table.dim() != 2:
            raise ValueError(f"expected [rows, dim] text table, got {tuple(table.shape)}")
        rows, dim = table.shape

        self.zero_idx = int(blob.get("zero_idx", -1))
        self.vocab = int(blob.get("vocab", rows - 1))
        self.text_padding = int(blob.get("text_padding", 3))
        self.end_of_text_padding = int(blob.get("end_of_text_padding", 0))
        self.text_encoded_dim = int(dim)                 # 4096  -- read by the model
        self.audio_encoded_dim = int(dim)               # alias (model reads either name)
        self.trainable = bool(trainable)

        self.emb = nn.Embedding(rows, dim)
        with torch.no_grad():
            self.emb.weight.copy_(table)

        self.feature_norm = str(feature_norm).lower()
        if self.feature_norm == "layernorm":
            self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        elif self.feature_norm in ("none", "null", "identity"):
            self.norm = nn.Identity()
        else:
            raise ValueError(f"feature_norm must be 'layernorm' or 'none', got {feature_norm!r}")

        for p in self.parameters():
            p.requires_grad = self.trainable

        logger.info(
            f"MoshiTextEmb: table {tuple(table.shape)} from '{emb_path}'  "
            f"trainable={self.trainable}  dim={self.text_encoded_dim}  "
            f"feature_norm={self.feature_norm}  "
            f"(PAD={self.text_padding}, EPAD={self.end_of_text_padding})"
        )

    def train(self, mode: bool = True):
        super().train(mode and self.trainable)
        return self

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: LongTensor [B, T] or [B, 1, T]  ->  [B, T, dim].

        ``zero_idx`` (-1) entries map to an exact zero vector; PAD/EPAD and real
        SentencePiece ids (0..vocab-1) are ordinary table rows.
        """
        if codes.dim() == 3:
            codes = codes.squeeze(1)
        if codes.dim() != 2:
            raise ValueError(f"expected text codes [B, T] or [B, 1, T], got {tuple(codes.shape)}")
        codes = codes.long()
        is_zero = codes == self.zero_idx
        e = self.emb(codes.clamp(min=0))                 # [B, T, dim]
        if bool(is_zero.any()):
            e = torch.where(is_zero.unsqueeze(-1), torch.zeros_like(e), e)
        return self.norm(e)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
