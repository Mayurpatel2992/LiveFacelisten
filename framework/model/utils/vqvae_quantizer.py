# Code adapted from [Esser, Rombach 2021]: https://compvis.github.io/taming-transformers/

import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        # Layout version 2 quantizes independent e_dim-sized subcodes inside
        # every temporal frame.  Keeping this persistent makes checkpoints
        # trained with the old sequence-mixing layout fail loudly on load.
        self.register_buffer("layout_version", torch.tensor(2, dtype=torch.int64))

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def _flatten_per_frame(self, z):
        """Convert [B, T, D] to [B*T*num_subcodes, e_dim].

        The previous implementation permuted to [B, D, T] before ``view``.
        That made each code vector span multiple time steps whenever T != 1,
        so token identity depended on sequence length and leaked future frames.
        """
        if z.ndim != 3:
            raise ValueError(f"Expected z with shape [B,T,D], got {tuple(z.shape)}")
        batch, time, dim = z.shape
        if dim % self.e_dim != 0:
            raise ValueError(
                f"Latent dimension {dim} must be divisible by code dimension {self.e_dim}"
            )
        num_subcodes = dim // self.e_dim
        flat = z.reshape(batch, time, num_subcodes, self.e_dim)
        flat = flat.reshape(batch * time * num_subcodes, self.e_dim)
        return flat, (batch, time, dim, num_subcodes)

    @staticmethod
    def _restore_per_frame(z_flat, layout):
        batch, time, dim, num_subcodes = layout
        return z_flat.reshape(batch, time, num_subcodes, -1).reshape(batch, time, dim)

    def forward(self, z, sample=False, temperature=None, k=None):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        quantization pipeline:
            1. get encoder input
            2. flatten input
        """

        # One or more independent e_dim-sized code vectors per frame.
        z_flattened, layout = self._flatten_per_frame(z)

        # distances from z to embeddings e_j: (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        if not sample:      # no stochastic sampling (trianing or set sample to False)
            min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        else:               # stochastic sampling
            logits = -d     # Convert to probabilities
            if temperature is None or temperature <= 0:
                raise ValueError("temperature must be > 0 when sample=True")
            logits = logits / temperature   # temperature scaling
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            k = 1 if k is None else k
            min_encoding_indices = torch.multinomial(probabilities, num_samples=k)
            if k > 1:
                min_encoding_indices = min_encoding_indices[:, [-1]]
        assert min_encoding_indices is not None, "min_encoding_indices is None"

        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e,
            device=z.device, dtype=z.dtype,
        )
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q_flat = torch.matmul(min_encodings, self.embedding.weight)

        # compute loss for embedding
        loss = self.beta * torch.mean((z_q_flat.detach() - z_flattened) ** 2) + \
                   torch.mean((z_q_flat - z_flattened.detach()) ** 2)

        # preserve gradients
        z_q_flat = z_flattened + (z_q_flat - z_flattened).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # Restore [B,T,D] without mixing temporal positions.
        z_q = self._restore_per_frame(z_q_flat, layout)
        batch, time, _, num_subcodes = layout
        shaped_indices = min_encoding_indices.reshape(batch, time, num_subcodes)

        return z_q, loss, (perplexity, min_encodings, shaped_indices)

    def gumbel_softmax_quantize(self, z, temperature=1.0, hard=True):
        """
        Differentiable quantization using Gumbel-Softmax.
        During training: soft attention over codebook (gradients flow through)
        During inference (hard=True): argmax but with straight-through gradient

        z: [B, T, e_dim]  continuous input
        returns: z_q [B, T, e_dim]  quantized output (differentiable)
        """
        z_flattened, layout = self._flatten_per_frame(z)

        # distances from z to embeddings e_j: (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())  # [B*T, n_e]

        # convert distances to logits (negative distance = higher similarity)
        logits = -d  # [B*T, n_e]

        # Gumbel-Softmax: differentiable discrete sampling
        soft_one_hot = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)  # [B*T, n_e]

        # weighted sum of codebook entries
        z_q_flat = torch.matmul(soft_one_hot, self.embedding.weight)
        z_q = self._restore_per_frame(z_q_flat, layout)

        return z_q
