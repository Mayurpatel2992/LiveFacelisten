"""Option A (UniLS, Chu et al. 2026, Fig. 3 stage 2) on the DualTalk speaking model.

Each causal backbone block of the trained speaking model becomes
    x = x + SelfAttn(LN(x))                      LoRA on qkv / proj
    x = x + CrossAttn_own(LN(x), own audio)      new, trained from scratch
    x = x + CrossAttn_partner(LN(x), partner)    new, trained from scratch
    x = x + FFN(LN(x))                           LoRA on both linears
UniLS's stage 1 (audio-free generator) is skipped: the speaking model's
history dropout / uncertainty masking already trains it to continue motion
with weak audio. Cross-attention output projections and LoRA B matrices are
zero-initialised, so training starts exactly at the speaking model.
The own audio also still enters through the backbone input (as in the
speaking model); the AR code heads and flow head are fine-tuned when
duplex.train_heads is true.
"""
import torch.nn as nn
from torch import Tensor
from typing import Optional

from framework.model.model_dualtalk_duplex import CausalCrossAttention, DualTalkDuplexBase, LoRALinear


class CrossBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window: int):
        super().__init__()
        self.norm_own = nn.LayerNorm(dim)
        self.mem_own = nn.LayerNorm(dim)
        self.ca_own = CausalCrossAttention(dim, num_heads, window)
        self.norm_par = nn.LayerNorm(dim)
        self.mem_par = nn.LayerNorm(dim)
        self.ca_par = CausalCrossAttention(dim, num_heads, window)

    def forward(self, x: Tensor, own: Tensor, par: Tensor, cache: Optional[dict]) -> Tensor:
        x = x + self.ca_own(self.norm_own(x), self.mem_own(own), None if cache is None else cache["own"])
        x = x + self.ca_par(self.norm_par(x), self.mem_par(par), None if cache is None else cache["par"])
        return x


class DualTalkUniLS(DualTalkDuplexBase):
    def build_duplex(self, dcfg):
        a = self.hparams.arflow
        rank, alpha = int(dcfg.get('lora_rank', 16)), float(dcfg.get('lora_alpha', 16))
        for layer in self.backbone.layers:
            layer.attn.qkv = LoRALinear(layer.attn.qkv, rank, alpha)
            layer.attn.proj = LoRALinear(layer.attn.proj, rank, alpha)
            layer.ffn[0] = LoRALinear(layer.ffn[0], rank, alpha)
            layer.ffn[2] = LoRALinear(layer.ffn[2], rank, alpha)
        self.cross_blocks = nn.ModuleList([
            CrossBlock(a.d_model, a.num_heads, int(dcfg.get('cross_attn_window', 0)))
            for _ in self.backbone.layers
        ])
        if dcfg.get('train_heads', True):
            for m in (self.ar_code_heads, self.flow_head):
                for p in m.parameters():
                    p.requires_grad = True

    def duplex_param_prefixes(self):
        return ("cross_blocks",)

    def new_duplex_cache(self):
        return [{"sa": {}, "own": {}, "par": {}} for _ in self.backbone.layers]

    def duplex_hidden(self, x: Tensor, own_tok: Tensor, par_tok: Tensor,
                      gate: Optional[Tensor], cache=None) -> Tensor:
        for i, layer in enumerate(self.backbone.layers):
            c = None if cache is None else cache[i]
            x = x + layer.attn(layer.norm1(x), kv_cache=None if c is None else c["sa"])
            x = self.cross_blocks[i](x, own_tok, par_tok, c)
            x = x + layer.ffn(layer.norm2(x))
        return self.backbone.norm(x)
