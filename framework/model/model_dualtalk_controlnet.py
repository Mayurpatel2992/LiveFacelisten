"""Option B: ControlNet-style listening branch on the frozen DualTalk speaking model.

    main branch  : the trained speaking model, fully frozen; always driven by
                   the agent's OWN track (its own voice, silence while listening)
    listen branch: trainable copy of the main causal backbone (same init),
                   input = main input + Z_in(partner audio tokens)
    connections  : after every block i,  h_i <- h_i + g_t * Z_i(c_i)
                   Z_in, Z_i zero-initialised Linear (ControlNet zero convs)
    gate g_t     : 1 - P(agent is speaking at t), from a small causal classifier
                   on the own-track audio tokens (trained with BCE on the VAD
                   labels). Own voice -> g ~ 0 -> pure speaking model;
                   partner talking or silence -> g ~ 1 -> listening residual on.

Gate sources (duplex.gate_train_source / duplex.gate_infer_source):
    vad   ground-truth own VAD (teacher-forced switch; oracle at inference)
    pred  the classifier (deployable)
    ones  no gate (plain ControlNet, ablation)
"""
import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from framework.model.model_dualtalk_duplex import DualTalkDuplexBase

GATE_SOURCES = ("vad", "pred", "ones")


class DualTalkControlNet(DualTalkDuplexBase):
    def build_duplex(self, dcfg):
        d = self.hparams.arflow.d_model
        self.ctrl_layers = copy.deepcopy(self.backbone.layers)
        for p in self.ctrl_layers.parameters():
            p.requires_grad = True
        self.ctrl_in = nn.Linear(d, d)
        self.ctrl_out = nn.ModuleList([nn.Linear(d, d) for _ in self.ctrl_layers])
        for lin in [self.ctrl_in, *self.ctrl_out]:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.gate_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))
        self.gate_train_source = dcfg.get('gate_train_source', 'vad')
        self.gate_infer_source = dcfg.get('gate_infer_source', 'pred')
        for s in (self.gate_train_source, self.gate_infer_source):
            if s not in GATE_SOURCES:
                raise ValueError(f"gate source must be one of {GATE_SOURCES}, got {s!r}")
        self.lambda_gate = float(dcfg.get('lambda_gate', 0.1))

    def duplex_param_prefixes(self):
        return ("ctrl_layers", "ctrl_in", "ctrl_out", "gate_head")

    # ---- gate ----------------------------------------------------------- #
    def speak_logits(self, own_tok: Tensor) -> Tensor:
        """[B,T,d] own audio tokens (causal, frozen front-end) -> [B,T] logits."""
        return self.gate_head(own_tok.detach()).squeeze(-1)

    def _gate_from(self, source: str, own_tok: Tensor, vad: Optional[Tensor]) -> Tensor:
        if source == "vad":
            if vad is None:
                raise ValueError("gate source 'vad' needs VAD labels")
            return (1.0 - vad).unsqueeze(-1)
        if source == "pred":
            return (1.0 - torch.sigmoid(self.speak_logits(own_tok))).unsqueeze(-1)
        return torch.ones_like(own_tok[..., :1])

    def duplex_gate(self, own_tok, vad, split):
        source = self.gate_train_source if split == "train" else self.gate_infer_source
        return self._gate_from(source, own_tok, vad)

    def infer_gate(self, own_tok, vad):
        return self._gate_from(self.gate_infer_source, own_tok, vad)

    def extra_losses(self, own_tok, vad, mask, split):
        logits = self.speak_logits(own_tok)
        bce = F.binary_cross_entropy_with_logits(logits, vad, reduction='none')
        loss = (bce * mask).sum() / mask.sum().clamp(min=1.0)
        with torch.no_grad():
            acc = (((logits > 0).float() == vad).float() * mask).sum() / mask.sum().clamp(min=1.0)
        return {"gate_bce": (loss, self.lambda_gate), "gate_acc": (acc, 0.0)}

    # ---- backbone -------------------------------------------------------- #
    def new_duplex_cache(self):
        return {"main": self.backbone.new_kv_caches(), "ctrl": [dict() for _ in self.ctrl_layers]}

    def duplex_hidden(self, x: Tensor, own_tok: Tensor, par_tok: Tensor,
                      gate: Optional[Tensor], cache=None) -> Tensor:
        h = x
        c = x + self.ctrl_in(par_tok)
        for i, (main, ctrl) in enumerate(zip(self.backbone.layers, self.ctrl_layers)):
            h = main(h, kv_cache=None if cache is None else cache["main"][i])
            c = ctrl(c, kv_cache=None if cache is None else cache["ctrl"][i])
            r = self.ctrl_out[i](c)
            h = h + (r if gate is None else gate * r)
        return self.backbone.norm(h)
