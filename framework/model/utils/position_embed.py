from torch import nn
import torch
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=True):
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x, start_pos: int = 0):
        end_pos = start_pos + (x.shape[1] if self.batch_first else x.shape[0])
        if end_pos > self.pe.shape[0]:
            raise ValueError(
                f"Position {end_pos} exceeds positional encoding limit {self.pe.shape[0]}"
            )
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, start_pos:end_pos, :]
        else:
            x = x + self.pe[start_pos:end_pos, :]
        return self.dropout(x)
