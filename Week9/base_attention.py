import torch
import torch.nn as nn


class BaseAttention(nn.Module):
    def __init__(
        self, 
        embed_dim,
        num_heads,
        dropout: float = 0.0,
        **kwargs
        ):
        super().__init__()

        self.mha = nn.MultiHeadAttention(**kwargs)
        self.layernorm = nn.LayerNormalization()
        self.dropout = nn.Dropout(dropout)     