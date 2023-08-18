import torch
import torch.nn as nn
import numpy as np


class PositionalEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int, 
        d_model: int, 
        max_length=2048):
        super(PositionalEmbedding, self).__init__()
        """
        Input: 
            - vocab_size: max number of tokens
            - d_model: 
        """
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)  # padding_idx for mask_zero=True equivalent
        self.pos_encoding = self.positional_encoding(max_length, d_model)

    def positional_encoding(self, length, depth):
        """
        Input: 
            - length
            - depth
        Output:
            - pos_encoding: shape depth x length
        """
        depth = depth / 2
        pos = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        i = torch.arange(0, depth, dtype=torch.float)
        angle = pos / torch.pow(10000, (2 * i) / depth)
        pos_encoding = torch.cat(
            [torch.sin(angle), torch.cos(angle)], 
            dim=-1)
        return pos_encoding

    def forward(self, x):
        length = x.size(1)
        x = self.embedding(x)
        scale_factor = torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x *= scale_factor
        x = x + self.pos_encoding[:length, :].unsqueeze(0)
        return x
