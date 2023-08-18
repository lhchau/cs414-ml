import torch
import torch.nn as nn
from Week9.base_attention import BaseAttention

class CrossAttention(BaseAttention):
    def forward(self, x, context):
        attn_output, _ = self.mha(
            query=x,
            key=context,
            value=context,
        )
        attn_output = self.dropout(attn_output)
        x = x + attn_output
        x = self.layernorm(x)
        return x