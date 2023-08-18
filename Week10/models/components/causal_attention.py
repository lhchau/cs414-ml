import torch
import torch.nn as nn
from Week9.base_attention import BaseAttention

class CausalSelfAttention(BaseAttention):
    def forward(self, x):
        attn_output, _ = self.mha(
            query=x,
            key=x,
            value=x,
            use_causal_mask=True
        )
        attn_output = self.dropout(attn_output)
        x = x + attn_output
        x = self.layernorm(x)
        return x