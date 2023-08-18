import torch
import torch.nn as nn

from Week10.models.components.causal_attention import CausalSelfAttention
from Week10.models.components.feed_forward_layer import FeedForward
from Week10.models.components.cross_attention import CrossAttention
from Week9.positional_embed import PositionalEmbedding

class DecoderLayer(nn.Module):
    def __init__(
        self, 
        d_model, 
        num_heads, 
        dff, 
        dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(d_model, num_heads, dropout=dropout_rate)
        
        self.cross_attention = CrossAttention(d_model, num_heads, dropout_rate)
        
        self.ffn = FeedForward(d_model, dff)
        
    def forward(self, x, context):
        x = self.causal_self_attention(x)
        x = self.cross_attention(x, context)
        x = self.ffn(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self, 
        *,
        num_layers,
        d_model, 
        num_heads, 
        dff, 
        vocab_size, 
        dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)

        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        return x
