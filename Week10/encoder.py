import torch
import torch.nn as nn

from Week10.global_selfattn import GlobalSelfAttention
from Week10.feed_forward_layer import FeedForward
from Week9.positional_embed import PositionalEmbedding

class EncoderLayer(nn.Module):
    def __init__(
        self, 
        d_model, 
        num_heads, 
        dff, 
        dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.self_attention = GlobalSelfAttention(d_model, num_heads, dropout=dropout_rate)
        self.ffn = FeedForward(d_model, dff)
        
    def forward(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self, 
        *,
        num_layers,
        d_model, 
        num_heads, 
        dff, 
        vocab_size, 
        dropout_rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)

        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        return x
