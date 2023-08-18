import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(
        self, 
        d_model,
        dff,
        dropout,
        *args, 
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        
        self.seq = nn.Sequential([
            nn.Linear(dff),
            nn.ReLU(),
            nn.Linear(d_model),
            nn.Dropout(dropout)
        ])
        self.layer_norm = nn.LayerNorm()
        
    def forward(self, x):
        x = x + self.seq(x)
        x = self.layer_norm(x)
        return x