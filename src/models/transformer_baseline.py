import torch
import torch.nn as nn

class TransformerBaseline(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 256, dropout: float = 0.1, output_dim: int = 1):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=False)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x: [B, T, D]
        x = self.proj(x)              # [B, T, d_model]
        x = x.permute(1, 0, 2)        # [T, B, d_model]
        h = self.encoder(x)           # [T, B, d_model]
        last = h[-1]                  # [B, d_model]
        last = self.dropout(last)
        return self.fc(last)          # [B, output_dim]
