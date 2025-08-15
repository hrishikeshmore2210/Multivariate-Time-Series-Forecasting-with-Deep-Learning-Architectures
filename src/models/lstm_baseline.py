import torch
import torch.nn as nn

class LSTMBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [B, T, D]
        out, _ = self.lstm(x)
        last = out[:, -1, :]           # [B, H]
        last = self.dropout(last)
        return self.fc(last)           # [B, output_dim]
