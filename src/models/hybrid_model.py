import torch
import torch.nn as nn

class HybridLSTMTransformer(nn.Module):
    '''
    Simple hybrid: project→LSTM→(optional)Transformer→pool→FC
    If num_transformer_layers == 0, this reduces to an LSTM + pooling.
    '''
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        num_transformer_layers: int = 2,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        output_dim: int = 1,
        pooling: str = "last",  # "last" or "mean"
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=lstm_hidden, num_layers=lstm_layers, batch_first=True, dropout=dropout if lstm_layers>1 else 0.0)
        self.to_trans = nn.Linear(lstm_hidden, d_model)

        if num_transformer_layers > 0:
            enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=False)
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_transformer_layers)
        else:
            self.encoder = None

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, output_dim)
        self.pooling = pooling

    def forward(self, x):
        # x: [B, T, D]
        x = self.proj(x)                    # [B, T, d_model]
        h, _ = self.lstm(x)                 # [B, T, lstm_hidden]
        h = self.to_trans(h)                # [B, T, d_model]

        if self.encoder is not None:
            h = h.permute(1, 0, 2)          # [T, B, d_model]
            h = self.encoder(h)             # [T, B, d_model]
            h = h.permute(1, 0, 2)          # [B, T, d_model]

        if self.pooling == "mean":
            pooled = h.mean(dim=1)          # [B, d_model]
        else:
            pooled = h[:, -1, :]            # [B, d_model]

        pooled = self.dropout(pooled)
        return self.fc(pooled)              # [B, output_dim]
