import torch.nn as nn
import torch

class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,input_dim,d_model=64,nhead=4,
        num_layers=3,dim_feedforward=128,dropout=0.1
    ):
        super().__init__()
        # go from input_dim to d_model
        self.input_proj = nn.Linear(input_dim,d_model)
        # define what a single layer looks like
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        # define head
        self.head = nn.Sequential(
            nn.Linear(d_model,d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2,1)
        )
    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        x = self.input_proj(x)
        x = self.encoder(x)
        x_last = x[:, -1, :]  # take the last output
        out = self.head(x_last)
        return out
