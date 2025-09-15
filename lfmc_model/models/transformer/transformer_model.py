import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Sinusoidal Positional Encoding
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # [max_len, d_model]
        self.alpha = nn.Parameter(torch.tensor(0.1))  # learnable scaling
    def forward(self, x):               # x: [B, T, d]
        T = x.size(1)
        return x + self.alpha * self.pe[:T].unsqueeze(0) 

# Multihead Attention Pooling
class MultiheadAttnPool(nn.Module):
    def __init__(self, d_model, nhead, num_queries=2, dropout=0.0):
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(num_queries, d_model))
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.out_drop = nn.Dropout(dropout)
        self.proj = nn.Sequential(
            nn.Linear(num_queries * d_model, d_model),
            nn.ReLU()
        )
    def forward(self, x):  # x: [B, T, d]
        B = x.size(0)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, Q, d]
        y, _ = self.mha(q, x, x)                         # [B, Q, d]
        y = self.out_drop(y)
        return self.proj(y.reshape(B, -1))               # [B, d]

# Linear Attention Pooling
class LinearAttnPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn_proj = nn.Linear(d_model, 1)
    def forward(self, x):
        weights = torch.softmax(self.attn_proj(x), dim=1)
        return (weights * x).sum(dim=1)

# FiLM conditioner
class FiLMConditioner(nn.Module):
    def __init__(self, in_dim: int, d_model: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * d_model)
        )
        # init close to identity: gamma≈0, beta=0
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, s):  # s: [B, in_dim]
        film = self.mlp(s)                # [B, 2*d]
        gamma, beta = film.chunk(2, -1)   # [B,d], [B,d]
        return gamma, beta

class LFMCTransformer(nn.Module):
    def __init__(
        self,
        short_input_dim: int,
        static_input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        num_queries: int = 2,
    ):
        super().__init__()
        # project input to d_model
        self.short_proj = nn.Linear(
            short_input_dim, d_model
        )
        self.static_proj = nn.Linear(
            static_input_dim, d_model
        )
        # transformer encoder
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
        # positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model=d_model)
        # FiLM conditioner
        self.film_cond = FiLMConditioner(
            in_dim=static_input_dim, d_model=d_model,
            hidden=64
        )
        # attention pooling
        self.pooler = MultiheadAttnPool(
            d_model=d_model,
            nhead=nhead,
            num_queries=num_queries,
            dropout=dropout
        )
        # head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
    def forward(
        self,
        short_history, # [B, T_short, D_short]
        static_features # [B, D_static]
    ):
        # project our static vector
        s_vec = self.static_proj(static_features).unsqueeze(1)  # [B, 1, d]
        # project short history
        x_tok = self.short_proj(short_history)                       # [B, T, d]
        # project our static history
        s_tok = self.static_proj(static_features).unsqueeze(1)  # [B, 1, d]
        # FiLM conditioning
        gamma, beta = self.film_cond(static_features)  # [B,d], [B,d]
        x_tok = gamma.unsqueeze(1) * x_tok + beta.unsqueeze(1)
        # prepend static token and encode
        x_seq = torch.cat([s_tok, x_tok], dim=1)  # [B, T+1, d]
        # positional encoding
        x_seq = self.pos_enc(x_seq)                # [B, T+1, d]
        # transformer encoding
        x_enc = self.encoder(x_seq)              # [B, T+1, d]
        # aggreatee with attention pooling
        x_met = x_enc[:,1:,:]  # remove static token
        h_met = self.pooler(x_met)               # [B, d]
        h_stat = x_enc[:,0,:]  # static token
        h = h_met + h_stat               # [B, d]
        # head
        logits_pred = self.head(h)                    # [B, 1]
        return logits_pred