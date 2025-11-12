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
    def __init__(self, short_input_dim, static_input_dim,
                 d_model=64, nhead=4, num_layers=3,
                 dim_feedforward=128, dropout=0.1,
                 num_queries=2):
        super().__init__()
        self.short_proj  = nn.Linear(short_input_dim, d_model)
        self.static_proj = nn.Linear(static_input_dim,  d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=enc_layer, num_layers=num_layers
        )
        self.pos_enc  = SinusoidalPositionalEncoding(d_model)
        self.film_cond = FiLMConditioner(
            in_dim=static_input_dim, d_model=d_model, hidden=64
        )
        self.pooler = LinearAttnPool(d_model)
        #self.pooler = MultiheadAttnPool(
        #    d_model=d_model, nhead=nhead,
        #    num_queries=num_queries, dropout=dropout
        #)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
    def forward(self, short_history, static_features):
        # static_features: [B, 1, static_input_dim]
        static_features = static_features.squeeze(1)   # [B, static_input_dim]
        # Project
        x_tok = self.short_proj(short_history)         # [B,T,d]
        s_tok = self.static_proj(static_features)      # [B,d]
        s_tok = s_tok.unsqueeze(1)                     # [B,1,d]
        ## FiLM: stable residual scaling
        #gamma, beta = self.film_cond(static_features)  # [B,d],[B,d]
        #gamma = 0.5 * torch.tanh(gamma)                # bound
        #x_tok = (1 + gamma).unsqueeze(1) * x_tok \
        #        + beta.unsqueeze(1)                    # [B,T,d]
        # Pack sequence (static token first)
        x_seq = torch.cat([s_tok, x_tok], dim=1)       # [B,T+1,d]
        #x_seq = self.pos_enc(x_seq)
        x_enc = self.encoder(x_seq)                    # [B,T+1,d]
        # Pool (met tokens) + add static token
        h_met  = self.pooler(x_enc[:, 1:, :])          # [B,d]
        h_stat = x_enc[:, 0, :]                        # [B,d]
        h = h_met + h_stat
        y = self.head(h).squeeze(-1)                   # [B]
        return y