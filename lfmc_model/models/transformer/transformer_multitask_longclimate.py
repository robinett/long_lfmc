import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float
                          ).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        self.alpha = nn.Parameter(torch.tensor(0.1))
    def forward(self, x):               # x: [B, T, d]
        T = x.size(1)
        return x + self.alpha * self.pe[:T].unsqueeze(0)

class MultiheadAttnPool(nn.Module):
    def __init__(self, d_model, nhead, num_queries=2, dropout=0.0):
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(num_queries,
                                                d_model))
        self.mha = nn.MultiheadAttention(embed_dim=d_model,
                                         num_heads=nhead,
                                         dropout=dropout,
                                         batch_first=True)
        self.out_drop = nn.Dropout(dropout)
        self.proj = nn.Sequential(
            nn.Linear(num_queries * d_model, d_model),
            nn.ReLU()
        )
    def forward(self, x):               # x: [B, T, d]
        B = x.size(0)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        y, _ = self.mha(q, x, x)        # [B, Q, d]
        y = self.out_drop(y)
        return self.proj(y.reshape(B, -1))  # [B, d]

class LinearAttnPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn_proj = nn.Linear(d_model, 1)
    def forward(self, x):               # x: [B, T, d]
        w = torch.softmax(self.attn_proj(x), dim=1)
        return (w * x).sum(dim=1)       # [B, d]

class FiLMConditioner(nn.Module):
    def __init__(self, in_dim: int, d_model: int, hidden: int=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * d_model)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
    def forward(self, s):               # s: [B, in_dim]
        film = self.mlp(s)              # [B, 2d]
        g, b = film.chunk(2, -1)        # [B,d],[B,d]
        return g, b

class GaussianHead(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )
    def forward(self, h):               # h: [B, d]
        out = self.net(h)               # [B, 2]
        mu, logv = out[:, 0], out[:, 1]
        logv = torch.clamp(logv, -10, 10)
        return mu, logv

# ---------------- new: LongEncoder branch (adds out_dim) --------------------

class LongEncoder(nn.Module):
    """
    Encodes long-term daily climate into a single [B, out_dim]
    embedding via Transformer + pooling.
    """
    def __init__(self,
                 long_input_dim: int,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 128,
                 dropout: float = 0.1,
                 pool: str = "multihead",   # or "linear"
                 num_queries: int = 2,
                 out_dim: int = None):
        super().__init__()
        self.out_dim = out_dim if out_dim is not None else d_model

        self.long_proj = nn.Linear(long_input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=enc_layer, num_layers=num_layers
        )
        self.pos_enc = SinusoidalPositionalEncoding(d_model)

        #if pool == "multihead":
        #    self.pool = MultiheadAttnPool(d_model, nhead,
        #                                  num_queries, dropout)  # -> [B, d_model]
        #else:
        self.pool = LinearAttnPool(d_model)                  # -> [B, d_model]

        # Optional shrink to out_dim
        self.out_proj = (
            nn.Identity() if self.out_dim == d_model
            else nn.Sequential(nn.LayerNorm(d_model),
                               nn.Linear(d_model, self.out_dim))
        )

    def forward(self, long_hist):       # [B, Tl, Din_long]
        x = self.long_proj(long_hist)   # [B, Tl, d]
        #x = self.pos_enc(x)             # [B, Tl, d]
        x = self.encoder(x)             # [B, Tl, d]
        z_long = self.pool(x)           # [B, d]
        z_long = self.out_proj(z_long)  # [B, out_dim]
        return z_long


# ---------------- updated main model (adds long_* overrides) ----------------

class LFMCTransformer(nn.Module):
    """
    short_history:  [B, Ts, Din_short]
    long_history:   [B, Tl, Din_long]
    static_features:[B, 1, D_static]
    1) encode long_history -> z_long [B, long_out_dim]
    2) s_aug = concat(static, z_long)
    3) static token + FiLM from s_aug condition short_history
    """
    def __init__(self,
                 short_input_dim: int,
                 static_input_dim: int,
                 long_input_dim: int,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_layers: int = 3,
                 dim_feedforward: int = 128,
                 dropout: float = 0.1,
                 num_queries: int = 2,
                 pool_long: str = "multihead",
                 long_num_layers: int = 2,
                 long_d_model: int = None,
                 long_nhead: int = None,
                 long_dim_feedforward: int = None,
                 long_out_dim: int = None,
                 num_task_weights: int = 3
        ):
        super().__init__()

        # Resolve long-branch hyperparameters (fallback to main)
        _long_d_model = (
            long_d_model if long_d_model is not None else d_model
        )
        _long_nhead = (
            long_nhead if long_nhead is not None else nhead
        )
        _long_dim_feedforward = (
            long_dim_feedforward if long_dim_feedforward is not None else dim_feedforward
        )
        _long_out_dim = (
            long_out_dim if long_out_dim is not None else _long_d_model
        )

        # for gradnorm
        self.task_weights = nn.Parameter(
            torch.ones(
                num_task_weights,
                dtype=torch.float32
            )
        )

        # ----- long branch
        self.long_enc = LongEncoder(
            long_input_dim=long_input_dim,
            d_model=_long_d_model,
            nhead=_long_nhead,
            num_layers=long_num_layers,
            dim_feedforward=_long_dim_feedforward,
            dropout=dropout,
            pool=pool_long,
            num_queries=num_queries,
            out_dim=_long_out_dim
        )

        # ----- short branch + shared pieces
        self.short_proj = nn.Linear(short_input_dim, d_model)

        # static gets augmented with z_long → sizes depend on _long_out_dim
        self.static_aug_dim = static_input_dim + _long_out_dim
        self.static_proj = nn.Linear(self.static_aug_dim, d_model)

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
            in_dim=self.static_aug_dim, d_model=d_model, hidden=64
        )

        self.pooler = LinearAttnPool(d_model)

        #self.pooler = MultiheadAttnPool(
        #    d_model=d_model, nhead=nhead,
        #    num_queries=num_queries, dropout=dropout
        #)

        self.head_insitu = GaussianHead(d_model, dropout)
        self.head_vv     = GaussianHead(d_model, dropout)
        self.head_vh     = GaussianHead(d_model, dropout)

    def forward(self,
                short_history,      # [B, Ts, Din_short]
                long_history,       # [B, Tl, Din_long]
                static_features):   # [B, 1, D_static]

        # 1) long → z_long
        z_long = self.long_enc(long_history)       # [B, long_out_dim]

        # 2) augment static with z_long
        s = static_features.squeeze(1)             # [B, D_static]
        s_aug = torch.cat([s, z_long], dim=-1)     # [B, D_static+long_out_dim]
        s_tok = self.static_proj(s_aug)            # [B, d]
        s_tok = s_tok.unsqueeze(1)                 # [B, 1, d]

        # 3) FiLM-condition short seq with augmented static
        x_tok = self.short_proj(short_history)     # [B, Ts, d]
        #gamma, beta = self.film_cond(s_aug)        # [B, d], [B, d]
        #gamma = 0.5 * torch.tanh(gamma)
        #x_tok = (1 + gamma).unsqueeze(1) * x_tok + beta.unsqueeze(1)

        # 4) pack [static_token | short_seq] and encode
        x_seq = torch.cat([s_tok, x_tok], dim=1)   # [B, Ts+1, d]
        #x_seq = self.pos_enc(x_seq)
        x_enc = self.encoder(x_seq)                # [B, Ts+1, d]

        # 5) pool short tokens + fuse with static token
        #h_met  = self.pooler(x_enc[:, 1:, :])      # [B, d]
        #h_stat = x_enc[:, 0, :]                    # [B, d]
        #h_allpooled = self.pooler(x_enc)
        #h_add = h_met + h_stat          # [B, d]
        #h_cat = torch.cat([h_met, h_stat], dim=-1)  # [B, 2d]
        #print(x_seq.shape, x_enc.shape)
        #print(h_met.shape, h_stat.shape, h_add.shape, h_cat.shape, h_allpooled.shape)
        #sys.exit()
        h = self.pooler(x_enc)                     # [B, d]


        # 6) heads
        mu_i, logv_i = self.head_insitu(h)
        mu_vv, logv_vv = self.head_vv(h)
        mu_vh, logv_vh = self.head_vh(h)

        return {
            "mu_insitu": mu_i,
            "log_var_insitu": logv_i,
            "mu_vv": mu_vv,
            "log_var_vv": logv_vv,
            "mu_vh": mu_vh,
            "log_var_vh": logv_vh,
            "z_long": z_long  # [B, long_out_dim]
        }
