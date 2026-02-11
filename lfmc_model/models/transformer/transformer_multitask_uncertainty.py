import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys

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

class LinearHead(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.fc(x)


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
        self.cls_long = nn.Parameter(
            torch.randn(1, 1, d_model)
        )
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
        # Optional shrink to out_dim
        self.out_proj = (
            nn.Identity() if self.out_dim == d_model
            else nn.Sequential(nn.LayerNorm(d_model),
                               nn.Linear(d_model, self.out_dim))
        )

    def forward(self, long_hist):       # [B, Tl, Din_long]
        # project
        x_proj = self.long_proj(long_hist)   # [B, Tl, d]
        # add learnable cls token
        cls_token = self.cls_long.expand(x_proj.size(0), 1, x_proj.size(2))  # [B, 1, d]
        x_cls = torch.cat([cls_token, x_proj], dim=1)  # [B, Tl+1, d]
        # encode
        x_enc = self.encoder(x_cls)             # [B, Tl, d]
        # select cls token output and shrink to out_dim
        z_cls = x_enc[:, 0, :]                  # [B, d]
        z_long = self.out_proj(z_cls)  # [B, out_dim]
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
                 #long_input_dim: int,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_layers: int = 3,
                 dim_feedforward: int = 128,
                 dropout: float = 0.1,
                 num_queries: int = 2,
                 #pool_long: str = "multihead",
                 #long_num_layers: int = 2,
                 #long_d_model: int = None,
                 #long_nhead: int = None,
                 #long_dim_feedforward: int = None,
                 #long_out_dim: int = None,
                 num_task_weights: int = 3
        ):
        super().__init__()

        ## Resolve long-branch hyperparameters (fallback to main)
        #_long_d_model = (
        #    long_d_model if long_d_model is not None else d_model
        #)
        #_long_nhead = (
        #    long_nhead if long_nhead is not None else nhead
        #)
        #_long_dim_feedforward = (
        #    long_dim_feedforward if long_dim_feedforward is not None else dim_feedforward
        #)
        #_long_out_dim = (
        #    long_out_dim if long_out_dim is not None else _long_d_model
        #)

        # for gradnorm
        self.task_weights = nn.Parameter(
            torch.ones(
                num_task_weights,
                dtype=torch.float32
            )
        )

        # ----- long branch
        #self.long_enc = LongEncoder(
        #    long_input_dim=long_input_dim,
        #    d_model=_long_d_model,
        #    nhead=_long_nhead,
        #    num_layers=long_num_layers,
        #    dim_feedforward=_long_dim_feedforward,
        #    dropout=dropout,
        #    pool=pool_long,
        #    num_queries=num_queries,
        #    out_dim=_long_out_dim
        #)

        # ----- short branch + shared pieces
        self.short_proj = nn.Linear(short_input_dim, d_model)

        # static gets augmented with z_long → sizes depend on _long_out_dim
        self.static_aug_dim = static_input_dim #+ _long_out_dim
        self.static_proj = nn.Linear(self.static_aug_dim, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=enc_layer, num_layers=num_layers
        )
        #self.pos_enc  = SinusoidalPositionalEncoding(d_model)

        #self.film_cond = FiLMConditioner(
        #    in_dim=self.static_aug_dim, d_model=d_model, hidden=64
        #)

        #self.pooler = LinearAttnPool(d_model)

        #self.pooler = MultiheadAttnPool(
        #    d_model=d_model, nhead=nhead,
        #    num_queries=num_queries, dropout=dropout
        #)

        self.head_insitu = GaussianHead(d_model, dropout)
        self.head_vv     = GaussianHead(d_model, dropout)
        self.head_vh     = GaussianHead(d_model, dropout)

        #self.head_insitu = LinearHead(d_model, 1)
        #self.head_vv     = LinearHead(d_model, 1)
        #self.head_vh     = LinearHead(d_model, 1)

    def forward(self,
                short_history,      # [B, Ts, Din_short]
                #long_history,       # [B, Tl, Din_long]
                static_features):   # [B, 1, D_static]

        ## 1) encode long history
        #z_long = self.long_enc(long_history)       # [B, long_out_dim]

        # 2) augment static with z_long
        s_aug = static_features.squeeze(1)             # [B, D_static]
        #s_aug = torch.cat([s, z_long], dim=-1)     # [B, D_static+long_out_dim]
        s_tok = self.static_proj(s_aug)            # [B, d]
        s_tok = s_tok.unsqueeze(1)                 # [B, 1, d]
        
        # 3) project short history
        x_tok = self.short_proj(short_history)     # [B, Ts, d]

        # 4) concatenate long + static to short and encode
        x_seq = torch.cat([s_tok, x_tok], dim=1)   # [B, Ts+1, d]
        x_enc = self.encoder(x_seq)                # [B, Ts+1, d]

        # 5) use the static+long token as the output representation that we pass to heads.
        #    this is basically saying that we want the static+long information to be updated
        #    via the short history information
        h = x_enc[:, 0, :]                         # [B, d]

        # 6) heads
        mu_i, logv_i = self.head_insitu(h)
        mu_vv, logv_vv = self.head_vv(h)
        mu_vh, logv_vh = self.head_vh(h)
        #mu_i = self.head_insitu(h).squeeze(-1)
        #mu_vv = self.head_vv(h).squeeze(-1)
        #mu_vh = self.head_vh(h).squeeze(-1)
        #logv_i = torch.zeros_like(mu_i)
        #logv_vv = torch.zeros_like(mu_vv)
        #logv_vh = torch.zeros_like(mu_vh)

        return {
            "mu_insitu": mu_i,
            "log_var_insitu": logv_i,
            "mu_vv": mu_vv,
            "log_var_vv": logv_vv,
            "mu_vh": mu_vh,
            "log_var_vh": logv_vh,
            #"z_long": z_long  # [B, long_out_dim]
        }
