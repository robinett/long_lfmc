import torch
import torch.nn as nn
import torch.nn.functional as F


def build_tcn_dilations(max_dilation):
    dilations = []
    dilation = 1
    while dilation <= max_dilation:
        dilations.append(dilation)
        dilation *= 2
    return dilations


class LinearHead(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.fc(x)


class BranchMLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, x):
        if self.left_padding > 0:
            x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class ResidualTCNBlock(nn.Module):
    def __init__(self, d_model, kernel_size, dilation, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.conv1 = CausalConv1d(d_model, d_model, kernel_size, dilation)
        self.norm2 = nn.LayerNorm(d_model)
        self.conv2 = CausalConv1d(d_model, d_model, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        y = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        y = self.conv1(y)
        y = F.gelu(y)
        y = self.dropout(y)

        y = self.norm2(y.transpose(1, 2)).transpose(1, 2)
        y = self.conv2(y)
        y = F.gelu(y)
        y = self.dropout(y)

        return residual + y


class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x):
        attn_logits = self.score(x).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=1)
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)
        return pooled


class WeatherTCNEncoder(nn.Module):
    def __init__(self, input_dim, d_weather, kernel_size, dropout, max_dilation):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_weather)
        self.dilations = build_tcn_dilations(max_dilation)
        self.blocks = nn.ModuleList(
            [
                ResidualTCNBlock(
                    d_model=d_weather,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
                for dilation in self.dilations
            ]
        )
        self.pool = AttentionPooling(d_weather)

    def forward(self, x):
        x = self.input_proj(x)
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)
        return self.pool(x)


class StaticMLPEncoder(nn.Module):
    def __init__(self, input_dim, d_static, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2 * d_static),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_static, d_static),
        )

    def forward(self, x):
        return self.net(x)


class SequenceTransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, dropout, num_layers=2):
        super().__init__()
        nhead = max(2, d_model // 32)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

    def forward(self, x):
        x = self.input_proj(x)
        cls = self.cls_token.expand(x.size(0), 1, x.size(-1))
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)
        return x[:, 0, :]


class FusionTransformer(nn.Module):
    def __init__(self, d_common, dropout, num_layers=2):
        super().__init__()
        nhead = max(2, d_common // 32)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_common) * 0.02)
        self.modality_embeddings = nn.Parameter(torch.randn(1, 4, d_common) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_common,
            nhead=nhead,
            dim_feedforward=2 * d_common,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_common),
        )

    def forward(self, weather_token, static_token, modis_token):
        cls = self.cls_token.expand(weather_token.size(0), 1, weather_token.size(-1))
        tokens = torch.stack([weather_token, static_token, modis_token], dim=1)
        tokens = torch.cat([tokens, cls], dim=1)
        tokens = tokens + self.modality_embeddings
        fused = self.encoder(tokens)
        return fused[:, -1, :]


class LFMCMultiSourceFusion(nn.Module):
    def __init__(
        self,
        short_input_dim,
        static_input_dim,
        long_input_dim,
        weather_d_model=128,
        modis_d_model=128,
        static_d_model=64,
        common_d_model=128,
        weather_kernel_size=5,
        weather_max_dilation=32,
        shared_latent_dim=0,
        lfmc_private_dim=0,
        sar_private_dim=0,
        dropout=0.15,
        num_task_weights=3,
    ):
        super().__init__()
        self.task_weights = nn.Parameter(torch.ones(num_task_weights, dtype=torch.float32))
        self.shared_latent_dim = int(shared_latent_dim)
        self.lfmc_private_dim = int(lfmc_private_dim)
        self.sar_private_dim = int(sar_private_dim)
        self.use_task_private_latents = (
            self.shared_latent_dim > 0
            and self.lfmc_private_dim > 0
            and self.sar_private_dim > 0
        )

        self.weather_encoder = WeatherTCNEncoder(
            input_dim=long_input_dim,
            d_weather=weather_d_model,
            kernel_size=weather_kernel_size,
            dropout=dropout,
            max_dilation=weather_max_dilation,
        )
        self.static_encoder = StaticMLPEncoder(
            input_dim=static_input_dim,
            d_static=static_d_model,
            dropout=dropout,
        )
        self.modis_encoder = SequenceTransformerEncoder(
            input_dim=short_input_dim,
            d_model=modis_d_model,
            dropout=dropout,
            num_layers=2,
        )

        self.weather_to_common = nn.Linear(weather_d_model, common_d_model)
        self.static_to_common = nn.Linear(static_d_model, common_d_model)
        self.modis_to_common = nn.Linear(modis_d_model, common_d_model)

        self.fusion = FusionTransformer(
            d_common=common_d_model,
            dropout=dropout,
            num_layers=2,
        )

        if self.use_task_private_latents:
            self.shared_branch = BranchMLP(
                common_d_model,
                self.shared_latent_dim,
                self.shared_latent_dim,
                dropout,
            )
            self.lfmc_private_branch = BranchMLP(
                common_d_model,
                self.lfmc_private_dim,
                self.lfmc_private_dim,
                dropout,
            )
            self.vv_private_branch = BranchMLP(
                common_d_model,
                self.sar_private_dim,
                self.sar_private_dim,
                dropout,
            )
            self.vh_private_branch = BranchMLP(
                common_d_model,
                self.sar_private_dim,
                self.sar_private_dim,
                dropout,
            )
            self.head_insitu = LinearHead(self.shared_latent_dim + self.lfmc_private_dim, 1)
            self.head_vv = LinearHead(self.shared_latent_dim + self.sar_private_dim, 1)
            self.head_vh = LinearHead(self.shared_latent_dim + self.sar_private_dim, 1)
        else:
            self.head_insitu = LinearHead(common_d_model, 1)
            self.head_vv = LinearHead(common_d_model, 1)
            self.head_vh = LinearHead(common_d_model, 1)

    def forward(self, short_history, long_history, static_features):
        static_flat = static_features.squeeze(1)

        weather_token = self.weather_to_common(self.weather_encoder(long_history))
        static_token = self.static_to_common(self.static_encoder(static_flat))
        modis_token = self.modis_to_common(self.modis_encoder(short_history))

        fused_cls = self.fusion(weather_token, static_token, modis_token)

        if self.use_task_private_latents:
            shared_latent = self.shared_branch(fused_cls)
            lfmc_private_latent = self.lfmc_private_branch(fused_cls)
            vv_private_latent = self.vv_private_branch(fused_cls)
            vh_private_latent = self.vh_private_branch(fused_cls)

            lfmc_head_input = torch.cat([shared_latent, lfmc_private_latent], dim=-1)
            vv_head_input = torch.cat([shared_latent, vv_private_latent], dim=-1)
            vh_head_input = torch.cat([shared_latent, vh_private_latent], dim=-1)
        else:
            shared_latent = None
            lfmc_private_latent = None
            vv_private_latent = None
            vh_private_latent = None
            lfmc_head_input = fused_cls
            vv_head_input = fused_cls
            vh_head_input = fused_cls

        mu_i = self.head_insitu(lfmc_head_input).squeeze(-1)
        mu_vv = self.head_vv(vv_head_input).squeeze(-1)
        mu_vh = self.head_vh(vh_head_input).squeeze(-1)
        logv_i = torch.zeros_like(mu_i)
        logv_vv = torch.zeros_like(mu_vv)
        logv_vh = torch.zeros_like(mu_vh)

        return {
            "mu_insitu": mu_i,
            "log_var_insitu": logv_i,
            "mu_vv": mu_vv,
            "log_var_vv": logv_vv,
            "mu_vh": mu_vh,
            "log_var_vh": logv_vh,
            "weather_token": weather_token,
            "static_token": static_token,
            "modis_token": modis_token,
            "cls_token": fused_cls,
            "shared_latent": shared_latent,
            "lfmc_private_latent": lfmc_private_latent,
            "vv_private_latent": vv_private_latent,
            "vh_private_latent": vh_private_latent,
        }
