import torch
import torch.nn as nn
import torch.nn.functional as F

class LFMCRnn(nn.Module):
    def __init__(
        self,
        short_input_dim,
        static_input_dim,
        hidden_size = 10,
        num_layers = 4,
        dropout = 0.05,
        num_task_weights = 2
    ):
        super().__init__()
        self.short_input_dim = short_input_dim
        self.static_input_dim = static_input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        rnn_in = short_input_dim + static_input_dim

        self.task_weights = nn.Parameter(
            torch.ones(
                num_task_weights,
                dtype=torch.float32
            )
        )

        self.lstm = nn.LSTM(
            input_size = rnn_in,
            hidden_size = hidden_size,
            num_layers = num_layers,
            dropout = dropout,
            batch_first=True
        )
        self.head_insitu = nn.Linear(hidden_size, 1)
        self.head_vv = nn.Linear(hidden_size, 1)
        self.head_vh = nn.Linear(hidden_size, 1)

    def forward(
        self,
        short_history,
        static_features
    ):
        x = short_history
        if static_features.ndim == 3:
            s = static_features.squeeze(1)
        else:
            s = static_features
        B, T, _ = x.shape
        s_rep = s.unsqueeze(1).expand(B,T,-1)
        x_in = torch.cat([x, s_rep], dim=-1)
        out, (hn, cn) = self.lstm(x_in)
        h = out[:,-1,:]

        mu_i = self.head_insitu(h).squeeze(-1)
        mu_vv = self.head_vv(h).squeeze(-1)
        mu_vh = self.head_vh(h).squeeze(-1)
        z_i = torch.zeros_like(mu_i)
        z_vv = torch.zeros_like(mu_vv)
        z_vh = torch.zeros_like(mu_vh)

        return {
            'mu_insitu': mu_i,
            'log_var_insitu': z_i,
            'mu_vv': mu_vv,
            'log_var_vv': z_vv,
            'mu_vh': mu_vh,
            'log_var_vh': z_vh
        }
