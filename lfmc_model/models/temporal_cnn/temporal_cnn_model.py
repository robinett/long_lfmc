import torch.nn as nn
import torch

class TemporalCNN(nn.Module):
   def __init__(
       self,
       input_dim,
       d_model=64,
       num_layers=4,
       kernel_size=3,
       dilation_base=2,
       dropout=0.1,
       output_dim=1
   ):
       super().__init__()
       self.input_proj = nn.Conv1d(
           input_dim,d_model,kernel_size=1
       )
       layers = []
       for i in range(num_layers):
           dilation = dilation_base ** i
           layers.extend([
               nn.Conv1d(
                   in_channels=d_model,
                   out_channels=d_model,
                   kernel_size=kernel_size,
                   padding=dilation,
                   dilation=dilation
               ),
               nn.ReLU(),
               nn.BatchNorm1d(d_model),
               nn.Dropout(dropout)
           ])
       self.conv_layers = nn.Sequential(*layers)
       self.head = nn.Sequential(
           nn.Linear(d_model, d_model // 2),
           nn.ReLU(),
           nn.Linear(d_model // 2, 1)
       )

   def forward(self,x):
      # x: [batch_size, seq_len, input_dim]
      x = x.transpose(1, 2)
      x = self.input_proj(x)
      x = self.conv_layers(x)
      x_last = x[:, :, -1]  # take the last output
      out = self.head(x_last)
      return out
