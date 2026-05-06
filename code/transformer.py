import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim=1024, hidden_dims=512, seq_len=8, nhead=8, num_layers=2, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dims = hidden_dims

        self.input_linear = nn.Linear(input_dim, seq_len * hidden_dims)

        self.pos_encoder = PositionalEncoding(hidden_dims, max_len=seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dims,
            nhead=min(nhead, hidden_dims),
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(hidden_dims, 1)
    def forward(self, x):
        """
        x: [batch_size, 1, input_dim]
        """
        batch_size = x.size(0) #B
        x = x.squeeze(1) 
        #BATCH,1024                   # [batch, input_dim]
        x = self.input_linear(x) 
        #BATCH,4096           # [batch, seq_len*hidden_dims]
        x = x.view(batch_size, self.seq_len, self.hidden_dims) 
        #BATCH,8,512
        #  # [batch, seq_len, hidden_dims]
        x = self.pos_encoder(x)             # position
        x = x.transpose(0, 1)               # [seq_len, batch, hidden_dims]
        x = self.transformer_encoder(x)     # [seq_len, batch, hidden_dims]
        x = x.transpose(0, 1)  
        #BATCH,8,512             # [batch, seq_len, hidden_dims]
        return x




class RegMLP5(nn.Module):
    def __init__(self, input_dim=1024, hidden_dims=[512, 256], out_dim=5):
        super(RegMLP5, self).__init__()
        layers = []
        in_features = input_dim
        for out_features in hidden_dims:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.Dropout(0.3))
            in_features = out_features
        layers.append(nn.Linear(in_features, out_dim))  
        self.mlp = nn.Sequential(*layers)
        self.is_regression = True

    def forward(self, x):
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        logits = self.mlp(x)   # shape (B,out_dim)
        return logits


class TransformerRegMLP5(nn.Module):
    def __init__(self, num_layers, input_dim=1024, hidden_dims=512, mlp_hidden_dims=[512, 256],
                 seq_len=8, nhead=8, dim_feedforward=1024, dropout=0.1, out_dim=6):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dims = hidden_dims
        self.transformer_encoder = TransformerEncoder(
            input_dim, hidden_dims, seq_len, nhead, num_layers, dim_feedforward, dropout
        )
        flat_dim = seq_len * hidden_dims
        self.head = RegMLP5(input_dim=flat_dim, hidden_dims=mlp_hidden_dims, out_dim=out_dim)

    def forward(self, x):
        x = self.transformer_encoder(x)         # [B, seq_len, hidden_dims]
        x = x.reshape(x.size(0), -1)            # [B, seq_len*hidden_dims]
        x = self.head(x)                        # [B,out_dim]
        return x