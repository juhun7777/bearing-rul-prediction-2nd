"""
Transformer 기반 모델들
"""

import math
import torch
import torch.nn as nn

class PositionalEncodingSimple(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pos = torch.arange(0,max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0,d_model,2, dtype=torch.float32) * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.pe = pe.unsqueeze(1)
    def forward(self, x):
        return x + self.pe[:x.size(0)]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
class TimeSeriesTransformer(nn.Module):
    def __init__(self, src_len, d_model=64, nhead=8, num_layers=3, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_emb = nn.Linear(2, d_model)
        self.pos_enc = PositionalEncodingSimple(d_model, max_len=src_len)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
        self.linear = nn.Linear(d_model, 1)
        
    def forward(self, src):
        x = src.unsqueeze(2)
        
        batch_size, seq_len = src.shape
        rates = torch.zeros((batch_size, seq_len, 1), device=src.device, dtype=torch.float32)
        rates[:, 1:, 0] = src[:, 1:] - src[:, :-1]
        
        x = torch.cat([x, rates], dim=2)
        x = self.input_emb(x)
        x = x.permute(1, 0, 2)
        x = self.pos_enc(x)
        out = self.transformer(x)
        last = out[-1, :, :]
        return self.linear(last).squeeze(1)

class TransformerRULModel(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=8, num_layers=3, dropout=0.1, seq_len=50):
        super(TransformerRULModel, self).__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1), nn.ReLU()
        )
        
    def forward(self, x):
        x = self.input_projection(x) * math.sqrt(self.d_model)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        transformer_out = self.transformer_encoder(x)
        output = self.output_layer(transformer_out[:, -1, :])
        return output
