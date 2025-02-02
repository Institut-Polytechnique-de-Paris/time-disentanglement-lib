# -*-Encoding: utf-8 -*-
"""
Authors:
    Khalid OUBLAL, PhD IPP/ OneTech
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding_pixel(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding_pixel, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, pixel='slot_pixel'):
        super(PositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.fc = nn.Linear(1, d_model)
    def forward(self, x):
        input_reshaped = x.view(x.size(0), -1, 1)
        output_tensor = self.fc(input_reshaped)
        output_tensor = output_tensor.view(output_tensor.size(0), x.size(1), self.d_model)
        return output_tensor


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, pixel='slot_pixel', dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding_pixel(d_model=d_model)
        self.temporal_embedding = PositionalEmbedding(d_model=d_model, pixel=pixel)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        x =  self.temporal_embedding(x) 
        return self.dropout(x)
