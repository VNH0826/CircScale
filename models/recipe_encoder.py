# models/recipe_encoder.py
import torch
import torch.nn as nn
import math


class RecipeEncoder(nn.Module):
    def __init__(self, num_heuristics, embedding_dim):
        super(RecipeEncoder, self).__init__()
        self.embedding = nn.Embedding(num_heuristics, embedding_dim)

    def forward(self, recipe_indices):
        return self.embedding(recipe_indices)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # sin(pos/10000^(2i/d_model))
        pe[:, 1::2] = torch.cos(position * div_term)  # cos(pos/10000^(2i/d_model))

        pe = pe.unsqueeze(0)  # 添加batch维度：[1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)