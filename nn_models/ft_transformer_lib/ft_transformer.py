import math
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ─────────────────────────────────────────────────────────
# Tokenizers
# ─────────────────────────────────────────────────────────

class NumericalFeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, d_token: int, bias: bool, initialization: str):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token)) if bias else None
        d_sqrt_inv = 1 / math.sqrt(d_token)
        torch.nn.init.uniform_(self.weight, -d_sqrt_inv, d_sqrt_inv)
        if self.bias is not None:
            torch.nn.init.uniform_(self.bias, -d_sqrt_inv, d_sqrt_inv)

    def forward(self, x: Tensor) -> Tensor:
        x = self.weight[None] * x[..., None]
        if self.bias is not None:
            x = x + self.bias[None]
        return x

class FeatureTokenizer(nn.Module):
    def __init__(self, n_num_features: int, d_token: int):
        super().__init__()
        self.num_tokenizer = NumericalFeatureTokenizer(n_num_features, d_token, True, 'uniform')

    def forward(self, x_num: Optional[Tensor]) -> Tensor:
        return self.num_tokenizer(x_num)

# ─────────────────────────────────────────────────────────
# Transformer Blocks
# ─────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, d_token: int, n_heads: int, dropout: float, ffn_d_hidden: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_token, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, ffn_d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_d_hidden, d_token),
        )
        self.norm1 = nn.LayerNorm(d_token)
        self.norm2 = nn.LayerNorm(d_token)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

class FTTransformer(nn.Module):
    def __init__(self, n_num_features: int, d_token: int, n_blocks: int, n_heads: int,
                 attention_dropout: float, ffn_d_hidden: int, ffn_dropout: float, d_out: int):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_num_features, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        self.transformer = nn.Sequential(*[
            TransformerBlock(d_token, n_heads, attention_dropout, ffn_d_hidden)
            for _ in range(n_blocks)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Linear(d_token, d_out)
        )

    def forward(self, x_num: Tensor) -> Tensor:
        x = self.tokenizer(x_num)
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([x, cls_token], dim=1)
        x = self.transformer(x)
        return self.head(x[:, -1])

def build_ft_transformer(input_dim: int, output_dim: int, config: Dict[str, Any]) -> FTTransformer:
    return FTTransformer(
        n_num_features=input_dim,
        d_token=config.get("d_token", 64),
        n_blocks=config.get("n_blocks", 3),
        n_heads=config.get("n_heads", 4),
        attention_dropout=config.get("attention_dropout", 0.1),
        ffn_d_hidden=config.get("ffn_d_hidden", 128),
        ffn_dropout=config.get("ffn_dropout", 0.1),
        d_out=output_dim,
    )