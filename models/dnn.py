# file: models/dnn.py
from __future__ import annotations
from typing import Sequence
import torch
import torch.nn as nn
from .registry import register_model

@register_model(
    "dnn",
    role="backbone",
    tags=["torch", "mlp", "gpu"],
    metadata={"notes": "PyTorch MLP backbone; training handled by train_dnn strategy"},
)
class DNNBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float = 0.0, batch_norm: bool = True):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
