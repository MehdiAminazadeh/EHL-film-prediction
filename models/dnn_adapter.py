# file: models/dnn_adapter.py
from __future__ import annotations
from typing import Sequence, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin

from .registry import register_model
from .base import Capabilities

class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float = 0.0, batch_norm: bool = True):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.squeeze(-1)

@register_model(
    "dnn",
    role="estimator",  # sklearn-like so pipelines & optuna_search can use it
    tags=["torch", "mlp", "gpu"],
    metadata={"notes": "PyTorch MLP wrapped as sklearn regressor with optional early stopping"},
)
class DNNAdapter(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        # model
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.0,
        batch_norm: bool = True,
        # training
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        epochs: int = 200,
        batch_size: int = 32,
        # early stopping
        patience: Optional[int] = None,     # used if early_stopping_rounds is None
        min_delta: float = 1e-4,
        # misc
        device: Optional[str] = None,
        verbose: int = 0,
        random_state: Optional[int] = 42,
    ):
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = float(dropout)
        self.batch_norm = bool(batch_norm)

        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)

        self.patience = patience
        self.min_delta = float(min_delta)

        self.device = device
        self.verbose = int(verbose)
        self.random_state = random_state

        self.model_ = None
        self.input_dim_ = None
        self.best_iteration_ = None

    # sklearn API
    def get_params(self, deep: bool = True) -> dict:
        return {
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "batch_norm": self.batch_norm,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "device": self.device,
            "verbose": self.verbose,
            "random_state": self.random_state,
        }

    def set_params(self, **params: Any):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _select_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build(self, input_dim: int):
        self.input_dim_ = input_dim
        self.model_ = _MLP(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            batch_norm=self.batch_norm,
        ).to(self._device)

    def _make_loader(self, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self._device)
        y_t = torch.as_tensor(y.reshape(-1, 1), dtype=torch.float32, device=self._device)
        ds = torch.utils.data.TensorDataset(X_t, y_t)
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    def fit(self, X: np.ndarray, y: np.ndarray, *, eval_set: Optional[list[Tuple[np.ndarray, np.ndarray]]] = None,
            early_stopping_rounds: Optional[int] = None, verbose: Optional[bool] = None):
        rng = np.random.default_rng(self.random_state)
        self._device = self._select_device()
        self._build(X.shape[1])

        rounds = early_stopping_rounds if early_stopping_rounds is not None else (self.patience or 0)
        use_es = rounds and eval_set and len(eval_set) > 0
        min_delta = self.min_delta

        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_fn = torch.nn.MSELoss()

        train_loader = self._make_loader(X, y, self.batch_size, shuffle=True)

        if use_es:
            Xv, yv = eval_set[0]
            val_loader = self._make_loader(Xv, yv, self.batch_size, shuffle=False)
        else:
            val_loader = None

        best_val = float("inf")
        best_state = None
        wait = 0
        self.best_iteration_ = None

        for epoch in range(self.epochs):
            # train
            self.model_.train()
            for xb, yb in train_loader:
                opt.zero_grad()
                pred = self.model_(xb)
                if pred.ndim == 1:
                    pred = pred.unsqueeze(1)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 5.0)
                opt.step()

            # validate for ES
            if use_es and val_loader is not None:
                self.model_.eval()
                vloss = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        pred = self.model_(xb)
                        if pred.ndim == 1:
                            pred = pred.unsqueeze(1)
                        vloss += loss_fn(pred, yb).item()

                if vloss < best_val - float(min_delta):
                    best_val = vloss
                    best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
                    wait = 0
                    self.best_iteration_ = epoch + 1
                else:
                    wait += 1
                    if wait >= rounds:
                        break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Estimator is not fitted.")
        self.model_.eval()
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self._device)
        out = self.model_(X_t)
        if out.ndim == 1:
            out = out.unsqueeze(1)
        return out.detach().cpu().numpy().reshape(-1)

    @classmethod
    def capabilities(cls) -> Capabilities:
        return Capabilities(
            backend="torch",
            supports_early_stopping=True,
            supports_gpu=True,
            supports_feature_importance=False,
            handles_categoricals=False,
            deterministic=False,
            notes="PyTorch MLP via sklearn adapter",
        )
