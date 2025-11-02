# file: models/node_adapter.py
from __future__ import annotations
from typing import Any, Optional, Tuple, List
import numpy as np
import torch
from torch import nn
from sklearn.base import BaseEstimator, RegressorMixin

from models.registry import register_model

# ---- import your custom libs ----
from nn_models.node_lib.arch import DenseBlock   # adjust import path to your project layout
from nn_models.node_lib.nn_utils import entmax15, sparsemoid


class _NODEBackbone(nn.Module):
    """Your NODE backbone + linear head."""
    def __init__(
        self,
        input_dim: int,
        layer_dim: int,
        num_layers: int,
        depth: int,
        input_dropout: float,
    ):
        super().__init__()
        self.backbone = DenseBlock(
            input_dim=input_dim,
            layer_dim=layer_dim,
            num_layers=num_layers,
            tree_dim=1,
            flatten_output=True,
            input_dropout=input_dropout,
            depth=depth,
            choice_function=entmax15,
            bin_function=sparsemoid,
        )
        self.head = nn.Linear(layer_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


@register_model(
    "node",
    role="estimator",
    tags=["torch", "neural", "tabular"],
    metadata={"family": "NODE", "author": "custom"},
)
class NodeSklearnAdapter(BaseEstimator, RegressorMixin):
    """
    Sklearn-like wrapper around your custom NODE to work with RegKit pipelines.

    Supports:
      • fit(X, y, eval_set=[(Xv, yv)], early_stopping_rounds=k, verbose=0)
      • predict(X)
      • params: layer_dim, num_layers, depth, input_dropout, lr/learning_rate, epochs, batch_size, patience
    """

    def __init__(
        self,
        # model hyperparams
        layer_dim: int = 64,
        num_layers: int = 2,
        depth: int = 8,
        input_dropout: float = 0.0,
        # training hyperparams
        learning_rate: float = 8e-3,
        lr: Optional[float] = None,             # alias; if provided and learning_rate not set explicitly, it will be used
        weight_decay: float = 0.0,
        epochs: int = 200,
        batch_size: int = 256,
        patience: Optional[int] = 20,           # internal ES if eval_set is not provided
        # misc
        device_name: Optional[str] = None,      # "cuda" | "cpu" | None -> auto
        num_workers: int = 0,
        drop_last: bool = False,
        random_state: Optional[int] = 42,
        verbose: int = 0,
        **kwargs: Any,                          # accept/ignore future keys safely
    ):
        # alias support
        if lr is not None and (learning_rate is None or learning_rate == 8e-3):
            learning_rate = float(lr)

        # store config
        self.layer_dim = int(layer_dim)
        self.num_layers = int(num_layers)
        self.depth = int(depth)
        self.input_dropout = float(input_dropout)

        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.patience = None if patience is None else int(patience)

        self.device_name = device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = int(num_workers)
        self.drop_last = bool(drop_last)
        self.random_state = random_state
        self.verbose = int(verbose)

        # runtime
        self.model_: Optional[_NODEBackbone] = None
        self.best_val_: Optional[float] = None
        self.best_state_: Optional[dict] = None
        self.best_iteration_: Optional[int] = None  # for compatibility with ES consumers

    # ---- sklearn API ----

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose: Optional[bool | int] = None,
        **fit_kwargs: Any,
    ):
        # basic I/O
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        n, d = X.shape

        device = torch.device(self.device_name)
        torch.manual_seed(self.random_state or 42)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(self.random_state or 42)

        self.model_ = _NODEBackbone(
            input_dim=d,
            layer_dim=self.layer_dim,
            num_layers=self.num_layers,
            depth=self.depth,
            input_dropout=self.input_dropout,
        ).to(device)

        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.MSELoss()

        Xtr = torch.from_numpy(X).to(device)
        ytr = torch.from_numpy(y).to(device)

        # optional ES via eval_set (matches RegKit ES helper expectations)
        use_es = False
        es_patience = 0
        if eval_set and early_stopping_rounds and early_stopping_rounds > 0:
            Xv_np, yv_np = eval_set[0]
            Xv = torch.from_numpy(np.asarray(Xv_np, dtype=np.float32)).to(device)
            yv = torch.from_numpy(np.asarray(yv_np, dtype=np.float32).reshape(-1, 1)).to(device)
            use_es = True
            es_patience = int(early_stopping_rounds)

        # Simple mini-batch training
        bs = min(self.batch_size, n) if self.batch_size > 0 else n

        best_val = float("inf")
        wait = 0
        best_epoch = None

        for epoch in range(int(self.epochs)):
            self.model_.train()
            epoch_loss = 0.0

            # manual batching to avoid DataLoader dependency here
            for start in range(0, n, bs):
                end = min(start + bs, n)
                xb = Xtr[start:end]
                yb = ytr[start:end]
                optimizer.zero_grad()
                preds = self.model_(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # validation/ES
            if use_es:
                self.model_.eval()
                with torch.no_grad():
                    val_preds = self.model_(Xv)
                    val_loss = criterion(val_preds, yv).item()

                improved = (best_val - val_loss) > 1e-6
                if improved:
                    best_val = val_loss
                    wait = 0
                    best_epoch = epoch
                    self.best_state_ = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
                else:
                    wait += 1
                    if wait >= es_patience:
                        break

        # load best state if we used ES
        if use_es and self.best_state_ is not None:
            self.model_.load_state_dict(self.best_state_)
            self.best_iteration_ = int(best_epoch) if best_epoch is not None else None
            self.best_val_ = float(best_val)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model not fitted yet.")
        device = torch.device(self.device_name)
        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            xb = torch.from_numpy(X).to(device)
            preds = self.model_(xb).detach().cpu().numpy().reshape(-1)
        return preds

    # provide a LightGBM/XGB-like hook so our Optuna helper can read best iteration
    @property
    def best_iteration(self) -> Optional[int]:
        return self.best_iteration_
