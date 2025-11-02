# file: models/ft_transformer_adapter.py
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import torch
from torch import nn
from sklearn.base import BaseEstimator, RegressorMixin

from models.registry import register_model

# Adjust this import to your actual builder path
from nn_models.ft_transformer_lib.ft_transformer import build_ft_transformer


@register_model(
    "ft_transformer",
    role="estimator",
    tags=["torch", "transformer", "tabular"],
    metadata={"family": "FT-Transformer", "author": "custom"},
)
class FTTransformerSklearnAdapter(BaseEstimator, RegressorMixin):
    """
    Sklearn-like wrapper around custom FT-Transformer.
    Supports: fit(X, y, eval_set=[(Xv,yv)], early_stopping_rounds=k), predict(X)
    """

    def __init__(self, **kwargs: Any):
        # ----- hyperparams with defaults -----
        # model
        self.d_token            = int(kwargs.get("d_token",            64))
        self.n_blocks           = int(kwargs.get("n_blocks",           4))
        self.n_heads            = int(kwargs.get("n_heads",            8))
        self.attention_dropout  = float(kwargs.get("attention_dropout", 0.0))
        self.ff_dropout         = float(kwargs.get("ff_dropout",        0.0))
        self.residual_dropout   = float(kwargs.get("residual_dropout",  0.0))
        self.ffn_d_hidden       = int(kwargs.get("ffn_d_hidden",       256))

        # training
        lr_alias = kwargs.get("lr", None)
        self.learning_rate      = float(lr_alias if lr_alias is not None else kwargs.get("learning_rate", 1e-3))
        self.weight_decay       = float(kwargs.get("weight_decay", 0.0))
        self.epochs             = int(kwargs.get("epochs", 300))
        self.batch_size         = int(kwargs.get("batch_size", 32))
        self.patience           = (None if kwargs.get("patience", None) is None
                                   else int(kwargs.get("patience", 10)))

        # misc
        self.device_name        = kwargs.get("device_name", None) or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose            = int(kwargs.get("verbose", 0))
        self.random_state       = kwargs.get("random_state", 42)

        # runtime
        self.model_: Optional[nn.Module] = None
        self.best_iteration_: Optional[int] = None
        self.best_state_: Optional[Dict[str, torch.Tensor]] = None
        self.best_val_: Optional[float] = None

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
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        n, d = X.shape

        device = torch.device(self.device_name)
        torch.manual_seed(self.random_state or 42)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(self.random_state or 42)

        params = dict(
            d_token=self.d_token,
            n_blocks=self.n_blocks,
            n_heads=self.n_heads,
            attention_dropout=self.attention_dropout,
            ff_dropout=self.ff_dropout,
            residual_dropout=self.residual_dropout,
            ffn_d_hidden=self.ffn_d_hidden,
        )
        self.model_ = build_ft_transformer(input_dim=d, output_dim=1, config=params).to(device)

        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        Xtr = torch.from_numpy(X).to(device)
        ytr = torch.from_numpy(y).to(device)

        use_es = False
        es_patience = 0
        if eval_set and early_stopping_rounds and early_stopping_rounds > 0:
            Xv_np, yv_np = eval_set[0]
            Xv = torch.from_numpy(np.asarray(Xv_np, dtype=np.float32)).to(device)
            yv = torch.from_numpy(np.asarray(yv_np, dtype=np.float32).reshape(-1, 1)).to(device)
            use_es = True
            es_patience = int(early_stopping_rounds)

        bs = min(self.batch_size, n) if self.batch_size > 0 else n
        best_val = float("inf")
        wait = 0
        best_epoch = None

        for epoch in range(int(self.epochs)):
            self.model_.train()
            for start in range(0, n, bs):
                end = min(start + bs, n)
                xb = Xtr[start:end]
                yb = ytr[start:end]
                optimizer.zero_grad()
                preds = self.model_(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

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

    @property
    def best_iteration(self) -> Optional[int]:
        return self.best_iteration_
