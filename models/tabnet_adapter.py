# file: models/tabnet_adapter.py
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, Sequence
import numpy as np
import torch

from sklearn.base import BaseEstimator, RegressorMixin

from nn_models.tabnet_lib.tab_model import TabNetRegressor 
from .registry import register_model
from .base import Capabilities


@register_model(
    "tabnet",
    role="estimator",
    tags=["torch", "tabnet", "gpu"],
    metadata={"notes": "Custom TabNet wrapped as sklearn estimator; compatible with generic Optuna HPO."},
)
class TabNetSklearnAdapter(BaseEstimator, RegressorMixin):
    """
    Sklearn-like adapter for your custom TabNetRegressor.

    Accepted constructor params mirror your script (with sensible defaults):
      - n_d, n_a, n_steps, gamma, lambda_sparse
      - learning_rate (lr), epochs, batch_size, virtual_batch_size
      - patience (ES), num_workers, drop_last, mask_type, device_name
      - optimizer is Adam by default; you can extend via params if needed.

    Early stopping:
      - If fit(..., early_stopping_rounds=k, eval_set=[(Xv, yv)]) is provided,
        we pass patience=k and eval_set through to TabNetRegressor.fit().
      - If patience was set in __init__ and early_stopping_rounds is not provided,
        we use the constructor patience.
    """

    def __init__(
        self,
        # model hyperparams
        n_d: int = 32,
        n_a: int = 32,
        n_steps: int = 5,
        gamma: float = 1.5,
        lambda_sparse: float = 1e-3,
        mask_type: str = "entmax",  # or "sparsemax"
        # training hyperparams
        learning_rate: float = 1e-3,
        epochs: int = 200,
        batch_size: int = 256,
        virtual_batch_size: int = 128,
        patience: Optional[int] = 20,
        num_workers: int = 0,
        drop_last: bool = False,
        device_name: Optional[str] = None,  # "cuda" | "cpu" | None->auto
        verbose: int = 0,
        # optimizer override (optional)
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        # random state
        random_state: Optional[int] = 42,
        cat_emb_dim: Optional[int] = None,   # accepted for config compatibility (not used yet)
        **kwargs: Any,                       # swallow any future/unknown keys
    ):
        # model
        self.n_d = int(n_d)
        self.n_a = int(n_a)
        self.n_steps = int(n_steps)
        self.gamma = float(gamma)
        self.lambda_sparse = float(lambda_sparse)
        self.mask_type = mask_type

        # training
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.virtual_batch_size = int(virtual_batch_size)
        self.patience = None if patience is None else int(patience)
        self.num_workers = int(num_workers)
        self.drop_last = bool(drop_last)
        self.device_name = (
            device_name if device_name is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.verbose = int(verbose)

        self.optimizer = optimizer
        self.weight_decay = float(weight_decay)
        self.random_state = random_state
        
        self.cat_emb_dim = cat_emb_dim

        # internal
        self.model_: Optional[TabNetRegressor] = None
        self.best_iteration_: Optional[int] = None

    # --- sklearn API ---
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "n_d": self.n_d,
            "n_a": self.n_a,
            "n_steps": self.n_steps,
            "gamma": self.gamma,
            "lambda_sparse": self.lambda_sparse,
            "mask_type": self.mask_type,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "virtual_batch_size": self.virtual_batch_size,
            "patience": self.patience,
            "num_workers": self.num_workers,
            "drop_last": self.drop_last,
            "device_name": self.device_name,
            "verbose": self.verbose,
            "optimizer": self.optimizer,
            "weight_decay": self.weight_decay,
            "random_state": self.random_state,
        }

    def set_params(self, **params: Any):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _make_model(self) -> TabNetRegressor:
        # choose optimizer
        opt_fn = torch.optim.Adam if self.optimizer.lower() == "adam" else torch.optim.Adam
        opt_params = dict(lr=self.learning_rate, weight_decay=self.weight_decay)

        return TabNetRegressor(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            lambda_sparse=self.lambda_sparse,
            optimizer_fn=opt_fn,
            optimizer_params=opt_params,
            mask_type=self.mask_type,
            verbose=self.verbose,
            device_name=self.device_name,
            seed=self.random_state,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        eval_set: Optional[list[Tuple[np.ndarray, np.ndarray]]] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose: Optional[bool] = None,
    ):
        self.model_ = self._make_model()

        # decide ES patience
        patience = early_stopping_rounds if early_stopping_rounds is not None else self.patience

        # TabNetRegressor expects y to be 2D
        y2 = y.reshape(-1, 1)

        # Build fit kwargs mapping
        fit_kwargs: Dict[str, Any] = dict(
            X_train=X,
            y_train=y2,
            max_epochs=self.epochs,
            patience=patience,
            batch_size=self.batch_size,
            virtual_batch_size=self.virtual_batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )
        if eval_set:
            # convert eval y to 2D
            eval_set2 = [(Xe, ye.reshape(-1, 1)) for (Xe, ye) in eval_set]
            fit_kwargs["eval_set"] = eval_set2
            fit_kwargs["eval_metric"] = ["r2"]

        self.model_.fit(**fit_kwargs)

        # best iteration if provided by underlying model
        self.best_iteration_ = getattr(self.model_, "best_iteration_", None)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Estimator is not fitted.")
        yhat = self.model_.predict(X)
        # ensure 1D
        return np.asarray(yhat).reshape(-1)

    @classmethod
    def capabilities(cls) -> Capabilities:
        return Capabilities(
            backend="torch",
            supports_early_stopping=True,
            supports_gpu=True,
            supports_feature_importance=False,
            handles_categoricals=False,  # set True if you implement cat handling
            deterministic=False,
            notes="Custom TabNet via sklearn adapter",
        )
