# file: make_pipeline.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from models import get_model  # <- registry factory


# ---------------------------
# Model family / preprocessing rules
# ---------------------------

TREE_LIKE = {
    "random_forest", "extra_trees", "gradient_boosting", "xgboost", "lightgbm"
}
NEEDS_SCALING = {
    "svr", "ridge", "lasso", "elasticnet", "kneighbors"
}
NATIVE_CATEGORICAL = {"catboost"}

# --- add near the top with the other sets ---
NEURAL_LIKE = {"dnn", "tabnet", "ft_transformer", "node"}  # models that benefit from scaling + dense inputs



# ---------------------------
# Column typing
# ---------------------------

def infer_column_types(
    df: pd.DataFrame,
    target: str,
    explicit_numeric: Optional[List[str]] = None,
    explicit_categorical: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Infer numeric vs categorical columns. You can override with explicit lists.
    """
    features = [c for c in df.columns if c != target]

    if explicit_numeric is not None or explicit_categorical is not None:
        num = explicit_numeric or []
        cat = explicit_categorical or [c for c in features if c not in (explicit_numeric or [])]
        return num, cat

    num_cols, cat_cols = [], []
    for c in features:
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return num_cols, cat_cols


# ---------------------------
# Preprocessor builders
# ---------------------------

def _build_preprocessor(
    model_name: str,
    numeric: List[str],
    categorical: List[str],
    cfg: Optional[Dict[str, Any]] = None,
) -> Union[ColumnTransformer, str]:
    cfg = cfg or {}
    # For neural-like models we prefer scaling and dense OHE by default
    is_neural = model_name in NEURAL_LIKE

    force_scale = bool(cfg.get("force_scale", False)) or is_neural
    drop_cats_for_trees = bool(cfg.get("drop_unused_cats_for_trees", False))
    # Dense by default for neural models; can still be overridden via cfg
    one_hot_sparsity = bool(cfg.get("ohe_sparse", False) and not is_neural)

    if model_name in NATIVE_CATEGORICAL:
        return "catboost_native"

    # numeric branch
    do_scale = force_scale or model_name in NEEDS_SCALING
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if do_scale:
        num_steps.append(("scaler", StandardScaler()))

    # categorical branch
    if len(categorical) == 0:
        cat_transformer = "drop"
    else:
        if (model_name in TREE_LIKE) and drop_cats_for_trees:
            cat_transformer = "drop"
        else:
            cat_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                # neural-like -> dense OHE; others -> respect cfg
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=one_hot_sparsity is True)),
            ])

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=num_steps), numeric),
            ("cat", cat_transformer, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )



# ---------------------------
# CatBoost native DF pipeline
# ---------------------------

class CatBoostDFPipeline:
    """
    A light wrapper that:
      - imputes numeric (median) and categorical (most_frequent),
      - preserves raw categorical values (strings/objects),
      - passes cat feature indices to CatBoost's fit().
    Compatible with sklearn's .fit/.predict.
    """
    def __init__(self, model, numeric: List[str], categorical: List[str]):
        self.model = model
        self.numeric = numeric
        self.categorical = categorical
        self.num_imp = SimpleImputer(strategy="median") if numeric else None
        self.cat_imp = SimpleImputer(strategy="most_frequent") if categorical else None
        self.columns: List[str] = []       # [num..., cat...]
        self.cat_indices: List[int] = []   # positional indices in combined matrix

    def _ensure_layout(self, X_df: pd.DataFrame):
        if not self.columns:
            self.columns = self.numeric + self.categorical
            self.cat_indices = list(range(len(self.numeric), len(self.numeric) + len(self.categorical)))

    def _fit_transform(self, X_df: pd.DataFrame) -> np.ndarray:
        self._ensure_layout(X_df)
        Xn = X_df[self.numeric] if self.numeric else pd.DataFrame(index=X_df.index)
        Xc = X_df[self.categorical] if self.categorical else pd.DataFrame(index=X_df.index)

        if self.num_imp and self.numeric:
            Xn = pd.DataFrame(self.num_imp.fit_transform(Xn), index=X_df.index, columns=self.numeric)
        if self.cat_imp and self.categorical:
            Xc = pd.DataFrame(self.cat_imp.fit_transform(Xc), index=X_df.index, columns=self.categorical)

        X = pd.concat([Xn, Xc], axis=1)[self.columns]
        return X.values

    def _transform(self, X_df: pd.DataFrame) -> np.ndarray:
        self._ensure_layout(X_df)
        # transform path (imputers already fit)
        Xn = (pd.DataFrame(self.num_imp.transform(X_df[self.numeric]), columns=self.numeric, index=X_df.index)
              if (self.num_imp and self.numeric) else X_df[self.numeric] if self.numeric else pd.DataFrame(index=X_df.index))
        Xc = (pd.DataFrame(self.cat_imp.transform(X_df[self.categorical]), columns=self.categorical, index=X_df.index)
              if (self.cat_imp and self.categorical) else X_df[self.categorical] if self.categorical else pd.DataFrame(index=X_df.index))
        X = pd.concat([Xn, Xc], axis=1)[self.columns]
        return X.values

    # sklearn-like API
    def fit(self, X_df: pd.DataFrame, y, **kwargs):
        X = self._fit_transform(X_df)
        fit_kwargs = dict(kwargs)
        if self.cat_indices:
            fit_kwargs.setdefault("cat_features", self.cat_indices)
        self.model.fit(X, y, **fit_kwargs)
        return self

    def predict(self, X_df: pd.DataFrame):
        X = self._transform(X_df)
        return self.model.predict(X)


# ---------------------------
# Public factory
# ---------------------------

def make_pipeline(
    model_name: str,
    model_params: Dict[str, Any],
    *,
    df_train: pd.DataFrame,
    target_col: str,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    preproc_cfg: Optional[Dict[str, Any]] = None,
    target_scaling: bool = False,
):
    """
    Build a full preprocessing → model pipeline using the model registry.
    Returns:
      • sklearn Pipeline (ColumnTransformer → model), or
      • TransformedTargetRegressor wrapping the pipeline if target_scaling=True, or
      • CatBoostDFPipeline (wrapped in TransformedTargetRegressor if target_scaling=True).
    """
    model_name = model_name.lower()
    num_cols, cat_cols = infer_column_types(
        df_train, target_col, explicit_numeric=numeric_cols, explicit_categorical=categorical_cols
    )

    # instantiate model via registry (SklearnAdapter / XGB / LGBM / CatBoostAdapter)
    model = get_model(model_name, model_params)

    # build preprocessor
    preproc = _build_preprocessor(model_name, num_cols, cat_cols, cfg=preproc_cfg)

    # CatBoost native path
    if preproc == "catboost_native":
        pipe = CatBoostDFPipeline(model=model, numeric=num_cols, categorical=cat_cols)
        return TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler()) if target_scaling else pipe

    # sklearn-compatible pipeline
    pipe = Pipeline(steps=[("preprocess", preproc), ("model", model)], verbose=False)
    return TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler()) if target_scaling else pipe


# ---------------------------
# Feature name extractor (optional, handy for FI)
# ---------------------------

def get_feature_names_from_pipeline(
    pipe_or_ttr: Union[Pipeline, TransformedTargetRegressor, CatBoostDFPipeline],
    input_df: pd.DataFrame,
    target_col: str,
) -> List[str]:
    """
    Retrieve transformed feature names (for FI plots, debugging).
    - Works for sklearn Pipeline with ColumnTransformer.
    - For CatBoostDFPipeline, returns [numeric + categorical] original names (no OHE).
    - For TransformedTargetRegressor, unwraps and delegates.
    """
    # unwrap TTR
    if isinstance(pipe_or_ttr, TransformedTargetRegressor):
        reg = pipe_or_ttr.regressor
        return get_feature_names_from_pipeline(reg, input_df, target_col)

    # CatBoost path
    if isinstance(pipe_or_ttr, CatBoostDFPipeline):
        return (pipe_or_ttr.numeric or []) + (pipe_or_ttr.categorical or [])

    # sklearn pipeline path
    if isinstance(pipe_or_ttr, Pipeline):
        preprocess = pipe_or_ttr.named_steps.get("preprocess")
        if isinstance(preprocess, ColumnTransformer):
            try:
                # sklearn >= 1.0
                return list(preprocess.get_feature_names_out())
            except Exception:
                # Fallback: build names manually
                names: List[str] = []
                for name, trans, cols in preprocess.transformers_:
                    if trans == "drop":
                        continue
                    if hasattr(trans, "get_feature_names_out"):
                        try:
                            out = trans.get_feature_names_out(cols)
                            names.extend(list(out))
                        except Exception:
                            names.extend(cols if isinstance(cols, list) else [cols])
                    else:
                        names.extend(cols if isinstance(cols, list) else [cols])
                return names
    # fallback: raw columns minus target
    return [c for c in input_df.columns if c != target_col]
