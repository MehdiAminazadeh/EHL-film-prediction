from __future__ import annotations
from .adapters import SklearnAdapter
from .registry import register_model
from .base import Capabilities

# --- Core sklearn regressors ---
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

def _mk_adapter(cls, name: str):
    return type(f"{name}Adapter", (SklearnAdapter,), {"_cls": cls})

# Linear models
@register_model("ridge", tags=["sklearn","linear"])
class RidgeRegressor(_mk_adapter(Ridge, "Ridge")):
    @classmethod
    def capabilities(cls) -> Capabilities:
        return Capabilities(backend="sklearn", notes="Ridge regression")

@register_model("lasso", tags=["sklearn","linear"])
class LassoRegressor(_mk_adapter(Lasso, "Lasso")):
    @classmethod
    def capabilities(cls) -> Capabilities:
        return Capabilities(backend="sklearn", notes="Lasso regression")

@register_model("elasticnet", tags=["sklearn","linear"])
class ElasticNetRegressor(_mk_adapter(ElasticNet, "ElasticNet")):
    @classmethod
    def capabilities(cls) -> Capabilities:
        return Capabilities(backend="sklearn", notes="ElasticNet regression")

# Tree ensembles
@register_model("random_forest", tags=["sklearn","tree","bagging"])
class RandomForestAdapter(_mk_adapter(RandomForestRegressor, "RandomForest")):
    @classmethod
    def capabilities(cls) -> Capabilities:
        return Capabilities(
            backend="sklearn",
            supports_feature_importance=True,
            notes="RandomForestRegressor with impurity-based feature_importances_",
        )

@register_model("extra_trees", tags=["sklearn","tree","bagging"])
class ExtraTreesAdapter(_mk_adapter(ExtraTreesRegressor, "ExtraTrees")):
    @classmethod
    def capabilities(cls) -> Capabilities:
        return Capabilities(
            backend="sklearn",
            supports_feature_importance=True,
            notes="ExtraTreesRegressor with impurity-based feature_importances_",
        )

@register_model("gradient_boosting", tags=["sklearn","boosting"])
class SklearnGBRAdapter(_mk_adapter(GradientBoostingRegressor, "GradientBoosting")):
    @classmethod
    def capabilities(cls) -> Capabilities:
        return Capabilities(
            backend="boosting",
            supports_feature_importance=True,  # has feature_importances_
            notes="Sklearn GradientBoostingRegressor",
        )

# Kernel / instance-based
@register_model("svr", tags=["sklearn","kernel"])
class SVRAdapter(_mk_adapter(SVR, "SVR")):
    @classmethod
    def capabilities(cls) -> Capabilities:
        return Capabilities(backend="sklearn", notes="Support Vector Regression")

@register_model("kneighbors", tags=["sklearn","instance"])
class KNNAdapter(_mk_adapter(KNeighborsRegressor, "KNN")):
    @classmethod
    def capabilities(cls) -> Capabilities:
        return Capabilities(backend="sklearn", notes="KNeighborsRegressor")

# --- Optional boosters (only if installed) ---
try:
    import xgboost as xgb
    @register_model("xgboost", tags=["boosting","trees"])
    class XGBAdapter(_mk_adapter(xgb.XGBRegressor, "XGBoost")):
        @classmethod
        def capabilities(cls) -> Capabilities:
            return Capabilities(
                backend="boosting",
                supports_feature_importance=True,
                supports_gpu=True,  # if tree_method='gpu_hist'
                notes="XGBoost (set tree_method='gpu_hist' for GPU)",
            )
except Exception:
    pass

try:
    import lightgbm as lgb
    @register_model("lightgbm", tags=["boosting","trees"])
    class LGBMAdapter(_mk_adapter(lgb.LGBMRegressor, "LightGBM")):
        @classmethod
        def capabilities(cls) -> Capabilities:
            return Capabilities(
                backend="boosting",
                supports_feature_importance=True,
                supports_gpu=True,  # if device='gpu' and GPU build
                notes="LightGBM (set device='gpu' for GPU build)",
            )
except Exception:
    pass

try:
    from catboost import CatBoostRegressor
    from .adapters import SklearnAdapter
    from .registry import register_model
    from .base import Capabilities

    def _mk_adapter(cls, name: str):
        return type(f"{name}Adapter", (SklearnAdapter,), {"_cls": cls})

    @register_model("catboost", tags=["boosting","trees"])
    class CatBoostAdapter(_mk_adapter(CatBoostRegressor, "CatBoost")):
        def __init__(self, **params):
            # set default verbosity ONCE at init (CatBoost forbids changing params after fit)
            params = dict(params)
            params.setdefault("verbose", False)
            super().__init__(**params)

        # DO NOT call set_params here; just fit
        def fit(self, X, y, **kwargs):
            self.model.fit(X, y, **kwargs)
            return self

        @classmethod
        def capabilities(cls) -> Capabilities:
            return Capabilities(
                backend="boosting",
                supports_feature_importance=True,
                supports_gpu=True,        # if task_type='GPU'
                handles_categoricals=True,
                notes="CatBoost (set task_type='GPU' for GPU; native categoricals with cat_features)",
            )
except Exception:
    pass
