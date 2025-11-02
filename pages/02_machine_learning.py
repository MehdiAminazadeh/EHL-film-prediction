import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from ui_shared import topbar, guidance_text, suitability_report
from sklearn.model_selection import train_test_split, KFold, cross_validate, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance



MODEL_DESCRIPTIONS = {
    "Linear Regression": {
        "title": "Linear Regression",
        "body": (
            "Simple baseline model. Assumes a linear relation between inputs and the target "
            "(`Average Film`). Works best when features are not highly correlated and the "
            "relationship is roughly linear.\n\n"
            "Desired outcome: small MAE/RMSE, R² close to 1. If residuals show a pattern, "
            "the data is not purely linear."
        ),
    },
    "Ridge": {
        "title": "Ridge Regression",
        "body": (
            "Linear model with L2 regularization. Useful when features are correlated or "
            "you have many numeric inputs. Shrinks coefficients but keeps all features.\n\n"
            "Desired outcome: similar or slightly better R² than Linear Regression, and "
            "more stable coefficients."
        ),
    },
    "Lasso": {
        "title": "Lasso Regression",
        "body": (
            "Linear model with L1 regularization. Can drive some coefficients to zero, "
            "so it can act like feature selection. Good for noisy or wide datasets.\n\n"
            "Desired outcome: good R² with fewer effective features. If R² drops too much, "
            "L1 is too strong."
        ),
    },
    "Random Forest": {
        "title": "Random Forest Regressor",
        "body": (
            "Nonlinear, tree-based ensemble. Captures interactions and nonlinearities "
            "without manual feature engineering. Usually a strong default on tabular data.\n\n"
            "Desired outcome: R² high (often higher than linear), residuals centered around 0, "
            "permutation importance showing sensible top features.\n\n"
            "Plots:\n"
            "- Predicted vs Actual: points close to diagonal → good fit.\n"
            "- Residuals histogram: bell-ish around 0 → good.\n"
            "- Learning curve: if CV R² still rising → add more data."
        ),
    },
    "SVR (RBF)": {
        "title": "SVR (RBF)",
        "body": (
            "Support Vector Regression with RBF kernel. Good for smooth nonlinear relations, "
            "but can be slower on large datasets. Sensitive to scaling (we scale inputs above).\n\n"
            "Desired outcome: R² comparable to Random Forest; residuals not wildly skewed. "
            "If many points are off, try tuning C/gamma."
        ),
    },
}



st.set_page_config(page_title="Machine Learning • EHL", layout="wide")
topbar("Machine Learning", show_sidebar=True)
st.title("Machine Learning — Predict Average Film (Unified Classical + Nonlinear Models)")

df = st.session_state.get("last_clean_df")
if df is None or df.empty:
    st.info("Upload and clean data on Home first.")
    st.stop()

numeric_df = df.select_dtypes(include=[np.number]).copy()

banned = [c for c in numeric_df.columns if re.search(r"\border\b", c, re.I) or re.search(r"average\s*wavelength", c, re.I)]
numeric_df = numeric_df.drop(columns=banned, errors="ignore")

target_col = next((c for c in numeric_df.columns if re.search(r"average\s*film", c, re.I)), None)
if not target_col:
    st.error("No 'Average Film' column found.")
    st.stop()

test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
cv_folds = st.slider("Cross-validation folds", 3, 10, 5, 1)

models = {
    "Linear Regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
    "Lasso": Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=0.01, max_iter=10000))]),
    "Random Forest": Pipeline([("model", RandomForestRegressor(n_estimators=400, random_state=42))]),
    "SVR (RBF)": Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="rbf", C=10.0, gamma="scale"))]),
}
model_name = st.selectbox("Model", list(models.keys()), index=3)

with st.popover("What is this model?"):
    info = MODEL_DESCRIPTIONS.get(model_name)
    if info:
        st.subheader(info["title"])
        st.write(info["body"])
        st.markdown(
            "- **Predicted vs Actual** (below): checks fit.\n"
            "- **Residuals distribution**: checks errors → should be around 0.\n"
            "- **Learning curve**: shows if more data would help.\n"
            "- **Permutation importance**: which inputs matter most."
        )
    else:
        st.write("No description available for this model.")

st.code(guidance_text(cv_folds, test_size, numeric_df.shape[0]), language="text")


X = numeric_df.drop(columns=[target_col], errors="ignore")
y = numeric_df[target_col].values

if X.shape[1] < 2:
    st.warning("Not enough features after cleaning.")
    st.stop()

pipe = models[model_name]
kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

scoring = {
    "mae": "neg_mean_absolute_error",
    "mse": "neg_mean_squared_error",
    "r2": "r2",
}
cv_res = cross_validate(pipe, X, y, cv=kf, scoring=scoring)
cv_table = pd.DataFrame({
    "MAE (cv)": -cv_res["test_mae"],
    "RMSE (cv)": np.sqrt(-cv_res["test_mse"]),
    "R2 (cv)": cv_res["test_r2"],
})
st.markdown("### Cross-validation metrics")
st.dataframe(cv_table.describe().loc[["mean", "std"]].T, use_container_width=True)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)
pipe.fit(X_tr, y_tr)
y_pred = pipe.predict(X_te)

metrics = {
    "MAE": mean_absolute_error(y_te, y_pred),
    "RMSE": np.sqrt(mean_squared_error(y_te, y_pred)),
    "R2": r2_score(y_te, y_pred),
}
st.markdown("### Holdout metrics")
st.dataframe(pd.Series(metrics, name="value").to_frame(), use_container_width=True)

fig_pred = go.Figure()
fig_pred.add_scatter(x=y_te, y=y_pred, mode="markers", name="Predicted")
mn, mx = float(np.nanmin(y_te)), float(np.nanmax(y_te))
fig_pred.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines", name="Ideal"))
fig_pred.update_layout(template="plotly_dark", title="Predicted vs Actual", xaxis_title="Actual", yaxis_title="Predicted", height=360)
st.plotly_chart(fig_pred, use_container_width=True)

resid = y_te - y_pred
fig_resid = go.Figure()
fig_resid.add_histogram(x=resid, nbinsx=40)
fig_resid.update_layout(template="plotly_dark", title="Residuals distribution", height=320)
st.plotly_chart(fig_resid, use_container_width=True)

train_sizes, train_scores, test_scores = learning_curve(pipe, X, y, cv=kf, scoring="r2", train_sizes=np.linspace(0.2, 1.0, 5))
fig_lc = go.Figure()
fig_lc.add_scatter(x=train_sizes, y=train_scores.mean(axis=1), mode="lines+markers", name="Train R²")
fig_lc.add_scatter(x=train_sizes, y=test_scores.mean(axis=1), mode="lines+markers", name="CV R²")
fig_lc.update_layout(template="plotly_dark", title="Learning curve (R²)", height=340)
st.plotly_chart(fig_lc, use_container_width=True)

feature_names = list(X.columns)
perm = permutation_importance(pipe, X_te, y_te, n_repeats=10, random_state=42, scoring="r2")
imp = pd.Series(perm.importances_mean, index=feature_names).sort_values(ascending=False)
fig_imp = go.Figure()
fig_imp.add_bar(y=imp.index[::-1], x=imp.values[::-1], orientation="h")
fig_imp.update_layout(template="plotly_dark", title="Permutation importance", height=360)
st.plotly_chart(fig_imp, use_container_width=True)

st.divider()
st.caption(suitability_report(numeric_df))
