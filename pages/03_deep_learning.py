from __future__ import annotations
import io
import sys
from pathlib import Path
from typing import Any, Dict, List
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import pandas as pd
import streamlit as st

from ui_shared import topbar, inline_plotly_fig, find_avg_film

ROOT = Path(__file__).resolve().parents[1]
for sub in [
    ".",
    "utils",
    "configs",
    "pipelines",
    "pipelines/train_strategies",
    "models",
    "nn_models",
]:
    p = ROOT / sub if sub != "." else ROOT
    if p.exists():
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)

from utils.config_parser import load_config, ConfigError
from utils.logger import ExperimentLogger
from pipelines.make_pipeline import make_pipeline
from pipelines.train_strategies.train_dnn import train_dnn
from pipelines.train_strategies.train_tabnet import train_tabnet
from pipelines.train_strategies.train_ft import train_ft_transformer
from pipelines.train_strategies.train_node import train_node

st.set_page_config(page_title="Deep Learning • EHL", layout="wide")
topbar("Deep Learning", show_sidebar=True)

st.title("Deep Learning — EHL Film Prediction")
st.caption("Runs your RegKit framework (TabNet / FT-Transformer / NODE / DNN) on the preprocessed EHL data from Home.")

df_clean = st.session_state.get("last_clean_df")
if df_clean is None or df_clean.empty:
    st.info("No data in memory. Go to **Home** → upload TXT/ZIP → preprocessing → come back.")
    st.stop()

target_col = find_avg_film(df_clean.columns)
if not target_col:
    st.error("Target column (like 'Average Film (nm)') not found.")
    st.stop()

st.markdown("#### 1. Model & run setup")

MODEL_OPTIONS = {
    "TabNet (recommended)": "tabnet",
    "FT-Transformer": "ft_transformer",
    "NODE": "node",
    "DNN (MLP)": "dnn",
}

c1, c2, c3 = st.columns([1.4, 1.0, 1.0])
with c1:
    model_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()))
    model_name = MODEL_OPTIONS[model_label]
with c2:
    num_runs = st.slider("Repeat runs", 1, 10, 1, 1)
with c3:
    show_plots = st.checkbox("Show plots", True)

st.info("For more stable DL results and HP tuning, retrain ≈10× and compare MAE / RMSE / R².")

if "dl_stop" not in st.session_state:
    st.session_state["dl_stop"] = False

st.markdown("#### 2. Train / evaluate")
col_run, col_stop = st.columns([3, 1])
with col_run:
    run_clicked = st.button("Run Deep Learning", type="primary", use_container_width=True)
with col_stop:
    stop_clicked = st.button("Stop current run", use_container_width=True)
    if stop_clicked:
        st.session_state["dl_stop"] = True

placeholder_metrics = st.empty()
placeholder_plots = st.empty()
placeholder_download = st.empty()
logs_container = st.container()

if not run_clicked:
    st.stop()

st.session_state["dl_stop"] = False

cfg_path = ROOT / "configs" / "config.yaml"
if not cfg_path.exists():
    cfg_path = ROOT / "config.yaml"

if not cfg_path.exists():
    st.error("config.yaml not found (checked ./configs/ and project root).")
    st.stop()

try:
    cfg = load_config(cfg_path)
    cfg.switch_model(model_name)
except ConfigError as e:
    st.error(f"Config error: {e}")
    st.stop()

TRAINERS = {
    "dnn": train_dnn,
    "tabnet": train_tabnet,
    "ft_transformer": train_ft_transformer,
    "node": train_node,
}
train_fn = TRAINERS[model_name]

progress = st.progress(0.0, text="Starting...")
logger = ExperimentLogger(enabled=False, excel_path=str(ROOT / "experiment_log.xlsx"), verbosity=0)

all_runs: List[Dict[str, Any]] = []
last_best_params: Dict[str, Any] = []
run_logs: List[str] = []

for i in range(num_runs):
    if st.session_state.get("dl_stop"):
        progress.progress(i / max(1, num_runs), text=f"Stopped at run {i}/{num_runs}")
        st.warning(f"Stopped by user at run {i}/{num_runs}.")
        break

    progress.progress(i / num_runs, text=f"Training run {i+1}/{num_runs} ...")

    f_out = io.StringIO()
    f_err = io.StringIO()
    with redirect_stdout(f_out), redirect_stderr(f_err):
        res = train_fn(cfg=cfg, df=df_clean, logger=logger)

    res["__run"] = i + 1
    all_runs.append(res)

    out_txt = f_out.getvalue().strip()
    err_txt = f_err.getvalue().strip()
    combined = ""
    if out_txt:
        combined += out_txt
    if err_txt:
        if combined:
            combined += "\n"
        combined += err_txt
    run_logs.append(combined if combined else "(no messages)")

    last_best_params = res.get("Best_Params", {}) or {}

progress.progress(1.0, text="Finished.")

with logs_container:
    st.markdown("#### 3. Run logs")
    for idx, log_txt in enumerate(run_logs, start=1):
        with st.expander(f"Run {idx} logs", expanded=(idx == len(run_logs))):
            st.code(log_txt, language="text")

if not all_runs:
    st.warning("No runs were completed.")
    st.stop()

results_df = pd.DataFrame(all_runs)
placeholder_metrics.markdown("#### 4. Metrics (per run)")
placeholder_metrics.dataframe(results_df, use_container_width=True)

default_model_params = {k: v for k, v in cfg.model.items() if not isinstance(v, dict)}
if isinstance(last_best_params, dict):
    final_params = {**default_model_params, **last_best_params}
else:
    final_params = default_model_params

final_pipe = make_pipeline(
    model_name=model_name,
    model_params=final_params,
    df_train=df_clean,
    target_col=target_col,
    preproc_cfg={"force_scale": True},
    target_scaling=False,
)

X_full = df_clean.drop(columns=[target_col])
y_true = df_clean[target_col].values

defaults = {k: v for k, v in cfg.model.items() if not isinstance(v, dict)}
final_params = {**defaults, **last_best_params}

final_pipe = make_pipeline(
    model_name=model_name,
    model_params=final_params,
    df_train=df_clean,
    target_col=target_col,
    preproc_cfg={"force_scale": True},
    target_scaling=False,
)

X_full = df_clean.drop(columns=[target_col])
y_true = df_clean[target_col].values

final_pipe.fit(X_full, y_true)
y_pred = final_pipe.predict(X_full)

pred_df = pd.DataFrame(
    {
        "y_true": y_true,
        "y_pred": y_pred,
        "model": model_name,
    }
)

if show_plots:
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_scatter(x=y_true, y=y_pred, mode="markers", name="pred")
    lo, hi = float(np.nanmin(y_true)), float(np.nanmax(y_true))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="ideal"))
    fig.update_layout(
        title="Predicted vs Actual",
        xaxis_title="Actual",
        yaxis_title="Predicted",
    )
    with placeholder_plots.container():
        inline_plotly_fig(fig, height=360)

    err = y_pred - y_true
    fig2 = go.Figure()
    fig2.add_histogram(x=err, nbinsx=40)
    fig2.update_layout(title="Residuals", xaxis_title="error", yaxis_title="count")
    inline_plotly_fig(fig2, height=300)

buf_m = io.StringIO()
results_df.to_csv(buf_m, index=False)
placeholder_download.download_button(
    "Download metrics CSV",
    buf_m.getvalue().encode("utf-8"),
    "dl_metrics.csv",
    "text/csv",
)

buf_p = io.StringIO()
pred_df.to_csv(buf_p, index=False)
st.download_button(
    "Download predictions CSV",
    buf_p.getvalue().encode("utf-8"),
    "dl_predictions.csv",
    "text/csv",
)

st.success("Done. You can change model or increase runs now.")
