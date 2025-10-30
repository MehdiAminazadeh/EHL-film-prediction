import base64
import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def hide_sidebar_nav():
    st.markdown(
        """
        <style>[data-testid="stSidebarNav"]{display:none!important}</style>
        """,
        unsafe_allow_html=True,
    )


def _logo_data_uri(path: str = "assets/megt_logo.png"):
    try:
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return None


def topbar(active: str = "Home", show_sidebar: bool = True):
    if not show_sidebar:
        st.markdown(
            "<style>[data-testid='stSidebarNav']{display:none!important}</style>",
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <style>
        .top-shell {
            position: sticky;
            top: 0;
            z-index: 999;
            backdrop-filter: blur(8px);
            background: rgba(9, 13, 22, 0.7);
            border-bottom: 1px solid rgba(255,255,255,0.08);
        }
        .top-inner {
            max-width: 1180px;
            margin: 0 auto;
            padding: 12px 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .brand {
            font-weight: 800;
            letter-spacing: 0.3px;
            font-size: 1.05rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    nav = {
        "Home": "app.py",
        "Statistics": "pages/01_statistics.py",
        "Machine Learning": "pages/02_machine_learning.py",
        "Deep Learning": "pages/03_deep_learning.py",
    }

    active_index_map = {"Home": 0, "Statistics": 1, "ML": 2, "DL": 3}
    active_index = active_index_map.get(active, 0)
    labels = list(nav.keys())

    with st.container():
        st.markdown('<div class="top-shell"><div class="top-inner">', unsafe_allow_html=True)

        logo_data = _logo_data_uri()
        col_brand, col_nav = st.columns([0.35, 0.65])

        with col_brand:
            if logo_data:
                st.image(logo_data, width=85)
            else:
                st.markdown('<div class="brand">EHL Film Prediction</div>', unsafe_allow_html=True)

        with col_nav:
            cols = st.columns(len(labels))
            for idx, label in enumerate(labels):
                with cols[idx]:
                    clicked = st.button(label, key=f"nav_btn_{idx}", use_container_width=True)
                    st.markdown(
                        f"""
                        <style>
                        button[data-testid="baseButton-secondary"][key="nav_btn_{idx}"] {{
                            background: {'rgba(37,99,235,0.22)' if idx == active_index else 'transparent'};
                            border: 1px solid {'rgba(37,99,235,0.35)' if idx == active_index else 'rgba(255,255,255,0.15)'};
                            color: {'#eaf0ff' if idx == active_index else '#dbe2f2'};
                            border-radius: 8px;
                            font-weight: 600;
                        }}
                        button[data-testid="baseButton-secondary"][key="nav_btn_{idx}"]:hover {{
                            background: rgba(255,255,255,0.08);
                        }}
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )
                    if clicked and idx != active_index:
                        st.switch_page(nav[label])

        st.markdown("</div></div>", unsafe_allow_html=True)


def suitability_report(df: pd.DataFrame) -> str:
    rows, cols = df.shape
    messages: list[str] = [f"Rows = {rows}, Columns = {cols}"]

    if rows < 200:
        messages.append("• Small (<200): DL not recommended; use classical ML.")
    elif rows < 1000:
        messages.append("• Moderate: ML fine; DL only with regularization.")
    else:
        messages.append("• Large: ML & DL feasible; monitor training time.")

    if cols > rows:
        messages.append("• p>n: overfitting risk → PCA/regularization advised.")

    if cols > 50:
        messages.append("• Many features (>50): consider feature selection/PCA.")

    if df.isna().any().any():
        messages.append("• Missing values remain: some models still need imputation.")

    return "\n".join(messages)


def guidance_text(cv_folds: int, test_size: float, n: int) -> str:
    train_n = int((1 - test_size) * n)
    lines = [
        f"• Samples: {n} → train≈{train_n}, test≈{n - train_n}",
        f"• CV={cv_folds}: each fold validates on ≈{1 / cv_folds:.0%}.",
        "• Higher CV → better generalization estimate, more compute.",
        "• Larger test size → stronger final eval, less training data.",
        "• Small datasets: test_size 0.15–0.20 and CV=5 are good defaults.",
    ]
    return "\n".join(lines)


def find_avg_film(cols):
    for c in cols:
        if re.search(r"\baverage\s*film\b", str(c), re.IGNORECASE):
            return c
    return None


def inline_plotly_fig(fig, height: int = 360):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
