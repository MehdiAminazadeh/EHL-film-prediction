
import streamlit as st
from ui_shared import topbar
from statistics import render_statistics_panel



st.set_page_config(page_title="Statistics • EHL", layout="wide")
topbar("Statistics", show_sidebar=True)

st.title("Statistics")

df = st.session_state.get("last_clean_df")
if df is None or df.empty:
    st.info("Upload and clean data on Home first.")
    st.stop()

render_statistics_panel(df)
