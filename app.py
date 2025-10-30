import io
import time
import re

import pandas as pd
import streamlit as st

from ui_shared import topbar
from data_processor import process_files, extract_supported_from_zip
from preprocess import preprocess_dataframe


st.set_page_config(page_title="Home • EHL Film Prediction", layout="wide")

topbar("Home", show_sidebar=True)

st.markdown('<div class="wrap">', unsafe_allow_html=True)
st.title("Home")
st.caption(
    "Upload TXT (or a ZIP of TXT files). We merge and auto-clean so Statistics, ML and DL pages can use the data immediately."
)

if "uploader_ver" not in st.session_state:
    st.session_state["uploader_ver"] = 0

upload_mode = st.radio("Upload mode", ["Files (.txt)", "Folder (.zip)"], horizontal=True)

col_upload, col_clear = st.columns([3, 1])

with col_upload:
    uploads = st.file_uploader(
        "Upload files",
        type=["txt", "zip"],
        accept_multiple_files=True if upload_mode.startswith("Files") else False,
        key=f"uploader_home_{st.session_state['uploader_ver']}",
    )

with col_clear:
    clear_clicked = st.button("Clear", use_container_width=True)
    if clear_clicked:
        st.session_state.clear()
        st.session_state["uploader_ver"] = st.session_state.get("uploader_ver", 0) + 1
        st.rerun()


def is_safe_txt_name(name: str) -> bool:
    pattern = r"^[\w ()\-\.\[\]]+\.txt$"
    return name.lower().endswith(".txt") and re.match(pattern, name, re.IGNORECASE)


if uploads:
    if upload_mode.startswith("Files"):
        file_payloads = []
        for uploaded in uploads:
            if not is_safe_txt_name(uploaded.name):
                st.error(f"Unsupported file name/type (TXT only): {uploaded.name}")
                st.stop()
            file_payloads.append((uploaded.name, uploaded.read()))
    else:
        zip_bytes = uploads.read()
        file_payloads = extract_supported_from_zip(zip_bytes)
        if not file_payloads:
            st.error("No valid .txt found in the ZIP.")
            st.stop()

    start_time = time.time()
    try:
        merged_df = process_files(file_payloads, parallel=True)
    except Exception as exc:
        st.error(f"Failed while parsing uploaded files. Details:\n\n{exc}")
        st.stop()

    st.session_state["last_df"] = merged_df

    if merged_df is not None and not merged_df.empty:
        clean_df, clean_report = preprocess_dataframe(merged_df, numeric_threshold=0.80)
        st.session_state["last_clean_df"] = clean_df
        st.session_state["last_clean_report"] = clean_report
        elapsed = time.time() - start_time
        st.success(f"Uploaded {len(file_payloads)} TXT file(s), merged and auto-cleaned in {elapsed:.2f}s.")
    else:
        st.warning("No valid data extracted from the TXT files.")

raw_df = st.session_state.get("last_df")

if raw_df is not None and not raw_df.empty:
    col_left, col_right = st.columns([1, 1.4], gap="large")

    with col_left:
        st.markdown("#### Clean")
        st.caption("Auto-clean is already applied. Click to recompute if needed.")
        reclean_clicked = st.button("Re-clean data", use_container_width=True)
        if reclean_clicked:
            reclean_df, reclean_report = preprocess_dataframe(raw_df, numeric_threshold=0.80)
            st.session_state["last_clean_df"] = reclean_df
            st.session_state["last_clean_report"] = reclean_report
            st.success("Re-cleaned.")

    with col_right:
        st.markdown('<div class="card"><h4>Preview</h4></div>', unsafe_allow_html=True)
        st.dataframe(raw_df.head(10), use_container_width=True, height=240)

st.markdown("</div>", unsafe_allow_html=True)
