import re
import time
import hashlib

import pandas as pd
import streamlit as st

from styling import show_glowing_logo
from ui_shared import topbar
from data_processor import process_files, extract_supported_from_zip
from preprocess import preprocess_dataframe


st.set_page_config(page_title="Home • EHL Film Prediction", layout="wide")
topbar("Home", show_sidebar=True)
show_glowing_logo("assets/megt_logo.jpg")


st.title("Home")
st.caption("Upload TXT (or ZIP) → merge & normalize to 15-column EHL dataset for analysis and ML.")

if "uploader_ver" not in st.session_state:
    st.session_state["uploader_ver"] = 0
if "last_df" not in st.session_state:
    st.session_state["last_df"] = None
if "last_clean_df" not in st.session_state:
    st.session_state["last_clean_df"] = None
if "last_clean_report" not in st.session_state:
    st.session_state["last_clean_report"] = None
if "uploaded_hashes" not in st.session_state:
    st.session_state["uploaded_hashes"] = set()

upload_mode = st.radio("Upload mode", ["Files (.txt)", "Folder (.zip)"], horizontal=True)

col_upload, col_clear = st.columns([4, 1])

with col_upload:
    uploads = st.file_uploader(
        "Upload files",
        type=["txt", "zip"],
        accept_multiple_files=upload_mode.startswith("Files"),
        key=f"uploader_home_{st.session_state['uploader_ver']}",
    )

with col_clear:
    clear_clicked = st.button("Clear", use_container_width=True)
    if clear_clicked:
        st.session_state["last_df"] = None
        st.session_state["last_clean_df"] = None
        st.session_state["last_clean_report"] = None
        st.session_state["uploaded_hashes"] = set()
        st.session_state["uploader_ver"] += 1
        st.rerun()


def valid_name(name: str) -> bool:
    return bool(re.match(r"^[\w ()\-\.\[\]]+\.txt$", name or "", re.I))


def file_digest(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


if uploads:
    new_payloads: list[tuple[str, bytes]] = []

    if upload_mode.startswith("Files"):
        for uploaded in uploads:
            if not valid_name(uploaded.name):
                continue

            content = uploaded.read()
            digest = file_digest(content)

            if uploaded.name in st.session_state["uploaded_hashes"]:
                continue
            if digest in st.session_state["uploaded_hashes"]:
                continue

            new_payloads.append((uploaded.name, content))
            st.session_state["uploaded_hashes"].add(uploaded.name)
            st.session_state["uploaded_hashes"].add(digest)
    else:
        extracted = extract_supported_from_zip(uploads.read())
        for name, content in extracted:
            digest = file_digest(content)
            if name in st.session_state["uploaded_hashes"]:
                continue
            if digest in st.session_state["uploaded_hashes"]:
                continue
            new_payloads.append((name, content))
            st.session_state["uploaded_hashes"].add(name)
            st.session_state["uploaded_hashes"].add(digest)

    if not new_payloads:
        st.warning("No new (non-duplicate) TXT files found.")
    else:
        start = time.time()
        merged_df = process_files(new_payloads)
        st.session_state["last_df"] = merged_df

        clean_df, clean_report = preprocess_dataframe(merged_df)
        st.session_state["last_clean_df"] = clean_df
        st.session_state["last_clean_report"] = clean_report

        elapsed = time.time() - start
        st.success(f"Processed {len(new_payloads)} new file(s) in {elapsed:.2f}s")

clean_df = st.session_state.get("last_clean_df")

if clean_df is not None and not clean_df.empty:
    st.markdown("### Preview (Cleaned 15-column Dataset)")
    st.dataframe(clean_df.head(40), use_container_width=True, height=360)

    report = st.session_state.get("last_clean_report")
    if report:
        with st.expander("Cleaning report"):
            st.code(report, language="text")
