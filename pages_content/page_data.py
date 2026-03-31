"""
pages_content/page_data.py
Upload, preview, and validate the dataset.
All heavy logic lives in core/data/loader.py.
"""

import streamlit as st
from pandas.errors import EmptyDataError

from state.session import clear_state, init_state, set_state, get_value
from core.data.loader import (
    load_csv, load_excel, list_excel_sheets,
    validate_df, extended_describe,
)
from components.dataset_summary import render_dataset_summary
from config.settings import SUPPORTED_FILE_TYPES, MAX_ROWS_PREVIEW


def render():
    init_state()

    st.title("📂 Data Loading")
    st.divider()

    # ── File uploader ─────────────────────────────────────────────────────
    st.subheader("📥 Import Dataset")
    uploaded_file = st.file_uploader(
        "**Upload dataset (CSV or Excel)**",
        type=SUPPORTED_FILE_TYPES,
        key="data_uploader",
    )

    if uploaded_file is not None:
        clear_state()
        fname = uploaded_file.name.lower()
        try:
            if fname.endswith(".csv"):
                df = load_csv(uploaded_file)
                set_state("data.raw", df)
                st.success("✅ CSV loaded successfully.")

            elif fname.endswith((".xls", ".xlsx")):
                sheets = list_excel_sheets(uploaded_file)
                sheet = st.selectbox("📄 Select Excel sheet", sheets)
                df = load_excel(uploaded_file, sheet)
                set_state("data.raw", df)
                st.success(f"✅ Sheet **'{sheet}'** loaded successfully.")

        except EmptyDataError:
            st.error("❌ The file is empty or has no readable columns.")
        except UnicodeDecodeError:
            st.error("❌ Could not decode the file even after trying multiple encodings. Try re-saving as UTF-8.")
        except Exception as e:
            st.error("❌ Failed to load file.")
            st.exception(e)

    # ── Always read from session state ───────────────────────────────────
    st.divider()
    df = get_value("data.raw")

    if df is None:
        st.info("No dataset loaded yet. Upload a file above.")
        st.stop()

    # ── Validation warnings ───────────────────────────────────────────────
    warnings = validate_df(df)
    if warnings:
        with st.expander("⚠️ Data Quality Warnings", expanded=True):
            for w in warnings:
                st.warning(w)

    # ── Dataset summary card ──────────────────────────────────────────────
    render_dataset_summary(df)

    # ── Full preview ──────────────────────────────────────────────────────
    st.subheader("💻 Dataset Preview")
    st.dataframe(df.head(MAX_ROWS_PREVIEW), use_container_width=True)

    st.divider()

    # ── Extended describe ─────────────────────────────────────────────────
    st.subheader("📝 Statistical Summary")
    summary_df = extended_describe(df)
    st.dataframe(summary_df, use_container_width=True)

    st.info("👉 Proceed to **📊 EDA** or **⚙️ Preprocessing**.")