"""
utils/data_filter.py
--------------------
Shared, stateless data-filtering widget used by BOTH the Preprocessing
page and the EDA page.

Usage
-----
    from utils.data_filter import render_data_filters

    df_filtered = render_data_filters(df, key_prefix="prep")
    # or
    df_filtered = render_data_filters(df, key_prefix="eda")

All Streamlit widget keys are namespaced by key_prefix so the two pages
never share widget state.

Filter groups
-------------
  Categorical  : Brine Type, Brine Name, Concentrate Method
  Numeric      : Dilution Ratio, Temperature, Initial Foam Temp, Oil %
  Custom rules : user-defined column + range/multiselect
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

# ── Process column candidates (checked in order, first match wins) ────────────
_DILUTION_KEYS   = ["dilution ratio_corrected", "dilution ratio", "dilution"]
_TEMP_KEYS       = ["temperature_corrected", "temperature", "temp foam monitoring"]
_INIT_TEMP_KEYS  = ["initial foam temp", "foam temp (dilution"]
_OIL_PCT_KEYS    = ["oil  (%)_corrected", "oil  (%)", "oil (%)"]
_OIL_TYPE_KEYS   = ["oil type", "oil name"]
_CONC_RATIO_KEYS = ["concentrate  (ratio)", "concentrate ratio"]
_METHOD_KEYS     = ["concentrate manufacturing method"]
_BRINE_TYPE_KEYS = ["brine type", "brinetype"]
_BRINE_NAME_KEYS = ["brine name"]


def _find(df: pd.DataFrame, keywords: list[str]) -> str | None:
    """Case-insensitive substring search. Returns first matching column name."""
    lower_map = {c.lower().strip(): c for c in df.columns}
    for kw in keywords:
        for cl, orig in lower_map.items():
            if kw in cl:
                return orig
    return None


def _numeric_filter(df: pd.DataFrame, col: str,
                    label: str, key: str) -> pd.DataFrame:
    """Slider filter for a numeric column. Returns filtered df."""
    s   = pd.to_numeric(df[col], errors="coerce")
    lo  = float(s.min(skipna=True))
    hi  = float(s.max(skipna=True))
    if lo >= hi:
        st.caption(f"{label}: single value ({lo}) — no filter applied.")
        return df
    rng = st.slider(label, lo, hi, (lo, hi), key=key)
    return df[(s >= rng[0]) & (s <= rng[1])]


def _categorical_filter(df: pd.DataFrame, col: str,
                         label: str, key: str) -> pd.DataFrame:
    """Multiselect filter for a categorical column. Returns filtered df."""
    opts = sorted(df[col].dropna().unique().tolist(), key=str)
    if len(opts) <= 1:
        st.caption(f"{label}: only one unique value — no filter applied.")
        return df
    picks = st.multiselect(label, opts, default=opts, key=key)
    return df[df[col].isin(picks)]


# ─────────────────────────────────────────────────────────────────────────────

def render_data_filters(df: pd.DataFrame, key_prefix: str = "filt") -> pd.DataFrame:
    """
    Render the full data-filter UI and return the filtered DataFrame.
    key_prefix must be unique per page ("prep" / "eda") to avoid widget collisions.
    """
    data = df.copy()
    n0   = len(data)

    # # ── SECTION A: Categorical / categorical-like filters ─────────────────
    # with st.container(border=True):
    #     st.markdown("##### 🏷️ Categorical Filters")
    #     ca1, ca2, ca3 = st.columns(3)

    #     brine_type_col = _find(data, _BRINE_TYPE_KEYS)
    #     brine_name_col = _find(data, _BRINE_NAME_KEYS)
    #     method_col     = _find(data, _METHOD_KEYS)

    #     with ca1:
    #         if brine_type_col:
    #             data = _categorical_filter(data, brine_type_col,
    #                                         f"Brine Type  ({brine_type_col})",
    #                                         f"{key_prefix}_brine_type")
    #         else:
    #             st.caption("Brine Type column not found.")

    #     with ca2:
    #         if brine_name_col:
    #             data = _categorical_filter(data, brine_name_col,
    #                                         f"Brine Name  ({brine_name_col})",
    #                                         f"{key_prefix}_brine_name")
    #         else:
    #             st.caption("Brine Name column not found.")

    #     with ca3:
    #         if method_col:
    #             data = _categorical_filter(data, method_col,
    #                                         f"Manufacturing Method",
    #                                         f"{key_prefix}_method")
    #         else:
    #             st.caption("Concentrate Method column not found.")

    #     # Oil type (categorical) if present
    #     oil_type_col = _find(data, _OIL_TYPE_KEYS)
    #     if oil_type_col:
    #         data = _categorical_filter(data, oil_type_col,
    #                                     f"Oil Type  ({oil_type_col})",
    #                                     f"{key_prefix}_oil_type")

    # # ── SECTION B: Process condition numeric filters ───────────────────────
    # with st.container(border=True):
    #     st.markdown("##### ⚙️ Process Condition Filters")

    #     # Dilution Ratio
    #     dil_col = _find(data, _DILUTION_KEYS)
    #     if dil_col:
    #         data_num = data.copy()
    #         data_num[dil_col] = pd.to_numeric(data_num[dil_col], errors="coerce")
    #         data = _numeric_filter(data_num, dil_col,
    #                                 f"Dilution Ratio  ({dil_col})",
    #                                 f"{key_prefix}_dilution")
    #     else:
    #         st.caption("Dilution Ratio column not found.")

    #     # Temperature (monitoring)
    #     pc1, pc2 = st.columns(2)
    #     with pc1:
    #         temp_col = _find(data, _TEMP_KEYS)
    #         if temp_col:
    #             data_num = data.copy()
    #             data_num[temp_col] = pd.to_numeric(data_num[temp_col], errors="coerce")
    #             data = _numeric_filter(data_num, temp_col,
    #                                     f"Temperature  ({temp_col})",
    #                                     f"{key_prefix}_temp")
    #         else:
    #             st.caption("Temperature column not found.")

    #     # Initial foam temperature
    #     with pc2:
    #         init_temp_col = _find(data, _INIT_TEMP_KEYS)
    #         if init_temp_col:
    #             data_num = data.copy()
    #             data_num[init_temp_col] = pd.to_numeric(data_num[init_temp_col], errors="coerce")
    #             data = _numeric_filter(data_num, init_temp_col,
    #                                     f"Initial Foam Temp  ({init_temp_col})",
    #                                     f"{key_prefix}_init_temp")
    #         else:
    #             st.caption("Initial Foam Temp column not found.")

    #     # Oil %
    #     oil_pct_col = _find(data, _OIL_PCT_KEYS)
    #     if oil_pct_col:
    #         data_num = data.copy()
    #         data_num[oil_pct_col] = pd.to_numeric(data_num[oil_pct_col], errors="coerce")
    #         data = _numeric_filter(data_num, oil_pct_col,
    #                                 f"Oil %  ({oil_pct_col})",
    #                                 f"{key_prefix}_oil_pct")
    #     else:
    #         st.caption("Oil % column not found.")

    #     # Concentrate Ratio
    #     conc_col = _find(data, _CONC_RATIO_KEYS)
    #     if conc_col:
    #         data_num = data.copy()
    #         data_num[conc_col] = pd.to_numeric(data_num[conc_col], errors="coerce")
    #         data = _numeric_filter(data_num, conc_col,
    #                                 f"Concentrate Ratio  ({conc_col})",
    #                                 f"{key_prefix}_conc_ratio")

    # ── SECTION C: Custom dynamic rules ───────────────────────────────────
    with st.container(border=True):
        st.markdown("##### ➕ Custom Rules")
        st.caption("Add any additional column-level filter.")

        dyn_key = f"{key_prefix}_dynamic_filters"
        if dyn_key not in st.session_state:
            st.session_state[dyn_key] = []

        r1, r2 = st.columns([4, 1])
        new_col = r1.selectbox("Column for new rule",
                                options=data.columns.tolist(),
                                key=f"{key_prefix}_dyn_new_col")
        if r2.button("➕ Add", key=f"{key_prefix}_add_rule", use_container_width=True):
            st.session_state[dyn_key].append({"column": new_col})

        for i, rule in enumerate(st.session_state[dyn_key]):
            col_name = rule["column"]
            if col_name not in data.columns:
                continue
            with st.expander(f"Rule {i+1}: {col_name}", expanded=True):
                rc1, rc2 = st.columns([5, 1])
                base_series = df[col_name]          # always reference the BASE df
                if pd.api.types.is_numeric_dtype(base_series):
                    lo = float(pd.to_numeric(base_series, errors="coerce").min(skipna=True))
                    hi = float(pd.to_numeric(base_series, errors="coerce").max(skipna=True))
                    if lo < hi:
                        rng = rc1.slider(col_name, lo, hi, (lo, hi),
                                          key=f"{key_prefix}_dyn_r_{i}")
                        num_s = pd.to_numeric(data[col_name], errors="coerce")
                        data  = data[(num_s >= rng[0]) & (num_s <= rng[1])]
                    else:
                        rc1.caption(f"{col_name}: single value ({lo}).")
                else:
                    opts = base_series.dropna().unique().tolist()
                    sel  = rc1.multiselect(col_name, opts, default=opts,
                                            key=f"{key_prefix}_dyn_c_{i}")
                    data = data[data[col_name].isin(sel)]

                if rc2.button("🗑️", key=f"{key_prefix}_del_{i}"):
                    st.session_state[dyn_key].pop(i)
                    st.rerun()

    # ── Summary ───────────────────────────────────────────────────────────
    removed = n0 - len(data)
    if removed > 0:
        st.warning(f"🔽 Filters removed **{removed:,}** rows — "
                   f"**{len(data):,}** of {n0:,} remaining.")
    else:
        st.success(f"✅ No rows removed — **{len(data):,}** rows.")

    return data
