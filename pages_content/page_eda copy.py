"""
pages_content/page_eda.py
Exploratory Data Analysis — fully decoupled from preprocessing.

Tabs
----
  1  Preview
  2  Data Filtering          ← shared render_data_filters (same as preprocessing)
  3  Feature Engineering     ← foam groups + auto-interactions for EDA only
  4  Summary
  5  Scatter + Histograms
  6  3D Scatter
  7  Box Plots
  8  Correlation
  9  Categorical
"""

from __future__ import annotations

import math
import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from itertools import product as iter_product

from state.session import init_state, get_value
from components.column_selector import render_column_selector
from core.data.preprocessor import (
    extended_describe, categorical_summary,
    categorical_warnings, categorical_imbalance,
)
from core.viz.eda import (
    draw_correlation_heatmap, draw_boxplots,
    draw_histograms, draw_scatter,
    draw_pairwise_scatter_with_hist,
)
from core.viz.style import fig_to_st
from config.settings import MAX_HEATMAP_FEATURES
from utils.data_filter import render_data_filters

# ─────────────────────────────────────────────────────────────────────────────
#  Foam taxonomy (mirror of preprocessing page)
# ─────────────────────────────────────────────────────────────────────────────
DEF_NANO    = ["HS (%)", "BLH5 (%)", "HSA (%)"]
DEF_ANIONIC = ["AOS (%)", "DB45 (%)", "alpha-step (%)", "SDS (%)", "SLES (%)", "n. LABS (%)"]
DEF_NONION  = ["APG (%)", "decyl glucoside (%)", "caprylyl glucoside (%)",
               "Tween 80 (%)", "PG (%)"]
DEF_ZW      = ["CapB (%)", "2C (%)", "DM (%)","Cola 2C (%)",  "Cola SLAA (%)", "Cola SC (%)","amine oxide (%)",
               "Cola LMB (%)", "Amphosol 1c (%)", "SC (%)", "LBHP (%)", "CS50(%)","LAO (%)"]
DEF_POLY    = ["HPAM (%)", "xanthan gum (%) ", "Guar Gum (%)", "FPAM (%)", "PAA (%)", "PA (%)"]
DEF_CITRIC  = ["Citric (%)",
               "31.2(%) citric+ 13.3(%) KOH  (pH not adjusted, pH=4.46 )",
               "31.2% citric+ 13.3% KOH  (adjusted pH=4) (%)",
               "38.1% citric+   KOH  (pH=5) (%)",
               "potassium citrate (9.7%)/citric acid buffer (19.22%)  pH=3 (%)",
               "potassium citrate (%)"]
DEF_ACID    = ["EDTA (%)", "etidronic acid (%)", "acetic acid (%)"]
DEF_ANTI    = ["Mem 2000-clear tech (%)", "Mem 2500-clear tech (%)",
               "Mem 4000-clear tech (%)", "Mem 3500-clear tech (%)",
               "Mem 3000-clear tech (%)", "ClearHib 1000 (%)"]
DEF_BRINE   = ["Divalent", "Monovalent"]
DEF_OIL     = ["Alkane (linear HC) ", "Aromatics", "Branched HC",
               "Light HC (up to C10)", "Sulfur content",
               "Acid & ester content", "Chlorinated components", "Polarity "]
DEF_PROCESS = ["Temperature", "Dilution Ratio", "Oil  (%)",
               "concentrate manufacturing method (Ratio)",
               "Initial Foam Temp (dilution Temp) "]

GRP_CLR = {
    "Nanoparticle":  "#1565C0", "Anionic":       "#E53935",
    "Nonionic":      "#FB8C00", "Zwitterionic":  "#8E24AA",
    "Surfactant":    "#FF8F00", "Polymer":       "#6A1B9A",
    "Citric Group": "#2E7D32", "Acid/Chelant":  "#C62828",
    "Antiscalant":   "#00695C", "Brine":         "#4527A0",
    "Oil":           "#BF360C", "Process":       "#546E7A",
    "Interaction":   "#795548", "Custom":        "#37474F",
}

CHEM_GROUPS = {"Nanoparticle","Anionic","Nonionic","Zwitterionic",
               "Polymer","Citric Group","Acid/Chelant","Antiscalant"}


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers (local copies so EDA page has no import dependency on prep page)
# ─────────────────────────────────────────────────────────────────────────────

def _clean_label(col: str) -> str:
    s = str(col).strip()
    s = re.sub(r'\s*\(%\)\s*$', '', s)
    s = re.sub(r'\(%\)', '', s)
    s = re.sub(r'%', ' pct', s)
    s = re.sub(r'\(|\)', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _avail(defaults, df_cols):
    return [c for c in defaults if c in df_cols]


def _safe_sum_zero(df, cols):
    ex = [c for c in cols if c in df.columns]
    if not ex:
        return pd.Series(0.0, index=df.index)
    result = pd.Series(0.0, index=df.index)
    for c in ex:
        result += pd.to_numeric(df[c], errors="coerce").fillna(0)
    return result


def _interact(feats, a, b, name):
    if a not in feats or b not in feats:
        return
    if feats[a].isna().mean() > 0.10 or feats[b].isna().mean() > 0.10:
        return
    feats[name] = feats[a] * feats[b]


def _build_eda_features(df: pd.DataFrame, sel: dict,
                         include_ix: bool,
                         custom_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """
    Build engineered features for EDA exploration only.
    Returns (df_with_new_cols, feat_to_group).
    Chemical NaN → 0; condition NaN left as-is (rows NOT dropped for EDA).
    """
    feats: dict = {}
    fg:    dict = {}

    def add(name, series, group):
        s = series.reset_index(drop=True) if hasattr(series, "reset_index") else series
        feats[name] = s
        fg[name]    = group

    used: dict = {}
    for group, cols in sel.items():
        for col in cols:
            if col not in df.columns:
                continue
            lbl = _clean_label(col)
            if lbl in used:
                lbl = lbl + " " + group
            used[lbl] = True
            add(lbl, pd.to_numeric(df[col], errors="coerce"), group)

    nano_tot   = _safe_sum_zero(df, sel.get("Nanoparticle", []))
    surf_an    = _safe_sum_zero(df, sel.get("Anionic",      []))
    surf_non   = _safe_sum_zero(df, sel.get("Nonionic",     []))
    surf_zw    = _safe_sum_zero(df, sel.get("Zwitterionic", []))
    surf_tot   = surf_an + surf_non + surf_zw
    poly_tot   = _safe_sum_zero(df, sel.get("Polymer",       []))
    citric_tot = _safe_sum_zero(df, sel.get("Citric Group", []))
    acid_tot   = _safe_sum_zero(df, sel.get("Acid/Chelant",  []))
    anti_tot   = _safe_sum_zero(df, sel.get("Antiscalant",   []))

    if sel.get("Nanoparticle"):
        add("Nano Total",  nano_tot,                    "Nanoparticle")
        add("Has Nano",   (nano_tot > 0).astype(float), "Nanoparticle")

    if sel.get("Anionic") or sel.get("Nonionic") or sel.get("Zwitterionic"):
        add("Surfactant Total",        surf_tot,                   "Surfactant")
        add("Surfactant Anionic",      surf_an,                    "Anionic")
        add("Surfactant Nonionic",     surf_non,                   "Nonionic")
        add("Surfactant Zwitterionic", surf_zw,                    "Zwitterionic")
        add("Zwitterionic Fraction",   surf_zw  / (surf_tot+1e-9), "Zwitterionic")
        add("Anionic Fraction",        surf_an  / (surf_tot+1e-9), "Anionic")
        add("Nonionic Fraction",       surf_non / (surf_tot+1e-9), "Nonionic")

    if sel.get("Polymer"):
        add("Polymer Total", poly_tot,                     "Polymer")
        add("Has Polymer",  (poly_tot > 0).astype(float),  "Polymer")

    if sel.get("Citric Group"):
        add("Citric Total",         citric_tot,                     "Citric Group")
        add("Citric to Surfactant", citric_tot / (surf_tot + 1e-9), "Citric Group")

    if sel.get("Acid/Chelant"):
        add("Acid Total", acid_tot, "Acid/Chelant")

    if sel.get("Antiscalant"):
        add("Antiscalant Total", anti_tot, "Antiscalant")

    if sel.get("Brine"):
        div_s  = pd.to_numeric(df.get("Divalent",  pd.Series(np.nan, index=df.index)), errors="coerce")
        mono_s = pd.to_numeric(df.get("Monovalent",pd.Series(np.nan, index=df.index)), errors="coerce")
        add("Divalent",                  div_s,                   "Brine")
        add("Monovalent",                mono_s,                  "Brine")
        add("Ionic Strength",            mono_s + 2*div_s,        "Brine")
        add("Divalent Monovalent Ratio",  div_s/(mono_s+1e-9),    "Brine")
        add("Log Divalent",              np.log1p(div_s),         "Brine")
        add("Log Monovalent",            np.log1p(mono_s),        "Brine")
        add("Log Ionic Strength",        np.log1p(mono_s+2*div_s),"Brine")

    oil_lbl = {
        "Alkane (linear HC) ":   "Oil Alkane",
        "Aromatics":              "Oil Aromatics",
        "Branched HC":            "Oil Branched HC",
        "Light HC (up to C10)":  "Oil Light HC",
        "Sulfur content":         "Oil Sulfur",
        "Acid & ester content":  "Oil Acid Ester",
        "Chlorinated components": "Oil Chlorinated",
        "Polarity ":              "Oil Polarity",
    }
    if sel.get("Oil"):
        for raw in sel["Oil"]:
            lbl = oil_lbl.get(raw, "Oil " + _clean_label(raw))
            if raw in df.columns:
                add(lbl, pd.to_numeric(df[raw], errors="coerce"), "Oil")

    # Process — driven by individual selections when _proc_map present
    pm = sel.get("_proc_map", {})
    if pm:
        if pm.get("temp") and pm["temp"] in df.columns:
            add("Temperature", pd.to_numeric(df[pm["temp"]], errors="coerce"), "Process")
        if pm.get("init_temp") and pm["init_temp"] in df.columns:
            add("Initial Foam Temp", pd.to_numeric(df[pm["init_temp"]], errors="coerce"), "Process")
        if pm.get("dilution") and pm["dilution"] in df.columns:
            add("Dilution Ratio", pd.to_numeric(df[pm["dilution"]], errors="coerce"), "Process")
        if pm.get("oil_pct") and pm["oil_pct"] in df.columns:
            add("Oil Percent", pd.to_numeric(df[pm["oil_pct"]], errors="coerce"), "Process")
        if pm.get("method") and pm["method"] in df.columns:
            import re as _re
            _mc = df[pm["method"]].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
            for _val in sorted(_mc.unique(), key=str):
                _slug = _re.sub(r'[^a-z0-9]+', '_', _val.lower())[:40].strip('_')
                add(f"Concen_Ratio_Method_{_slug}", (_mc == _val).astype(float), "Process")
    elif sel.get("Process"):   # fallback: flat list (backward-compat)
        for tc in ["Temperature", "Initial Foam Temp (dilution Temp) "]:
            if tc in df.columns:
                add("Temperature", pd.to_numeric(df[tc], errors="coerce"), "Process")
                break
        if "Dilution Ratio" in df.columns:
            add("Dilution Ratio", pd.to_numeric(df["Dilution Ratio"], errors="coerce"), "Process")
        if "Oil  (%)" in df.columns:
            add("Oil Percent", pd.to_numeric(df["Oil  (%)"], errors="coerce"), "Process")
        mc = "concentrate manufacturing method (Ratio)"
        if mc in df.columns:
            import re as _re
            _mc = df[mc].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
            for _val in sorted(_mc.unique(), key=str):
                _slug = _re.sub(r'[^a-z0-9]+', '_', _val.lower())[:40].strip('_')
                add(f"Concen_Ratio_Method_{_slug}", (_mc == _val).astype(float), "Process")

    for col in custom_cols:
        if col in df.columns:
            lbl = "Custom " + _clean_label(col)
            add(lbl, pd.to_numeric(df[col], errors="coerce"), "Custom")

    if include_ix:
        _interact(feats, "Nano Total",           "Surfactant Total",        "Nano x Surfactant Total")
        _interact(feats, "Nano Total",           "Surfactant Anionic",      "Nano x Surfactant Anionic")
        _interact(feats, "Nano Total",           "Surfactant Zwitterionic", "Nano x Surfactant Zwitterionic")
        _interact(feats, "Nano Total",           "Log Divalent",            "Nano x Log Divalent")
        _interact(feats, "Nano Total",           "Temperature",             "Nano x Temperature")
        _interact(feats, "Nano Total",           "Dilution Ratio",          "Nano x Dilution Ratio")
        _interact(feats, "Nano Total",           "Oil Polarity",            "Nano x Oil Polarity")
        _interact(feats, "Nano Total",           "Oil Aromatics",           "Nano x Oil Aromatics")
        _interact(feats, "Surfactant Total",     "Log Divalent",            "Surfactant x Log Divalent")
        _interact(feats, "Surfactant Anionic",   "Log Divalent",            "Surfactant Anionic x Log Divalent")
        _interact(feats, "Surfactant Total",     "Temperature",             "Surfactant x Temperature")
        _interact(feats, "Surfactant Total",     "Dilution Ratio",          "Surfactant x Dilution Ratio")
        _interact(feats, "Surfactant Total",     "Oil Percent",             "Surfactant x Oil Percent")
        _interact(feats, "Surfactant Total",     "Oil Polarity",            "Surfactant x Oil Polarity")
        _interact(feats, "Surfactant Anionic",   "Oil Aromatics",           "Surfactant Anionic x Oil Aromatics")
        _interact(feats, "Zwitterionic Fraction","Log Divalent",            "Zwitterionic Fraction x Log Divalent")
        _interact(feats, "Polymer Total",        "Surfactant Total",        "Polymer x Surfactant")
        _interact(feats, "Polymer Total",        "Log Divalent",            "Polymer x Log Divalent")
        _interact(feats, "Polymer Total",        "Temperature",             "Polymer x Temperature")
        _interact(feats, "Polymer Total",        "Nano Total",              "Polymer x Nano")
        _interact(feats, "Citric Total",         "Log Divalent",            "Citric x Log Divalent")
        _interact(feats, "Citric Total",         "Surfactant Total",        "Citric x Surfactant")
        _interact(feats, "Citric Total",         "Nano Total",              "Citric x Nano")
        _interact(feats, "Citric Total",         "Temperature",             "Citric x Temperature")
        _interact(feats, "Acid Total",           "Log Divalent",            "Acid x Log Divalent")
        _interact(feats, "Log Divalent",         "Log Monovalent",          "Log Divalent x Log Monovalent")
        _interact(feats, "Divalent Monovalent Ratio","Surfactant Anionic",  "Hardness x Surfactant Anionic")
        _interact(feats, "Divalent Monovalent Ratio","Nano Total",          "Hardness x Nano")
        _interact(feats, "Temperature",          "Dilution Ratio",          "Temperature x Dilution Ratio")
        _interact(feats, "Temperature",          "Oil Percent",             "Temperature x Oil Percent")
        _interact(feats, "Dilution Ratio",       "Surfactant Total",        "Dilution x Surfactant")
        _interact(feats, "Oil Polarity",         "Surfactant Total",        "Oil Polarity x Surfactant")
        _interact(feats, "Oil Aromatics",        "Nano Total",              "Oil Aromatics x Nano")

    X = pd.DataFrame(feats).reset_index(drop=True)
    X = X.loc[:, X.nunique() > 0]
    X = X.loc[:, ~X.isnull().all()]
    fg = {k: v for k, v in fg.items() if k in X.columns}
    return X, fg


def _render_fe_tab(df_sub: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Render the EDA Feature Engineering tab.
    Returns df_sub with engineered columns added (for exploration only).
    A button controls whether new columns are merged into the active EDA dataset.
    """
    all_cols = df_sub.columns.tolist()

    st.markdown(
        "Configure chemical groups below. Click **▶ Apply to EDA dataset** "
        "to add the engineered features to the current EDA session — "
        "**no changes are saved to preprocessing or the model**."
    )

    with st.container(border=True):
        st.markdown("##### 🧴 Surfactants")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown("**🔴 Anionic**")
            sel_an  = st.multiselect("Anionic",       all_cols, default=_avail(DEF_ANIONIC, all_cols), format_func=_clean_label, key="eda_fe_an")
        with sc2:
            st.markdown("**🟣 Zwitterionic**")
            sel_zw  = st.multiselect("Zwitterionic",  all_cols, default=_avail(DEF_ZW,      all_cols), format_func=_clean_label, key="eda_fe_zw")
        with sc3:
            st.markdown("**🟠 Nonionic**")
            sel_non = st.multiselect("Nonionic",      all_cols, default=_avail(DEF_NONION,  all_cols), format_func=_clean_label, key="eda_fe_non")

        st.markdown("##### 🔵 Nanoparticles")
        sel_nano = st.multiselect("Nanoparticle", all_cols, default=_avail(DEF_NANO, all_cols),
                                   format_func=_clean_label, key="eda_fe_nano")

        st.markdown("##### ⚗️ Other Chemistry & Conditions")
        oc1, oc2 = st.columns(2)
        with oc1:
            sel_poly   = st.multiselect("Polymer",       all_cols, default=_avail(DEF_POLY,    all_cols), format_func=_clean_label, key="eda_fe_poly")
            sel_citric = st.multiselect("Citric Group", all_cols, default=_avail(DEF_CITRIC,  all_cols), format_func=_clean_label, key="eda_fe_cit")
            sel_acid   = st.multiselect("Acid/Chelant",  all_cols, default=_avail(DEF_ACID,    all_cols), format_func=_clean_label, key="eda_fe_acid")
            sel_anti   = st.multiselect("Antiscalant",   all_cols, default=_avail(DEF_ANTI,    all_cols), format_func=_clean_label, key="eda_fe_anti")
        with oc2:
            sel_brine = st.multiselect("Brine",          all_cols, default=_avail(DEF_BRINE, all_cols), format_func=_clean_label, key="eda_fe_brine")
            sel_oil   = st.multiselect("Oil Composition",all_cols, default=_avail(DEF_OIL,   all_cols), format_func=_clean_label, key="eda_fe_oil")

        # ── Process conditions — one picker per variable ──────────────────
        st.markdown("##### ⚙️ Process Conditions")
        st.caption("Select the column for each process variable. '(none)' to exclude.")
        NONE_E = "(none)"
        none_opt_e = [NONE_E] + all_cols

        ep1, ep2, ep3 = st.columns(3)
        with ep1:
            _edef_dil = next((c for c in ["Dilution Ratio_Corrected","Dilution Ratio"] if c in all_cols), NONE_E)
            eproc_dil = st.selectbox("Dilution Ratio", none_opt_e, index=none_opt_e.index(_edef_dil),
                                      format_func=_clean_label, key="eda_fe_proc_dil")
            _edef_temp = next((c for c in ["Temperature_Corrected","Temperature"] if c in all_cols), NONE_E)
            eproc_temp = st.selectbox("Temperature", none_opt_e, index=none_opt_e.index(_edef_temp),
                                       format_func=_clean_label, key="eda_fe_proc_temp")
        with ep2:
            _edef_oilpct = next((c for c in ["Oil  (%)_Corrected","Oil  (%)","Oil (%)"] if c in all_cols), NONE_E)
            eproc_oil_pct = st.selectbox("Oil %", none_opt_e, index=none_opt_e.index(_edef_oilpct),
                                          format_func=_clean_label, key="eda_fe_proc_oilpct")
            _edef_it = next((c for c in ["Initial Foam Temp (dilution Temp) ","Initial Foam Temp (dilution Temp)_Corrected"] if c in all_cols), NONE_E)
            eproc_init_temp = st.selectbox("Initial Foam Temp", none_opt_e, index=none_opt_e.index(_edef_it),
                                            format_func=_clean_label, key="eda_fe_proc_inittemp")
        with ep3:
            _edef_method = next((c for c in ["concentrate manufacturing method (Ratio)_corrected",
                                              "concentrate manufacturing method (Ratio)"] if c in all_cols), NONE_E)
            eproc_method = st.selectbox("Manufacturing Method", none_opt_e, index=none_opt_e.index(_edef_method),
                                         format_func=_clean_label, key="eda_fe_proc_method")


        proc_cols_e = [c for c in [eproc_dil, eproc_temp, eproc_oil_pct,
                                    eproc_init_temp, eproc_method]
                       if c != NONE_E]

        already = set(sel_an+sel_zw+sel_non+sel_nano+sel_poly+sel_citric+sel_acid+sel_anti
                      +sel_brine+sel_oil+proc_cols_e)
        remaining = [c for c in all_cols if c not in already]
        sel_custom = st.multiselect("➕ Custom extra columns", remaining, default=[],
                                     format_func=_clean_label, key="eda_fe_custom")
        include_ix = st.checkbox("✨ Include interaction features", value=True, key="eda_fe_ix")

    eproc_map = {
        "dilution":   eproc_dil       if eproc_dil       != NONE_E else None,
        "temp":       eproc_temp      if eproc_temp      != NONE_E else None,
        "oil_pct":    eproc_oil_pct   if eproc_oil_pct   != NONE_E else None,
        "init_temp":  eproc_init_temp if eproc_init_temp != NONE_E else None,
        "method":     eproc_method    if eproc_method    != NONE_E else None,

    }

    sel = {
        "Nanoparticle":  sel_nano,   "Anionic":       sel_an,
        "Nonionic":      sel_non,    "Zwitterionic":  sel_zw,
        "Polymer":       sel_poly,   "Citric Group": sel_citric,
        "Acid/Chelant":  sel_acid,   "Antiscalant":   sel_anti,
        "Brine":         sel_brine,  "Oil":           sel_oil,
        "Process":       proc_cols_e,
        "_proc_map":     eproc_map,
    }

    # Fill chem NaN → 0 before building (for chem groups only)
    df_fe = df_sub.copy()
    all_chem_cols = sum([sel[g] for g in CHEM_GROUPS if g in sel], [])
    for col in all_chem_cols:
        if col in df_fe.columns:
            df_fe[col] = pd.to_numeric(df_fe[col], errors="coerce").fillna(0)

    # Always build preview
    X_eng, fg = _build_eda_features(df_fe, sel, include_ix, sel_custom)

    # ── Overview ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown("##### 📊 Engineered Feature Preview")

    n_ix   = sum(1 for g in fg.values() if g == "Interaction")
    n_chem = X_eng.shape[1] - n_ix

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total new features", X_eng.shape[1])
    m2.metric("Chemistry features", n_chem)
    m3.metric("Interaction features", n_ix)
    m4.metric("Rows", len(X_eng))

    group_counts = pd.Series(fg).value_counts()
    fig_gc = px.bar(
        x=group_counts.index, y=group_counts.values,
        color=group_counts.index, color_discrete_map=GRP_CLR,
        labels={"x": "Group", "y": "# Features"},
        title="Features per Group",
        template="plotly_white",
    )
    fig_gc.update_layout(height=260, showlegend=False,
                          margin=dict(t=40, b=20, l=10, r=10))
    st.plotly_chart(fig_gc, use_container_width=True)

    with st.expander("Preview engineered columns (first 10 rows)"):
        st.dataframe(X_eng.head(10), use_container_width=True)

    skipped = [name for (a, b, name) in [
        ("Nano Total","Oil Polarity","Nano x Oil Polarity"),
        ("Nano Total","Oil Aromatics","Nano x Oil Aromatics"),
        ("Surfactant Total","Oil Polarity","Surfactant x Oil Polarity"),
    ] if name not in X_eng.columns]
    if skipped:
        st.info(
            f"ℹ️ {len(skipped)} interaction(s) skipped (>10% NaN): "
            + ", ".join(skipped)
        )

    # ── Apply button ──────────────────────────────────────────────────────
    st.divider()
    if st.button(
        "▶ Apply to EDA dataset",
        type="primary",
        use_container_width=True,
        key="eda_fe_apply",
        help="Adds engineered columns to the EDA dataset for this session only. "
             "Does NOT affect preprocessing or model training.",
    ):
        # Merge engineered cols into df_sub (avoid duplicates)
        df_out = df_sub.copy().reset_index(drop=True)
        for col in X_eng.columns:
            if col not in df_out.columns:
                df_out[col] = X_eng[col].values
        st.session_state["eda_enriched_df"] = df_out
        st.session_state["eda_fe_applied"]  = True
        st.success(
            f"✅ Added **{X_eng.shape[1]}** engineered features to EDA dataset. "
            f"All other EDA tabs now show these features too."
        )
        st.rerun()

    # Return enriched df if already applied, else original
    if st.session_state.get("eda_fe_applied") and "eda_enriched_df" in st.session_state:
        enriched = st.session_state["eda_enriched_df"]
        # Validate shape matches current filtered df
        if len(enriched) == len(df_sub):
            st.caption(
                f"🟢 Engineered features are active — EDA dataset has "
                f"**{enriched.shape[1]}** columns."
            )
            if st.button("↩ Remove engineered features", key="eda_fe_reset"):
                st.session_state.pop("eda_enriched_df", None)
                st.session_state["eda_fe_applied"] = False
                st.rerun()
            return enriched
        else:
            # Shape mismatch (filter changed) — reset
            st.session_state.pop("eda_enriched_df", None)
            st.session_state["eda_fe_applied"] = False

    return df_sub


# ─────────────────────────────────────────────────────────────────────────────
#  Main render
# ─────────────────────────────────────────────────────────────────────────────

def render():
    init_state()

    st.title("📊 Exploratory Data Analysis")
    st.divider()

    df = get_value("data.raw")
    if df is None:
        st.warning("⚠️ No dataset loaded. Go to **📂 Data** first.")
        st.stop()

    # ── Column selector ───────────────────────────────────────────────────
    st.subheader("🎯 Select Columns to Explore")
    with st.expander("Column selector", expanded=True):
        features, target = render_column_selector(
            df,
            default_features=get_value("data.feature_names"),
            default_target=get_value("data.target_name"),
            key_prefix="eda",
        )

    if not features or target is None:
        st.info("Select at least one feature and a target column above.")
        st.stop()

    selected_cols = list(dict.fromkeys(features + [target]))
    df_base = df[selected_cols].copy()

    # ── EDA tabs ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔎 EDA Dashboard")

    (tab_prev, tab_filt, tab_fe, tab_sum,
     tab_scat, tab_3d, tab_box, tab_corr, tab_cat) = st.tabs([
        "🔎 Preview",
        "🔍 Data Filtering",
        "⚗️ Feature Engineering",
        "📝 Summary",
        "📊 Scatter + Histograms",
        "🧊 3D Scatter",
        "📦 Box Plots",
        "🔗 Correlation",
        "⚠️ Categorical",
    ])

    # ── Tab 1: Preview ────────────────────────────────────────────────────
    with tab_prev:
        st.write(f"**Shape:** {df_base.shape[0]:,} rows × {df_base.shape[1]} cols")
        st.dataframe(df_base, use_container_width=True)

    # ── Tab 2: Data Filtering ─────────────────────────────────────────────
    with tab_filt:
        st.subheader("🔍 Filter Rows for EDA")
        st.caption(
            "Filters here affect all EDA tabs below — they do **not** touch "
            "the preprocessing pipeline or model data."
        )
        # Use the shared filter widget (eda-namespaced keys)
        df_sub = render_data_filters(df_base, key_prefix="eda")

        # Store filtered df in session so other tabs can use it
        st.session_state["eda_filtered_df"] = df_sub

        # Reset FE enrichment when filter changes (shape mismatch guard)
        if st.session_state.get("eda_fe_applied"):
            enriched = st.session_state.get("eda_enriched_df")
            if enriched is not None and len(enriched) != len(df_sub):
                st.session_state.pop("eda_enriched_df", None)
                st.session_state["eda_fe_applied"] = False
                st.info("ℹ️ Filter changed — engineered features reset. Re-apply in the Feature Engineering tab.")

    # Resolve working df for all downstream tabs
    # Priority: FE-enriched → filtered → base
    if st.session_state.get("eda_fe_applied") and "eda_enriched_df" in st.session_state:
        df_work = st.session_state["eda_enriched_df"]
    elif "eda_filtered_df" in st.session_state:
        df_work = st.session_state["eda_filtered_df"]
    else:
        df_work = df_base

    # Resolve feature list for df_work
    work_features = [c for c in df_work.columns if c != target]
    num_cols      = [c for c in work_features if pd.api.types.is_numeric_dtype(df_work[c])]

    # ── Tab 3: Feature Engineering ────────────────────────────────────────
    with tab_fe:
        st.subheader("⚗️ Feature Engineering — EDA Exploration")
        # Pass the currently filtered (not enriched) df so the FE tab
        # always starts from raw features, not stacked engineered ones
        filt_df = st.session_state.get("eda_filtered_df", df_base)
        df_work = _render_fe_tab(filt_df, target)

    # ── Tab 4: Summary ────────────────────────────────────────────────────
    with tab_sum:
        st.markdown("#### 🔢 Numeric Summary")
        st.dataframe(extended_describe(df_work), use_container_width=True)
        cat_df = categorical_summary(df_work)
        if not cat_df.empty:
            st.markdown("#### 🏷️ Categorical Summary")
            st.dataframe(cat_df, use_container_width=True)

    # ── Tab 5: Scatter + Histograms ───────────────────────────────────────
    with tab_scat:
        num_cols_work = [c for c in work_features
                         if c in df_work.columns
                         and pd.api.types.is_numeric_dtype(df_work[c])]

        if not num_cols_work:
            st.warning("No numeric feature columns available.")
        else:
            st.markdown("#### 🔄 Scatter + Marginal Histograms")
            y_s = df_work[target]
            n_classes = y_s.dropna().nunique()

            if pd.api.types.is_numeric_dtype(y_s) and n_classes > 5:
                st.success("Detected: Regression")
                show_trend = st.checkbox("Show R² trendline", value=True, key="eda_trend")
                figs = draw_pairwise_scatter_with_hist(
                    df_work[num_cols_work + [target]], target, show_trend
                )
            elif n_classes <= 3:
                y_enc, mapping = pd.factorize(y_s)
                df_plot = df_work.copy()
                df_plot[target] = y_enc
                st.info(f"Detected: Classification ({n_classes} classes) — "
                        f"mapping: {dict(enumerate(mapping))}")
                figs = draw_pairwise_scatter_with_hist(
                    df_plot[num_cols_work + [target]], target, False
                )
            else:
                st.warning(f"⚠️ Target has {n_classes} classes — scatter disabled for >3.")
                figs = []

            if figs:
                for row_start in range(0, len(figs), 2):
                    cols_ui = st.columns(2)
                    for ci, fig in enumerate(figs[row_start: row_start+2]):
                        with cols_ui[ci]:
                            fig_to_st(fig)
            else:
                st.info("Not enough data to render plots.")

            # st.divider()
            # st.markdown("#### 📊 Histograms")
            # bins = st.slider("Histogram bins", 5, 100, 30, key="eda_bins")
            # fig_h = draw_histograms(df_work, num_cols_work, bins=bins)
            # fig_to_st(fig_h)

            # st.divider()
            # st.markdown("#### 🟢 All Features vs Target")
            # plot_all_one = st.checkbox("One combined figure", value=True, key="eda_all_one")
            # if not plot_all_one:
            #     for feat in num_cols_work:
            #         fig_s = draw_scatter(df_work, feat, target, hue_col=None)
            #         fig_to_st(fig_s)
            # else:
            #     n     = len(num_cols_work)
            #     ncols = 3
            #     nrows = math.ceil(n / ncols)
            #     fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
            #     axes = axes.flatten()
            #     for i, feat in enumerate(num_cols_work):
            #         pdf = df_work[[feat, target]].dropna()
            #         xv  = pd.to_numeric(pdf[feat],   errors="coerce")
            #         yv  = pd.to_numeric(pdf[target],  errors="coerce")
            #         ok  = xv.notna() & yv.notna()
            #         axes[i].scatter(xv[ok], yv[ok], alpha=0.6, s=18)
            #         axes[i].set_xlabel(feat, fontsize=8)
            #         axes[i].set_ylabel(target, fontsize=8)
            #         axes[i].set_title(f"{feat} vs {target}", fontsize=9)
            #     for j in range(i+1, len(axes)):
            #         fig.delaxes(axes[j])
            #     plt.tight_layout()
            #     fig_to_st(fig)

    # ── Tab 6: 3D Scatter ─────────────────────────────────────────────────
    with tab_3d:
        num_cols_3d = [c for c in df.columns
                       if pd.api.types.is_numeric_dtype(df[c])]

        enable_3d = st.checkbox("Enable Interactive Scatter", value=True, key="eda_3d_enable")
        if enable_3d:
            st.markdown("#### 🧊 2D Interactive Scatter")
            if len(num_cols_3d) >= 2:
                ca, cb, cc = st.columns(3)
                xi = ca.selectbox("X", num_cols_3d, index=0, key="i2d_x")
                yi = cb.selectbox("Y", num_cols_3d, index=min(1,len(num_cols_3d)-1), key="i2d_y")
                co_opts = ["None"] + df.columns.tolist()
                co_def  = co_opts.index(target) if target in co_opts else 0
                ci = cc.selectbox("Colour by", co_opts, index=co_def, key="i2d_c")
                if ci == "None": ci = None
                hov_opts = [c for c in df.columns if c not in [xi, yi]]
                hov_cols = st.multiselect("Hover columns", hov_opts,
                                           default=[target] if target in hov_opts else hov_opts[:2],
                                           key="i2d_hov")
                keep = list(dict.fromkeys([xi, yi]
                             + ([ci] if ci else [])
                             + [c for c in hov_cols if c not in [xi, yi, ci]]))
                pdf = df[keep].copy()
                if ci and not pd.api.types.is_numeric_dtype(pdf[ci]):
                    pdf[ci] = pdf[ci].astype(str)
                fig2d = px.scatter(pdf, x=xi, y=yi, color=ci,
                                    hover_data={c: True for c in hov_cols},
                                    color_continuous_scale="Viridis"
                                    if ci and pd.api.types.is_numeric_dtype(pdf[ci]) else None,
                                    opacity=0.75, height=500)
                fig2d.update_traces(marker=dict(size=6))
                fig2d.update_layout(margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig2d, use_container_width=True)
            else:
                st.warning("Need ≥ 2 numeric columns.")

            st.divider()
            st.markdown("#### 🧊 3D Interactive Scatter")
            if len(num_cols_3d) >= 3:
                ca, cb, cc, cd = st.columns(4)
                xi = ca.selectbox("X", num_cols_3d, index=0, key="i3d_x")
                yi = cb.selectbox("Y", num_cols_3d, index=min(1,len(num_cols_3d)-1), key="i3d_y")
                zi = cc.selectbox("Z", num_cols_3d, index=min(2,len(num_cols_3d)-1), key="i3d_z")
                co_opts = ["None"] + df.columns.tolist()
                co_def  = co_opts.index(target) if target in co_opts else 0
                ci = cd.selectbox("Colour by", co_opts, index=co_def, key="i3d_c")
                if ci == "None": ci = None
                hov_opts = [c for c in df.columns if c not in [xi, yi, zi]]
                hov_cols = st.multiselect("Hover columns", hov_opts,
                                           default=[target] if target in hov_opts else hov_opts[:2],
                                           key="i3d_hov")
                keep = list(dict.fromkeys([xi, yi, zi]
                             + ([ci] if ci else [])
                             + [c for c in hov_cols if c not in [xi, yi, zi, ci]]))
                pdf = df[keep].copy()
                if ci and not pd.api.types.is_numeric_dtype(pdf[ci]):
                    pdf[ci] = pdf[ci].astype(str)
                fig3d = px.scatter_3d(pdf, x=xi, y=yi, z=zi, color=ci,
                                       hover_data={c: True for c in hov_cols},
                                       color_continuous_scale="Viridis"
                                       if ci and pd.api.types.is_numeric_dtype(pdf[ci]) else None,
                                       opacity=0.75, height=650)
                fig3d.update_traces(marker=dict(size=4))
                fig3d.update_layout(margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig3d, use_container_width=True)
            else:
                st.warning("Need ≥ 3 numeric columns.")

    # ── Tab 7: Box Plots ──────────────────────────────────────────────────
    with tab_box:
        st.markdown("#### 📦 Boxplots")
        num_w = [c for c in work_features
                 if c in df_work.columns and pd.api.types.is_numeric_dtype(df_work[c])]
        use_std = st.checkbox("Standardise for visualization", value=False, key="eda_box_std")
        df_plot = df_work.copy()
        if use_std and num_w:
            df_plot[num_w] = StandardScaler().fit_transform(df_work[num_w])
        if num_w:
            fig_bp = draw_boxplots(df_plot, num_w)
            fig_to_st(fig_bp)
        else:
            st.info("No numeric feature columns.")

    # ── Tab 8: Correlation ────────────────────────────────────────────────
    with tab_corr:
        st.markdown("### 🟣 Correlation Matrix")
        num_w = [c for c in work_features
                 if c in df_work.columns and pd.api.types.is_numeric_dtype(df_work[c])]
        if not num_w:
            st.warning("No numeric features.")
        elif len(num_w) > MAX_HEATMAP_FEATURES:
            st.warning(f"Too many features ({len(num_w)}) — showing table only.")
            st.dataframe(df_work[num_w].corr(), use_container_width=True)
        else:
            st.dataframe(df_work[num_w + [target]].corr(), use_container_width=True)
            fig_hm, corr, rec, high_corr = draw_correlation_heatmap(
                df_work, columns=num_w + [target], target=target
            )
            fig_to_st(fig_hm)
            st.markdown("### ⭐ Recommended Features")
            if rec is not None:
                st.dataframe(rec, use_container_width=True)
            st.markdown("### ⚠️ Highly Correlated Feature Pairs")
            if not high_corr.empty:
                st.dataframe(high_corr, use_container_width=True)
            else:
                st.success("No severe multicollinearity detected.")

    # ── Tab 9: Categorical ────────────────────────────────────────────────
    with tab_cat:
        warn_df = categorical_warnings(df_work)
        if warn_df.empty:
            st.success("✅ No major categorical issues detected.")
        else:
            st.dataframe(warn_df, use_container_width=True)
        with st.expander("📊 Class Imbalance Details"):
            imb_df = categorical_imbalance(df_work)
            if imb_df.empty:
                st.info("No categorical columns found.")
            else:
                st.dataframe(imb_df, use_container_width=True)

    st.info("👉 Proceed to **⚙️ Preprocessing** when ready.")