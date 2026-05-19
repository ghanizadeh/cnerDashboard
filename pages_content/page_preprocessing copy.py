"""
pages_content/page_preprocessing.py
Sequential preprocessing pipeline — no tabs.

Sections (in order):
  0  Target Conversion   — keep regression OR bin to Low/High or Low/Mid/High
  1  Data Filtering      — brine/dilution/concentrate + dynamic custom rules
  2  Feature Engineering — foam chemical groups + auto interactions (foam_app style)
  3  Outlier Removal     — IQR / Z-score with live boxplot preview
  4  Scaling             — standard / minmax / robust (fit on full set; scaler stored for train page)
  5  Final Overview      — shape, class balance, feature matrix preview → Apply & Save

Data-isolation contract (unchanged from previous design):
  - EDA page reads only  data.raw  (never touched here).
  - Shared state keys are written ONLY when "Apply & Save" is clicked.
  - Working data is rebuilt from scratch every rerun from session state flags.
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from itertools import product as iter_product

from state.session import init_state, get_value, set_state
from components.column_selector import render_column_selector
from core.data.preprocessor import (
    impute_missing, detect_outliers, remove_outliers,
    encode_categoricals, scale_features,
)
from core.viz.eda import draw_boxplots
from core.viz.style import fig_to_st
from config.settings import DEFAULT_IQR_FACTOR, DEFAULT_ZSCORE_THRESHOLD
from utils.data_filter import render_data_filters
from core.data.foam_feature_engineering import (
    build_foam_features,
    render_feature_engineering_ui,
    GRP_CLR, CHEM_GROUPS,
    DEF_NANO, DEF_ANIONIC, DEF_NONION, DEF_ZW, DEF_POLY,
    DEF_CITRIC, DEF_ACID, DEF_ANTI, DEF_BRINE, DEF_OIL, DEF_PROCESS,
)




# ─────────────────────────────────────────────────────────────────────────────
#  Section renderers (pure UI — each returns updated `data` DataFrame)
# ─────────────────────────────────────────────────────────────────────────────

def _section_header(icon: str, number: int, title: str, subtitle: str = ""):
    st.markdown(f"### {icon} Step {number} — {title}")
    if subtitle:
        st.caption(subtitle)


def _render_target_conversion(df: pd.DataFrame, target_col: str) -> pd.Series:
    """
    Section 0: optionally bin a numeric target into 2 or 3 classes.
    Returns the (possibly transformed) target Series and writes
    task_mode to session state.
    """
    _section_header("🎯", 0, "Target Conversion",
                    "Keep as continuous regression OR bin to classification classes.")

    with st.container(border=True):
        y_orig = pd.to_numeric(df[target_col], errors="coerce").dropna()
        c1, c2 = st.columns([2, 3])

        mode = c1.radio(
            "Target mode",
            ["Regression (continuous)", "Binary: Low / High", "Ternary: Low / Mid / High"],
            key="tconv_mode",
        )

        y_out = pd.to_numeric(df[target_col], errors="coerce")

        if mode == "Regression (continuous)":
            c2.info(
                f"Target kept as **continuous** ({target_col}).\n\n"
                f"Range: {y_orig.min():.2f} – {y_orig.max():.2f}  |  "
                f"Median: {y_orig.median():.2f}"
            )
            st.session_state["tconv_task"] = "regression"

        elif mode == "Binary: Low / High":
            default_cut = float(y_orig.median())
            cut = c2.slider(
                "Cut-off  (Low < cut ≤ High)",
                min_value=float(y_orig.min()),
                max_value=float(y_orig.max()),
                value=default_cut,
                key="tconv_cut2",
            )
            y_out = (y_out > cut).map({False: "Low", True: "High"})
            counts = y_out.value_counts()
            c2.success(
                f"Low: **{counts.get('Low', 0)}** rows  |  "
                f"High: **{counts.get('High', 0)}** rows"
            )
            st.session_state["tconv_task"] = "classification"

        else:  # ternary
            lo_default = float(y_orig.quantile(0.33))
            hi_default = float(y_orig.quantile(0.67))
            rng = c2.slider(
                "Boundaries  (Low < lo ≤ Mid < hi ≤ High)",
                min_value=float(y_orig.min()),
                max_value=float(y_orig.max()),
                value=(lo_default, hi_default),
                key="tconv_cut3",
            )
            lo_cut, hi_cut = rng

            def _bin3(v):
                if pd.isna(v): return np.nan
                if v <= lo_cut:  return "Low"
                if v <= hi_cut:  return "Mid"
                return "High"

            y_out = y_out.apply(_bin3)
            counts = y_out.value_counts()
            c2.success(
                f"Low: **{counts.get('Low', 0)}**  |  "
                f"Mid: **{counts.get('Mid', 0)}**  |  "
                f"High: **{counts.get('High', 0)}**"
            )
            st.session_state["tconv_task"] = "classification"

        # Mini histogram of original target
        fig = px.histogram(
            x=y_orig, nbins=40, title="Original Target Distribution",
            labels={"x": target_col}, template="plotly_white",
            color_discrete_sequence=["#1565C0"],
        )
        fig.update_layout(height=260, margin=dict(t=35, b=10, l=10, r=10),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    return y_out


def _render_data_filtering(data: pd.DataFrame) -> pd.DataFrame:
    """Section 1: delegates to shared render_data_filters (utils/data_filter.py)."""
    _section_header("🔍", 1, "Data Filtering",
                    "Filter rows sent to the model. Each process condition has its own control. "
                    "Filters are non-destructive — adjust any time.")
    return render_data_filters(data, key_prefix="prep")


def _render_feature_engineering(
    df: pd.DataFrame,
    selected_features: list[str],
) -> tuple[pd.DataFrame, list[str], dict]:
    """
    Section 2 — delegates to shared render_feature_engineering_ui.
    key_prefix="prep" keeps widget keys unique from EDA page.
    """
    _section_header("⚗️", 2, "Feature Engineering",
                    "Select chemical groups — group totals, fractions and "
                    "interactions are built automatically.")
    df_out, feature_cols, fg = render_feature_engineering_ui(
        df, key_prefix="prep", header=False,
    )
    # Save meta for Apply step
    st.session_state["fe_meta"] = {
        "feature_cols": feature_cols,
        "feat_to_group": fg,
    }
    if not feature_cols:
        return df, selected_features, {}
    return df_out, feature_cols, fg



def _render_outlier_removal(data: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Section 3: outlier detection and removal with live preview."""
    _section_header("⚠️", 3, "Outlier Removal",
                    "Detect and remove outliers from numeric feature columns.")

    num_cols = [c for c in feature_cols if c in data.columns
                and pd.api.types.is_numeric_dtype(data[c])]

    with st.container(border=True):
        if not num_cols:
            st.info("No numeric feature columns — skipping.")
            return data

        method = st.selectbox("Detection method",
                               ["None", "IQR", "Z-score"],
                               key="out_method")

        if method == "None":
            if st.session_state.get("prep_confirmed_outlier"):
                st.info(f"✔ Outlier removal active — "
                        f"{st.session_state.get('prep_outlier_params', {}).get('method')}")
                if st.button("↩ Reset outlier step", key="out_reset"):
                    st.session_state.prep_confirmed_outlier = False
                    st.session_state.pop("prep_outlier_params", None)
                    st.rerun()
            else:
                st.success("No outlier removal selected.")
            return data

        oc1, oc2 = st.columns(2)
        iqr_f = oc1.slider("IQR factor",    1.0, 3.0, DEFAULT_IQR_FACTOR,      0.1, key="out_iqr")
        z_t   = oc2.slider("Z-score threshold", 2.0, 5.0, DEFAULT_ZSCORE_THRESHOLD, 0.1, key="out_z")

        summary = detect_outliers(data, num_cols, method=method,
                                   iqr_factor=iqr_f, z_thresh=z_t)
        has_outliers = summary["Outliers"].sum() > 0

        with st.expander("Outlier summary table", expanded=has_outliers):
            st.dataframe(summary.style.background_gradient(subset=["Outliers"], cmap="Reds"),
                         use_container_width=True)

        # Boxplots for top-5 most affected features
        top5 = summary.sort_values("Outliers", ascending=False).head(5)["Feature"].tolist()
        if top5:
            with st.expander("Boxplots (top 5 most affected features)", expanded=False):
                fig_bp = go.Figure()
                for col in top5:
                    fig_bp.add_trace(go.Box(y=data[col].dropna(), name=col, boxpoints="outliers"))
                fig_bp.update_layout(title="Distribution before removal",
                                     height=350, template="plotly_white")
                st.plotly_chart(fig_bp, use_container_width=True)

        if has_outliers:
            total_out = int(summary["Outliers"].sum())
            st.warning(f"{total_out} potential outlier rows detected across {(summary['Outliers']>0).sum()} features.")

        col_btn1, col_btn2 = st.columns(2)
        if col_btn1.button("🧹 Remove Outliers & Confirm", key="out_apply", type="primary"):
            cleaned, n_removed = remove_outliers(
                data, num_cols, method=method, iqr_factor=iqr_f, z_thresh=z_t,
            )
            st.session_state.prep_outlier_params    = {"method": method, "iqr_factor": iqr_f, "z_thresh": z_t}
            st.session_state.prep_confirmed_outlier = True
            st.success(f"✅ Removed **{n_removed}** outlier rows. Remaining: **{len(cleaned):,}**")
            st.rerun()

        if st.session_state.get("prep_confirmed_outlier"):
            p = st.session_state.get("prep_outlier_params", {})
            st.info(f"✔ Outlier removal active — method: **{p.get('method')}**, "
                    f"IQR: {p.get('iqr_factor')}, Z: {p.get('z_thresh')}")
            if col_btn2.button("↩ Reset outlier step", key="out_reset2"):
                st.session_state.prep_confirmed_outlier = False
                st.session_state.pop("prep_outlier_params", None)
                st.rerun()

    # Apply confirmed outlier removal
    if st.session_state.get("prep_confirmed_outlier") and "prep_outlier_params" in st.session_state:
        p = st.session_state.prep_outlier_params
        data, _ = remove_outliers(data, num_cols,
                                   method=p["method"],
                                   iqr_factor=p["iqr_factor"],
                                   z_thresh=p["z_thresh"])
    return data


def _render_scaling(feature_cols: list[str]) -> str:
    """Section 4: scaling strategy picker. Returns strategy string."""
    _section_header("📏", 4, "Feature Scaling",
                    "Scaler is stored and applied properly inside the Train page (fit on train, transform test). "
                    "Original feature values are preserved for plots.")

    with st.container(border=True):
        sc1, sc2 = st.columns([2, 3])
        strategy = sc1.selectbox(
            "Scaling strategy",
            ["none", "standard", "robust", "minmax"],
            index=1,
            key="prep_scaling",
            help=(
                "standard = zero mean, unit variance\n"
                "robust   = median / IQR (outlier-resistant)\n"
                "minmax   = scale to [0, 1]"
            ),
        )
        descriptions = {
            "none":     "No scaling applied.",
            "standard": "StandardScaler: subtract mean, divide std. Best for linear models.",
            "robust":   "RobustScaler: subtract median, divide IQR. Best when outliers remain.",
            "minmax":   "MinMaxScaler: scale to [0, 1]. Sensitive to outliers.",
        }
        sc2.info(descriptions[strategy])
        st.caption("⚠️ Scaling is done internally at train time. All plots and SHAP show **original scale** values.")
    return strategy


def _render_final_overview(
    X: pd.DataFrame,
    y: pd.Series,
    fg: dict,
    task: str,
) -> None:
    """Section 5: final summary before Apply."""
    _section_header("📋", 5, "Final Dataset Overview",
                    "Review the prepared dataset before saving to session.")

    with st.container(border=True):
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Rows", f"{len(X):,}")
        r2.metric("Features", X.shape[1])
        r3.metric("Task type", task)
        r4.metric("Missing values", int(X.isnull().sum().sum()))

        # Target distribution
        ct1, ct2 = st.columns(2)
        with ct1:
            if task == "regression":
                fig_y = px.histogram(x=y, nbins=40,
                                     title="Target Distribution",
                                     template="plotly_white",
                                     color_discrete_sequence=["#1565C0"])
                fig_y.update_layout(height=260, margin=dict(t=35,b=10,l=10,r=10),
                                    showlegend=False,
                                    xaxis_title=y.name or "Target")
                st.plotly_chart(fig_y, use_container_width=True)
            else:
                vc = y.value_counts()
                fig_y = px.bar(x=vc.index, y=vc.values,
                               color=vc.index, title="Class Distribution",
                               template="plotly_white",
                               labels={"x": "Class", "y": "Count"})
                fig_y.update_layout(height=260, showlegend=False,
                                    margin=dict(t=35,b=10,l=10,r=10))
                st.plotly_chart(fig_y, use_container_width=True)

        with ct2:
            # Group importance bar
            if fg:
                gc = pd.Series(fg).value_counts()
                fig_gc = px.bar(x=gc.index, y=gc.values,
                                color=gc.index, color_discrete_map=GRP_CLR,
                                title="Features per Group",
                                template="plotly_white",
                                labels={"x": "Group", "y": "# Features"})
                fig_gc.update_layout(height=260, showlegend=False,      
                                     margin=dict(t=35,b=10,l=10,r=10))
                st.plotly_chart(fig_gc, use_container_width=True)

        with st.expander("Feature matrix preview (first 20 rows)"):
            st.dataframe(X, use_container_width=True)

        # NaN report
        nan_cols = X.isnull().mean()
        nan_cols = nan_cols[nan_cols > 0].sort_values(ascending=False)
        if len(nan_cols):
            with st.expander(f"⚠️ {len(nan_cols)} features have remaining NaN"):
                st.dataframe(nan_cols.rename("NaN fraction").to_frame()
                              .style.background_gradient(cmap="Oranges"),
                             use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Main render
# ─────────────────────────────────────────────────────────────────────────────

def render():
    init_state()

    st.title("⚙️ Preprocessing & Feature Engineering")
    st.caption(
        "Work through each step sequentially. "
        "Changes only take effect when you click **✅ Apply & Save** at the bottom."
    )
    st.divider()

    # ── Guard ─────────────────────────────────────────────────────────────
    df_raw = get_value("data.raw")
    if df_raw is None:
        st.warning("⚠️ Load data first — go to **📂 Data**.")
        st.stop()

    # ── Feature / Target selection ────────────────────────────────────────
    st.markdown("### 📌 Feature & Target Selection")
    with st.expander("Select features and target column", expanded=True):
        selected_features, selected_target = render_column_selector(
            df_raw,
            default_features=get_value("data.feature_names"),
            default_target=get_value("data.target_name"),
            key_prefix="prep",
        )

    if not selected_features or selected_target is None:
        st.info("Select at least one feature and a target column above, then continue.")
        st.stop()

    # ── Working data management ───────────────────────────────────────────
    sel_key = tuple(sorted(selected_features)) + (selected_target,)
    if sel_key != st.session_state.get("prep_selection_key") or "working_data_base" not in st.session_state:
        st.session_state.working_data_base      = df_raw[selected_features + [selected_target]].copy()
        st.session_state.prep_selection_key     = sel_key
        st.session_state.prep_confirmed_outlier = False
        st.session_state.pop("prep_outlier_params", None)

    data = st.session_state.working_data_base.copy()

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 0 — Target Conversion
    # ══════════════════════════════════════════════════════════════════════
    y_converted = _render_target_conversion(data, selected_target)
    task_mode   = st.session_state.get("tconv_task", "regression")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 1 — Data Filtering
    # ══════════════════════════════════════════════════════════════════════
    data = _render_data_filtering(data)

    # Align y to filtered rows
    y_converted = y_converted.loc[data.index].reset_index(drop=True)
    data        = data.reset_index(drop=True)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 2 — Feature Engineering
    # ══════════════════════════════════════════════════════════════════════
    data, feature_cols, fg = _render_feature_engineering(data, selected_features)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 3 — Outlier Removal
    # ══════════════════════════════════════════════════════════════════════
    data = _render_outlier_removal(data, feature_cols)

    # Re-align y after outlier removal
    y_converted = y_converted.loc[data.index].reset_index(drop=True)
    data        = data.reset_index(drop=True)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 4 — Scaling
    # ══════════════════════════════════════════════════════════════════════
    scaling_strategy = _render_scaling(feature_cols)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 5 — Final Overview
    # ══════════════════════════════════════════════════════════════════════
    # Build X from feature_cols only (drop target if present)
    # ── Include BOTH original selected features + engineered features ──

    engineered_cols = [
        c for c in feature_cols
        if c in data.columns and c != selected_target
    ]

    original_cols = [
        c for c in selected_features
        if c in data.columns and c != selected_target
    ]

    # Combine while preserving order and removing duplicates
    #feat_cols_final = list(dict.fromkeys(original_cols + engineered_cols))
    feat_cols_final = list(dict.fromkeys(engineered_cols))

    X_preview = data[feat_cols_final].copy()

    _render_final_overview(X_preview, y_converted, fg, task_mode)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 6 — Unscaled Data Preview
    # ══════════════════════════════════════════════════════════════════════
    _section_header("🔬", 6, "Unscaled Data Preview",
                    "Inspect the final dataset in original units before any scaling. "
                    "Use this to verify filters, engineered features and outlier removal are correct.")

    with st.container(border=True):

        # ── Pipeline row-count summary ────────────────────────────────────
        raw_n     = len(df_raw)
        curr_n    = len(X_preview)
        removed_n = raw_n - curr_n

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Raw rows",     f"{raw_n:,}")
        m2.metric("Current rows", f"{curr_n:,}",
                  delta=f"-{removed_n:,} removed",
                  delta_color="off" if removed_n == 0 else "inverse")
        m3.metric("Features",     X_preview.shape[1])
        m4.metric("Target",       selected_target)

        st.divider()

        # ── Column picker ─────────────────────────────────────────────────
        all_prev_cols = X_preview.columns.tolist()
        sel_prev_cols = st.multiselect(
            "Columns to display",
            options=all_prev_cols,
            default=all_prev_cols[:min(15, len(all_prev_cols))],
            key="preview_cols",
            help="Select which columns to inspect. Defaults to first 15.",
        )
        show_tgt = st.checkbox("Include target column", value=True, key="preview_show_target")

        display_df = X_preview[sel_prev_cols].copy()
        if show_tgt:
            display_df.insert(0, selected_target, y_converted.values)

        # ── Quick value filter ────────────────────────────────────────────
        qf1, qf2 = st.columns([2, 3])
        search_feature = qf1.selectbox(
            "Quick-filter by column",
            options=["(none)"] + sel_prev_cols,
            key="preview_search_col",
        )
        if search_feature != "(none)":
            col_vals = X_preview[search_feature]
            if pd.api.types.is_numeric_dtype(col_vals):
                lo, hi = float(col_vals.min()), float(col_vals.max())
                if lo < hi:
                    rng = qf2.slider(
                        f"{search_feature} range",
                        lo, hi, (lo, hi), key="preview_num_range",
                    )
                    display_df = display_df[
                        (col_vals >= rng[0]) & (col_vals <= rng[1])
                    ]
            else:
                opts  = col_vals.dropna().unique().tolist()
                picks = qf2.multiselect(
                    f"{search_feature} values",
                    options=opts, default=opts, key="preview_cat_vals",
                )
                display_df = display_df[col_vals.isin(picks)]

        st.caption(
            f"Showing **{len(display_df):,}** rows × **{len(display_df.columns)}** columns "
            f"— original scale, no scaling applied."
        )
        st.dataframe(display_df, use_container_width=True, height=420)

        # ── Descriptive stats ─────────────────────────────────────────────
        with st.expander("📈 Descriptive statistics (unscaled)"):
            num_prev = display_df.select_dtypes(include="number").columns.tolist()
            if num_prev:
                desc = display_df[num_prev].describe().T
                desc["missing %"] = (
                    display_df[num_prev].isnull().mean() * 100
                ).round(2)
                st.dataframe(
                    desc.style
                        .format("{:.4f}", na_rep="—")
                        .background_gradient(subset=["missing %"], cmap="Oranges"),
                    use_container_width=True,
                )
            else:
                st.info("No numeric columns in current selection.")

        # ── Distribution spot-check ───────────────────────────────────────
        with st.expander("📊 Distribution spot-check"):
            num_opts = [c for c in sel_prev_cols
                        if pd.api.types.is_numeric_dtype(X_preview[c])]
            cat_opts = ["(none)"] + [
                c for c in display_df.columns
                if not pd.api.types.is_numeric_dtype(display_df[c])
            ]
            sp1, sp2 = st.columns([2, 3])
            if num_opts:
                spot_col   = sp1.selectbox("Feature to plot", num_opts, key="preview_spot_col")
                color_col  = sp2.selectbox("Colour by",       cat_opts, key="preview_spot_color")
                import plotly.express as _px
                if color_col == "(none)":
                    fig_sp = _px.histogram(
                        display_df, x=spot_col, nbins=40,
                        title=f"Distribution of  {spot_col}  (original scale)",
                        template="plotly_white",
                        color_discrete_sequence=["#1565C0"],
                    )
                else:
                    fig_sp = _px.histogram(
                        display_df, x=spot_col, color=color_col,
                        nbins=40, barmode="overlay", opacity=0.7,
                        title=f"{spot_col}  by  {color_col}",
                        template="plotly_white",
                    )
                fig_sp.update_layout(height=300, margin=dict(t=40, b=10, l=10, r=10))
                st.plotly_chart(fig_sp, use_container_width=True)
            else:
                st.info("No numeric columns selected.")

        # ── Download ──────────────────────────────────────────────────────
        st.download_button(
            "⬇️ Download this view as CSV",
            data=display_df.to_csv(index=False).encode(),
            file_name="unscaled_preview.csv",
            mime="text/csv",
        )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # APPLY & SAVE
    # ══════════════════════════════════════════════════════════════════════
    if st.button("✅ Apply Preprocessing & Save", type="primary", use_container_width=True):

        pipeline_data = data[feat_cols_final + [selected_target]].copy()
        pipeline_data[selected_target] = y_converted.values

        # Safety net: drop any remaining NaN rows
        n_before_drop = len(pipeline_data)
        pipeline_data = pipeline_data.dropna()
        if len(pipeline_data) < n_before_drop:
            st.warning(f"Auto-dropped {n_before_drop - len(pipeline_data)} rows with remaining NaN.")

        X = pipeline_data[feat_cols_final].copy()
        y = pipeline_data[selected_target].copy()

        # Encode categoricals if any slipped through
        cat_cols = X.select_dtypes(include="object").columns.tolist()
        encoder_map = {}
        if cat_cols:
            X, encoder_map = encode_categoricals(X, strategy="label")

        # Build and store scaler (fit on full X here; train page applies properly)
        scaler = None
        if scaling_strategy != "none":
            from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
            scaler = {"standard": StandardScaler(),
                      "minmax":   MinMaxScaler(),
                      "robust":   RobustScaler()}[scaling_strategy]
            num_sc = X.select_dtypes(include="number").columns.tolist()
            # Store original X (unscaled) for plots/SHAP
            X_original = X.copy()
            X[num_sc] = scaler.fit_transform(X[num_sc])
        else:
            X_original = X.copy()

        # Infer task type from converted target
        task_type = (
            "regression"
            if task_mode == "regression" and pd.api.types.is_numeric_dtype(y) and y.nunique() > 20
            else "classification" if task_mode == "classification"
            else "regression"
        )

        # ── Write to shared session state ─────────────────────────────
        set_state("data.cleaned",                   pipeline_data)
        set_state("data.X",                         X)
        set_state("data.X_original",                X_original)   # unscaled for plots
        set_state("data.y",                         y)
        set_state("data.feature_names",             selected_features)
        set_state("data.target_name",               selected_target)
        set_state("data.processed_feature_names",   list(X.columns))
        set_state("preprocessing.missing_strategy", "none")
        set_state("preprocessing.outlier_method",
                  st.session_state.get("prep_outlier_params", {}).get("method", "none")
                  if st.session_state.get("prep_confirmed_outlier") else "none")
        set_state("preprocessing.scaler",           scaler)
        set_state("preprocessing.scaler_strategy",  scaling_strategy)
        set_state("preprocessing.encoder",          encoder_map)
        set_state("preprocessing.rows_before",      df_raw.shape[0])
        set_state("preprocessing.rows_after",       len(pipeline_data))
        set_state("preprocessing.applied",          True)
        set_state("model.task_type",                task_type)
        set_state("preprocessing.feature_engineering_enabled", st.session_state.get("fe_enabled", False))
        set_state("preprocessing.feature_engineering_meta",    st.session_state.get("fe_meta", {}))
        set_state("preprocessing.feat_to_group",               fg)

        st.success("✅ Preprocessing complete and saved to session!")
        st.info(
            f"Rows: {df_raw.shape[0]:,} → {len(pipeline_data):,}  |  "
            f"Features: {X.shape[1]}  |  "
            f"Task: **{task_type}**  |  "
            f"Scaling: **{scaling_strategy}**"
        )

        with st.expander("📊 Final processed data preview (first 50 rows)", expanded=True):
            st.dataframe(X.head(50), use_container_width=True)

        st.info("👉 Proceed to **🤖 Train** to choose your validation strategy and run the model.")