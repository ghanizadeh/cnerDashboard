"""
pages_content/page_preprocessing.py
Feature & target selection, missing values, outliers, encoding & scaling.
Train/Test split has been moved to Model → Train → Validation Method.
All math lives in core/data/preprocessor.py.

Data isolation design
----------------------
- EDA page is fully read-only; it never writes to shared session state.
- Preprocessing keeps its own working_data_base (raw slice) and builds
  the pipeline state fresh every rerun from base + active filter widgets.
- Shared state keys (data.feature_names, data.target_name, data.X, etc.)
  are written ONLY when the user clicks "Apply Preprocessing & Save".
  This guarantees EDA always works on the original raw data and is never
  affected by any preprocessing action.
"""

import pandas as pd
import streamlit as st
from core.data.feature_engineering import (
    FeatureEngineeringConfig,
    apply_feature_engineering,
)
from utils.feature_interaction import (
    render_surfactant_np_interactions,
    render_custom_ratio_features,
)

from state.session import init_state, get_value, set_state
from components.column_selector import render_column_selector
from core.data.preprocessor import (
    impute_missing, detect_outliers, remove_outliers,
    encode_categoricals, scale_features,
    extended_describe, categorical_summary,
    categorical_warnings, categorical_imbalance,
)
from core.viz.eda import draw_boxplots, draw_correlation_heatmap
from core.viz.style import fig_to_st
from config.settings import (
    DEFAULT_IQR_FACTOR, DEFAULT_ZSCORE_THRESHOLD,
    MAX_HEATMAP_FEATURES,
)


def render():
    init_state()

    st.title("⚙️ Data Preprocessing")
    st.divider()

    # ── Guard: need raw data ──────────────────────────────────────────────
    df = get_value("data.raw")
    if df is None:
        st.warning("⚠️ Please load data first — go to **📂 Data**.")
        st.stop()

    # ─────────────────────────────────────────────────────────────────────
    # 1. Feature & Target Selection
    # ─────────────────────────────────────────────────────────────────────
    st.subheader("🎯 Feature & Target Selection")
    with st.expander("Select features and target", expanded=True):
        selected_features, selected_target = render_column_selector(
            df,
            default_features=get_value("data.feature_names"),
            default_target=get_value("data.target_name"),
            key_prefix="prep",
        )

    # NOTE: We do NOT write feature_names / target_name to shared state here.
    # They are persisted only on "Apply Preprocessing & Save".
    # This prevents EDA from inheriting preprocessing's column selection mid-session.

    if not selected_features or selected_target is None:
        st.info("Select at least one feature and a target column above, then continue.")
        st.stop()

    # ─────────────────────────────────────────────────────────────────────
    # 2. Working-data management
    #
    # working_data_base  — clean slice from raw; reset only when column
    #                      selection changes. Never mutated directly.
    # prep_confirmed_*   — flags set when the user confirms a destructive
    #                      step (missing imputation, outlier removal).
    #
    # Every rerun we rebuild `data` from scratch:
    #   base  →  filter widgets  →  confirmed missing step  →  confirmed outlier step
    #
    # This means filter changes are always live, and confirmed steps stack
    # on top in a deterministic order without double-application.
    # ─────────────────────────────────────────────────────────────────────
    current_selection_key = tuple(sorted(selected_features)) + (selected_target,)
    prev_selection_key    = st.session_state.get("prep_selection_key")
    selection_changed     = current_selection_key != prev_selection_key

    if selection_changed or "working_data_base" not in st.session_state:
        st.session_state.working_data_base    = df[selected_features + [selected_target]].copy()
        st.session_state.prep_selection_key   = current_selection_key
        # Reset confirmed steps when columns change
        st.session_state.prep_confirmed_missing = False
        st.session_state.prep_confirmed_outlier = False
        st.session_state.pop("prep_outlier_params", None)

    # ─── Rebuild pipeline data for this rerun ───────────────────────────
    data = st.session_state.working_data_base.copy()   # Step 0: clean base

    # (Filter step applied inside Tab 1 and returned via a local variable;
    #  we thread `data` through the tab rendering below.)

    st.divider()
    st.subheader("⚙️ Preprocessing Steps")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔂 Data Filtering",
        "🛠️ Feature Engineering",
        "🚫 Missing Value Handling",
        "⚠️ Outlier Removal",
        "🔤 Categorical Encoding",
        "📏 Feature Scaling",
    ])

    # ── Tab 1: Data Filtering ─────────────────────────────────────────────
    with tab1:
        st.subheader("🔍 Optional Data Filtering")
        st.caption(
            "Filters here define which rows go to the model. "
            "EDA always shows the full raw dataset regardless."
        )

        df_cols_lower = {col.lower(): col for col in data.columns}

        def find_column(lookup, keywords):
            for col_lower, original_col in lookup.items():
                if all(k in col_lower for k in keywords):
                    return original_col
            return None

        filter_cols = {
            "Concentrate Ratio": find_column(df_cols_lower, ["concentrate", "ratio"]),
            "Dilution Ratio":    find_column(df_cols_lower, ["dilution", "ratio"]),
            "Brine Type":        find_column(df_cols_lower, ["brine", "type"]),
            "Brine Name":        find_column(df_cols_lower, ["brine", "name"]),
        }

        for label, actual_col in filter_cols.items():
            if actual_col is None:
                continue
            if label.lower() not in actual_col.lower():
                st.info(f"ℹ️ Using detected column for {label}: **{actual_col}**")

            # Always read options from the base (unfiltered) so deselected
            # values can be added back without a page reload.
            unique_vals = st.session_state.working_data_base[actual_col].dropna().unique()

            if len(unique_vals) > 1:
                st.markdown(f"**Filter by {actual_col}:**")
                selected_vals = st.multiselect(
                    f"Choose {actual_col} values to keep",
                    options=unique_vals,
                    default=list(unique_vals),
                    key=f"prep_filter_{actual_col}",  # prep-namespaced — never collides with EDA
                )
                data = data[data[actual_col].isin(selected_vals)]
                st.info(
                    f"✅ Filtered {actual_col}: {len(selected_vals)} values selected "
                    f"→ {len(data):,} rows remain."
                )
            else:
                st.warning(f"⚠️ {actual_col} has only one unique value — no filtering applied.")

        st.success(
            f"✅ Active filter result — Raw: {df.shape[0]:,} rows "
            f"→ After filter: {len(data):,} rows"
        )

    # ─── Layer confirmed steps on top of the filtered data ───────────────
    # Step 2: confirmed missing-value imputation
    if st.session_state.get("prep_confirmed_missing") and "prep_missing_strategy" in st.session_state:
        data = impute_missing(
            data,
            strategy=st.session_state.prep_missing_strategy,
            columns=selected_features,
        )

    # Step 3: confirmed outlier removal
    if st.session_state.get("prep_confirmed_outlier") and "prep_outlier_params" in st.session_state:
        p = st.session_state.prep_outlier_params
        num_cols_out = [c for c in selected_features if pd.api.types.is_numeric_dtype(data[c])]
        data, _ = remove_outliers(
            data, num_cols_out,
            method=p["method"],
            iqr_factor=p["iqr_factor"],
            z_thresh=p["z_thresh"],
        )

    # ── Tab 2: Feature Engineering ────────────────────────────────────────
    with tab2:
        st.subheader("🛠️ Feature Engineering")
        data = render_surfactant_np_interactions(data)
        st.divider()
        data = render_custom_ratio_features(data)
        #st.info("Feature engineering panel — uncomment the helpers above to enable.")

    # ── Tab 3: Missing Value Handling ─────────────────────────────────────
    with tab3:
        st.subheader("🚫 Missing Value Handling")
        n_missing = data.isnull().sum().sum()

        if n_missing == 0:
            st.success("✅ No missing values detected in current data.")
        else:
            st.warning(f"⚠️ {n_missing} missing value(s) found.")
            mv_strategy = st.selectbox(
                "Strategy (numeric columns)",
                ["drop", "mean", "median", "mode"],
                key="prep_missing_strategy_widget",
            )
            if st.button("Apply missing-value strategy"):
                st.session_state.prep_missing_strategy  = mv_strategy
                st.session_state.prep_confirmed_missing = True
                st.rerun()

        if st.session_state.get("prep_confirmed_missing"):
            strat = st.session_state.get("prep_missing_strategy", "—")
            st.info(f"✔ Missing-value strategy **{strat}** is active.")
            if st.button("↩ Reset missing-value step"):
                st.session_state.prep_confirmed_missing = False
                st.session_state.pop("prep_missing_strategy", None)
                st.rerun()

    # ── Tab 4: Outlier Detection & Removal ────────────────────────────────
    with tab4:
        st.subheader("🧹 Outlier Detection & Removal")
        num_cols = [c for c in selected_features if pd.api.types.is_numeric_dtype(data[c])]
        if not num_cols:
            st.info("No numeric feature columns — skipping outlier detection.")
        else:
            method_rem = st.selectbox(
                "Detection and Removal Method",
                ["None", "IQR", "Z-score"],
                key="prep_outlier_method_widget",
            )
            if method_rem != "None":
                iqr_f = st.slider("IQR factor", 1.0, 3.0, DEFAULT_IQR_FACTOR, 0.1, key="prep_iqr")
                z_t   = st.slider("Z-score threshold", 2.0, 5.0, DEFAULT_ZSCORE_THRESHOLD, 0.1, key="prep_z")
                summary = detect_outliers(
                    data, num_cols, method=method_rem,
                    iqr_factor=iqr_f, z_thresh=z_t,
                )
                st.dataframe(summary, use_container_width=True)
                fig = draw_boxplots(data, num_cols)
                fig_to_st(fig, "Boxplots before removal")

                if st.button("🧹 Remove Outliers"):
                    cleaned, n_removed = remove_outliers(
                        data, num_cols, method=method_rem,
                        iqr_factor=iqr_f, z_thresh=z_t,
                    )
                    st.session_state.prep_outlier_params    = {
                        "method":     method_rem,
                        "iqr_factor": iqr_f,
                        "z_thresh":   z_t,
                    }
                    st.session_state.prep_confirmed_outlier = True
                    st.success(f"Removed {n_removed} outlier rows. Remaining: {len(cleaned):,}")
                    st.rerun()

            if st.session_state.get("prep_confirmed_outlier"):
                p = st.session_state.get("prep_outlier_params", {})
                st.info(
                    f"✔ Outlier removal active — method: **{p.get('method')}**, "
                    f"IQR factor: {p.get('iqr_factor')}, Z-thresh: {p.get('z_thresh')}"
                )
                if st.button("↩ Reset outlier step"):
                    st.session_state.prep_confirmed_outlier = False
                    st.session_state.pop("prep_outlier_params", None)
                    st.rerun()

    # ── Tab 5: Categorical Encoding ───────────────────────────────────────
    with tab5:
        st.subheader("🔤 Categorical Encoding")
        with st.expander("Encoding options", expanded=False):
            cat_cols = [
                c for c in selected_features
                if pd.api.types.is_object_dtype(data[c]) or str(data[c].dtype) == "category"
            ]
            if not cat_cols:
                st.success("✅ No categorical feature columns — encoding not needed.")
                encoding_strategy = "none"
            else:
                st.write(f"Categorical columns: **{cat_cols}**")
                encoding_strategy = st.selectbox(
                    "Encoding strategy",
                    ["onehot", "label", "ordinal"],
                    key="prep_encoding",
                )

    # ── Tab 6: Feature Scaling ────────────────────────────────────────────
    with tab6:
        st.subheader("📏 Feature Scaling")
        with st.expander("Scaling options", expanded=False):
            scaling_strategy = st.selectbox(
                "Scaling strategy",
                ["none", "standard", "minmax", "robust"],
                key="prep_scaling",
                help="Applied after train/test split to prevent data leakage.",
            )
            st.caption("Scaler is fit on training data only and applied to both sets.")

    # ─────────────────────────────────────────────────────────────────────
    # Live pipeline summary
    # ─────────────────────────────────────────────────────────────────────
    st.divider()
    with st.container(border=True):
        st.markdown("**📋 Current preprocessing pipeline state** (sent to model on Apply)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(data):,}", f"raw: {df.shape[0]:,}")
        c2.metric("Features selected", len(selected_features))
        c3.metric("Missing values remaining", int(data.isnull().sum().sum()))

    # ─────────────────────────────────────────────────────────────────────
    # Apply All & Save — writes to shared session state
    # ─────────────────────────────────────────────────────────────────────
    if st.button("✅ Apply Preprocessing & Save", type="primary", use_container_width=True):
        pipeline_data = data.copy()

        # Safety net: drop any remaining NaNs
        n_remaining = pipeline_data.isnull().sum().sum()
        if n_remaining > 0:
            st.warning(f"⚠️ {n_remaining} missing value(s) still present — auto-dropping rows.")
            pipeline_data = impute_missing(pipeline_data, strategy="drop", columns=selected_features)

        # Separate X / y
        X = pipeline_data.drop(columns=[selected_target]).copy()
        y = pipeline_data[selected_target].copy()

        # Encode categoricals
        encoder_map: dict = {}
        if encoding_strategy != "none":
            X, encoder_map = encode_categoricals(X, strategy=encoding_strategy)

        # Scale (fit on full X here; train page splits and applies the saved scaler)
        scaler = None
        if scaling_strategy != "none":
            from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
            scaler_map = {
                "standard": StandardScaler(),
                "minmax":   MinMaxScaler(),
                "robust":   RobustScaler(),
            }
            scaler = scaler_map[scaling_strategy]
            num_cols_scale = X.select_dtypes(include="number").columns.tolist()
            X[num_cols_scale] = scaler.fit_transform(X[num_cols_scale])

        # Infer task type
        task_type = (
            "regression"
            if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20
            else "classification"
        )

        missing_label = (
            st.session_state.get("prep_missing_strategy", "none")
            if st.session_state.get("prep_confirmed_missing") else "none"
        )
        outlier_label = (
            st.session_state.get("prep_outlier_params", {}).get("method", "none")
            if st.session_state.get("prep_confirmed_outlier") else "none"
        )

        # ── Write to shared state — the ONLY place this happens ──────────
        set_state("data.cleaned",                  pipeline_data)
        set_state("data.X",                        X)
        set_state("data.y",                        y)
        set_state("data.feature_names",            selected_features)      # written only here
        set_state("data.target_name",              selected_target)         # written only here
        set_state("data.processed_feature_names",  list(X.columns))
        set_state("preprocessing.missing_strategy", missing_label)
        set_state("preprocessing.outlier_method",   outlier_label)
        set_state("preprocessing.scaler",           scaler)
        set_state("preprocessing.encoder",          encoder_map)
        set_state("preprocessing.rows_before",      df.shape[0])
        set_state("preprocessing.rows_after",       len(pipeline_data))
        set_state("preprocessing.applied",          True)
        set_state("model.task_type",                task_type)
        set_state("preprocessing.feature_engineering_enabled",
                  st.session_state.get("fe_enabled", False))
        set_state("preprocessing.feature_engineering_meta",
                  st.session_state.get("fe_meta", {}))

        st.success("✅ Preprocessing complete and saved to session!")
        st.info(
            f"Rows: {df.shape[0]:,} → {len(pipeline_data):,} &nbsp;|&nbsp; "
            f"Features: {X.shape[1]} &nbsp;|&nbsp; "
            f"Task: **{task_type}**"
        )
        st.subheader("📊 Final Processed Data (preview)")
        st.dataframe(X.head(50), use_container_width=True)
        st.info("👉 Proceed to **🤖 Train** — choose your validation strategy there.")