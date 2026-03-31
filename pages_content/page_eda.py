"""
pages_content/page_eda.py
Exploratory Data Analysis — fully decoupled from preprocessing.
Reuses components and core/viz/eda.py functions.
"""

import streamlit as st
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import StandardScaler
from state.session import init_state, get_value, set_state
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
import pandas as pd

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

    # EDA is read-only — do NOT persist selections to shared state.
    # Column selections here are local to EDA only.

    if not features or target is None:
        st.info("Select at least one feature and a target column above.")
        st.stop()

    selected_cols = features + [target]
    df_sub = df[selected_cols].copy()

    # ── EDA tabs ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔎 EDA Dashboard")
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8  = st.tabs(
        ["🔎 Preview", "Data Filtering", "📝 Summary", "📊 Scatter + Histograms", "📊 3D Scatter plots", "📦 Box plots", "🔗 Correlation", "⚠️ Categorical" ]
    )

    # ── Tab 1: Preview ────────────────────────────────────────────────────
    with tab1:
        st.write(f"**Shape:** {df_sub.shape[0]:,} rows × {df_sub.shape[1]} cols")
        st.dataframe(df_sub, use_container_width=True)

    # ── Tab 2: Data Filtering ────────────────────────────────────────────────────
    with tab2:
        st.subheader("🔍 Optional Filtering Before Cleaning")
        # Normalize column names for case-insensitive matching
        df_cols_lower = {col.lower(): col for col in df_sub.columns}
        # 🔥 Smart column finder
        def find_column(df_cols_lower, keywords):
            for col_lower, original_col in df_cols_lower.items():
                if all(k in col_lower for k in keywords):
                    return original_col
            return None
        # 🔥 Define filters with fallback logic
        filter_cols = {
            "Concentrate Ratio": find_column(df_cols_lower, ["concentrate", "ratio"]),
            "Dilution Ratio": find_column(df_cols_lower, ["dilution", "ratio"]),
            "Brine Type": find_column(df_cols_lower, ["brine", "type"]),
            "Brine Name": find_column(df_cols_lower, ["brine", "name"]),
        }

        for label, actual_col in filter_cols.items():

            if actual_col is None:
                #st.warning(f"⚠️ No column found for **{label}**")
                continue

            # 🔥 Inform user if fallback happened
            if label.lower() not in actual_col.lower():
                st.info(f"ℹ️ Using detected column for {label}: **{actual_col}**")

            unique_vals = df_sub[actual_col].dropna().unique()

            if len(unique_vals) > 1:
                st.markdown(f"**Filter by {actual_col}:**")

                selected_vals = st.multiselect(
                    f"Choose {actual_col} values to keep",
                    options=unique_vals,
                    default=list(unique_vals),
                    key=f"eda_filter_{actual_col}"   # eda-namespaced key — never collides with preprocessing
                )

                df_sub = df_sub[df_sub[actual_col].isin(selected_vals)]

                st.info(
                    f"✅ Filtered {actual_col}: {len(selected_vals)} values selected → {len(df_sub)} rows remain."
                )

            else:
                st.warning(f"⚠️ {actual_col} has only one unique value — no filtering applied.")

        # Update after filtering
        st.success(f"✅ Data filtered successfully. Original Shape: {df.shape} - Current shape: {df_sub.shape}")


    # ── Tab 3: Statistical summary ────────────────────────────────────────
    with tab3:
        st.markdown("#### 🔢 Numeric Summary")
        st.dataframe(extended_describe(df_sub), use_container_width=True)

        cat_df = categorical_summary(df_sub)
        if not cat_df.empty:
            st.markdown("#### 🏷️ Categorical Summary")
            st.dataframe(cat_df, use_container_width=True)

    # ── Tab 4: Scatter + Histograms ─────────────────────────────────────────────
    with tab4:
        num_cols = [c for c in features if pd.api.types.is_numeric_dtype(df_sub[c])]
        if not num_cols:
            st.warning("No numeric feature columns available for distribution plots.")
        else:

            st.markdown("#### 🔄 Scatter + Marginal Histograms")
            num_cols = [c for c in features if pd.api.types.is_numeric_dtype(df_sub[c])]
            if not num_cols:
                st.warning("No numeric feature columns available.")
            else:
                y = df_sub[target]  # use filtered df_sub to keep index aligned
                unique_classes = y.dropna().unique()
                n_classes = len(unique_classes)
                # =========================
                # CASE 1: Regression
                # =========================
                if pd.api.types.is_numeric_dtype(y) and n_classes > 5:
                    st.success("Detected: Regression")
                    show_trendline = st.checkbox("Show R² trendline", value=True, key="scatter_trendline")
                    figs = draw_pairwise_scatter_with_hist(
                        df_sub[num_cols + [target]],
                        target,
                        show_trendline
                    )
                # =========================
                # CASE 2: Classification (2–3 classes)
                # =========================
                elif n_classes <= 3:
                    
                    # 🔥 Convert target for plotting
                    y_encoded, mapping = pd.factorize(y)
                    df_plot = df_sub.copy()
                    df_plot[target] = y_encoded
                    st.info(f"Detected: Classification ({n_classes} classes) --- Class mapping: {dict(enumerate(mapping))}")
                    show_trendline = False
                    figs = draw_pairwise_scatter_with_hist(
                        df_plot[num_cols + [target]],
                        target,
                        show_trendline
                    )
                # =========================
                # CASE 3: Too many classes
                # =========================
                else:
                    st.warning(
                        f"⚠️ Target has {n_classes} classes. "
                        "Scatter plots are disabled for >3 classes."
                    )
                    figs = []
                if not figs:
                    st.info("Not enough data to render plots.")
                else:
                    for row_start in range(0, len(figs), 2):
                        cols = st.columns(2)
                        for col_idx, fig in enumerate(figs[row_start : row_start + 2]):
                            with cols[col_idx]:
                                fig_to_st(fig)

            st.divider()
            st.markdown("#### 📊 Histograms")
            bins = st.slider("Histogram bins", 5, 100, 30, key="eda_bins")
            fig = draw_histograms(df_sub, num_cols, bins=bins)
            fig_to_st(fig)
            st.divider()

            st.markdown("#### 🟢 Scatter Plot - All Features vs Target")
            # Checkbox option
            plot_all_one = st.checkbox("Show all features in one figure",value=True,key="eda_scatter_all_one")
            # --------------------------------------------------
            # OPTION 1 — Separate figures (your current style)
            # --------------------------------------------------
            if not plot_all_one:
                for feature in num_cols:
                    fig = draw_scatter(
                        df_sub,
                        x_col=feature,
                        y_col=target,
                        hue_col=None
                    )
                    fig_to_st(fig)
            # -------------------------------------------------
            # OPTION 2 — One image with subplots
            # --------------------------------------------------
            else:
                n = len(num_cols)
                cols = 3
                rows = math.ceil(n / cols)
                fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
                axes = axes.flatten()
                for i, feature in enumerate(num_cols):
                    ax = axes[i]
                    # Drop NaNs and coerce to numeric to avoid matplotlib
                    # category-axis errors when target contains mixed types
                    plot_df = df_sub[[feature, target]].dropna()
                    x_vals = pd.to_numeric(plot_df[feature], errors="coerce")
                    y_vals = pd.to_numeric(plot_df[target],  errors="coerce")
                    valid  = x_vals.notna() & y_vals.notna()
                    ax.scatter(x_vals[valid], y_vals[valid], alpha=0.7)
                    ax.set_xlabel(feature)
                    ax.set_ylabel(target)
                    ax.set_title(f"{feature} vs {target}")
                # Remove empty axes
                for j in range(i+1, len(axes)):
                    fig.delaxes(axes[j])
                plt.tight_layout()
                fig_to_st(fig)


    # ── Tab 5: 3D Scatter ───────────────────────────────────────────────
    with tab5:
        st.markdown("#### 📊 3D Scatter Plotsddddd")
        if len(num_cols) >= 2:
            c1, c2, c3 = st.columns(3)
            # X axis
            x_col = c1.selectbox("X axis",num_cols,key="eda_3d_x")
            # Y axis
            y_col = c2.selectbox("Y axis",num_cols,index=1,key="eda_3d_y")
            # Color column (default = target)
            color_options = ["None"] + df_sub.columns.tolist()
            default_index = color_options.index(target) if target in color_options else 0
            hue_col = c3.selectbox("Color by",color_options,index=default_index,key="eda_3d_color")
            if hue_col == "None":
                hue_col = None
            # 🔥 FIX: ensure column names are strings
            fig = draw_scatter(df_sub, x_col, y_col, hue_col=hue_col)
            fig_to_st(fig)
 
    # ── Tab 6: Boxplots ───────────────────────────────────────────────
    with tab6:
        st.markdown("#### 📦 Boxplots")
        use_standardize = st.checkbox("Standardize features for visualization (recommended when scales differ)",value=False)
        df_plot = df_sub.copy()
        if use_standardize:
            scaler = StandardScaler()
            df_plot[num_cols] = scaler.fit_transform(df_sub[num_cols])
        fig = draw_boxplots(df_plot, num_cols)
        fig_to_st(fig)
        st.divider()

    # ── Tab 7: Correlation ────────────────────────────────────────────────
    with tab7:
        st.markdown("### 🟣 Correlation Matrix")
        num_cols = [c for c in features if pd.api.types.is_numeric_dtype(df_sub[c])]
        if not num_cols:
            st.warning("No numeric features available for correlation analysis.")
        elif len(num_cols) > MAX_HEATMAP_FEATURES:
            st.warning(
                f"⚠️ Too many numeric features ({len(num_cols)}) — "
                f"heatmap limited to {MAX_HEATMAP_FEATURES}. "
                "Showing raw correlation table only."
            )
            st.dataframe(df_sub[num_cols].corr(), use_container_width=True)
        else:
            st.dataframe(df_sub[num_cols + [target]].corr(), use_container_width=True)
            fig, corr, rec, high_corr = draw_correlation_heatmap(
                df_sub,
                columns=num_cols + [target],
                target=target
            )
            fig_to_st(fig)
            st.markdown("### ⭐ Recommended Features")
            if rec is not None:
                st.dataframe(rec, use_container_width=True)
            st.markdown("### ⚠️ Highly Correlated Features")
            if not high_corr.empty:
                st.dataframe(high_corr, use_container_width=True)
            else:
                st.success("No severe multicollinearity detected.")

    # ── Tab 8: Categorical diagnostics ───────────────────────────────────
    with tab8:
        warn_df = categorical_warnings(df_sub)
        if warn_df.empty:
            st.success("✅ No major categorical issues detected.")
        else:
            st.dataframe(warn_df, use_container_width=True)

        with st.expander("📊 Class Imbalance Details"):
            imb_df = categorical_imbalance(df_sub)
            if imb_df.empty:
                st.info("No categorical columns found.")
            else:
                st.dataframe(imb_df, use_container_width=True)

    # EDA is purely read-only — no shared state is written here.
    st.info("👉 Proceed to **⚙️ Preprocessing** when ready.")