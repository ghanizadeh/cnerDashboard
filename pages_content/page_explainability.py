"""
pages_content/page_explainability.py
=====================================
🧠 Explainability Tab — SHAP, PDP, Surrogate Rules.

Ported from code0/tabs/explainability.py into the refactored architecture.
No changes to SHAP logic — only re-wired to use the refactored session state.
"""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from state.session import init_state, get_value
from core.models.explainability import (
    get_shap_values,
    shap_importance_df,
    plot_shap_beeswarm,
    plot_shap_dependence,
    plot_shap_dependence_2d,
    plot_pdp_1d,
    plot_pdp_2d,
    extract_rules,
    MAX_SHAP_SAMPLES,
    SURROGATE_MAX_DEPTH,
    SURROGATE_DEPTH_RANGE,
)
from core.viz.style import fig_to_st
from config.settings import RANDOM_STATE


def render():
    init_state()

    st.title("🧠 Explainability")
    st.divider()

    # ── Guards ────────────────────────────────────────────────────────────
    model      = get_value("model.object")
    X_train    = get_value("split.X_train")
    y_train    = get_value("split.y_train")
    task_type  = get_value("model.task_type")

    if model is None or X_train is None:
        st.warning("⚠️ No trained model found. Go to **🤖 Train** first.")
        st.stop()

    # Always derive feature names from X_train columns — these are guaranteed
    # to match shap_vals dimensions. data.processed_feature_names may differ
    # after encoding and would cause index() lookups to fail silently.
    feature_names = list(X_train.columns)

    class_names = []
    if task_type == "classification":
        y = get_value("data.y")
        if y is not None:
            class_names = [str(c) for c in sorted(y.unique())]

    # ── Class selector (classification only) ─────────────────────────────
    class_idx = 0
    if task_type == "classification" and class_names:
        class_idx = st.selectbox(
            "Class for SHAP / PDP",
            options=list(range(len(class_names))),
            format_func=lambda i: f"{i} – {class_names[i]}",
        )

    # ── Subsample for SHAP ────────────────────────────────────────────────
    sample_n = min(MAX_SHAP_SAMPLES, len(X_train))
    X_shap = (
        X_train.sample(sample_n, random_state=RANDOM_STATE)
        if len(X_train) > sample_n
        else X_train.copy()
    )

    # ─────────────────────────────────────────────────────────────────────
    # SHAP
    # ─────────────────────────────────────────────────────────────────────
    st.subheader("🐝 SHAP Analysis")

    shap_key = f"shap_{id(model)}_{class_idx}"
    if shap_key not in st.session_state:
        with st.spinner("Computing SHAP values…"):
            try:
                st.session_state[shap_key] = get_shap_values(model, X_shap, class_idx)
            except Exception as exc:
                st.warning(f"SHAP computation failed: {exc}")
                st.session_state[shap_key] = None

    shap_vals = st.session_state[shap_key]

    if shap_vals is not None:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("**Beeswarm (global impact)**")
            try:
                st.pyplot(plot_shap_beeswarm(shap_vals, X_shap))
            except Exception as exc:
                st.warning(f"Beeswarm plot failed: {exc}")

        with col2:
            st.markdown("**Mean |SHAP| bar**")
            imp_df = shap_importance_df(shap_vals, list(X_shap.columns))
            fig = px.bar(
                imp_df.sort_values("MeanAbsSHAP", ascending=True),
                x="MeanAbsSHAP",
                y="Feature",
                orientation="h",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Dependence plot (LOWESS trend)**")
        dep_feat = st.selectbox("Feature for SHAP dependence", feature_names, key="shap_dep")
        try:
            fig = plot_shap_dependence(shap_vals, X_shap, dep_feat)
            fig_to_st(fig)
        except Exception as exc:
            st.warning(f"Dependence plot failed: {exc}")

        st.markdown("**Dependence plot (interaction coloring)**")
        dep_feat_int = st.selectbox("X-axis feature", feature_names, key="shap_dep_int")
        int_options = ["Auto"] + [f for f in feature_names if f != dep_feat_int]
        int_feat = st.selectbox(
            "Interaction (color) feature — Auto lets SHAP choose",
            int_options, key="shap_dep_color"
        )
        try:
            import matplotlib.pyplot as plt
            import shap as _shap

            fig_int, ax_int = plt.subplots(figsize=(9, 5))
            interaction_index = None if int_feat == "Auto" else int_feat
            _shap.dependence_plot(
                dep_feat_int,
                shap_vals,
                X_shap,
                interaction_index=interaction_index,
                ax=ax_int,
                show=False,
            )
            fig_int.tight_layout()
            st.pyplot(fig_int)
            plt.close(fig_int)
        except Exception as exc:
            st.warning(f"Interaction dependence plot failed: {exc}")

        # ── 2D SHAP dependence plot (bars + coloured scatter) ─────────────
        st.markdown("**2D SHAP Dependence — bin means + interaction colouring**")
        st.caption(
            "Grey bars = mean SHAP per feature-value bin. "
            "Dots = individual samples coloured by the interaction feature. "
            "Matches the reference chart style with a colour bar on the right."
        )
        c1, c2 = st.columns(2)
        with c1:
            feat_2d = st.selectbox(
                "Select X-axis feature", feature_names, key="shap_2d_x"
            )
        with c2:
            int_options_2d = ["Auto"] + [f for f in feature_names if f != feat_2d]
            int_feat_2d = st.selectbox(
                "Select interaction (color) feature (optional)",
                int_options_2d, key="shap_2d_color"
            )
        n_bins_2d = st.slider("Number of X bins", 5, 20, 10, key="shap_2d_bins")
        try:
            fig_2d = plot_shap_dependence_2d(
                shap_vals,
                X_shap,
                feature=feat_2d,
                interaction_feature=None if int_feat_2d == "Auto" else int_feat_2d,
                n_bins=n_bins_2d,
            )
            st.pyplot(fig_2d)
            import matplotlib.pyplot as _plt
            _plt.close(fig_2d)
        except Exception as exc:
            st.error(f"2D SHAP dependence plot failed: {exc}")
            st.exception(exc)

    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # Partial Dependence Plots
    # ─────────────────────────────────────────────────────────────────────
    st.subheader("📉 PDP - Partial Dependence Plots")

    pdp_feat = st.selectbox("Feature for 1D PDP", feature_names, key="pdp_1d")
    try:
        fig = plot_pdp_1d(
            model, X_train, pdp_feat,
            class_idx if task_type == "classification" else None,
            task_type,
        )
        fig_to_st(fig)
    except Exception as exc:
        st.warning(f"1D PDP failed: {exc}")

    if len(feature_names) >= 2:
        st.markdown("**2D PDP (interaction surface)**")
        c1, c2 = st.columns(2)
        with c1:
            f1 = st.selectbox("Feature 1", feature_names, key="pdp_2d_f1")
        with c2:
            f2_options = [f for f in feature_names if f != f1]
            f2 = st.selectbox("Feature 2", f2_options, key="pdp_2d_f2")
        try:
            fig = plot_pdp_2d(
                model, X_train, f1, f2,
                class_idx if task_type == "classification" else None,
                task_type,
            )
            fig_to_st(fig)
        except Exception as exc:
            st.warning(f"2D PDP failed: {exc}")

    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # Surrogate Rules
    # ─────────────────────────────────────────────────────────────────────
    st.subheader("📜 Surrogate Rule Extraction")

    depth = st.slider(
        "Rule tree max depth",
        SURROGATE_DEPTH_RANGE[0],
        SURROGATE_DEPTH_RANGE[1],
        SURROGATE_MAX_DEPTH,
        key="rule_depth",
    )

    _, rules_text = extract_rules(
        X_train, y_train, feature_names, class_names, task_type, depth
    )
    st.code(rules_text)
    st.caption(
        "These rules come from a shallow decision tree trained on the same data as the main model. "
        "They approximate the model's logic in human-readable form."
    )