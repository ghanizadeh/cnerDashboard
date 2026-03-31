"""
pages_content/page_evaluate.py
Detailed metrics, plots, feature importance, explainability, and model comparison.
"""

import numpy as np
import pandas as pd
import streamlit as st

from state.session import init_state, get_value
from components.metrics_card import render_metrics_card
from core.viz.evaluation import (
    draw_confusion_matrix, draw_roc_curve,
    draw_feature_importance, draw_residuals, draw_pred_vs_actual,
)
from core.models.evaluator import get_feature_importance
from core.viz.style import fig_to_st
from config.settings import RANDOM_STATE


def render():
    init_state()

    st.title("📈 Model Evaluation")
    st.divider()

    # ── Guards ────────────────────────────────────────────────────────────
    model       = get_value("model.object")
    model_name  = get_value("model.name")
    task_type   = get_value("model.task_type")
    metrics     = get_value("results.metrics")
    X_test      = get_value("split.X_test")
    X_train     = get_value("split.X_train")
    y_test      = get_value("split.y_test")
    y_train     = get_value("split.y_train")

    if model is None or X_test is None:
        st.warning("⚠️ No trained model found. Go to **🤖 Train** first.")
        st.stop()

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    feature_names = (
        get_value("data.processed_feature_names")
        or get_value("data.feature_names")
        or list(X_test.columns)
    )

    st.info(f"**Model:** {model_name} &nbsp;|&nbsp; **Task:** {task_type.capitalize()}")

    # ─────────────────────────────────────────────────────────────────────
    # Tabs
    # ─────────────────────────────────────────────────────────────────────
    tab_metrics, tab_plots,  tab_explain, tab_compare = st.tabs(
        ["📊 Metrics", "📉 Plots",  "🧠 Explainability", "⚖️ Model Comparison"]
    )

    # ── Tab 1: Metrics ────────────────────────────────────────────────────
    with tab_metrics:
        st.subheader("Test-Set Metrics")
        render_metrics_card(metrics, task_type=task_type)

        if task_type == "classification":
            report = metrics.get("classification_report", {})
            if report:
                st.subheader("Classification Report")
                st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

    # ── Tab 2: Plots ──────────────────────────────────────────────────────
    with tab_plots:
        if task_type == "classification":
            st.subheader("Confusion Matrix")
            classes = list(np.unique(y_test))
            fig = draw_confusion_matrix(y_test, y_pred, labels=classes)
            fig_to_st(fig)

            if y_prob is not None:
                st.subheader("ROC Curve")
                fig = draw_roc_curve(y_test, y_prob)
                fig_to_st(fig)
            else:
                st.info("Model does not support probability estimates — ROC curve unavailable.")
        else:
            st.subheader("Predicted vs Actual")
            fig = draw_pred_vs_actual(y_test, y_pred)
            fig_to_st(fig)
            st.subheader("Residuals")
            fig = draw_residuals(y_test, y_pred)
            fig_to_st(fig)

    # ── Tab 4: Explainability ─────────────────────────────────────────────
    with tab_explain:
        if X_train is None:
            st.warning("Training data not found in session. Re-train the model first.")
            st.stop()

        try:
            from core.models.explainability import (
                get_shap_values, shap_importance_df,
                plot_shap_beeswarm, plot_shap_dependence,
                plot_shap_dependence_2d,
                plot_pdp_1d, plot_pdp_2d, extract_rules,
                MAX_SHAP_SAMPLES, SURROGATE_MAX_DEPTH, SURROGATE_DEPTH_RANGE,
            )
            import plotly.express as px

            # ── Class selector ────────────────────────────────────────────
            class_names = []
            if task_type == "classification":
                y = get_value("data.y")
                if y is not None:
                    class_names = [str(c) for c in sorted(y.unique())]

            class_idx = 0
            if task_type == "classification" and class_names:
                class_idx = st.selectbox(
                    "Class for SHAP / PDP",
                    options=list(range(len(class_names))),
                    format_func=lambda i: f"{i} – {class_names[i]}",
                    key="eval_shap_class",
                )

            # ── Subsample ─────────────────────────────────────────────────
            sample_n = min(MAX_SHAP_SAMPLES, len(X_train))
            X_shap = (
                X_train.sample(sample_n, random_state=RANDOM_STATE)
                if len(X_train) > sample_n
                else X_train.copy()
            )

            # ── SHAP ──────────────────────────────────────────────────────
            st.subheader("🐝 SHAP Analysis")
            shap_key = f"eval_shap_{id(model)}_{class_idx}"
            if shap_key not in st.session_state:
                with st.spinner("Computing SHAP values…"):
                    try:
                        st.session_state[shap_key] = get_shap_values(model, X_shap, class_idx)
                    except Exception as exc:
                        st.warning(f"SHAP failed: {exc}")
                        st.session_state[shap_key] = None

            shap_vals = st.session_state[shap_key]

            if shap_vals is not None:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown("**🟢 Beeswarm (global impact)**")
                    try:
                        st.pyplot(plot_shap_beeswarm(shap_vals, X_shap))
                    except Exception as exc:
                        st.warning(f"Beeswarm failed: {exc}")
                with col2:
                    st.markdown("**🟡 Mean |SHAP| bar**")
                    imp_df = shap_importance_df(shap_vals, list(X_shap.columns))
                    fig = px.bar(
                        imp_df.sort_values("MeanAbsSHAP", ascending=True),
                        x="MeanAbsSHAP", y="Feature", orientation="h",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("**SHAP Dependence Plot (1D)**")
                dep_feat = st.selectbox("Feature", feature_names, key="eval_shap_dep")
                try:
                    fig = plot_shap_dependence(shap_vals, X_shap, dep_feat)
                    fig_to_st(fig)
                except Exception as exc:
                    st.warning(f"Dependence plot failed: {exc}")

                st.divider()
                st.markdown("**📊 2D SHAP Dependence Plot**")
                st.caption("X axis = feature value · Y axis = SHAP value · Colour = interaction feature")
                col_x, col_c = st.columns(2)
                with col_x:
                    shap_2d_x = st.selectbox(
                        "X axis — primary feature",
                        feature_names,
                        key="eval_shap_2d_x",
                    )
                with col_c:
                    other_feats = [f for f in feature_names if f != shap_2d_x]
                    shap_2d_color = st.selectbox(
                        "Colour — interaction feature",
                        other_feats,
                        key="eval_shap_2d_color",
                    )
                try:
                    fig = plot_shap_dependence_2d(
                        shap_vals, X_shap,
                        feature=shap_2d_x,
                        interaction_feature=shap_2d_color,
                    )
                    fig_to_st(fig)
                except Exception as exc:
                    st.warning(f"2D SHAP plot failed: {exc}")

            st.divider()

            # ── PDP ───────────────────────────────────────────────────────
            st.subheader("📉 Partial Dependence Plots")
            pdp_feat = st.selectbox("Feature for 1D PDP", feature_names, key="eval_pdp_1d")
            try:
                fig = plot_pdp_1d(
                    model, X_train, pdp_feat,
                    class_idx if task_type == "classification" else None,
                    task_type,
                )
                fig_to_st(fig)
            except Exception as exc:
                st.warning(f"1D PDP failed: {exc}")

            st.divider()

            # ── Surrogate Rules ───────────────────────────────────────────
            st.subheader("📜 Surrogate Rule Extraction")
            depth = st.slider(
                "Rule tree max depth",
                SURROGATE_DEPTH_RANGE[0], SURROGATE_DEPTH_RANGE[1],
                SURROGATE_MAX_DEPTH, key="eval_rule_depth",
            )
            _, rules_text = extract_rules(
                X_train, y_train, feature_names, class_names, task_type, depth
            )
            st.code(rules_text)
            st.caption(
                "Shallow decision tree trained on the same data as the main model. "
                "Approximates model logic in human-readable form."
            )

        except ImportError as e:
            st.warning(
                f"Explainability requires extra packages: {e}  \n"
                "Run: `pip install shap statsmodels`"
            )

    # ── Tab 5: Model Comparison ───────────────────────────────────────────
    with tab_compare:
        trained_models = get_value("results.trained_models") or []
        if len(trained_models) < 2:
            st.info("Train at least **2 models** to compare them here.")
        else:
            rows = []
            for m in trained_models:
                row = {"Model": m["name"]}
                row.update({k: round(v, 4) for k, v in m["metrics"].items()
                            if isinstance(v, (int, float))})
                rows.append(row)
            compare_df = pd.DataFrame(rows)
            st.subheader(f"Comparison of {len(trained_models)} models")
            st.dataframe(compare_df, use_container_width=True)

            if task_type == "classification" and "accuracy" in compare_df.columns:
                best = compare_df.loc[compare_df["accuracy"].idxmax(), "Model"]
                st.success(f"🏆 Best accuracy: **{best}**")
            elif task_type == "regression" and "r2" in compare_df.columns:
                best = compare_df.loc[compare_df["r2"].idxmax(), "Model"]
                st.success(f"🏆 Best R²: **{best}**")

    st.divider()
    st.info("👉 Proceed to **🔮 Predict** to make new predictions.")