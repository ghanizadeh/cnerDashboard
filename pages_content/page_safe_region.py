"""
pages_content/page_safe_region.py
===================================
🟢 Safe Region & Optimizer

Ported from code0/tabs/safe_region.py into the refactored architecture.
Safe Region Search, Active Learning, and Bayesian Optimization live here
— NOT in the sidebar.

No changes to existing optimisation logic — only re-wired to use
refactored session state (get_value / set_state).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from state.session import init_state, get_value, set_state

# ── Optimisation core (ported from code0) ────────────────────────────────────
from core.models.optimisation import (
    sample_uniform,
    sample_dirichlet_mixture,
    apply_constraints,
    score_synthetic_classification,
    score_synthetic_regression,
    filter_safe_classification,
    filter_optimal_regression,
    build_recommended_ranges,
    format_recommendation_text,
    bayesian_optimise,
    suggest_next_experiments,
)

# ── Plots (ported from code0) ─────────────────────────────────────────────────
from utils.plots_safe_region import (
    plot_safe_region_2d,
    plot_safe_region_3d,
    plot_bo_history,
)

_MIN_SYNTH       = 100
_DECIMAL_PLACES  = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_constraint_controls(features: list[str], X_train: pd.DataFrame) -> dict:
    """Constraint expanders rendered inside the tab."""
    st.markdown("### Constraint configuration")

    with st.expander("Feature bounds (optional overrides)", expanded=False):
        feature_bounds: dict = {}
        cols = st.columns(2)
        for i, feat in enumerate(features):
            col_vals = pd.to_numeric(X_train[feat], errors="coerce")
            data_min = float(col_vals.min())
            data_max = float(col_vals.max())
            with cols[i % 2]:
                st.write(f"**{feat}**")
                lo = st.number_input(f"Min {feat}", value=round(data_min, 2), key=f"lo_{feat}")
                hi = st.number_input(f"Max {feat}", value=round(data_max, 2), key=f"hi_{feat}")
                feature_bounds[feat] = (lo, hi)

    use_mixture   = False
    mixture_cols  = None
    mixture_total = 100.0
    use_dirichlet = False

    with st.expander("Mixture / sum constraint (optional)", expanded=False):
        use_mixture = st.checkbox("Enable sum-to-constant constraint")
        if use_mixture:
            mixture_cols  = st.multiselect("Columns that must sum to constant", features)
            mixture_total = st.number_input("Target sum", value=100.0, step=1.0)
            use_dirichlet = st.checkbox(
                "Use Dirichlet sampling (ensures exact sum)",
                value=True,
                help="Recommended — guarantees every sample satisfies the sum constraint.",
            )

    return {
        "feature_bounds": feature_bounds,
        "use_mixture":    use_mixture,
        "mixture_cols":   mixture_cols,
        "mixture_total":  mixture_total,
        "use_dirichlet":  use_dirichlet,
    }


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render():
    init_state()

    st.title("🟢 Safe Region & Optimizer")
    st.divider()

    # ── Guards ────────────────────────────────────────────────────────────
    model     = get_value("model.object")
    X_train   = get_value("split.X_train")
    y         = get_value("data.y")
    task_type = get_value("model.task_type")

    if model is None or X_train is None:
        st.warning("⚠️ No trained model found. Go to **🤖 Train** first.")
        st.stop()

    feature_names = (
        get_value("data.processed_feature_names")
        or get_value("data.feature_names")
        or list(X_train.columns)
    )

    # Build class_names / safe_idx
    class_names = []
    safe_idx    = 0
    if task_type == "classification" and y is not None:
        class_names = [str(c) for c in sorted(y.unique())]

    random_state = 42

    # ── Target class / regression objective ───────────────────────────────
    if task_type == "classification":
        safe_class_idx = st.selectbox(
            "Target (safe) class",
            options=list(range(len(class_names))),
            format_func=lambda i: f"{i} – {class_names[i]}",
            index=safe_idx,
        )
        safe_label = class_names[safe_class_idx] if class_names else "safe"
        objective  = "maximize"
    else:
        safe_class_idx = 0
        safe_label     = "optimal"
        objective      = st.radio("Objective", ["maximize", "minimize"], horizontal=True)

    # ── Confidence threshold (sidebar moved here) ─────────────────────────
    st.markdown("### Scoring parameters")
    conf_thr = st.slider(
        "Confidence threshold (classification)",
        0.50, 0.99, 0.75, 0.01,
        key="safe_conf_thr",
        help="Minimum predicted probability for a point to be flagged as 'safe'.",
        disabled=(task_type != "classification"),
    )

    # ── Synthetic data toggle ─────────────────────────────────────────────
    st.markdown("### Design space exploration")
    use_synth = st.toggle(
        "Use synthetic design space",
        value=True,
        help=(
            "ON: generate synthetic formulations to map the full design space. "
            "OFF: score only the real training data."
        ),
    )

    n_synth = 0
    if use_synth:
        n_synth = max(_MIN_SYNTH, st.number_input(
            "Number of synthetic samples", value=5000, min_value=100, max_value=50000, step=500,
            key="n_synth"
        ))
        st.caption(f"Generating {n_synth:,} synthetic samples.")
    else:
        st.info("Synthetic sampling disabled — scoring real training data only.")

    # ── Constraints ───────────────────────────────────────────────────────
    constraint_cfg = _render_constraint_controls(feature_names, X_train)
    feature_bounds = constraint_cfg["feature_bounds"]
    mixture_cols   = constraint_cfg["mixture_cols"]
    mixture_total  = constraint_cfg["mixture_total"]
    use_dirichlet  = constraint_cfg["use_dirichlet"]
    sum_constraint = (
        (mixture_cols, mixture_total)
        if constraint_cfg["use_mixture"] and mixture_cols
        else None
    )

    # ── Build data to score ───────────────────────────────────────────────
    if use_synth:
        synth_key = (
            f"synth|{id(model)}|{n_synth}"
            f"|{str(sorted(feature_bounds.items()))}"
            f"|{str(mixture_cols)}|{mixture_total}|{use_dirichlet}|{random_state}"
        )
        if synth_key not in st.session_state:
            stale = [k for k in list(st.session_state.keys()) if k.startswith("synth|")]
            for k in stale:
                del st.session_state[k]
            with st.spinner(f"Generating {n_synth:,} synthetic samples…"):
                if use_dirichlet and mixture_cols:
                    base = sample_dirichlet_mixture(
                        mixture_cols, feature_bounds, mixture_total,
                        n_synth, 1.0, random_state,
                    )
                    remaining = [f for f in feature_names if f not in mixture_cols]
                    if remaining:
                        uni = sample_uniform(
                            X_train[remaining], n_synth,
                            {k: v for k, v in feature_bounds.items() if k in remaining},
                            random_state,
                        )
                        for c in remaining:
                            base[c] = uni[c].values
                    generated = base[feature_names]
                else:
                    generated = sample_uniform(X_train, n_synth, feature_bounds, random_state)
                    generated = apply_constraints(generated, feature_bounds, sum_constraint)
            st.session_state[synth_key] = generated

        data_to_score = st.session_state[synth_key]
        source_label  = "synthetic"
    else:
        data_to_score = apply_constraints(X_train.copy(), feature_bounds, sum_constraint)
        if data_to_score.empty:
            st.warning("All real data excluded by current bounds. Widening bounds will restore data.")
            data_to_score = X_train.copy()
        source_label = "experimental"

    if data_to_score.empty:
        st.error("No data available to score. Check your constraints.")
        return

    # ── Score ─────────────────────────────────────────────────────────────
    if task_type == "classification":
        scored      = score_synthetic_classification(model, data_to_score, class_names, safe_class_idx)
        safe_all, safe_hi = filter_safe_classification(scored, safe_class_idx, conf_thr)
        score_col   = "safe_probability"
        _X_orig     = get_value("data.X")
        X_orig      = _X_orig if _X_orig is not None else X_train
        orig_labels = [str(v) for v in y.values] if y is not None else []
    else:
        scored      = score_synthetic_regression(model, data_to_score, objective)
        safe_hi     = filter_optimal_regression(scored, top_pct=0.10)
        safe_all    = scored
        score_col   = "predicted_value"
        _X_orig     = get_value("data.X")
        X_orig      = _X_orig if _X_orig is not None else X_train
        orig_labels = [str(round(float(v), 2)) for v in y.values] if y is not None else []

    # ── Summary metrics ───────────────────────────────────────────────────
    st.markdown("### Search summary")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Scored ({source_label})", f"{len(scored):,}")
    c2.metric("Predicted safe / optimal",  f"{len(safe_all):,}")
    c3.metric("High-confidence",           f"{len(safe_hi):,}")

    if safe_hi.empty:
        st.warning(
            "No high-confidence safe samples found. Try lowering the confidence threshold, "
            "widening feature bounds, or enabling synthetic sampling."
        )

    # ── Recommended ranges ────────────────────────────────────────────────
    st.markdown("### Recommended formulation ranges")
    source_df = safe_hi if not safe_hi.empty else safe_all
    ranges_df = build_recommended_ranges(source_df, feature_names)
    if not ranges_df.empty:
        st.dataframe(ranges_df.round(_DECIMAL_PLACES), use_container_width=True)
        st.code(format_recommendation_text(ranges_df, safe_label))
    else:
        st.info("Not enough data to compute recommended ranges.")
        ranges_df = pd.DataFrame()

    # ── Top candidates ────────────────────────────────────────────────────
    top_k = st.slider("Top-K formulations to display", 5, 100, 20, key="safe_top_k")
    st.markdown(f"### Top {top_k} recommended formulations")
    sort_by = (
        ["safe_probability", "safety_margin"]
        if task_type == "classification"
        else ["objective_score"]
    )
    top_df = source_df.sort_values(sort_by, ascending=False).head(top_k).copy()
    top_df.insert(0, "Rank", range(1, len(top_df) + 1))
    st.dataframe(top_df.round(_DECIMAL_PLACES), use_container_width=True)

    # ── 2D region map ─────────────────────────────────────────────────────
    if len(feature_names) >= 2:
        st.markdown("### 2D region map")
        c1, c2 = st.columns(2)
        with c1:
            x2 = st.selectbox("X axis", feature_names, key="map2_x")
        with c2:
            y2_opts = [f for f in feature_names if f != x2]
            y2 = st.selectbox("Y axis", y2_opts, key="map2_y")
        color_col = "pred_class" if task_type == "classification" else "predicted_value"
        try:
            fig = plot_safe_region_2d(
                scored, X_orig, orig_labels, x2, y2,
                color_col=color_col,
                class_names=class_names if task_type == "classification" else None,
                sample_n=n_synth if use_synth else len(scored),
                random_state=random_state,
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.warning(f"2D map failed: {exc}")

    # ── 3D region map ─────────────────────────────────────────────────────
    if len(feature_names) >= 3:
        st.markdown("### 3D region map")
        c1, c2, c3 = st.columns(3)
        with c1:
            x3 = st.selectbox("X", feature_names, key="map3_x")
        with c2:
            y3_opts = [f for f in feature_names if f != x3]
            y3 = st.selectbox("Y", y3_opts, key="map3_y")
        with c3:
            z3_opts = [f for f in feature_names if f not in [x3, y3]]
            z3 = st.selectbox("Z", z3_opts, key="map3_z")
        try:
            fig = plot_safe_region_3d(
                scored, x3, y3, z3,
                color_col=color_col,
                class_names=class_names if task_type == "classification" else None,
                sample_n=n_synth if use_synth else len(scored),
                random_state=random_state,
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.warning(f"3D map failed: {exc}")

    st.divider()

    # ── Bayesian Optimisation ─────────────────────────────────────────────
    st.markdown("### 🎯 Bayesian Optimisation")
    run_bo = st.checkbox("Run Bayesian Optimisation", value=False, key="run_bo")
    bo_results, bo_best = None, None

    if run_bo:
        n_bo_calls = st.number_input("BO evaluations", value=50, min_value=10, max_value=300, step=10, key="n_bo_calls")
        bo_key = f"bo_{id(model)}_{n_bo_calls}_{random_state}"
        if bo_key not in st.session_state:
            with st.spinner(f"Running {n_bo_calls} Bayesian evaluations…"):
                bo_results, bo_best = bayesian_optimise(
                    model=model,
                    feature_cols=feature_names,
                    X_ref=X_train,
                    feature_bounds=feature_bounds,
                    safe_class_idx=safe_class_idx,
                    class_names=class_names,
                    problem_type=task_type,
                    objective=objective,
                    n_calls=n_bo_calls,
                    random_state=random_state,
                )
            st.session_state[bo_key] = (bo_results, bo_best)
        bo_results, bo_best = st.session_state[bo_key]

        st.success("Bayesian optimisation complete.")
        st.markdown("**Best point found:**")
        st.json({k: round(v, _DECIMAL_PLACES) for k, v in bo_best.items()})
        if bo_results is not None and score_col in bo_results.columns:
            st.plotly_chart(plot_bo_history(bo_results, score_col), use_container_width=True)

    st.divider()

    # ── Active Learning ───────────────────────────────────────────────────
    st.markdown("### 🔬 Active Learning — Next Experiment Suggestions")
    run_al = st.checkbox("Run Active Learning", value=False, key="run_al")
    al_suggestions = None

    if run_al:
        n_al = st.number_input("Suggestions", value=5, min_value=1, max_value=50, step=1, key="n_al")
        al_method = st.selectbox(
            "Uncertainty method",
            ["entropy", "margin", "least_confident"],
            key="al_method",
        )
        al_key = f"al_{id(model)}_{n_al}_{al_method}_{random_state}"
        if al_key not in st.session_state:
            with st.spinner("Identifying most informative experiment candidates…"):
                al_suggestions = suggest_next_experiments(
                    model=model,
                    X_ref=X_train,
                    feature_bounds=feature_bounds or None,
                    problem_type=task_type,
                    class_names=class_names,
                    safe_class_idx=safe_class_idx,
                    n_suggestions=n_al,
                    method=al_method,
                    random_state=random_state,
                )
            st.session_state[al_key] = al_suggestions
        al_suggestions = st.session_state[al_key]

        st.info(
            "These formulations have the **highest model uncertainty** — running these "
            "experiments will most efficiently improve model accuracy."
        )
        st.dataframe(al_suggestions.round(_DECIMAL_PLACES), use_container_width=True)

    # ── Industrial interpretation ─────────────────────────────────────────
    st.divider()
    st.markdown("### Industrial interpretation")
    if not ranges_df.empty:
        cols_r   = ranges_df.columns.tolist()
        low_col  = next((c for c in cols_r if "Low"  in c), cols_r[1])
        high_col = next((c for c in cols_r if "High" in c), cols_r[3])
        insights = [
            f"- **{row['Feature']}**: {row[low_col]} – {row[high_col]}  (median ≈ {row['Median']})"
            for _, row in ranges_df.iterrows()
        ]
        mode_note = (
            "real experimental data" if not use_synth
            else f"{len(scored):,} synthetic samples"
        )
        st.markdown(
            f"For class **'{safe_label}'** the model recommends:\n\n"
            + "\n".join(insights)
            + f"\n\nRanges derived from the **5th–95th percentile** of the high-confidence "
              f"safe region ({mode_note}), giving a robust industrial operating window."
        )
