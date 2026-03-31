# Refactoring Changelog

## Summary

This document describes every structural change made during the refactor of the ML Dashboard.
Existing model logic, visualisation styles, and core math are **unchanged**.

---

## 1. Train/Test Split → Moved OUT of Preprocessing

**Before:** `pages_content/page_preprocessing.py` contained Tab 6 "✂️ Train / Test Split"
and the `Apply` button ran `split_data()` before persisting `X_train / X_test / y_train / y_test`.

**After:**
- Tab 6 is **removed** from Preprocessing entirely.
- The `Apply` button now saves only `data.X` and `data.y` (full pre-processed features).
- Splitting is performed inside `pages_content/page_train.py` at training time.

**Why:** Splitting is a model validation choice, not a preprocessing step.
Moving it enables proper cross-validation strategies that require access to the full dataset.

---

## 2. New Module — `core/models/validation.py`

Unified validation strategy factory with a clean public API:

```python
get_cv_strategy(method, config, task_type)      # returns sklearn splitter
split_for_training(X, y, test_size, ...)        # returns X_train, X_test, y_train, y_test
run_validation(method, config, model, X, y, scoring, task_type, groups)  # returns results dict
```

### Supported Validation Methods

| Constant | UI Label | Notes |
|---|---|---|
| `METHOD_TRAIN_TEST` | Train / Test Split | Configurable test size, shuffle, stratify |
| `METHOD_KFOLD` | K-Fold Cross Validation | n_splits, shuffle, random_state |
| `METHOD_STRATIFIED` | Stratified K-Fold | Classification only |
| `METHOD_LOOCV` | Leave-One-Out CV | Warning shown for >500 samples |
| `METHOD_LOGO` | Leave-One-Group-Out | Requires user to select a group column |

---

## 3. Updated `pages_content/page_train.py`

Now contains three clear sections:

1. **🤖 Model Selection** — unchanged, uses `render_model_picker`
2. **🔬 Validation Method** — NEW: full UI for all 5 validation strategies
3. **📈 Training Output** — CV scores table + hold-out metrics + feature importance

For CV methods, the model is **refit on the full dataset** after CV scoring,
then a 20% hold-out is used purely for metric display.

---

## 4. New Page — `pages_content/page_explainability.py`

🧠 Explainability page added to the Model sub-menu (both Regression and Classification).

Contains:
- **SHAP Beeswarm** (global feature impact)
- **Mean |SHAP| Bar Chart** (feature importance via SHAP)
- **SHAP Dependence Plot** (LOWESS trend)
- **SHAP Interaction Dependence** (color-coded by a second feature)
- **1D PDP** with ICE curves and marginal histograms
- **2D PDP** interaction surface (contour plot)
- **Surrogate Rule Extraction** (shallow decision tree)

Logic ported **verbatim** from `code0/tabs/explainability.py` and `code0/core/explainability.py`.
No SHAP logic was changed.

---

## 5. New Page — `pages_content/page_safe_region.py`

🟢 Safe Region & Optimizer page added to the Model sub-menu.

Contains (all moved FROM code0 sidebar INTO this dedicated page):
- **Safe Region Search** — synthetic/real data scoring, recommended feature ranges, top candidates
- **2D and 3D Region Maps**
- **Bayesian Optimisation** — configurable n_calls, exposed via checkbox
- **Active Learning** — next-experiment suggestions, configurable method + count
- **Industrial Interpretation** — narrative summary of recommended ranges

Logic ported **verbatim** from `code0/tabs/safe_region.py` and `code0/core/optimisation.py`.

---

## 6. New Module — `core/models/explainability.py`

Ported from `code0/core/explainability.py` into the refactored package namespace.
No logic changes. Functions:

- `get_shap_values`, `shap_importance_df`
- `plot_shap_beeswarm`, `plot_shap_dependence`
- `plot_pdp_1d`, `plot_pdp_2d`
- `extract_rules`

---

## 7. New Module — `core/models/optimisation.py`

Ported from `code0/core/optimisation.py`. Config dependency replaced with
inline dataclass defaults (no external config file required). Functions:

- `sample_uniform`, `sample_dirichlet_mixture`, `apply_constraints`
- `score_synthetic_classification`, `score_synthetic_regression`
- `filter_safe_classification`, `filter_optimal_regression`
- `build_recommended_ranges`, `format_recommendation_text`
- `bayesian_optimise`, `suggest_next_experiments`

---

## 8. New Utility — `utils/plots_safe_region.py`

Ported from `code0/utils/plots.py`. Contains plotly chart builders for:
- `plot_safe_region_2d`, `plot_safe_region_3d`
- `plot_bo_history`
(Other chart builders in the file are also available.)

---

## 9. Updated `app.py`

Sidebar sub-menu for "Model" now exposes 5 options (was 3):

```
Train  |  Evaluate  |  Predict  |  🧠 Explainability  |  🟢 Safe Region & Optimizer
```

Home page welcome table updated to document all 8 steps.

---

## 10. Updated `requirements.txt`

New dependencies added:

| Package | Reason |
|---|---|
| `shap>=0.45.0` | Explainability page |
| `statsmodels>=0.14.0` | LOWESS smoothing in SHAP/PDP plots |
| `scikit-optimize>=0.9.0` | Bayesian optimisation |
| `scipy>=1.12.0` | Dirichlet sampling |
| `catboost>=1.2.0` | CatBoost models (optional) |
| `plotly>=5.20.0` | Interactive safe region maps |

---

## Files Changed / Added

| File | Status | Notes |
|---|---|---|
| `app.py` | Modified | New routes for Explainability + Safe Region |
| `pages_content/page_train.py` | Rewritten | Validation Method UI added; split moved here |
| `pages_content/page_preprocessing.py` | Modified | Tab 6 removed; Apply simplified |
| `pages_content/page_explainability.py` | **NEW** | SHAP + PDP + Rules |
| `pages_content/page_safe_region.py` | **NEW** | Safe region + BO + AL |
| `core/models/validation.py` | **NEW** | Unified CV strategy module |
| `core/models/explainability.py` | **NEW** | Ported from code0 |
| `core/models/optimisation.py` | **NEW** | Ported from code0 (patched imports) |
| `utils/plots_safe_region.py` | **NEW** | Ported from code0 |
| `requirements.txt` | Modified | New dependencies |

## Files NOT Changed

Everything else in the refactored codebase is **unchanged**:
`page_data`, `page_eda`, `page_evaluate`, `page_predict`,
`page_feature_engineering`, `core/data/*`, `core/viz/*`,
`core/models/trainer.py`, `core/models/evaluator.py`, `core/models/registry.py`,
`state/session.py`, `config/settings.py`, `config/models.yaml`, `components/*`.
