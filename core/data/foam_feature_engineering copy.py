"""
foam_eda_app.py
================
Standalone Streamlit app for foam half-life EDA scatter plots.

Run:
    streamlit run foam_eda_app.py

Dependencies:
    pip install streamlit pandas numpy matplotlib seaborn openpyxl
"""
from __future__ import annotations

import io
import os
import re
import zipfile
from itertools import product as iproduct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
#  Group taxonomy (mirrors foam_feature_engineering.py)
# ─────────────────────────────────────────────────────────────────────────────
GROUPS: dict[str, list[str]] = {
    "Nanoparticle": [
        "HS (%)", "BLH5 (%)", "HSA (%)",
    ],
    "Anionic": [
        "AOS (%)", "alpha-step (%)", "SDS (%)", "SLES (%)",
        "n. LABS (%)", "DB45 (%)", "Cola SLAA (%)", "Cola SC (%)",
    ],
    "Nonionic": [
        "APG (%)", "decyl glucoside (%)", "caprylyl glucoside (%)",
        "Tween 80 (%)", "PG (%)", "LAO (%)",
    ],
    "Zwitterionic": [
        "CapB (%)", "2C (%)", "Cola 2C (%)", "amine oxide (%)",
        "Cola LMB (%)", "Amphosol 1c (%)", "SC (%)", "LBHP (%)", "CS50(%)", "DM (%)",
    ],
    "Polymer": [
        "HPAM (%)", "xanthan gum (%) ", "Guar Gum (%)", "FPAM (%)",
        "PAA (%)", "PA (%)", "ClearHib 1000 (%)",
    ],
    "Citric": [
        "Citric (%)",
        "31.2(%) citric+ 13.3(%) KOH  (pH not adjusted, pH=4.46 )",
        "31.2% citric+ 13.3% KOH  (adjusted pH=4) (%)",
        "38.1% citric+   KOH  (pH=5) (%)",
        "potassium citrate (9.7%)/citric acid buffer (19.22%)  pH=3 (%)",
        "potassium citrate (%)",
    ],
    "Acid": [
        "EDTA (%)", "etidronic acid (%)", "acetic acid (%)",
    ],
    "Antiscalant": [
        "Mem 2000-clear tech (%)", "Mem 2500-clear tech (%)",
        "Mem 4000-clear tech (%)", "Mem 3500-clear tech (%)",
        "Mem 3000-clear tech (%)",
    ],
    "Brine": [
        "Divalent", "Monovalent",
    ],
    "Oil": [
        "Alkane (linear HC) ", "Aromatics", "Branched HC",
        "Light HC (up to C10)", "Sulfur content",
        "Acid & ester content", "Chlorinated components", "Polarity ",
    ],
    "Process": [
        "Temperature", "Dilution Ratio", "Oil  (%)",
        "concentrate manufacturing method (Ratio)",
        "Initial Foam Temp (dilution Temp) ",
    ],
}

# Sum feature definitions: name → list of raw columns to sum
SUM_FEATURES: dict[str, str] = {
    "Anionic (All Types)":      "Anionic",
    "Nonionic (All Types)":     "Nonionic",
    "Zwitterionic (All Types)": "Zwitterionic",
    "Sum Surfactant":           "Surfactant",   # special: Anionic+Nonionic+Zwitterionic
    "Nanoparticle (All Types)": "Nanoparticle",
    "Polymer (All Types)":      "Polymer",
    "Acid (All Types)":         "Acid",
    "Citric (All Types)":       "Citric",
}

GRP_CLR = {
    "Nanoparticle": "#1565C0", "Anionic":    "#E53935",
    "Nonionic":     "#FB8C00", "Zwitterionic":"#8E24AA",
    "Surfactant":   "#FF8F00", "Polymer":    "#6A1B9A",
    "Citric":       "#2E7D32", "Acid":       "#C62828",
    "Antiscalant":  "#00695C", "Brine":      "#4527A0",
    "Oil":          "#BF360C", "Process":    "#546E7A",
    "Sum":          "#37474F",
}

TARGET_CANDIDATES = [
    "Half life (method: poly) [h]",
    "Half life (method: poly) [h] 95%",
    "Calculated Half Life (hr_Linear)",
    "Calculated Half Life (hr)",
]

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _slugify(text: str) -> str:
    """Safe folder/file name from arbitrary text."""
    s = str(text).strip()
    s = re.sub(r'[^\w\s\-]', '', s)
    s = re.sub(r'[\s]+', '_', s)
    return s[:60]


def _fname(x_col: str, target: str, color_col: str) -> str:
    """
    Build file name in the format:
        X_{x}_Y_{target}_Color_{color}.png
    All parts are slugified so the name is filesystem-safe.
    """
    x_s   = _slugify(x_col)
    y_s   = _slugify(target)
    c_s   = _slugify(color_col)
    return f"X_{x_s}_Y_{y_s}_Color_{c_s}.png"


def _normalise(name: str) -> str:
    """Lowercase + strip trailing ' (%)' and whitespace for fuzzy matching."""
    s = str(name).strip().lower()
    s = re.sub(r'\s*\(%\)\s*$', '', s).strip()
    return s


def _avail(candidates: list[str], cols: list[str]) -> list[str]:
    """
    Return items from `cols` that fuzzy-match any candidate,
    case-insensitively and ignoring a trailing ' (%)'.
    e.g. candidate 'AOS (%)' matches column 'AOS' and vice-versa.
    `cols` is the list of ACTUAL DataFrame columns.
    `candidates` is the hardcoded default list for a group.
    """
    # Build lookup: normalised_actual_col → actual_col
    norm_to_col = {_normalise(c): c for c in cols}
    matched = []
    seen = set()
    for cand in candidates:
        key = _normalise(cand)
        if key in norm_to_col:
            actual = norm_to_col[key]
            if actual not in seen:
                matched.append(actual)
                seen.add(actual)
    return matched


def _resolve_col(name: str, df: pd.DataFrame) -> str | None:
    """Find the actual DataFrame column that matches `name` fuzzily."""
    if name in df.columns:
        return name
    key = _normalise(name)
    for col in df.columns:
        if _normalise(col) == key:
            return col
    return None


def compute_sum_features(df: pd.DataFrame,
                          sel_groups: dict[str, list[str]]) -> pd.DataFrame:
    """
    Add sum feature columns to df (copy returned).
    Column lookup is fuzzy: case-insensitive, ignores trailing ' (%)'.
    Only adds a sum feature when at least one member column is found.
    """
    out = df.copy()
    for feat_name, group_key in SUM_FEATURES.items():
        if group_key == "Surfactant":
            cols = (sel_groups.get("Anionic", []) +
                    sel_groups.get("Nonionic", []) +
                    sel_groups.get("Zwitterionic", []))
        else:
            cols = sel_groups.get(group_key, [])
        # Resolve each column name fuzzily against the actual DataFrame
        resolved = [_resolve_col(c, out) for c in cols]
        present  = [c for c in resolved if c is not None]
        if present:
            s = pd.Series(0.0, index=out.index)
            for c in present:
                s += _safe_num(out[c]).fillna(0)
            out[feat_name] = s
    return out


def compute_ratio_features(df: pd.DataFrame,
                            ratios: list[tuple[str, str]]) -> pd.DataFrame:
    """Add user-defined ratio columns (numerator / denominator)."""
    out = df.copy()
    for num, den in ratios:
        if num in out.columns and den in out.columns:
            col_name = f"{num} / {den}"
            out[col_name] = _safe_num(out[num]) / _safe_num(out[den]).replace(0, np.nan)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────────────────────────────────────

CMAP = "viridis"
POINT_ALPHA = 0.72
POINT_SIZE  = 38
FIG_DPI     = 150


def _scatter(df: pd.DataFrame,
             x_col: str, y_col: str, color_col: str,
             title: str) -> plt.Figure:
    """
    Single 2D scatter: X = x_col, Y = y_col, colour = color_col.
    Returns a matplotlib Figure.
    """
    sub = df[[x_col, y_col, color_col]].dropna()
    if sub.empty:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=FIG_DPI)
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="grey")
        ax.set_title(title, fontsize=10, fontweight="bold")
        return fig

    x = sub[x_col].values
    y = sub[y_col].values
    c = sub[color_col].values

    # Colour: numeric → continuous cmap, categorical → discrete
    is_num_color = pd.api.types.is_numeric_dtype(sub[color_col])

    fig, ax = plt.subplots(figsize=(7, 5), dpi=FIG_DPI)
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#F5F5F5")

    if is_num_color:
        sc = ax.scatter(x, y, c=c, cmap=CMAP, alpha=POINT_ALPHA,
                        s=POINT_SIZE, edgecolors="white", linewidths=0.4)
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(color_col, fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    else:
        cats = sub[color_col].astype(str)
        uniq = sorted(cats.unique())
        palette = plt.cm.get_cmap("tab10", len(uniq))
        for i, cat in enumerate(uniq):
            mask = cats == cat
            ax.scatter(x[mask], y[mask], color=palette(i),
                       alpha=POINT_ALPHA, s=POINT_SIZE,
                       edgecolors="white", linewidths=0.4, label=cat)
        ax.legend(fontsize=7, title=color_col, title_fontsize=7,
                  loc="upper left", framealpha=0.7)

    ax.set_xlabel(x_col, fontsize=9)
    ax.set_ylabel(y_col, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=8)
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#CCCCCC")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Plot generation — one function per subfolder type
# ─────────────────────────────────────────────────────────────────────────────

def _render_to_bytes(fig: plt.Figure) -> bytes:
    """Render a matplotlib figure to PNG bytes and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _generate_subfolder(
    df: pd.DataFrame,
    subfolder: str,
    x_cols: list[str],
    color_cols_per_x: dict[str, list[str]],
    target: str,
    prog_offset: int = 0,
    prog_total: int  = 1,
    prog_bar = None,
) -> dict[str, bytes]:
    """
    Core loop: for each x_col create one PNG per colour column.

    Parameters
    ----------
    subfolder          : top-level subfolder name inside "2D Plots/"
    x_cols             : list of X-axis feature names (→ one sub-folder each)
    color_cols_per_x   : {x_col: [colour_col, ...]}  — colour list per X
    prog_offset/total  : for combined progress tracking across subfolders
    """
    figures: dict[str, bytes] = {}
    done = 0
    for x_col in x_cols:
        x_folder = _slugify(x_col)
        for col_col in color_cols_per_x.get(x_col, []):
            title = (f"Y: {target}\n"
                     f"X: {x_col}   |   Colour: {col_col}")
            fig   = _scatter(df, x_col, target, col_col, title)
            fname = _fname(x_col, target, col_col)
            figures[f"{subfolder}/{x_folder}/{fname}"] = _render_to_bytes(fig)
            done += 1
            if prog_bar is not None:
                pct = min((prog_offset + done) / prog_total, 1.0)
                prog_bar.progress(pct,
                    text=f"[{subfolder[:30]}]  X={x_col}  colour={col_col}…")
    return figures


def generate_all_plots(
    df: pd.DataFrame,
    x_cols: list[str],
    color_cols: list[str],
    ratio_cols: list[str],
    target: str,
) -> dict[str, bytes]:
    """
    Generate both plot subfolders and return {relative_path: png_bytes}.

    Folder structure
    ----------------
    2D Plots/
      Effect of individual component to each other on half life/
        AOS/
          X_AOS_Y_HalfLife_Color_APG.png        ← peer component colour
          X_AOS_Y_HalfLife_Color_Polarity.png   ← user colour-by feature
          ...
        APG/  ...
        Anionic (All Types)/  ...

      Effect of component Ratio to individual components/
        AOS/
          X_AOS_Y_HalfLife_Color_Anionic_All_Types_Sum_Surfactant.png
          ...
        APG/  ...

    Colour rules
    ------------
    Subfolder 1 — individual components:
      colours = other x_cols (peers) + user colour-by features
    Subfolder 2 — ratios:
      colours = all ratio_cols (must-have + user-defined)
      (self-colour always skipped)
    """
    SF1 = "Effect of individual component to each other on half life"
    SF2 = "Effect of component Ratio to individual components"

    # ── Build colour maps ────────────────────────────────────────────────
    # SF1: peers + extra colour-by
    sf1_colors: dict[str, list[str]] = {}
    for x in x_cols:
        peers  = [c for c in x_cols    if c != x]
        extras = [c for c in color_cols if c != x and c not in peers]
        sf1_colors[x] = list(dict.fromkeys(peers + extras))

    # SF2: ratio cols only (skip self)
    sf2_colors: dict[str, list[str]] = {}
    for x in x_cols:
        sf2_colors[x] = [r for r in ratio_cols if r != x]

    # ── Count total plots for progress bar ───────────────────────────────
    n_sf1 = sum(len(v) for v in sf1_colors.values())
    n_sf2 = sum(len(v) for v in sf2_colors.values())
    total = n_sf1 + n_sf2

    if total == 0:
        st.warning("No plots to generate — check selections.")
        return {}

    prog = st.progress(0, text="Starting…")
    figures: dict[str, bytes] = {}

    # ── Subfolder 1 ──────────────────────────────────────────────────────
    f1 = _generate_subfolder(df, SF1, x_cols, sf1_colors, target,
                              prog_offset=0, prog_total=total, prog_bar=prog)
    figures.update(f1)

    # ── Subfolder 2 ──────────────────────────────────────────────────────
    f2 = _generate_subfolder(df, SF2, x_cols, sf2_colors, target,
                              prog_offset=n_sf1, prog_total=total, prog_bar=prog)
    figures.update(f2)

    prog.empty()
    return figures


def figures_to_zip(figures: dict[str, bytes]) -> bytes:
    """Pack pre-rendered PNG bytes into an in-memory ZIP.
    Path inside ZIP: 2D Plots/<subfolder>/<x_folder>/<filename>.png
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path_key, png_bytes in figures.items():
            zf.writestr(f"2D Plots/{path_key}", png_bytes)
    buf.seek(0)
    return buf.read()


def save_figures_to_disk(figures: dict[str, bytes],
                          base_dir: str) -> int:
    """Save pre-rendered PNG bytes.
    Output: base_dir/2D Plots/<subfolder>/<x_folder>/<filename>.png
    """
    root = Path(base_dir) / "2D Plots"
    count = 0
    for path_key, png_bytes in figures.items():
        dest = root / Path(path_key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(png_bytes)
        count += 1
    return count


# ─────────────────────────────────────────────────────────────────────────────
#  Streamlit app
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Foam EDA — Scatter Plots",
    page_icon="🧪",
    layout="wide",
)

# ── Custom style ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Lora:wght@500;700&display=swap');
html, body, [class*="css"] { font-family: 'Lora', Georgia, serif; }
h1 { font-family: 'Lora', serif; font-weight: 700; letter-spacing: -0.5px; }
h2, h3 { font-family: 'Lora', serif; font-weight: 500; }
code, .stCode { font-family: 'DM Mono', monospace !important; }
.group-pill {
    display:inline-block; padding:3px 10px; border-radius:12px;
    font-size:0.75rem; font-weight:600; margin:2px;
    font-family:'DM Mono',monospace;
}
.section-card {
    background:#f8f9fa; border:1px solid #e0e0e0; border-radius:10px;
    padding:1.2rem 1.5rem; margin-bottom:1rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🧪 Foam EDA — 2D Scatter Plot Generator")
st.caption("Upload data → select components → generate and download organised scatter plots.")
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Upload
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 1 · Upload Data")
uploaded = st.file_uploader(
    "CSV (UTF-8) or Excel (.xlsx / .xls)",
    type=["csv", "xlsx", "xls"],
)
if uploaded is None:
    st.info("Upload a CSV or Excel file to continue.")
    st.stop()

is_excel = uploaded.name.lower().endswith((".xlsx", ".xls"))

# Sheet selector (Excel only)
sheet_name: str | int = 0
if is_excel:
    file_bytes_peek = uploaded.read()
    uploaded.seek(0)
    try:
        xl = pd.ExcelFile(io.BytesIO(file_bytes_peek))
        sheets = xl.sheet_names
        if len(sheets) > 1:
            sheet_name = st.selectbox(
                "Select sheet",
                options=sheets,
                index=0,
                key="sheet_selector",
            )
        else:
            sheet_name = sheets[0]
            st.caption(f"Single sheet detected: **{sheet_name}**")
    except Exception as e:
        st.error(f"Could not read Excel file: {e}")
        st.stop()

@st.cache_data(show_spinner="Reading file…")
def _load(file_bytes: bytes, excel: bool, sheet: str | int) -> pd.DataFrame:
    if excel:
        return (pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet)
                  .dropna(how="all"))
    return (pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8",
                        low_memory=False)
              .dropna(how="all"))

raw_bytes = uploaded.read()
df_raw = _load(raw_bytes, is_excel, sheet_name)
st.success(f"✅  **{len(df_raw):,} rows × {df_raw.shape[1]} columns**")

with st.expander("Preview raw data (first 5 rows)"):
    st.dataframe(df_raw.head(5), use_container_width=True)

all_cols = list(df_raw.columns)
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Target
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 2 · Select Target (Y axis)")
default_target = next((c for c in TARGET_CANDIDATES if c in all_cols), all_cols[-1])
target_col = st.selectbox("Target column", all_cols,
                           index=all_cols.index(default_target))
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Component groups (auto-populated from defaults, user can edit)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 3 · Chemical Component Groups")
st.caption(
    "Default columns for each group are pre-selected from your uploaded file. "
    "Add or remove columns freely."
)

sel_groups: dict[str, list[str]] = {}
non_target = [c for c in all_cols if c != target_col]

# Show groups in a 2-column grid
grp_items = [g for g in GROUPS if g not in ("Oil", "Process", "Brine")]
oil_proc   = ["Oil", "Brine"]  # Process handled separately below

with st.container():
    cols_ui = st.columns(2)
    for gi, grp in enumerate(grp_items):
        with cols_ui[gi % 2]:
            clr = GRP_CLR.get(grp, "#607D8B")
            st.markdown(
                f'<span class="group-pill" style="background:{clr}22;color:{clr};'
                f'border:1px solid {clr}66">{grp}</span>',
                unsafe_allow_html=True,
            )
            defaults = _avail(GROUPS[grp], non_target)
            sel_groups[grp] = st.multiselect(
                f"Columns for {grp}",
                options=non_target,
                default=defaults,
                key=f"grp_{grp}",
                label_visibility="collapsed",
            )

st.markdown("##### 🌊 Brine & 🛢️ Oil")
bp_cols = st.columns(2)
for ci, grp in enumerate(["Oil", "Brine"]):
    with bp_cols[ci]:
        clr = GRP_CLR.get(grp, "#607D8B")
        st.markdown(
            f'<span class="group-pill" style="background:{clr}22;color:{clr};'
            f'border:1px solid {clr}66">{grp}</span>',
            unsafe_allow_html=True,
        )
        defaults = _avail(GROUPS[grp], non_target)
        sel_groups[grp] = st.multiselect(
            f"Columns for {grp}",
            options=non_target,
            default=defaults,
            key=f"grp_{grp}",
            label_visibility="collapsed",
        )

# ── Process — individual pickers per variable ─────────────────────────────────
st.markdown("##### ⚙️ Process Conditions")
st.caption("Select the column for each process variable. '(none)' to exclude.")
NONE = "(none)"
none_opt = [NONE] + non_target

pc1, pc2, pc3 = st.columns(3)
with pc1:
    _def_temp = next((c for c in non_target
                      if _normalise(c) in ("temperature", "temperature_corrected",
                                           "temp")), NONE)
    proc_temp = st.selectbox(
        "🌡️ Temperature",
        none_opt,
        index=none_opt.index(_def_temp) if _def_temp in none_opt else 0,
        key="proc_temp",
    )

with pc2:
    _def_dil = next((c for c in non_target
                     if _normalise(c) in ("dilution ratio", "dilution ratio_corrected",
                                          "dilution")), NONE)
    proc_dil = st.selectbox(
        "💧 Dilution Ratio",
        none_opt,
        index=none_opt.index(_def_dil) if _def_dil in none_opt else 0,
        key="proc_dil",
    )

with pc3:
    _def_oilpct = next((c for c in non_target
                        if _normalise(c) in ("oil  (%)", "oil (%)",
                                             "oil  (%)_corrected", "oil percent",
                                             "oil%")), NONE)
    proc_oil_pct = st.selectbox(
        "🛢️ Oil Percent (%)",
        none_opt,
        index=none_opt.index(_def_oilpct) if _def_oilpct in none_opt else 0,
        key="proc_oil_pct",
    )

# Collect non-none process cols into sel_groups["Process"]
sel_groups["Process"] = [c for c in [proc_temp, proc_dil, proc_oil_pct]
                          if c != NONE]

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Compute sum & ratio features
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 4 · Computed Features")

# Sum features
df_work = compute_sum_features(df_raw, sel_groups)
computed_sums = [f for f in SUM_FEATURES if f in df_work.columns]

st.markdown("**Auto-computed Sum features** (based on your group selections):")
if computed_sums:
    pills = " ".join(
        f'<span class="group-pill" style="background:#e3f2fd;color:#1565C0;border:1px solid #90caf9">'
        f'{s}</span>'
        for s in computed_sums
    )
    st.markdown(pills, unsafe_allow_html=True)
else:
    st.info("No sum features computed — select at least one column per group above.")

# Ratio features
st.markdown("**Ratio features** — auto-computed must-haves + custom additions:")
avail_for_ratio = non_target + computed_sums
with st.expander("➕ Add custom ratio features", expanded=False):
    n_ratios = st.number_input("Number of ratio features to add",
                                min_value=0, max_value=20, value=0, step=1,
                                key="n_ratios")
    user_ratios: list[tuple[str, str]] = []
    for ri in range(int(n_ratios)):
        rc1, rc2 = st.columns(2)
        num = rc1.selectbox(f"Numerator {ri+1}", avail_for_ratio, key=f"ratio_num_{ri}")
        den = rc2.selectbox(f"Denominator {ri+1}", avail_for_ratio, key=f"ratio_den_{ri}")
        user_ratios.append((num, den))

    df_work = compute_ratio_features(df_work, user_ratios)
    ratio_cols = [f"{n} / {d}" for n, d in user_ratios
                  if f"{n} / {d}" in df_work.columns]
    if ratio_cols:
        st.success(f"Added {len(ratio_cols)} ratio feature(s): {', '.join(ratio_cols)}")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — X-axis columns and Colour columns
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 5 · Select Plot Axes")

# Flatten all selected individual columns (from ALL groups including Oil/Brine/Process)
individual_selected = list(dict.fromkeys(
    c for cols in sel_groups.values() for c in cols
))

# Oil properties (always available as colour even if not in selected groups)
oil_cols_in_data = _avail(GROUPS["Oil"], non_target)

# Must-have X columns — always included when present in df_work,
# regardless of what the user selected in the group pickers.
MUST_HAVE_X = [
    "Anionic (All Types)",
    "Nonionic (All Types)",
    "Zwitterionic (All Types)",
    "Sum Surfactant",
    "Nanoparticle (All Types)",
    "Polymer (All Types)",
    "Acid (All Types)",
    "Citric (All Types)",
]
must_have_present = [f for f in MUST_HAVE_X if f in df_work.columns]

# ── Must-have ratios: (numerator, denominator) pairs always computed ──────────
# Each pair generates TWO features: "A + B" (sum) and "A / B" (ratio).
# Oil Percent column is resolved dynamically from the process selection.
_oil_pct_col = next(iter(sel_groups.get("Process", [])), None)

MUST_HAVE_RATIOS: list[tuple[str, str]] = [
    ("Anionic (All Types)",     "Sum Surfactant"),
    ("Nonionic (All Types)",    "Sum Surfactant"),
    ("Zwitterionic (All Types)","Sum Surfactant"),
    ("Sum Surfactant",          "Sum Surfactant"),  # will be self — skipped in ratio
    ("Nanoparticle (All Types)","Sum Surfactant"),
    ("Polymer (All Types)",     "Sum Surfactant"),
    ("Acid (All Types)",        "Sum Surfactant"),
    ("Citric (All Types)",      "Sum Surfactant"),
]
if _oil_pct_col and _oil_pct_col in df_work.columns:
    MUST_HAVE_RATIOS += [
        (_oil_pct_col, "Sum Surfactant"),
        (_oil_pct_col, "Nanoparticle (All Types)"),
    ]

# Compute must-have ratio features into df_work
for _num, _den in MUST_HAVE_RATIOS:
    if _num == _den:
        continue
    if _num not in df_work.columns or _den not in df_work.columns:
        continue
    _sum_name   = f"{_num} + {_den}"
    _ratio_name = f"{_num} / {_den}"
    if _sum_name not in df_work.columns:
        df_work[_sum_name]   = df_work[_num] + df_work[_den]
    if _ratio_name not in df_work.columns:
        df_work[_ratio_name] = df_work[_num] / df_work[_den].replace(0, float("nan"))

# Collect all ratio feature names now in df_work (must-have + user-defined)
all_ratio_cols = [
    c for c in df_work.columns
    if (" + " in c or " / " in c)
    and c not in computed_sums
    and c not in non_target
]

# X-axis: individual + must-have sums + all computed sums + ratios
x_candidates = list(dict.fromkeys(
    individual_selected + must_have_present + computed_sums + ratio_cols
))

# Colour: same as X plus any oil columns not already included
color_candidates = list(dict.fromkeys(
    individual_selected + computed_sums + ratio_cols + oil_cols_in_data
))

st.markdown("**X-axis features** (one folder per X axis will be created):")
st.caption(
    "🔒 Sum features (Anionic, Nonionic, Zwitterionic, Sum Surfactant, "
    "Nanoparticle, Polymer, Acid, Citric) are always included as X axes "
    "when computable — you can add or remove individual columns below."
)

# Optional extra X columns (user can add/remove raw cols)
# Must-have sum features are injected automatically after widget
optional_x = [c for c in x_candidates if c not in must_have_present]
extra_x = st.multiselect(
    "Additional X axis columns",
    options=optional_x,
    default=optional_x,
    key="x_cols_extra",
    label_visibility="collapsed",
)
# Final x_cols = locked must-haves (if computed) + user selection
x_cols = list(dict.fromkeys(must_have_present + extra_x))

if must_have_present:
    st.info(
        f"✅ Always-included X axes: "
        + ", ".join(f"**{f}**" for f in must_have_present)
    )

st.markdown("**Colour-by features** (each X folder gets one plot per colour):")
color_cols = st.multiselect(
    "Colour-by columns",
    options=color_candidates,
    default=color_candidates[:min(6, len(color_candidates))],
    key="color_cols",
    label_visibility="collapsed",
)

if not x_cols:
    st.warning("Select at least one X-axis column.")
    st.stop()
if not color_cols:
    st.warning("Select at least one colour column.")
    st.stop()

# Per x_col: peers = other x_cols + extra color_cols
def _count_sf1(x):
    peers  = [c for c in x_cols    if c != x]
    extras = [c for c in color_cols if c != x and c not in peers]
    return len(peers) + len(extras)

n_sf1   = sum(_count_sf1(x) for x in x_cols)
n_sf2   = sum(len([r for r in all_ratio_cols if r != x]) for x in x_cols)
n_plots = n_sf1 + n_sf2
st.info(
    f"Will generate **{n_plots}** scatter plots total:  "
    f"**{n_sf1}** in 'Effect of individual component…' + "
    f"**{n_sf2}** in 'Effect of component Ratio…'  "
    f"across **{len(x_cols)}** X-axis folders."
)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — Save destination
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 6 · Save Destination")
save_mode = st.radio(
    "How to get the plots?",
    ["⬇️ Download as ZIP (browser)", "💾 Save directly to disk path"],
    horizontal=True,
    key="save_mode",
)

disk_path = ""
if "disk" in save_mode:
    disk_path = st.text_input(
        "Destination folder path",
        value=str(Path.home() / "foam_plots"),
        help="The folder will be created if it does not exist.",
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Step 7 — Generate
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 7 · Generate Plots")

if st.button("🚀 Generate All Plots", type="primary", use_container_width=True):

    with st.spinner("Building plots…"):
        figures = generate_all_plots(df_work, x_cols, color_cols, all_ratio_cols, target_col)

    st.success(f"✅ Generated **{len(figures)}** plots.")

    # ── Save / download ───────────────────────────────────────────────────
    if "disk" in save_mode:
        if not disk_path.strip():
            st.error("Enter a valid folder path.")
        else:
            try:
                n = save_figures_to_disk(figures, disk_path.strip())
                st.success(
                    f"💾 Saved **{n}** plots to:\n\n"
                    f"`{disk_path.strip()}/2D Plots/<x_folder>/<colour>.png`"
                )
            except Exception as e:
                st.error(f"Save failed: {e}")

    else:
        zip_bytes = figures_to_zip(figures)
        st.download_button(
            label="⬇️ Download all plots as ZIP",
            data=zip_bytes,
            file_name="foam_scatter_plots.zip",
            mime="application/zip",
            use_container_width=True,
        )
        st.caption(
            "ZIP structure:\n"
            "`2D Plots / Effect of individual component… / <X axis> / X_…_Color_….png`\n"
            "`2D Plots / Effect of component Ratio… / <X axis> / X_…_Color_….png`"
        )

    # ── Inline preview of first 6 plots ──────────────────────────────────
    st.divider()
    st.markdown("#### Preview (first 6 plots)")
    preview_items = list(figures.items())[:6]
    prev_cols = st.columns(2)
    for pi, (path_key, png_bytes) in enumerate(preview_items):
        with prev_cols[pi % 2]:
            st.image(io.BytesIO(png_bytes), caption=path_key,
                     use_column_width=True)