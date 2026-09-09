"""Streamlit dashboard for the XGBoost next-day-close prediction pipeline.

Reads artifacts produced by ``ml/train.py`` from ``models_out/`` plus the raw
close-price series from the DuckDB warehouse (``marts.fct_prices``), and
renders:

  1. A header with top-line holdout metrics.
  2. A "Train/Test Split" tab: per-symbol close/prediction lines with the
     train/test boundary called out, an actual-vs-predicted line for the test
     window, and a y_true-vs-y_pred scatter with a 45-degree reference line.
  3. A "Walk-Forward Validation" tab: per-fold RMSE/R2, a fold-metrics table,
     and a combined time plot of each fold's test window colored by fold with
     vertical lines marking the expanding-window boundaries.

This file is self-contained and defensive: every artifact is optional, and a
missing/unreadable file surfaces an ``st.warning`` (or, when nothing
renderable is left, an ``st.stop()``) instead of raising.

Palette / accessibility notes (per the project's dataviz skill):
  - Chart surfaces are left transparent so charts adopt Streamlit's active
    (light or dark) theme rather than fighting it.
  - Categorical series use the validated 8-hue palette from the dataviz skill
    reference (fixed hue order, never re-cycled/re-assigned per filter).
  - Train/test and actual/predicted are distinguished by both color AND
    line style (solid vs dashed) / marker shape, so identity never depends on
    color alone.
  - Axis titles, tick labels, and gridlines use muted/secondary ink so the
    data marks stay the most prominent element on the chart.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import duckdb
except ImportError:  # pragma: no cover - duckdb is an optional extra for this app
    duckdb = None


# --------------------------------------------------------------------------
# Paths (resolved relative to the project root, i.e. the parent of app/, so
# this works regardless of the process's current working directory).
# --------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
MODELS_OUT = ROOT_DIR / "models_out"
DB_PATH = ROOT_DIR / "warehouse" / "pipeline.duckdb"

METRICS_PATH = MODELS_OUT / "metrics.json"
PREDICTIONS_PATH = MODELS_OUT / "predictions.parquet"
WALKFORWARD_PATH = MODELS_OUT / "walkforward.parquet"
WALKFORWARD_METRICS_PATH = MODELS_OUT / "walkforward_metrics.json"


# --------------------------------------------------------------------------
# Dataviz palette (validated categorical order — fixed, never cycled).
# See the project's dataviz skill reference palette for the source values.
# --------------------------------------------------------------------------
PALETTE = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
}
CATEGORICAL_ORDER = [
    PALETTE["blue"],
    PALETTE["orange"],
    PALETTE["aqua"],
    PALETTE["yellow"],
    PALETTE["magenta"],
    PALETTE["green"],
    PALETTE["violet"],
    PALETTE["red"],
]

# Fixed semantic roles used throughout (identity by role, not by rank).
COLOR_ACTUAL = PALETTE["blue"]
COLOR_PREDICTED = PALETTE["orange"]
COLOR_TRAIN = PALETTE["aqua"]
COLOR_TEST = PALETTE["violet"]
COLOR_REFERENCE_LINE = "#898781"  # muted ink, non-data reference (45-degree line)
COLOR_CUTOFF_LINE = "#52514e"  # secondary ink, boundary marker

MUTED_INK = "#898781"
SECONDARY_INK = "#7a7972"
GRIDLINE = "rgba(137, 135, 129, 0.25)"


def _base_layout(fig: go.Figure, title: str, x_title: str, y_title: str) -> go.Figure:
    """Apply shared, theme-friendly layout: transparent surfaces, muted axes,
    a legend (since these charts always have >= 2 series), and a hover-ready
    unified tooltip."""
    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=SECONDARY_INK),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_xaxes(
        title=x_title,
        showgrid=True,
        gridcolor=GRIDLINE,
        zeroline=False,
        color=MUTED_INK,
        title_font=dict(color=SECONDARY_INK),
    )
    fig.update_yaxes(
        title=y_title,
        showgrid=True,
        gridcolor=GRIDLINE,
        zeroline=False,
        color=MUTED_INK,
        title_font=dict(color=SECONDARY_INK),
    )
    return fig


# --------------------------------------------------------------------------
# Cached loaders. Each returns None (or an empty frame) rather than raising,
# so the caller can decide whether to warn-and-continue or warn-and-stop.
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_metrics(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


@st.cache_data(show_spinner=False)
def load_parquet(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_walkforward_metrics(path: str) -> list | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            data = json.load(f)
        return data if isinstance(data, list) else None
    except (json.JSONDecodeError, OSError):
        return None


@st.cache_data(show_spinner=False)
def load_price_series(db_path: str) -> pd.DataFrame | None:
    """Underlying close series from marts.fct_prices, read-only. Optional —
    only used to backfill a 'close' line if predictions lack one."""
    p = Path(db_path)
    if duckdb is None or not p.exists():
        return None
    try:
        con = duckdb.connect(str(p), read_only=True)
        try:
            return con.execute(
                "SELECT symbol, bar_date, close, next_close "
                "FROM marts.fct_prices ORDER BY symbol, bar_date"
            ).fetch_df()
        finally:
            con.close()
    except Exception:
        return None


# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="XGBoost Next-Day-Close Predictor",
    layout="wide",
)

st.title("XGBoost Next-Day-Close Prediction Pipeline")
st.caption(
    "Holdout evaluation and walk-forward validation for the next-day-close "
    "regressor trained on `marts.fct_prices`."
)

metrics = load_metrics(str(METRICS_PATH))
predictions = load_parquet(str(PREDICTIONS_PATH))
walkforward = load_parquet(str(WALKFORWARD_PATH))
walkforward_metrics = load_walkforward_metrics(str(WALKFORWARD_METRICS_PATH))
price_series = load_price_series(str(DB_PATH))

# If literally nothing is available, there is no dashboard to show.
if metrics is None and predictions is None and walkforward is None and walkforward_metrics is None:
    st.warning(
        "No model artifacts found under `models_out/`. Run `ml/train.py` "
        "first to produce `metrics.json`, `predictions.parquet`, "
        "`walkforward.parquet`, and `walkforward_metrics.json`."
    )
    st.stop()

# --------------------------------------------------------------------------
# Header: top-line holdout metrics
# --------------------------------------------------------------------------
if metrics is None:
    st.warning(
        f"`{METRICS_PATH.name}` not found or unreadable under `models_out/`. "
        "Top-line metrics are unavailable."
    )
else:
    cutoff = metrics.get("cutoff_date", "n/a")
    split = metrics.get("split", "n/a")
    target = metrics.get("target", "n/a")
    st.subheader(f"Holdout metrics — split: `{split}` · target: `{target}` · cutoff: `{cutoff}`")

    col1, col2, col3, col4 = st.columns(4)
    rmse = metrics.get("rmse")
    mae = metrics.get("mae")
    r2 = metrics.get("r2")
    n_train = metrics.get("n_train")
    n_test = metrics.get("n_test")

    col1.metric("RMSE", f"{rmse:.4f}" if isinstance(rmse, (int, float)) else "n/a")
    col2.metric("MAE", f"{mae:.4f}" if isinstance(mae, (int, float)) else "n/a")
    col3.metric("R²", f"{r2:.4f}" if isinstance(r2, (int, float)) else "n/a")
    n_train_str = f"{n_train:,}" if isinstance(n_train, int) else "n/a"
    n_test_str = f"{n_test:,}" if isinstance(n_test, int) else "n/a"
    col4.metric("n_train / n_test", f"{n_train_str} / {n_test_str}")

    features = metrics.get("features")
    if isinstance(features, list) and features:
        with st.expander("Model features"):
            st.write(", ".join(str(f) for f in features))

st.divider()

tab_split, tab_walkforward = st.tabs(["Train/Test Split", "Walk-Forward Validation"])


# ==========================================================================
# TAB 1: Train / Test Split
# ==========================================================================
with tab_split:
    if predictions is None or predictions.empty:
        st.warning(
            f"`{PREDICTIONS_PATH.name}` not found, empty, or unreadable under "
            "`models_out/`. Nothing to plot for the train/test split."
        )
    else:
        required_cols = {"symbol", "bar_date", "y_true", "y_pred", "split"}
        missing_cols = required_cols - set(predictions.columns)
        if missing_cols:
            st.warning(
                f"`{PREDICTIONS_PATH.name}` is missing expected column(s): "
                f"{', '.join(sorted(missing_cols))}."
            )
        else:
            df = predictions.copy()
            # Coerce types defensively; bad rows become NaT/NaN and get dropped.
            df["bar_date"] = pd.to_datetime(df["bar_date"], errors="coerce")
            df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
            df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
            df = df.dropna(subset=["bar_date", "symbol"])

            symbols_available = sorted(df["symbol"].dropna().unique().tolist())
            if not symbols_available:
                st.warning("No symbols found in `predictions.parquet`.")
            else:
                selected_symbols = st.multiselect(
                    "Symbols",
                    options=symbols_available,
                    default=symbols_available,
                    key="split_symbols",
                )

                if not selected_symbols:
                    st.info("Select at least one symbol to plot.")
                else:
                    sub = df[df["symbol"].isin(selected_symbols)].sort_values(
                        ["symbol", "bar_date"]
                    )

                    cutoff_date = None
                    if metrics is not None:
                        raw_cutoff = metrics.get("cutoff_date")
                        if raw_cutoff:
                            cutoff_date = pd.to_datetime(raw_cutoff, errors="coerce")
                            if pd.isna(cutoff_date):
                                cutoff_date = None

                    # ---- Chart A: actual vs predicted over time, colored by
                    # symbol (categorical, fixed order), with train/test
                    # distinguished by line style (solid vs dashed) rather
                    # than color alone, plus a shaded test region / cutoff
                    # line for a second, redundant cue.
                    fig_ts = go.Figure()
                    for i, sym in enumerate(selected_symbols):
                        color = CATEGORICAL_ORDER[i % len(CATEGORICAL_ORDER)]
                        sym_df = sub[sub["symbol"] == sym]
                        if sym_df.empty:
                            continue
                        legend_shown_actual = False
                        legend_shown_pred = False
                        for split_name in ("train", "test"):
                            part = sym_df[sym_df["split"] == split_name]
                            if part.empty:
                                continue
                            fig_ts.add_trace(
                                go.Scatter(
                                    x=part["bar_date"],
                                    y=part["y_true"],
                                    mode="lines",
                                    name=f"{sym} actual",
                                    legendgroup=f"{sym}-actual",
                                    showlegend=not legend_shown_actual,
                                    line=dict(color=color, width=2, dash="solid"),
                                    opacity=0.5 if split_name == "train" else 1.0,
                                    hovertemplate=f"{sym} actual (%{{x|%Y-%m-%d}}): %{{y:.2f}}<extra></extra>",
                                )
                            )
                            legend_shown_actual = True
                            fig_ts.add_trace(
                                go.Scatter(
                                    x=part["bar_date"],
                                    y=part["y_pred"],
                                    mode="lines",
                                    name=f"{sym} predicted",
                                    legendgroup=f"{sym}-pred",
                                    showlegend=not legend_shown_pred,
                                    line=dict(color=color, width=2, dash="dot"),
                                    opacity=0.5 if split_name == "train" else 1.0,
                                    hovertemplate=f"{sym} predicted (%{{x|%Y-%m-%d}}): %{{y:.2f}}<extra></extra>",
                                )
                            )
                            legend_shown_pred = True

                    if cutoff_date is not None:
                        fig_ts.add_vline(
                            x=cutoff_date.timestamp() * 1000,
                            line_width=2,
                            line_dash="dash",
                            line_color=COLOR_CUTOFF_LINE,
                            annotation_text="cutoff",
                            annotation_position="top",
                        )

                    fig_ts = _base_layout(
                        fig_ts,
                        "Actual vs predicted close (solid = actual, dotted = predicted; faded = train)",
                        "Date",
                        "Price",
                    )
                    st.plotly_chart(fig_ts, use_container_width=True)

                    # ---- Chart B: actual vs predicted, TEST window only, one
                    # line per series (actual solid, predicted dashed) per
                    # symbol, so the holdout accuracy is easy to read alone.
                    test_sub = sub[sub["split"] == "test"]
                    if test_sub.empty:
                        st.info("No rows with split == 'test' to plot for the test-window comparison.")
                    else:
                        fig_test = go.Figure()
                        for i, sym in enumerate(selected_symbols):
                            color = CATEGORICAL_ORDER[i % len(CATEGORICAL_ORDER)]
                            part = test_sub[test_sub["symbol"] == sym].sort_values("bar_date")
                            if part.empty:
                                continue
                            fig_test.add_trace(
                                go.Scatter(
                                    x=part["bar_date"],
                                    y=part["y_true"],
                                    mode="lines+markers",
                                    name=f"{sym} actual",
                                    line=dict(color=color, width=2, dash="solid"),
                                    marker=dict(size=5, symbol="circle"),
                                    hovertemplate=f"{sym} actual: %{{y:.2f}}<extra></extra>",
                                )
                            )
                            fig_test.add_trace(
                                go.Scatter(
                                    x=part["bar_date"],
                                    y=part["y_pred"],
                                    mode="lines+markers",
                                    name=f"{sym} predicted",
                                    line=dict(color=color, width=2, dash="dash"),
                                    marker=dict(size=5, symbol="diamond"),
                                    hovertemplate=f"{sym} predicted: %{{y:.2f}}<extra></extra>",
                                )
                            )
                        fig_test = _base_layout(
                            fig_test,
                            "Test-window: actual vs predicted",
                            "Date",
                            "Price",
                        )
                        st.plotly_chart(fig_test, use_container_width=True)

                    # ---- Chart C: y_true vs y_pred scatter with 45-degree
                    # reference line (perfect-prediction diagonal).
                    if test_sub.empty:
                        st.info("No test rows available for the y_true vs y_pred scatter.")
                    else:
                        valid = test_sub.dropna(subset=["y_true", "y_pred"])
                        if valid.empty:
                            st.info("y_true/y_pred are all NaN in the test split; nothing to scatter.")
                        else:
                            fig_scatter = go.Figure()
                            lo = float(min(valid["y_true"].min(), valid["y_pred"].min()))
                            hi = float(max(valid["y_true"].max(), valid["y_pred"].max()))
                            for i, sym in enumerate(selected_symbols):
                                color = CATEGORICAL_ORDER[i % len(CATEGORICAL_ORDER)]
                                part = valid[valid["symbol"] == sym]
                                if part.empty:
                                    continue
                                fig_scatter.add_trace(
                                    go.Scatter(
                                        x=part["y_true"],
                                        y=part["y_pred"],
                                        mode="markers",
                                        name=sym,
                                        marker=dict(color=color, size=8, opacity=0.75),
                                        hovertemplate=f"{sym}<br>y_true: %{{x:.2f}}<br>y_pred: %{{y:.2f}}<extra></extra>",
                                    )
                                )
                            # 45-degree reference line: muted, non-data ink,
                            # so it never competes with a real series color.
                            fig_scatter.add_trace(
                                go.Scatter(
                                    x=[lo, hi],
                                    y=[lo, hi],
                                    mode="lines",
                                    name="perfect prediction (y = x)",
                                    line=dict(color=COLOR_REFERENCE_LINE, width=2, dash="dash"),
                                    hoverinfo="skip",
                                )
                            )
                            fig_scatter = _base_layout(
                                fig_scatter,
                                "Test window: y_true vs y_pred",
                                "Actual (y_true)",
                                "Predicted (y_pred)",
                            )
                            fig_scatter.update_yaxes(scaleanchor="x", scaleratio=1)
                            st.plotly_chart(fig_scatter, use_container_width=True)

                            with st.expander("Show underlying test predictions"):
                                st.dataframe(
                                    valid[["symbol", "bar_date", "y_true", "y_pred"]].sort_values(
                                        ["symbol", "bar_date"]
                                    ),
                                    use_container_width=True,
                                )


# ==========================================================================
# TAB 2: Walk-Forward Validation
# ==========================================================================
with tab_walkforward:
    if walkforward_metrics is None and (walkforward is None or walkforward.empty):
        st.warning(
            f"Neither `{WALKFORWARD_METRICS_PATH.name}` nor `{WALKFORWARD_PATH.name}` "
            "were found under `models_out/`. Run the walk-forward step of "
            "`ml/train.py` to populate this tab."
        )
    else:
        # ---- Fold metrics: RMSE/R2 chart + table.
        if walkforward_metrics is None:
            st.warning(f"`{WALKFORWARD_METRICS_PATH.name}` not found or unreadable.")
        else:
            wf_metrics_df = pd.DataFrame(walkforward_metrics)
            if wf_metrics_df.empty:
                st.info("`walkforward_metrics.json` contains no folds.")
            else:
                wf_metrics_df["fold"] = pd.to_numeric(wf_metrics_df.get("fold"), errors="coerce")
                wf_metrics_df = wf_metrics_df.sort_values("fold")

                fig_folds = go.Figure()
                if "rmse" in wf_metrics_df.columns:
                    fig_folds.add_trace(
                        go.Bar(
                            x=wf_metrics_df["fold"],
                            y=wf_metrics_df["rmse"],
                            name="RMSE",
                            marker_color=PALETTE["blue"],
                            hovertemplate="Fold %{x}<br>RMSE: %{y:.4f}<extra></extra>",
                        )
                    )
                if "r2" in wf_metrics_df.columns:
                    fig_folds.add_trace(
                        go.Scatter(
                            x=wf_metrics_df["fold"],
                            y=wf_metrics_df["r2"],
                            name="R²",
                            mode="lines+markers",
                            marker=dict(color=PALETTE["orange"], size=8, symbol="diamond"),
                            line=dict(color=PALETTE["orange"], width=2, dash="dot"),
                            yaxis="y2",
                            hovertemplate="Fold %{x}<br>R²: %{y:.4f}<extra></extra>",
                        )
                    )

                fig_folds = _base_layout(
                    fig_folds,
                    "Per-fold RMSE (bars) and R² (line)",
                    "Fold",
                    "RMSE",
                )
                # Note: RMSE and R2 are on genuinely different scales, so a
                # second axis is used deliberately here as an *overlay* pair
                # (bar + line), not a dual-scaled comparison of the same
                # series — kept to two clearly distinct mark types so the
                # reader isn't misled into comparing heights directly.
                fig_folds.update_layout(
                    yaxis2=dict(
                        title="R²",
                        overlaying="y",
                        side="right",
                        showgrid=False,
                        color=MUTED_INK,
                    ),
                    xaxis=dict(dtick=1),
                )
                st.plotly_chart(fig_folds, use_container_width=True)

                display_cols = [
                    c
                    for c in ["fold", "train_end_date", "n_train", "n_test", "rmse", "mae", "r2"]
                    if c in wf_metrics_df.columns
                ]
                st.dataframe(wf_metrics_df[display_cols], use_container_width=True)

        st.divider()

        # ---- Combined time plot of each fold's test predictions vs actuals,
        # colored by fold (fixed categorical order), with vertical lines at
        # each train_end_date marking the expanding-window boundaries.
        if walkforward is None or walkforward.empty:
            st.warning(
                f"`{WALKFORWARD_PATH.name}` not found, empty, or unreadable under "
                "`models_out/`. Fold-level prediction plot is unavailable."
            )
        else:
            required_wf_cols = {"fold", "train_end_date", "symbol", "bar_date", "y_true", "y_pred"}
            missing_wf_cols = required_wf_cols - set(walkforward.columns)
            if missing_wf_cols:
                st.warning(
                    f"`{WALKFORWARD_PATH.name}` is missing expected column(s): "
                    f"{', '.join(sorted(missing_wf_cols))}."
                )
            else:
                wf = walkforward.copy()
                wf["bar_date"] = pd.to_datetime(wf["bar_date"], errors="coerce")
                wf["train_end_date"] = pd.to_datetime(wf["train_end_date"], errors="coerce")
                wf["y_true"] = pd.to_numeric(wf["y_true"], errors="coerce")
                wf["y_pred"] = pd.to_numeric(wf["y_pred"], errors="coerce")
                wf = wf.dropna(subset=["bar_date", "fold"])

                if wf.empty:
                    st.info("No usable rows in `walkforward.parquet` after cleaning.")
                else:
                    folds = sorted(wf["fold"].unique().tolist())

                    fig_wf = go.Figure()
                    for i, fold in enumerate(folds):
                        color = CATEGORICAL_ORDER[i % len(CATEGORICAL_ORDER)]
                        part = wf[wf["fold"] == fold].sort_values("bar_date")
                        if part.empty:
                            continue
                        fig_wf.add_trace(
                            go.Scatter(
                                x=part["bar_date"],
                                y=part["y_true"],
                                mode="lines",
                                name=f"fold {fold} actual",
                                legendgroup=f"fold-{fold}",
                                line=dict(color=color, width=2, dash="solid"),
                                hovertemplate=f"fold {fold} actual (%{{x|%Y-%m-%d}}): %{{y:.2f}}<extra></extra>",
                            )
                        )
                        fig_wf.add_trace(
                            go.Scatter(
                                x=part["bar_date"],
                                y=part["y_pred"],
                                mode="lines",
                                name=f"fold {fold} predicted",
                                legendgroup=f"fold-{fold}",
                                line=dict(color=color, width=2, dash="dot"),
                                hovertemplate=f"fold {fold} predicted (%{{x|%Y-%m-%d}}): %{{y:.2f}}<extra></extra>",
                            )
                        )

                    # Expanding-window boundaries: one vertical line per
                    # distinct train_end_date, muted/secondary ink so they
                    # read as structure, not data.
                    boundary_dates = sorted(
                        d for d in wf["train_end_date"].dropna().unique().tolist()
                    )
                    for d in boundary_dates:
                        ts = pd.Timestamp(d)
                        fig_wf.add_vline(
                            x=ts.timestamp() * 1000,
                            line_width=1,
                            line_dash="dash",
                            line_color=COLOR_CUTOFF_LINE,
                        )

                    fig_wf = _base_layout(
                        fig_wf,
                        "Walk-forward: test predictions vs actuals by fold "
                        "(dashed vertical lines = train_end_date boundaries)",
                        "Date",
                        "Price",
                    )
                    st.plotly_chart(fig_wf, use_container_width=True)

                    with st.expander("Show underlying walk-forward rows"):
                        st.dataframe(
                            wf[["fold", "train_end_date", "symbol", "bar_date", "y_true", "y_pred"]]
                            .sort_values(["fold", "symbol", "bar_date"]),
                            use_container_width=True,
                        )
