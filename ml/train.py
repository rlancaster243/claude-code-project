"""Stage 3 - Predict.

Load marts.fct_prices from DuckDB, train an XGBoost regressor on a
TIME-ORDERED 70/30 split (no shuffling, to avoid look-ahead leakage), evaluate
(RMSE / MAE / R2), run an expanding-window walk-forward validation, and
persist the model + metrics + predictions.
"""

from __future__ import annotations

import json

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from features import DB_PATH, FEATURE_COLUMNS, MART_RELATION, OUT_DIR, target_column

RANDOM_STATE = 42
TRAIN_FRACTION = 0.70
N_WALKFORWARD_FOLDS = 5
MIN_FOLD_TRAIN_ROWS = 5


def load_mart() -> "pd.DataFrame":
    target = target_column()
    cols = ["symbol", "bar_date", target] + FEATURE_COLUMNS
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        df = con.execute(
            f"SELECT {', '.join(cols)} FROM {MART_RELATION}"
        ).fetch_df()
    finally:
        con.close()

    df = df.dropna(subset=[target])
    df = df.sort_values("bar_date").reset_index(drop=True)
    if len(df) < 10:
        raise SystemExit(
            f"ERROR: only {len(df)} rows in {MART_RELATION}; need >= 10 to split."
        )
    return df


def make_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def time_split(df: "pd.DataFrame", target: str) -> None:
    """Compute the 70/30 time-ordered split, train the final model, and
    write model.json, metrics.json, predictions.parquet.
    """
    dates = np.sort(df["bar_date"].unique())
    cutoff_idx = int(np.floor(TRAIN_FRACTION * (len(dates) - 1)))
    cutoff_date = dates[cutoff_idx]

    df = df.copy()
    df["split"] = np.where(df["bar_date"] <= cutoff_date, "train", "test")

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    model = make_model()
    model.fit(train_df[FEATURE_COLUMNS], train_df[target])

    df["y_pred"] = model.predict(df[FEATURE_COLUMNS])

    test_metrics = evaluate(test_df[target], model.predict(test_df[FEATURE_COLUMNS]))
    metrics = {
        "split": "time_70_30",
        "cutoff_date": str(cutoff_date),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        **test_metrics,
        "features": FEATURE_COLUMNS,
        "target": target,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(OUT_DIR / "model.json"))
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    predictions = df[["symbol", "bar_date", target, "y_pred", "split"]].rename(
        columns={target: "y_true"}
    )
    predictions.to_parquet(OUT_DIR / "predictions.parquet", index=False)

    print("[train] holdout (time 70/30) metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"[train] model -> {OUT_DIR / 'model.json'}")

    return metrics


def walk_forward(df: "pd.DataFrame", target: str) -> None:
    """Expanding-window walk-forward validation over N_WALKFORWARD_FOLDS
    folds, refitting a fresh model each fold. Writes walkforward.parquet and
    walkforward_metrics.json.
    """
    dates = np.sort(df["bar_date"].unique())
    n_dates = len(dates)

    # 5 evenly spaced cutoffs across the middle-to-late range of dates, plus
    # the final date as the end of the last test window -> 6 boundary points
    # bracketing N_WALKFORWARD_FOLDS expanding-window folds.
    fractions = np.linspace(0.5, 1.0, N_WALKFORWARD_FOLDS + 1)
    boundaries = [dates[int(np.floor(frac * (n_dates - 1)))] for frac in fractions]

    fold_metrics = []
    fold_predictions = []

    for k in range(1, N_WALKFORWARD_FOLDS + 1):
        train_end = boundaries[k - 1]
        test_end = boundaries[k]

        train_df = df[df["bar_date"] <= train_end]
        test_df = df[(df["bar_date"] > train_end) & (df["bar_date"] <= test_end)]

        if len(train_df) < MIN_FOLD_TRAIN_ROWS or len(test_df) == 0:
            continue

        model = make_model()
        model.fit(train_df[FEATURE_COLUMNS], train_df[target])
        y_pred = model.predict(test_df[FEATURE_COLUMNS])

        fold_eval = evaluate(test_df[target], y_pred)
        fold_metrics.append(
            {
                "fold": k,
                "train_end_date": str(train_end),
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                **fold_eval,
            }
        )

        fold_pred_df = pd.DataFrame(
            {
                "fold": k,
                "train_end_date": str(train_end),
                "symbol": test_df["symbol"].values,
                "bar_date": test_df["bar_date"].values,
                "y_true": test_df[target].values,
                "y_pred": y_pred,
            }
        )
        fold_predictions.append(fold_pred_df)

    if fold_predictions:
        walkforward_df = pd.concat(fold_predictions, ignore_index=True)
    else:
        walkforward_df = pd.DataFrame(
            columns=["fold", "train_end_date", "symbol", "bar_date", "y_true", "y_pred"]
        )
    walkforward_df.to_parquet(OUT_DIR / "walkforward.parquet", index=False)

    with open(OUT_DIR / "walkforward_metrics.json", "w") as f:
        json.dump(fold_metrics, f, indent=2)

    print("[train] walk-forward fold table:")
    header = f"{'fold':>4} {'train_end':>12} {'n_train':>8} {'n_test':>7} {'rmse':>10} {'mae':>10} {'r2':>8}"
    print(header)
    for m in fold_metrics:
        print(
            f"{m['fold']:>4} {m['train_end_date']:>12} {m['n_train']:>8} "
            f"{m['n_test']:>7} {m['rmse']:>10.4f} {m['mae']:>10.4f} {m['r2']:>8.4f}"
        )

    return fold_metrics


def main() -> None:
    target = target_column()
    df = load_mart()

    time_split(df, target)
    walk_forward(df, target)


if __name__ == "__main__":
    main()
