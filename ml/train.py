"""Stage 3 - Predict.

Load marts.fct_prices from DuckDB, train an XGBoost regressor on a train/test
split, evaluate (RMSE / MAE / R2), and persist the model + metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from features import DB_PATH, FEATURE_COLUMNS, MART_RELATION, ROOT, target_column

OUT_DIR = ROOT / "models_out"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_mart() -> "tuple[np.ndarray, np.ndarray, list[str]]":
    target = target_column()
    cols = FEATURE_COLUMNS + [target]
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        df = con.execute(
            f"SELECT {', '.join(cols)} FROM {MART_RELATION}"
        ).fetch_df()
    finally:
        con.close()

    df = df.dropna(subset=[target])
    # XGBoost handles NaN in features natively; no imputation needed.
    X = df[FEATURE_COLUMNS]
    y = df[target]
    if len(df) < 10:
        raise SystemExit(
            f"ERROR: only {len(df)} rows in {MART_RELATION}; need >= 10 to split."
        )
    return X, y


def main() -> None:
    X, y = load_mart()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
        "features": FEATURE_COLUMNS,
        "target": target_column(),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(OUT_DIR / "model.json"))
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("[train] metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"[train] model -> {OUT_DIR / 'model.json'}")


if __name__ == "__main__":
    main()
