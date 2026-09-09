"""Single source of truth for the model's feature set and target.

Target is read from config/source.yml so it stays in sync with the pipeline.
Feature list mirrors the numeric columns exposed by marts.fct_prices.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "source.yml"
DB_PATH = ROOT / "warehouse" / "pipeline.duckdb"
MART_RELATION = "marts.fct_prices"

# Numeric features exposed by fct_prices (excludes ids, target, and audit cols).
# All are known at the close of the current bar; the target is the next close.
FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "vwap",
    "range_abs",
    "change_abs",
    "return_1d",
    "close_lag_1",
    "close_lag_2",
    "close_lag_3",
    "volume_lag_1",
    "close_ma_5",
    "close_ma_10",
]


def target_column() -> str:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)["target_column"]
