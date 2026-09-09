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
FEATURE_COLUMNS = [
    "market_cap",
    "market_cap_rank",
    "fully_diluted_valuation",
    "total_volume",
    "price_change_pct_24h",
    "market_cap_change_pct_24h",
    "circulating_supply",
    "total_supply",
    "max_supply",
    "ath_change_pct",
    "atl_change_pct",
    "volume_to_mcap_ratio",
    "supply_utilization",
]


def target_column() -> str:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)["target_column"]
