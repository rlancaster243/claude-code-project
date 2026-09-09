-- Fact: one row per coin observation. Grain = coin_id. Carries the price
-- target plus the numeric features consumed by the XGBoost model.

select
    coin_id,                              -- FK -> dim_coin

    -- target
    current_price,

    -- features (no direct price leakage: high_24h/low_24h/absolute change omitted)
    market_cap,
    market_cap_rank,
    fully_diluted_valuation,
    total_volume,
    price_change_pct_24h,
    market_cap_change_pct_24h,
    circulating_supply,
    total_supply,
    max_supply,
    ath_change_pct,
    atl_change_pct,
    volume_to_mcap_ratio,
    supply_utilization,

    ingested_at
from {{ ref('int_markets_dedup') }}
