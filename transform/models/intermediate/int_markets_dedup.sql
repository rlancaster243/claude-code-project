-- Intermediate: deduplicate to one row per coin (latest ingest wins) and
-- derive analysis-ready fields. Rows without a price target are dropped here.

with staged as (
    select * from {{ ref('stg_coingecko_markets') }}
),

ranked as (
    select
        *,
        row_number() over (
            partition by coin_id
            order by ingested_at desc
        ) as _rn
    from staged
)

select
    coin_id,
    symbol,
    coin_name,
    current_price,
    market_cap,
    market_cap_rank,
    fully_diluted_valuation,
    total_volume,
    price_change_pct_24h,
    market_cap_change_pct_24h,
    circulating_supply,
    total_supply,
    max_supply,
    ath,
    ath_change_pct,
    atl,
    atl_change_pct,

    -- derived features
    case
        when market_cap > 0 then total_volume / market_cap
    end                                                as volume_to_mcap_ratio,
    case
        when max_supply > 0 then circulating_supply / max_supply
    end                                                as supply_utilization,

    ingested_at
from ranked
where _rn = 1
  and current_price is not null
