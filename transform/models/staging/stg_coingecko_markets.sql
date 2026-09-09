-- Staging: 1:1 with the raw source. Cast types, rename to snake_case,
-- light standardization. No dedup or business logic here.

with source as (
    select * from {{ source('raw', 'coingecko_markets') }}
)

select
    -- identifiers
    cast(id as varchar)                                as coin_id,
    lower(cast(symbol as varchar))                     as symbol,
    cast(name as varchar)                              as coin_name,

    -- target + market fundamentals
    cast(current_price as double)                      as current_price,
    cast(market_cap as double)                         as market_cap,
    cast(market_cap_rank as integer)                   as market_cap_rank,
    cast(fully_diluted_valuation as double)            as fully_diluted_valuation,
    cast(total_volume as double)                       as total_volume,

    -- 24h dynamics
    cast(price_change_percentage_24h as double)        as price_change_pct_24h,
    cast(market_cap_change_percentage_24h as double)   as market_cap_change_pct_24h,

    -- supply
    cast(circulating_supply as double)                 as circulating_supply,
    cast(total_supply as double)                       as total_supply,
    cast(max_supply as double)                         as max_supply,

    -- all-time high / low context
    cast(ath as double)                                as ath,
    cast(ath_change_percentage as double)              as ath_change_pct,
    cast(atl as double)                                as atl,
    cast(atl_change_percentage as double)              as atl_change_pct,

    -- audit
    cast(_ingested_at as timestamp)                    as ingested_at,
    cast(_source as varchar)                           as source_name
from source
