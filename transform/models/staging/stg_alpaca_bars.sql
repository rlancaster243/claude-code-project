-- Staging: 1:1 with the raw Alpaca bars source. Cast types, rename the terse
-- OHLCV keys to readable names. No dedup or business logic here.

with source as (
    select * from {{ source('raw', 'alpaca_stock_bars') }}
)

select
    cast(symbol as varchar)          as symbol,
    cast(t as timestamp)             as bar_ts,
    cast(t as date)                  as bar_date,

    -- OHLCV
    cast(o as double)                as open,
    cast(h as double)                as high,
    cast(l as double)                as low,
    cast(c as double)                as close,
    cast(v as bigint)                as volume,
    cast(n as bigint)                as trade_count,
    cast(vw as double)               as vwap,

    -- audit
    cast(_ingested_at as timestamp)  as ingested_at,
    cast(_source as varchar)         as source_name
from source
