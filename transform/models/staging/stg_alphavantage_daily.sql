-- Staging: 1:1 with the raw Alpha Vantage daily bars source. Cast types, rename
-- the terse OHLCV keys to readable names. No dedup or business logic here.

with source as (
    select * from {{ source('raw', 'alphavantage_daily') }}
)

select
    cast(symbol as varchar)          as symbol,
    cast(bar_date as date)           as bar_date,

    -- OHLCV
    cast("1. open" as double)        as open,
    cast("2. high" as double)        as high,
    cast("3. low" as double)         as low,
    cast("4. close" as double)       as close,
    cast("5. volume" as bigint)      as volume,

    -- audit
    cast(_ingested_at as timestamp)  as ingested_at,
    cast(_source as varchar)         as source_name
from source
