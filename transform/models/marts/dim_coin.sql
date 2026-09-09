-- Dimension: descriptive attributes of each coin.

select
    coin_id,
    symbol,
    coin_name,
    market_cap_rank
from {{ ref('int_markets_dedup') }}
