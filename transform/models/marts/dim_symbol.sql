-- Dimension: one row per traded symbol with its observed history window.

select
    symbol,
    min(bar_date)   as first_bar_date,
    max(bar_date)   as last_bar_date,
    count(*)        as n_bars
from {{ ref('int_bars_features') }}
group by symbol
