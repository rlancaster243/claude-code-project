-- Fact: one row per (symbol, bar) observation. Grain = symbol + bar_date.
-- Carries the forecasting target `next_return` (with `next_close` kept for
-- reference/plots) plus the features known at the close of the current bar.
-- Rows lacking a target (each symbol's last bar) or warm-up lags are dropped
-- so the model only sees complete examples.

select
    symbol,                               -- FK -> dim_symbol
    bar_date,

    -- target: next trading day's simple return; next_close kept for context
    next_return,
    next_close,

    -- current-bar features (all known at prediction time)
    open,
    high,
    low,
    close,
    volume,
    range_abs,
    change_abs,
    return_1d,

    -- lagged / trailing features
    close_lag_1,
    close_lag_2,
    close_lag_3,
    volume_lag_1,
    close_ma_5,
    close_ma_10
from {{ ref('int_bars_features') }}
where next_close is not null
  and close_lag_3 is not null             -- ensure warm-up lags are populated
