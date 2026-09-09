-- Intermediate: dedup to one row per (symbol, bar_ts) keeping the latest ingest,
-- then derive lag/lead features per symbol ordered by time. `next_close` is the
-- forecasting target (tomorrow's close). Rows without a next_close (each
-- symbol's last bar) are dropped downstream in the mart.

with staged as (
    select * from {{ ref('stg_alpaca_bars') }}
),

deduped as (
    select *
    from (
        select
            *,
            row_number() over (
                partition by symbol, bar_ts
                order by ingested_at desc
            ) as _rn
        from staged
    )
    where _rn = 1
),

featured as (
    select
        symbol,
        bar_ts,
        bar_date,
        open,
        high,
        low,
        close,
        volume,
        trade_count,
        vwap,

        -- intraday shape
        high - low                                          as range_abs,
        (close - open)                                      as change_abs,

        -- lagged closes (known at prediction time)
        lag(close, 1) over w                                as close_lag_1,
        lag(close, 2) over w                                as close_lag_2,
        lag(close, 3) over w                                as close_lag_3,
        lag(volume, 1) over w                               as volume_lag_1,

        -- trailing moving averages of close
        avg(close) over (
            partition by symbol order by bar_ts
            rows between 4 preceding and current row
        )                                                   as close_ma_5,
        avg(close) over (
            partition by symbol order by bar_ts
            rows between 9 preceding and current row
        )                                                   as close_ma_10,

        -- forecasting target: next trading day's close
        lead(close, 1) over w                               as next_close
    from deduped
    window w as (partition by symbol order by bar_ts)
)

select
    *,
    case when close_lag_1 is not null and close_lag_1 <> 0
         then close / close_lag_1 - 1 end                   as return_1d
from featured
