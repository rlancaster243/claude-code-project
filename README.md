# claude-code-project

End-to-end analytics pipeline demonstrating clean **separation of concerns**:
a public REST API is extracted into DuckDB, transformed with dbt-core into
fact/dimension marts, then a mart feeds an **XGBoost** regressor that predicts
next-day close price, evaluated with a time-ordered split and walk-forward
validation, and visualized in a **Streamlit** dashboard.

```
Alpha Vantage TIME_SERIES_DAILY
   │  extract/extract.py  (requests -> DuckDB, raw landing zone)
   ▼
DuckDB raw.<source>
   │  dbt-core (dbt-duckdb)
   ▼
staging  ->  intermediate  ->  marts (fct_prices + dim_symbol)
   │  ml/train.py  (reads marts.fct_prices)
   ▼
XGBoost regressor  ->  70/30 time-ordered split + 5-fold walk-forward
   ▼
models_out/  (model.json, metrics.json, predictions.parquet,
              walkforward.parquet, walkforward_metrics.json)
   │  app/streamlit_app.py
   ▼
Streamlit dashboard  (make app)
```

Each stage owns one responsibility and hands a defined contract to the next:
raw table → dbt sources → mart table → model features → dashboard.

## Stack

- **Extract:** Python (`requests`, `duckdb`, `pandas`) — config-driven via `config/source.yml`
- **Transform:** dbt-core + `dbt-duckdb`
- **Predict:** `xgboost` + `scikit-learn`
- **Visualize:** `streamlit` + `plotly`
- **Env/deps:** `uv`  ·  **Orchestration:** `make`

## Quickstart

```bash
uv sync            # or: make setup

cp .env.example .env
# edit .env and set ALPHAVANTAGE_API_KEY=...   (never commit this file)

make all           # extract -> transform -> train
make app           # launch the Streamlit dashboard
```

Individual stages: `make extract` · `make transform` · `make train` · `make app` · `make clean`

## Using a different API

The active source is Alpha Vantage's `TIME_SERIES_DAILY` endpoint (free tier,
`outputsize=compact`, ~100 most recent daily bars per symbol) for five symbols
(`AAPL`, `MSFT`, `GOOGL`, `AMZN`, `TSLA`), predicting each symbol's **next-day
close** from current + lagged OHLCV features. To point at another REST API,
edit `config/source.yml` (`base_url`, `params`, `record_style`, `record_path`,
`primary_key`, `target_column`) and adjust the staging model in
`transform/models/staging/` to match the new columns. The intermediate, mart,
and ML layers are driven off those names.

Auth is a query-parameter API key pulled from an env var, so **secrets never
live in the repo**: `config/source.yml` only stores the env-var name
(`ALPHAVANTAGE_API_KEY`), and its value is loaded from a gitignored `.env`
file (copy `.env.example` to get started). Never commit the key.

## Model evaluation

`ml/train.py` evaluates the model two ways:

- A **time-ordered 70/30 train/test split** (no shuffling, since this is a
  time series) — predictions and metrics land in `models_out/predictions.parquet`
  and `models_out/metrics.json`.
- A **5-fold expanding-window walk-forward validation**, which retrains on a
  growing prefix of history and evaluates on the next slice — results land in
  `models_out/walkforward.parquet` and `models_out/walkforward_metrics.json`.

`make app` launches a Streamlit dashboard (`app/streamlit_app.py`) that
visualizes both the train/test split and the walk-forward validation.

## Layout

| Path | Responsibility |
|------|----------------|
| `config/source.yml` | API endpoint, paging, primary key, target column |
| `extract/extract.py` | REST → DuckDB `raw.*` (idempotent upsert) |
| `transform/models/staging/` | type-cast / rename, 1:1 with raw |
| `transform/models/intermediate/` | dedup + derived features |
| `transform/models/marts/` | `fct_prices`, `dim_symbol` + schema tests |
| `ml/features.py` | shared feature list + target |
| `ml/train.py` | XGBoost train/eval (split + walk-forward), writes `models_out/*` |
| `app/streamlit_app.py` | Streamlit dashboard over `models_out/*` |

## Branches

`Staging` is the working/sandbox branch; `main` is the long-horizon branch.
Work merges into `Staging` via PR.
