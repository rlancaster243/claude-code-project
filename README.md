# claude-code-project

End-to-end analytics pipeline demonstrating clean **separation of concerns**:
a public REST API is extracted into DuckDB, transformed with dbt-core into
fact/dimension marts, then a mart feeds an **XGBoost** regressor that predicts
price via a train/test split.

```
Public REST API
   │  extract/extract.py  (requests -> DuckDB, raw landing zone)
   ▼
DuckDB raw.<source>
   │  dbt-core (dbt-duckdb)
   ▼
staging  ->  intermediate  ->  marts (fct_prices + dim_coin)
   │  ml/train.py  (reads marts.fct_prices)
   ▼
XGBoost regressor  ->  train/test split  ->  models_out/metrics.json
```

Each stage owns one responsibility and hands a defined contract to the next:
raw table → dbt sources → mart table → model features.

## Stack

- **Extract:** Python (`requests`, `duckdb`, `pandas`) — config-driven via `config/source.yml`
- **Transform:** dbt-core + `dbt-duckdb`
- **Predict:** `xgboost` + `scikit-learn`
- **Env/deps:** `uv`  ·  **Orchestration:** `make`

## Quickstart

```bash
uv sync            # or: make setup
make all           # extract -> transform -> train
```

Individual stages: `make extract` · `make transform` · `make train` · `make clean`

## Using a different API

The default source is CoinGecko `/coins/markets` (no API key, target =
`current_price`). To point at another REST API, edit `config/source.yml`
(`base_url`, `params`, `record_path`, `primary_key`, `target_column`) and adjust
the staging model `transform/models/staging/` to match the new columns. The
intermediate, mart, and ML layers are driven off those names.

If the API needs a key, set `auth_env` to the name of an environment variable
holding the key (never commit the key itself).

## Layout

| Path | Responsibility |
|------|----------------|
| `config/source.yml` | API endpoint, paging, primary key, target column |
| `extract/extract.py` | REST → DuckDB `raw.*` (idempotent upsert) |
| `transform/models/staging/` | type-cast / rename, 1:1 with raw |
| `transform/models/intermediate/` | dedup + derived features |
| `transform/models/marts/` | `fct_prices`, `dim_coin` + schema tests |
| `ml/features.py` | shared feature list + target |
| `ml/train.py` | XGBoost train/eval, writes `models_out/metrics.json` |

## Branches

`Staging` is the working/sandbox branch; `main` is the long-horizon branch.
Work merges into `Staging` via PR.
