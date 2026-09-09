.PHONY: all extract transform train app clean setup

setup:  ## Resolve & install the uv environment
	uv sync

extract:  ## Stage 1: REST API -> DuckDB raw table
	uv run python extract/extract.py

transform:  ## Stage 2: dbt build (run models + tests)
	cd transform && uv run dbt build --profiles-dir .

train:  ## Stage 3: train XGBoost on the mart, write metrics
	uv run python ml/train.py

app:  ## Launch the Streamlit dashboard
	uv run streamlit run app/streamlit_app.py

all: extract transform train  ## Full pipeline end-to-end

clean:  ## Remove the warehouse DB and model artifacts
	rm -f warehouse/*.duckdb warehouse/*.duckdb.wal
	rm -f models_out/model.json models_out/metrics.json models_out/predictions.parquet models_out/walkforward.parquet models_out/walkforward_metrics.json
	rm -rf transform/target transform/logs
