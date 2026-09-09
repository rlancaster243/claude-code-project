"""Stage 1 - Extract.

Pull raw records from a config-driven REST API into DuckDB, untouched.
The only transforms here are what's needed to *land* the data (flatten JSON,
add audit columns). All cleaning/typing is dbt's job downstream.

Idempotent: re-running replaces rows with the same primary key, so the raw
table row count is stable across runs.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "source.yml"
DB_PATH = ROOT / "warehouse" / "pipeline.duckdb"
RAW_SCHEMA = "raw"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def build_session() -> requests.Session:
    """Session with retry/backoff on transient errors."""
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session


def get_records(cfg: dict, session: requests.Session) -> list[dict]:
    """Page through the API and return the raw list of record dicts."""
    headers = {}
    auth_env = cfg.get("auth_env") or ""
    if auth_env:
        key = os.environ.get(auth_env)
        if not key:
            sys.exit(f"ERROR: config auth_env='{auth_env}' but that env var is unset.")
        headers["x-cg-demo-api-key"] = key

    paging = cfg.get("paging") or {}
    per_page = paging.get("per_page")
    max_pages = paging.get("max_pages", 1)
    record_path = cfg.get("record_path") or ""

    records: list[dict] = []
    for page in range(1, max_pages + 1):
        params = dict(cfg.get("params") or {})
        if per_page:
            params[paging["per_page_param"]] = per_page
            params[paging["page_param"]] = page

        resp = session.get(cfg["base_url"], params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        page_records = payload[record_path] if record_path else payload
        if not isinstance(page_records, list):
            sys.exit(f"ERROR: expected a list of records, got {type(page_records)}.")
        if not page_records:
            break
        records.extend(page_records)
        if per_page and len(page_records) < per_page:
            break  # last page
    return records


def land(cfg: dict, records: list[dict]) -> int:
    """Flatten records to a table and idempotently upsert into DuckDB raw.*"""
    if not records:
        sys.exit("ERROR: no records returned from the API; nothing to land.")

    source = cfg["source_name"]
    pk = cfg["primary_key"]

    df = pd.json_normalize(records)
    if pk not in df.columns:
        sys.exit(f"ERROR: primary_key '{pk}' not present in API response columns.")
    df["_ingested_at"] = datetime.now(timezone.utc)
    df["_source"] = source

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    try:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {RAW_SCHEMA}")
        table = f"{RAW_SCHEMA}.{source}"
        con.register("incoming", df)
        # Create table on first run from the incoming shape.
        con.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM incoming LIMIT 0")
        # Idempotent upsert: delete matching PKs, then insert the fresh rows.
        con.execute(
            f"DELETE FROM {table} WHERE {pk} IN (SELECT {pk} FROM incoming)"
        )
        con.execute(f"INSERT INTO {table} BY NAME SELECT * FROM incoming")
        count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        con.unregister("incoming")
        con.close()
    return count


def main() -> None:
    cfg = load_config()
    session = build_session()
    records = get_records(cfg, session)
    total = land(cfg, records)
    print(
        f"[extract] landed {len(records)} records into "
        f"{RAW_SCHEMA}.{cfg['source_name']} ({total} rows total) at {DB_PATH}"
    )


if __name__ == "__main__":
    main()
