"""Stage 1 - Extract.

Pull raw records from a config-driven REST API into DuckDB, untouched.
The only transforms here are what's needed to *land* the data (flatten JSON,
add audit columns). All cleaning/typing is dbt's job downstream.

Supports:
  * auth via a header -> env-var map (secrets stay in the environment)
  * pagination by numbered page OR opaque continuation token
  * responses that are a plain list, a list at `record_path`, or a
    dict-of-lists keyed by symbol (`record_style: symbol_dict`)
  * single or composite primary keys for idempotent upsert

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


def auth_headers(cfg: dict) -> dict:
    """Resolve {header: env_var_name} -> {header: value} from the environment."""
    headers: dict[str, str] = {}
    for header, env_var in (cfg.get("auth_headers") or {}).items():
        value = os.environ.get(env_var)
        if not value:
            sys.exit(
                f"ERROR: header '{header}' needs env var '{env_var}', which is unset.\n"
                f"       export {env_var}=... before running the extract."
            )
        headers[header] = value
    return headers


def extract_page_records(cfg: dict, payload) -> list[dict]:
    """Pull the list of record dicts out of one response payload."""
    record_style = cfg.get("record_style", "list")
    record_path = cfg.get("record_path") or ""

    node = payload[record_path] if record_path else payload

    if record_style == "symbol_dict":
        # node is {symbol: [record, ...], ...}; inject the key as a column.
        if not isinstance(node, dict):
            sys.exit(f"ERROR: record_style=symbol_dict but '{record_path}' is not a dict.")
        symbol_key = cfg.get("symbol_key", "symbol")
        rows: list[dict] = []
        for key, records in node.items():
            for rec in records or []:
                rows.append({symbol_key: key, **rec})
        return rows

    if not isinstance(node, list):
        sys.exit(f"ERROR: expected a list of records, got {type(node)}.")
    return node


def get_records(cfg: dict, session: requests.Session) -> list[dict]:
    """Page through the API and return the raw list of record dicts."""
    headers = auth_headers(cfg)
    paging = cfg.get("paging") or {}
    style = paging.get("style", "page")
    max_pages = paging.get("max_pages", 1)
    base_params = dict(cfg.get("params") or {})

    records: list[dict] = []
    token = None
    for page in range(1, max_pages + 1):
        params = dict(base_params)
        if style == "token":
            if token:
                params[paging["token_param"]] = token
        elif style == "page":
            per_page = paging.get("per_page")
            if per_page:
                params[paging["per_page_param"]] = per_page
                params[paging["page_param"]] = page

        resp = session.get(cfg["base_url"], params=params, headers=headers, timeout=60)
        if not resp.ok:
            sys.exit(f"ERROR: {resp.status_code} from API: {resp.text[:300]}")
        payload = resp.json()

        page_records = extract_page_records(cfg, payload)
        records.extend(page_records)

        if style == "token":
            token = payload.get(paging["token_response_field"])
            if not token:
                break
        else:  # numbered pages
            per_page = paging.get("per_page")
            if not page_records or (per_page and len(page_records) < per_page):
                break
    return records


def pk_columns(cfg: dict) -> list[str]:
    pk = cfg["primary_key"]
    return [pk] if isinstance(pk, str) else list(pk)


def land(cfg: dict, records: list[dict]) -> int:
    """Flatten records to a table and idempotently upsert into DuckDB raw.*"""
    if not records:
        sys.exit("ERROR: no records returned from the API; nothing to land.")

    source = cfg["source_name"]
    keys = pk_columns(cfg)

    df = pd.json_normalize(records)
    missing = [k for k in keys if k not in df.columns]
    if missing:
        sys.exit(f"ERROR: primary_key column(s) {missing} not in API response.")
    df["_ingested_at"] = datetime.now(timezone.utc)
    df["_source"] = source

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    try:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {RAW_SCHEMA}")
        table = f"{RAW_SCHEMA}.{source}"
        con.register("incoming", df)
        con.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM incoming LIMIT 0")
        # Idempotent upsert on the (composite) primary key.
        key_tuple = ", ".join(keys)
        con.execute(
            f"DELETE FROM {table} "
            f"WHERE ({key_tuple}) IN (SELECT {key_tuple} FROM incoming)"
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
