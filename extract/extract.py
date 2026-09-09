"""Stage 1 - Extract.

Pull raw records from a config-driven REST API into DuckDB, untouched.
The only transforms here are what's needed to *land* the data (flatten JSON,
add audit columns). All cleaning/typing is dbt's job downstream.

Supports:
  * auth via header map and/or query-param map (secrets stay in the environment)
  * one request per symbol (loop) OR numbered-page / token pagination
  * responses that are a plain list, a list at `record_path`, a dict-of-lists
    keyed by symbol (`symbol_dict`), or a dict keyed by date (`date_dict`)
  * single or composite primary keys for idempotent upsert

Idempotent: re-running replaces rows with the same primary key, so the raw
table row count is stable across runs.
"""

from __future__ import annotations

import os
import sys
import time
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

# Keys some APIs (e.g. Alpha Vantage) use to signal rate limits / bad keys
# instead of an HTTP error status.
API_MESSAGE_KEYS = ("Note", "Information", "Error Message")


def load_dotenv() -> None:
    """Load KEY=VALUE lines from a gitignored .env into the environment.

    Keeps secrets out of the config file and off the command line. Existing
    environment variables win, so `export`-ed values still take precedence.
    """
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


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


def _resolve_env_map(mapping: dict, what: str) -> dict:
    """Resolve {name: env_var_name} -> {name: value} from the environment."""
    resolved: dict[str, str] = {}
    for name, env_var in (mapping or {}).items():
        value = os.environ.get(env_var)
        if not value:
            sys.exit(
                f"ERROR: {what} '{name}' needs env var '{env_var}', which is unset.\n"
                f"       export {env_var}=... before running the extract."
            )
        resolved[name] = value
    return resolved


def auth_headers(cfg: dict) -> dict:
    return _resolve_env_map(cfg.get("auth_headers") or {}, "header")


def auth_params(cfg: dict) -> dict:
    return _resolve_env_map(cfg.get("auth_params") or {}, "query param")


def records_from_payload(cfg: dict, payload) -> list[dict]:
    """Pull the list of record dicts out of one response payload."""
    record_style = cfg.get("record_style", "list")
    record_path = cfg.get("record_path") or ""

    if record_path and record_path not in payload:
        # Surface API-level rate-limit / error messages instead of a KeyError.
        for key in API_MESSAGE_KEYS:
            if isinstance(payload, dict) and key in payload:
                sys.exit(f"ERROR: API responded with {key}: {payload[key]}")
        sys.exit(f"ERROR: expected '{record_path}' in response; got keys {list(payload)[:8]}.")

    node = payload[record_path] if record_path else payload

    if record_style == "symbol_dict":
        if not isinstance(node, dict):
            sys.exit("ERROR: record_style=symbol_dict but node is not a dict.")
        symbol_key = cfg.get("symbol_key", "symbol")
        return [
            {symbol_key: key, **rec}
            for key, recs in node.items()
            for rec in (recs or [])
        ]

    if record_style == "date_dict":
        if not isinstance(node, dict):
            sys.exit("ERROR: record_style=date_dict but node is not a dict.")
        date_key = cfg.get("date_key", "date")
        return [{date_key: d, **fields} for d, fields in node.items()]

    if not isinstance(node, list):
        sys.exit(f"ERROR: expected a list of records, got {type(node)}.")
    return node


def fetch(cfg, session, extra_params) -> list[dict]:
    """Issue one GET and return its records."""
    params = {**(cfg.get("params") or {}), **auth_params(cfg), **extra_params}
    resp = session.get(
        cfg["base_url"], params=params, headers=auth_headers(cfg), timeout=60
    )
    if not resp.ok:
        sys.exit(f"ERROR: {resp.status_code} from API: {resp.text[:300]}")
    return records_from_payload(cfg, resp.json())


def get_records(cfg: dict, session: requests.Session) -> list[dict]:
    """Fetch across symbols (and/or pages) and return all record dicts."""
    per_symbol = cfg.get("per_symbol")
    if per_symbol:
        symbols = per_symbol["symbols"]
        sym_param = per_symbol.get("symbol_param", "symbol")
        sym_key = per_symbol.get("symbol_key", "symbol")
        delay = per_symbol.get("delay_seconds", 0)

        records: list[dict] = []
        for i, symbol in enumerate(symbols):
            if i > 0 and delay:
                time.sleep(delay)  # respect per-minute rate limits
            rows = fetch(cfg, session, {sym_param: symbol})
            for row in rows:
                row.setdefault(sym_key, symbol)
            print(f"[extract]   {symbol}: {len(rows)} rows")
            records.extend(rows)
        return records

    # Single endpoint (optionally paginated).
    paging = cfg.get("paging") or {}
    style = paging.get("style", "page")
    max_pages = paging.get("max_pages", 1)

    records = []
    token = None
    for page in range(1, max_pages + 1):
        extra: dict = {}
        if style == "token" and token:
            extra[paging["token_param"]] = token
        elif style == "page":
            per_page = paging.get("per_page")
            if per_page:
                extra[paging["per_page_param"]] = per_page
                extra[paging["page_param"]] = page

        params = {**(cfg.get("params") or {}), **auth_params(cfg), **extra}
        resp = session.get(
            cfg["base_url"], params=params, headers=auth_headers(cfg), timeout=60
        )
        if not resp.ok:
            sys.exit(f"ERROR: {resp.status_code} from API: {resp.text[:300]}")
        payload = resp.json()
        page_records = records_from_payload(cfg, payload)
        records.extend(page_records)

        if style == "token":
            token = payload.get(paging["token_response_field"])
            if not token:
                break
        else:
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
        key_tuple = ", ".join(f'"{k}"' for k in keys)
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
    load_dotenv()
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
