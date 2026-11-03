"""Extract the engine's CSVs from factor-data (read-only Postgres mirror of Close / GHL / Meta / Typeform).

    python run.py --extract            # runs sql/*.sql and writes data/leads.csv, outcomes.csv, ad_spend.csv

Needs DATABASE_URL in .env: a READ-ONLY Postgres role (ask Hari). Never a Supabase service-role key.
The session is also forced read-only here, and the output is checked for raw PII columns before writing.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"
QUERIES = {"leads": "leads.sql", "outcomes": "outcomes.sql", "ad_spend": "ad_spend.sql"}
# raw personal data must never leave the database (guide section 9)
PII_COLUMNS = re.compile(r"^(email|phone|e_?mail|mobile|first_?name|last_?name|full_?name|name|address|ip|user_agent|notes?)$",
                         re.IGNORECASE)


def _redact(url: str) -> str:
    return re.sub(r"//[^@/]*@", "//***@", url)


def connect():
    import psycopg2

    load_dotenv()
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is not set. Copy .env.example to .env and add the read-only "
                           "factor-data connection string (a read-only role from Hari).")
    if "service_role" in url.lower():
        raise RuntimeError("DATABASE_URL looks like a service-role credential. Use a read-only role.")
    conn = psycopg2.connect(url, connect_timeout=20, application_name="factor-lead-value-engine")
    conn.set_session(readonly=True, autocommit=False)
    with conn.cursor() as cur:
        cur.execute("set statement_timeout = '10min'")
    print(f"[extract] connected to {_redact(url).split('?')[0]} (read-only session)")
    return conn


def run_query(conn, name: str, params: dict) -> pd.DataFrame:
    sql = (SQL_DIR / QUERIES[name]).read_text()
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [c.name for c in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=cols)
    pii = [c for c in df.columns if PII_COLUMNS.match(c)]
    if pii:
        raise RuntimeError(f"{name}.sql returned raw personal-data columns {pii}. Select person_id / *_hash only.")
    return df


def extract(cfg: dict, conn=None) -> dict[str, int]:
    d = cfg["data"]
    params = {"projects": d.get("projects") or None, "exclude": d.get("exclude_projects") or []}
    paths = {"leads": d["leads_csv"], "outcomes": d["outcomes_csv"], "ad_spend": d.get("ad_spend_csv")}
    own = conn is None
    conn = conn or connect()
    counts = {}
    try:
        for name in QUERIES:
            if not paths[name]:
                continue
            df = run_query(conn, name, params)
            Path(paths[name]).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(paths[name], index=False)
            counts[name] = len(df)
            print(f"[extract] {name}: {len(df):,} rows -> {paths[name]}")
        conn.rollback()                                   # read-only: nothing to commit
    finally:
        if own:
            conn.close()
    return counts
