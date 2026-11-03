"""Run sql/*.sql against a mock factor-data schema (tests/fixtures/mock_factor_data.sql).

Needs a throwaway Postgres: set TEST_DATABASE_URL (CI starts one). Skipped otherwise.
NEVER point this at factor-data - the fixture drops and recreates the core/track/crm/ads schemas.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from conftest import ROOT
from flv import extract

URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")
PARAMS = {"projects": ["nirvana"], "exclude": ["senior-care-leads", "insuracove-aus"]}


@pytest.fixture(scope="module")
def conn():
    import psycopg2

    c = psycopg2.connect(URL)
    with c.cursor() as cur:
        cur.execute((ROOT / "tests" / "fixtures" / "mock_factor_data.sql").read_text())
    c.commit()
    yield c
    c.close()


def test_leads(conn):
    df = extract.run_query(conn, "leads", PARAMS).set_index("lead_id")
    assert list(df.index) == ["nirvana:p1", "nirvana:p2", "nirvana:p3"]       # merged p1b -> p1; deleted / excluded gone
    assert df.loc["nirvana:p1", "opt_in_at"] == "2026-08-10T14:00:00+00:00"   # first contact (under the merged id)
    assert df.loc["nirvana:p1", "utm_campaign"] == "NIR | Prospecting"
    assert df.loc["nirvana:p1", "has_application"] == 1 and df.loc["nirvana:p1", "typeform_score"] == 72
    assert df.loc["nirvana:p3", "has_application"] == 0                       # applied 2 days after opt-in
    assert not {"email", "phone"} & set(df.columns)


def test_outcomes(conn):
    df = extract.run_query(conn, "outcomes", PARAMS).set_index("lead_id")
    p1, p2, p3 = df.loc["nirvana:p1"], df.loc["nirvana:p2"], df.loc["nirvana:p3"]
    assert p1["stage"] == "won" and p1["won_date"] == "2026-08-20" and p1["deal_value"] == 7500   # midnight UTC kept
    assert p2["stage"] == "contacted" and p2["is_lost"] == 1        # inferred 'qualified' row ignored
    assert p3["stage"] == "won" and p3["won_date"] == "2026-09-19"  # GHL timestamp -> New York date
    assert pd.isna(p3["deal_value"])                                # value_source = none


def test_ad_spend(conn):
    df = extract.run_query(conn, "ad_spend", PARAMS)
    assert df.to_dict("records") == [{"date": "2026-08-10", "project": "nirvana", "spend": 1234.56,
                                      "impressions": 1000, "link_clicks": 20, "leads": 3}]


def test_pii_columns_rejected(conn, tmp_path, monkeypatch):
    bad = tmp_path / "leads.sql"
    bad.write_text("select 'x' as lead_id, 'a@b.c' as email")
    monkeypatch.setattr(extract, "SQL_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="personal-data"):
        extract.run_query(conn, "leads", PARAMS)
