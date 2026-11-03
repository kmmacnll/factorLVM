"""End-to-end smoke test: small sandbox -> full pipeline -> every output file exists."""
from __future__ import annotations

import copy
import importlib.util
import json

import pandas as pd
import pytest

from conftest import ROOT
from flv import data

spec = importlib.util.spec_from_file_location("sandbox", ROOT / "scripts" / "generate_sandbox_data.py")
sandbox = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sandbox)
spec = importlib.util.spec_from_file_location("run", ROOT / "run.py")
runmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runmod)

OUTPUTS = ["dashboard.html", "lead_scores.csv", "campaign_simulation.csv", "project_simulation.csv",
           "backtest_by_campaign.csv", "value_model_effects.csv", "funnel_stage_rates.csv", "metrics.json", "models.pkl"]


def _cfg(tmp_path):
    cfg = data.load_config(ROOT / "config.yaml")
    cfg = copy.deepcopy(cfg)
    d = tmp_path / "data"
    cfg["data"].update(leads_csv=str(d / "leads.csv"), outcomes_csv=str(d / "outcomes.csv"),
                       ad_spend_csv=str(d / "ad_spend.csv"),
                       projects=None)                     # all high-ticket projects: enough wins at 3k leads
    cfg["simulation"]["n_sims"] = 500
    cfg["model"]["value_model"].update(advi_iterations=15000, draws=300)
    return cfg, d


def test_pipeline_end_to_end(tmp_path):
    cfg, d = _cfg(tmp_path)
    sandbox.generate(d, n_leads=3000, seed=11)
    out = tmp_path / "outputs"
    metrics = runmod.run(cfg, out)
    for f in OUTPUTS:
        assert (out / f).exists() and (out / f).stat().st_size > 0, f
    m = json.loads((out / "metrics.json").read_text())
    # 3k leads is too few wins for a meaningful AUC (trees may not split); just check it is computed
    assert 0.0 <= m["auc"] <= 1.0
    assert "backtest_value_error" in m and metrics["n_test"] > 0
    scores = pd.read_csv(out / "lead_scores.csv")
    assert not {"email", "phone", "email_hash", "phone_hash", "fbp"} & set(scores.columns)
    # lead-gen projects are excluded by default
    assert not set(scores["client_id"]) & {"senior-care-leads", "insuracove-aus"}
    # ad-level fields are null before 2026-07-01 -> "(missing)"
    early = scores[pd.to_datetime(scores["created_at"]) < "2026-06-30"]
    assert (early["campaign"] == "(missing)").all()


def test_leak_columns_in_leads_csv_are_blocked(tmp_path):
    cfg, d = _cfg(tmp_path)
    sandbox.generate(d, n_leads=500, seed=3, with_leaks=True)
    with pytest.raises(data.LeakageError, match="lead_quality"):
        data.load(cfg)


@pytest.mark.parametrize("col", ["funnel_stage", "status", "lead_quality", "appointment_at", "call_at",
                                 "booked_at", "won_at", "setter_notes", "deal_value"])
def test_leak_columns_as_features_are_blocked(col):
    cfg = data.load_config(ROOT / "config.yaml")
    cfg["features"]["categorical"] = cfg["features"]["categorical"] + [col]
    with pytest.raises(data.LeakageError):
        data.check_leakage(cfg)


def test_config_leakage_list_extends_builtin():
    cfg = data.load_config(ROOT / "config.yaml")
    cfg["leakage"] = {"forbidden_columns": ["my_setter_field"], "forbidden_prefixes": []}
    cfg["features"]["numeric"] = ["my_setter_field"]
    with pytest.raises(data.LeakageError):
        data.check_leakage(cfg)
    data.check_leakage(data.load_config(ROOT / "config.yaml"))     # the shipped config is clean


def test_won_dates_are_calendar_dates():
    s = pd.Series(["2026-08-20T00:00:00+00:00",   # Close: midnight UTC = the date itself
                   "2026-08-20",                  # bare date
                   "2026-09-20T02:30:00+00:00",   # GHL timestamp: 22:30 on the 19th in New York
                   ""])
    got = data.to_business_date(s, "America/New_York")
    assert list(got.dt.strftime("%Y-%m-%d").fillna("NaT")) == ["2026-08-20", "2026-08-20", "2026-09-19", "NaT"]


def test_opt_in_converted_to_new_york():
    got = data.to_local_naive(pd.Series(["2026-08-10T03:30:00+00:00"]), "America/New_York")
    assert str(got.iloc[0]) == "2026-08-09 23:30:00"
