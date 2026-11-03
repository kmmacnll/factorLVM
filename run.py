"""Factor Lead Value Engine - one command to run everything.

    python run.py --extract              # pull data/*.csv from factor-data (DATABASE_URL in .env), then run
    python run.py --demo                 # generate sandbox data into data/ and run the full pipeline
    python run.py                        # run on the CSVs in data/ (see config.yaml)
    python run.py --pull-meta 2026-07-01 # first pull daily campaign spend from Meta into data/ad_spend.csv
    python run.py --push-meta            # also send predicted values to Meta CAPI (respects meta.dry_run)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import time
from pathlib import Path

import pandas as pd

from flv import close_model, data, funnel, meta_api, report, simulate, value_model

ROOT = Path(__file__).resolve().parent
PII_COLS = ["email", "phone", "email_hash", "phone_hash", "fbp"]


def _sandbox():
    spec = importlib.util.spec_from_file_location("generate_sandbox_data", ROOT / "scripts" / "generate_sandbox_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(cfg: dict, out: Path, push_meta: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    ds = data.load(cfg)
    cm = close_model.train(ds, cfg)
    vm = value_model.train(ds, cfg)
    fm = funnel.fit(ds, cfg)
    scored = simulate.score(ds, cm, vm, fm, cfg)
    summ, total = simulate.monte_carlo(scored, vm, ds, cfg, group="campaign")
    proj_summ, _ = simulate.monte_carlo(scored, vm, ds, cfg, group="client_id")
    bt = simulate.backtest(scored, cm, "campaign", cfg["simulation"]["gross_margin"])
    cm.metrics.update(simulate.backtest_summary(bt))
    print(f"[backtest] value error {cm.metrics['backtest_value_error']:+.1%} | closes "
          f"{cm.metrics['backtest_pred_closes']:.1f} predicted vs {cm.metrics['backtest_actual_closes']} actual "
          f"| {cm.metrics['backtest_value_imputed_wins']} test wins had no value (imputed)")

    scored.drop(columns=[c for c in PII_COLS if c in scored.columns]).to_csv(out / "lead_scores.csv", index=False)
    summ.to_csv(out / "campaign_simulation.csv", index=False)
    proj_summ.to_csv(out / "project_simulation.csv", index=False)
    bt.to_csv(out / "backtest_by_campaign.csv", index=False)
    vm.summary.to_csv(out / "value_model_effects.csv", index=False)
    fm.counts.to_csv(out / "funnel_stage_rates.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(cm.metrics, indent=2))
    with open(out / "models.pkl", "wb") as f:
        pickle.dump({"close": cm, "value": vm, "funnel": fm}, f)

    report.build(out / "dashboard.html", ds, cm, vm, fm, scored, summ, total, bt, cfg, proj_summ=proj_summ)
    if push_meta:
        meta_api.push_values(scored, cfg, ds.as_of, str(out))
    return cm.metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--extract", action="store_true", help="extract data/*.csv from factor-data first (needs DATABASE_URL)")
    ap.add_argument("--extract-only", action="store_true", help="extract and stop (don't train)")
    ap.add_argument("--demo", action="store_true", help="generate sandbox data into data/ first")
    ap.add_argument("--pull-meta", metavar="SINCE", help="pull spend from Meta since YYYY-MM-DD (until today)")
    ap.add_argument("--push-meta", action="store_true", help="push predicted values to the Conversions API")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()

    t0 = time.time()
    cfg = data.load_config(a.config)
    out = Path(a.out)

    if a.extract or a.extract_only:
        from flv import extract
        extract.extract(cfg)
        if a.extract_only:
            return
    if a.demo:
        _sandbox().generate(Path(cfg["data"]["leads_csv"]).parent)
    if a.pull_meta:
        meta_api.pull_spend(cfg, a.pull_meta, pd.Timestamp.today().strftime("%Y-%m-%d"))

    run(cfg, out, push_meta=a.push_meta)
    print(f"\nDone in {time.time() - t0:.0f}s. Open {out / 'dashboard.html'} in your browser.")


if __name__ == "__main__":
    main()
