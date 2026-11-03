"""Factor Lead Value Engine - one command to run everything.

    python run.py --demo                 # generate fake data and run the full pipeline
    python run.py                        # run on your CSVs in data/ (see config.yaml)
    python run.py --pull-meta 2026-01-01 # first pull daily campaign spend from Meta into data/ad_spend.csv
    python run.py --push-meta            # also send predicted values to Meta CAPI (respects meta.dry_run)
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import pandas as pd

from flv import close_model, data, funnel, meta_api, report, simulate, synthetic, value_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--demo", action="store_true", help="generate synthetic data into data/ first")
    ap.add_argument("--pull-meta", metavar="SINCE", help="pull spend from Meta since YYYY-MM-DD (until today)")
    ap.add_argument("--push-meta", action="store_true", help="push predicted values to the Conversions API")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()

    t0 = time.time()
    cfg = data.load_config(a.config)
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    if a.demo:
        synthetic.generate(Path(cfg["data"]["leads_csv"]).parent)
    if a.pull_meta:
        meta_api.pull_spend(cfg, a.pull_meta, pd.Timestamp.today().strftime("%Y-%m-%d"))

    ds = data.load(cfg)
    cm = close_model.train(ds, cfg)
    vm = value_model.train(ds, cfg)
    fm = funnel.fit(ds, cfg)
    scored = simulate.score(ds, cm, vm, fm, cfg)
    summ, total = simulate.monte_carlo(scored, vm, ds, cfg, group="campaign")
    bt = simulate.backtest(scored, cm, "campaign", cfg["simulation"]["gross_margin"])

    drop = [c for c in ["email", "phone"] if c in scored.columns]
    scored.drop(columns=drop).to_csv(out / "lead_scores.csv", index=False)
    summ.to_csv(out / "campaign_simulation.csv", index=False)
    bt.to_csv(out / "backtest_by_campaign.csv", index=False)
    vm.summary.to_csv(out / "value_model_effects.csv", index=False)
    fm.counts.to_csv(out / "funnel_stage_rates.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(cm.metrics, indent=2))
    with open(out / "models.pkl", "wb") as f:
        pickle.dump({"close": cm, "value": vm, "funnel": fm}, f)

    report.build(out / "dashboard.html", ds, cm, vm, fm, scored, summ, total, bt, cfg)

    if a.push_meta:
        meta_api.push_values(scored, cfg, ds.as_of, str(out))
    print(f"\nDone in {time.time() - t0:.0f}s. Open {out / 'dashboard.html'} in your browser.")


if __name__ == "__main__":
    main()

