# Factor · Lead Value Engine

Predicts what every Meta lead is worth **at the moment it arrives**, so Factor can bid on value instead of lead count.

```
lead value = P(close within 90d)  ×  E[deal size | won]  ×  margin
             └ XGBoost + Platt     └ hierarchical Bayesian (PyMC)
               calibration           log-normal, client effects pooled
```

On top of that:
- **Funnel + survival layer.** Kaplan–Meier time-to-close, so young leads aren't counted as losses. Beta-Binomial stage conversion rates update a lead's probability as it moves through the CRM.
- **Monte Carlo.** Simulates total value per campaign (P10/P50/P90) and P(campaign beats target ROAS).
- **Meta Conversions API push.** Sends each new lead's predicted value back to Meta for value-based bidding (dry-run by default).
- **Dashboard.** One interactive HTML file with every chart.

## Quick start (5 minutes)

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py --demo          # fake data -> full pipeline -> outputs/dashboard.html
```

Open `outputs/dashboard.html` in a browser.

## Plugging in real data

**1. Drop your CSVs into `data/`:**

| File | One row per | Needs at minimum |
|---|---|---|
| `leads.csv` | lead | `lead_id`, `created_at`, plus anything known at submission: campaign/adset/ad, platform, placement, form answers |
| `outcomes.csv` | lead (from CRM) | `lead_id`, `stage` (furthest reached), `is_lost`, `closed_at`, `deal_value` |
| `ad_spend.csv` *(optional)* | campaign × day | `date`, `campaign_name`, `spend`. Or pull it from Meta: `python run.py --pull-meta 2026-01-01` |

**2. Edit `config.yaml`:**
- `columns:` maps *your* headers to the standard names (right-hand side = your CSV header).
- `funnel.stages:` lists your CRM stages in order, with "won" last.
- `features.categorical:` lists your form-question columns. **Only include things known when the lead submits.** Never include CRM activity (calls made, emails opened); that leaks the answer and makes the model look great while being useless.
- `simulation.gross_margin` and `target_roas` hold your client's economics.

**3. Add API keys:** `cp .env.example .env` and fill it in.

**4. Run:** `python run.py`. Add `--push-meta` to build CAPI payloads. Check `outputs/capi_payloads.json`, then set `meta.dry_run: false` (ideally with `META_TEST_EVENT_CODE` set first) to actually send.

## Outputs (`outputs/`)

| File | What it is |
|---|---|
| `dashboard.html` | All charts: campaign value, value-vs-CPL, calibration, lift, ROC, SHAP drivers, Bayesian effects, posterior check, funnel, survival, Monte Carlo, backtest |
| `lead_scores.csv` | Every lead with `p_close_at_submit`, `p_close_now`, `value_if_won` (+90% interval), `expected_value`, `max_cpl_at_target_roas`, `value_tier` |
| `campaign_simulation.csv` | Monte Carlo value per campaign, ROAS, P(beats target) |
| `backtest_by_campaign.csv` | Predicted vs actual on the held-out period |
| `value_model_effects.csv` | % effect of each attribute on deal size, with credible intervals |
| `metrics.json`, `models.pkl` | Model scores and the fitted models |

## Reading the results honestly

- **Calibration matters more than AUC for bidding.** If the calibration line drifts off the diagonal, the values you send Meta are biased.
- **Train only on mature leads** (older than `horizon_days`). Recent leads haven't had time to close.
- **`p_close_at_submit` goes to Meta.** `p_close_now` (which uses lead age and CRM stage) is for sales prioritisation only.
- **Small clients:** the Bayesian client effects shrink toward the group average. That's intended; it prevents a client with 10 deals from getting a wild estimate.
- **Wins needed:** you want a few hundred wins overall for the close model, and around 20 or more for the value model to say anything.
- **Proof is an A/B test,** not the dashboard. Run one campaign on value optimisation with these values against a matched control.

## Project layout

```
run.py              one-command pipeline
config.yaml         everything you'd change
.env                API keys (copy from .env.example)
flv/data.py         load, map columns, validate, labels, censoring
flv/close_model.py  XGBoost + Platt calibration + SHAP, time-split validation
flv/value_model.py  PyMC hierarchical log-normal deal-size model (ADVI or NUTS)
flv/funnel.py       Kaplan-Meier + Beta-Binomial funnel, stage/age updating
flv/simulate.py     lead scoring, Monte Carlo by campaign, backtest
flv/meta_api.py     Marketing API spend pull, Conversions API value push
flv/report.py       Plotly dashboard
flv/synthetic.py    realistic fake data for testing
```

Switch `model.value_model.method` to `nuts` for full MCMC once the data is real. It's slower but gold-standard, and it prints R-hat so you can check convergence.
