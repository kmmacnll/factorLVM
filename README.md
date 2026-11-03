# Factor · Lead Value Engine

Predicts what every Meta lead is worth **at the moment it arrives**, so Factor can bid on value instead of lead count.

```
lead value = P(close within 45d)  ×  E[deal size | won]  ×  margin
             └ XGBoost + Platt     └ hierarchical Bayesian (PyMC)
               calibration           log-normal, project effects pooled
```

On top of that:
- **Funnel + survival layer.** Kaplan–Meier time-to-close, so young leads aren't counted as losses. Beta-Binomial stage conversion rates update a lead's probability as it moves through the CRM.
- **Monte Carlo.** Simulates total value per campaign and per project (P10/P50/P90), and P(project beats target ROAS).
- **Meta Conversions API push.** Sends each new lead's predicted value back to Meta for value-based bidding (dry-run by default).
- **Dashboard.** One interactive HTML file with every chart.

The engine trains on **factor-data**, Factor's read-only Postgres mirror of Close/GHL, Meta Ads, Typeform and the Assist quiz. The modelling rules it follows are in [`docs/DATA_NOTES.md`](docs/DATA_NOTES.md).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # then fill it in, see below
```

### `.env` keys (git-ignored, never commit it)

| Key | Needed for | Notes |
|---|---|---|
| `DATABASE_URL` | `--extract` | Postgres connection string for a **read-only** factor-data role. Ask Hari. Never use a service-role key. |
| `META_ACCESS_TOKEN` | `--pull-meta`, `--push-meta` (live) | System-user token: `ads_read`, `ads_management` |
| `META_AD_ACCOUNT_ID` | `--pull-meta` | `act_…` |
| `META_PIXEL_ID` | `--push-meta` (live) | Dataset / pixel the CAPI events go to |
| `META_API_VERSION` | Meta calls | e.g. `v23.0` |
| `META_TEST_EVENT_CODE` | optional | Send to Events Manager → Test Events first |

## Try it on the sandbox (no credentials needed)

```bash
python scripts/generate_sandbox_data.py --out data/   # 40k synthetic leads shaped like factor-data
python run.py                                          # -> outputs/dashboard.html
```

`python run.py --demo` does both steps in one go. The sandbox writes the same three CSVs, with the same columns, that `--extract` writes from the real database. It also reproduces the database's quirks: null ad fields before 2026-07-01, canonical stage names, date-only won dates, wins without a value, lead-gen projects, and a frozen ScaleLogix. Everything in it is random; there is no real data or PII.

Sandbox results (seed 7, Nirvana, horizon 45 days): AUC **0.811**, Brier skill **+6.4%** (test base rate 3.3% on 2,860 leads), backtest value error **−2.2%** (95.8 predicted vs 94 actual closes). These are sanity checks, not a forecast for real data: the sandbox has a known ground truth.

`--with-leaks` adds post-contact CRM columns (`funnel_stage`, `status`, `lead_quality`, `appointment_at`, `call_at`) to `leads.csv`. The pipeline must refuse to run on it:

```bash
python scripts/generate_sandbox_data.py --out data/ --with-leaks && python run.py   # -> LeakageError
```

## Run on factor-data

```bash
python run.py --extract          # sql/*.sql -> data/*.csv over DATABASE_URL (read-only session), then train
python run.py --extract-only     # just refresh data/
```

`config.yaml` picks what gets extracted and modelled:
- `data.projects: [nirvana]`: projects to model (`core.projects.slug`). `null` means all of them. Each project becomes the engine's `client_id`, so the Bayesian deal-size effects pool across projects.
- `data.exclude_projects`: the lead-gen projects with no sales calls (`senior-care-leads` / Assist, `insuracove-aus`) are excluded by default.
- `model.horizon_days: 45`: Meta history only starts 2026-07-01, so a 90-day horizon would leave almost no mature leads with ad features. Raise it as history accumulates.

**Before the first real run**, resolve the `-- TODO verify` markers in `sql/`. In particular, map each project's lead-form **question titles** in `sql/leads.sql`. Until you do, `q_investment` / `q_timeline` / `q_situation` come through empty, and the model leans on project, campaign, Typeform score and time features only.

What the queries do:

| CSV | Built from |
|---|---|
| `leads.csv` | First `crm.contacts.date_created` per person × project (the opt-in). UTMs, `fbclid` and `fbp` from `crm.contacts.attribution`. Lead-form answers and score from `track.form_responses` (only `kind = 'lead'` forms, only responses submitted by opt-in). `person_id` plus email/phone **hashes** only; raw contact data never leaves the DB. |
| `outcomes.csv` | Furthest canonical stage from **non-inferred** `crm.stage_history`. Won date and value (`value_corrected_minor / 100`) from `crm.deals`. |
| `ad_spend.csv` | `ads.v_project_daily` (project × day). |

People are always resolved with `coalesce(merged_into, id)`.

## Leakage rules (enforced)

The close model may only see what is known **when the lead opts in**. `flv/data.py` raises `LeakageError`, and the run stops, if any of these is listed under `features`, or is even present in `leads.csv`:

- current CRM state: `funnel_stage`, `status`, `stage_label`, `current_stage`, `ltv_minor`, …
- anything a setter or closer fills in after contact: `lead_quality`, `appointment_at`, `call_at`, `sold_at`, `closing_type`, `sale_*`, `payment_*`, `cash_collected`, `contract_signed`, `refund_state`, `setter_*`, …
- dates from `decide.v_lead_outcomes` (`booked_at`, `showed_at`, `won_at`): they are **not point-in-time**
- the engine's own outcome columns (`stage`, `is_lost`, `deal_value`, `closed_at`, …)

The built-in list can't be weakened from config. Add project-specific post-contact fields (e.g. a Close custom field the setter fills in) under `leakage.forbidden_columns`.

`p_close_at_submit` is what goes to Meta. `p_close_now` also uses lead age and CRM stage; use it for sales prioritisation only, never for training.

## factor-data caveats

- **Dates and time zones.** Opt-in timestamps are UTC and are converted to America/New_York before the hour, day-of-week and daily bucketing features. Close won dates are stored as midnight UTC and treated as **calendar dates**; GHL won times are converted to New York.
- **Ad-level features before July.** Meta history starts 2026-07-01. Campaign, ad set and ad are null before that, and for projects with no ad account; they become a `(missing)` level the model learns separately. Spend exists only from July.
- **Deal values.** 85% of deals have no value. The deal-size model trains only on won deals that **have** a value, and logs how many it used (sandbox: 346 of 546). The backtest imputes E[value | won] for valueless wins and reports how many.
- **Typeform survivorship.** 36% of Typeform responses have no `person_id`, mostly applicants the form disqualified before the contact questions. They can't be linked to a lead, so answers exist mostly for applicants who got far enough to leave details, and `has_application = 0` does not mean "didn't apply". Linked answers are biased toward qualified applicants; keep this in mind when reading answer effects.
- **Stage maps are per project and partly automatic.** Review a project's map before trusting its funnel.
- **Spend is per project.** `ads.v_project_daily` has no campaign split, so ROAS and P(beats target) are reported per project (`project_simulation.csv`); campaign-level output is value only.
- **Freshness.** ScaleLogix is frozen from 2026-10-21. Lead-gen projects are excluded.

## The dashboard (runs locally, no hosting)

`outputs/dashboard.html` is a single self-contained file: Plotly and all data are embedded, so it opens straight from disk and works offline. There is nothing to deploy. Every run rebuilds it from whatever is in `data/`:

```bash
python run.py --demo --open      # sandbox data -> dashboard opens in your browser
python run.py --extract --open   # real factor-data (DATABASE_URL in .env) -> dashboard
python run.py --open             # your own CSVs already in data/ -> dashboard
```

Or double-click `outputs/dashboard.html`. To share a run, send the HTML file. It contains lead-level scores (lead ids, campaigns, values) but no emails, phones or hashes, so treat it as internal.

To plug in data without the database, drop `leads.csv`, `outcomes.csv` and (optionally) `ad_spend.csv` into `data/` with the columns `config.yaml` maps. `python scripts/generate_sandbox_data.py --out data/` writes an example of each.

## Outputs (`outputs/`, git-ignored)

| File | What it is |
|---|---|
| `dashboard.html` | All charts: campaign value, value vs CPL, calibration, lift, ROC, SHAP drivers, Bayesian effects, posterior check, funnel, survival, Monte Carlo, backtest |
| `lead_scores.csv` | Every lead with `p_close_at_submit`, `p_close_now`, `value_if_won` (+90% interval), `expected_value`, `max_cpl_at_target_roas`, `value_tier`. Contains no email/phone hashes. |
| `campaign_simulation.csv` / `project_simulation.csv` | Monte Carlo value per campaign; per project with spend, ROAS and P(beats target) |
| `backtest_by_campaign.csv` | Predicted vs actual on the held-out period |
| `value_model_effects.csv` | % effect of each attribute on deal size, with credible intervals |
| `metrics.json`, `models.pkl` | Model scores (incl. backtest error) and the fitted models |

## Reading the results honestly

- **Calibration matters more than AUC for bidding.** If the calibration line drifts off the diagonal, the values you send Meta are biased.
- **Train only on mature leads** (older than `horizon_days`). Recent leads haven't had time to close.
- **Small projects:** the Bayesian project effects shrink toward the group average. That's intended.
- **Wins needed:** a few hundred wins overall for the close model, and around 20 or more with a value for the deal-size model.
- **Proof is an A/B test,** not the dashboard. Run one campaign on value optimisation with these values against a matched control.
- **CAPI hashes:** the push uses factor-data's sha256 email/phone hashes directly. Confirm its normalisation matches Meta's before going live (see TODO in `flv/meta_api.py`).

## Tests

```bash
pytest -q                                                        # smoke test on a 3k-lead sandbox + leakage + date rules
TEST_DATABASE_URL=postgresql://postgres@localhost/scratch pytest  # also runs sql/*.sql against a mock schema
```

`TEST_DATABASE_URL` must point to a **throwaway** database: the fixture drops and recreates schemas. Never point it at factor-data. GitHub Actions runs everything on every push, with a Postgres service container.

## Project layout

```
run.py                         one-command pipeline (--extract, --demo, --pull-meta, --push-meta)
config.yaml                    everything you'd change
.env                           secrets (copy from .env.example)
sql/                           leads / outcomes / ad_spend extraction from factor-data
scripts/generate_sandbox_data.py  synthetic data in the extraction's CSV shape
docs/DATA_NOTES.md             modelling rules from the factor-data guide
flv/extract.py                 runs sql/ over DATABASE_URL (read-only), PII check
flv/data.py                    load, map columns, time zones, leakage guard, labels, censoring
flv/close_model.py             XGBoost + Platt calibration + SHAP, time-split validation
flv/value_model.py             PyMC hierarchical log-normal deal-size model (ADVI or NUTS)
flv/funnel.py, flv/km.py       Kaplan–Meier + Beta-Binomial funnel, stage/age updating
flv/simulate.py                lead scoring, Monte Carlo, backtest
flv/meta_api.py                Marketing API spend pull, Conversions API value push
flv/report.py                  Plotly dashboard
tests/                         pytest suite (+ mock schema for the SQL)
```

Switch `model.value_model.method` to `nuts` for full MCMC once the data is real. It's slower but gold-standard, and it prints R-hat so you can check convergence.
