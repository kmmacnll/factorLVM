# factor-data: modelling notes

These are the rules from Factor's internal data guide that the engine depends on. The guide itself is
internal and is **not** in this public repo, and neither are any connection details. Ask the Factor team
for it.

## Source

- factor-data is a **read-only Postgres mirror** of the agency's systems: Close and GoHighLevel CRMs,
  Meta Ads, Typeform, and the Assist quiz. Nothing in it is typed by hand.
- Connect with a **read-only role** (ask Hari) through `DATABASE_URL` in `.env`. Never use a
  service-role key.

## Keys

- **`person_id`** is one real human across every source and project. Always resolve merges first:
  `coalesce(track.people.merged_into, track.people.id)`.
- **`project_id`** is one client offer and the partition key for modelling. The engine maps the project
  slug to `client_id`, so the hierarchical deal-size model pools across projects. One lead is one
  person × project (`lead_id = "<slug>:<person_id>"`).
- Default project: **Nirvana**. The lead-gen projects (`senior-care-leads` / Assist and
  `insuracove-aus`) have no sales calls: "booked" and "showed" mean nothing there, and "won" means a
  sold lead. They are excluded by default (`data.exclude_projects`).

## What each CSV is built from (`sql/`)

| CSV | Source | Rules |
|---|---|---|
| `leads.csv` | `crm.contacts`, `track.form_responses` + `track.forms`, `track.people` | Opt-in is the first `crm.contacts.date_created`. UTMs, `fbclid` and `fbp` come from `crm.contacts.attribution`. Application answers come from **lead** forms only (`kind = 'lead'`), and only responses submitted by opt-in. Output is `person_id` and hashes only. |
| `outcomes.csv` | `crm.stage_history` (non-inferred), `crm.deals` | Furthest canonical stage from rows with `inferred = false`. The won date and value come from `crm.deals` (`status = 'won'`, `value_corrected_minor / 100`). |
| `ad_spend.csv` | `ads.v_project_daily` | Project × day. Money is in minor units, so divide by 100. |

## Time

- Every timestamp is UTC. The business day is **America/New_York** for every project, so convert before
  bucketing by day or week.
- **Close won dates are dates**, stored as midnight UTC. Keep the calendar date: converting to New York
  would move it to the previous day. GHL won times are real timestamps; convert them.
- **Meta history starts 2026-07-01**, so ad-level fields are null before then and for projects with no
  linked ad account. `model.horizon_days` is 45 so that July–September opt-ins can mature.

## Money

- Use `value_corrected_minor` (normalised into the project's plausible band), not `value_minor`.
- **85% of deals have no value** (`value_source = 'none'`). The deal-size model trains only on won deals
  that have a value and logs how many it used. The backtest imputes E[value | won] for valueless wins.

## Leakage: never features

- Current-state columns: `funnel_stage`, `status`, `stage_label`, `current_stage`, `ltv_minor`.
- Anything filled in during the sales process: `lead_quality` (often set by the setter after a
  conversation), `appointment_at`, `call_at`, `sold_at`, `closing_type`, `sale_*`, `payment_*`,
  `cash_collected`, `contract_signed`, `refund_state`, and many CRM custom fields.
- `date_updated` / `synced_at` show the last change, not the opt-in.
- **Dates from `decide.v_lead_outcomes`**: they are not point-in-time. Its `booked_at` is when a deal
  was opened, filtered by the deal's current stage.
- Typeform `sale` and `customer` forms are post-sale records. Typeform `score` is computed at submit,
  so it is safe.

`flv/data.py` enforces these. It raises `LeakageError` if one appears in `features` or in `leads.csv`.

## Label caveats

- Stage maps are per project and partly automatic, so they can be wrong. Review each project's map
  before trusting `funnel_stage`.
- Sales teams update CRMs late or in bulk, so some won dates are marking dates, not close dates.
- `inferred = true` stage rows mean "happened by roughly this time". The extraction ignores them for
  stages; wins come from `crm.deals`.
- **Typeform survivorship:** 36% of responses have no `person_id`, mostly applicants the form
  disqualified before the contact questions. Training only on linked responses biases the data toward
  qualified applicants, and `has_application = 0` does not mean "did not apply".
- ScaleLogix data is frozen from 2026-10-21 (its Close key was rejected) until it is reconnected.

## Privacy

Personal data (names, emails, phones, free-text answers) lives in `crm.contacts`, `crm.leads`,
`track.people`, `track.form_responses.answers` and some views. Never export it into notebooks, model
artefacts, logs, this repo, or third-party tools. Use `person_id` and the `*_hash` columns.
`flv/extract.py` refuses query output with raw contact columns.
