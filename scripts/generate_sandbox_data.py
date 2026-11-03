"""Synthetic sandbox data shaped exactly like the output of sql/*.sql run against factor-data.

No real data, no PII: every id, hash and click id is random. The point is to exercise the
pipeline on the *shape* and *quirks* of the real database before touching it:

  * several projects (Close high-ticket offers plus the two lead-gen projects that the config
    excludes by default), each with its own CRM start date
  * Meta history only from 2026-07-01: ad-level fields (utm_campaign / utm_term / utm_content,
    fbclid, fbp) are null before then, and always null for projects with no ad account
  * opt-in timestamps in UTC, with the business day in America/New_York
  * outcome stages use the canonical funnel vocabulary (incl. no_show / deposit / onboarding)
  * won dates are calendar dates (Close stores them as midnight UTC)
  * ~35% of won deals have no deal value (value_source = none)
  * ~40% of leads have no linked lead-form application (Typeform survivorship / no form)
  * ScaleLogix frozen after 2026-10-21 (rejected Close key)
  * --with-leaks adds post-contact CRM columns (funnel_stage, status, lead_quality,
    appointment_at, call_at) to leads.csv so you can check the leakage guard blocks them.

    python scripts/generate_sandbox_data.py --out data/
    python scripts/generate_sandbox_data.py --out data/ --n-leads 3000 --with-leaks
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "America/New_York"
AS_OF = pd.Timestamp("2026-11-02 18:00", tz=TZ)
META_START = pd.Timestamp("2026-07-01", tz=TZ)

# slug: (lane, crm start, share of leads, meta (first, last) or None, close-rate offset, median deal $, freeze date)
PROJECTS = {
    "nirvana":           ("high_ticket", "2026-02-19", 0.46, ("2026-07-05", None), 0.45, 6500, None),
    "build-remote":      ("high_ticket", "2026-02-15", 0.24, ("2026-07-01", None), -0.55, 4200, None),
    "scalelogix":        ("high_ticket", "2026-03-03", 0.06, None, 0.15, 5200, "2026-10-21"),
    "elevate-agency":    ("high_ticket", "2026-05-29", 0.10, ("2026-07-01", None), 0.0, 8800, None),
    "kairos":            ("high_ticket", "2026-07-27", 0.04, ("2026-07-27", None), -0.2, 7000, None),
    "youtubify":         ("high_ticket", "2026-06-18", 0.03, ("2026-07-01", "2026-10-16"), 0.0, 3500, None),
    "senior-care-leads": ("lead_gen",    "2026-07-13", 0.05, ("2026-07-13", None), None, 250, None),
    "insuracove-aus":    ("lead_gen",    "2026-08-19", 0.02, ("2026-08-17", None), None, 120, None),
}
INVEST = {"Under $1k": (-1.2, 0.75), "$1k-$5k": (-0.2, 0.95), "$5k-$10k": (0.45, 1.1), "$10k+": (0.8, 1.35)}
TIMELINE = {"Right now": 0.55, "Within 30 days": 0.2, "1-3 months": -0.3, "Just looking": -1.0}
SITUATION = {"Employed": 0.0, "Business owner": 0.35, "Self-employed": 0.15, "Student": -0.8, "Retired": -0.2}
CAMPAIGN_TYPES = {"Prospecting | Broad": -0.25, "Prospecting | LAL": 0.1, "Retargeting | 30d": 0.5,
                  "ASC+ | Advantage": 0.0, "Prospecting | Interest": -0.1}
LADDER = ["lead", "contacted", "booked", "showed", "qualified", "proposal", "won"]


def _hex(rng: np.random.Generator, n: int, k: int = 64) -> np.ndarray:
    return np.array([hashlib.sha256(rng.bytes(16)).hexdigest()[:k] for _ in range(n)])


def _uuid(rng: np.random.Generator, n: int) -> np.ndarray:
    h = _hex(rng, n, 32)
    return np.array([f"{x[:8]}-{x[8:12]}-4{x[13:16]}-a{x[17:20]}-{x[20:32]}" for x in h])


def generate(out_dir: str | Path = "data", n_leads: int = 40000, seed: int = 7, with_leaks: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    slugs = list(PROJECTS)
    shares = np.array([PROJECTS[s][2] for s in slugs])
    proj = rng.choice(slugs, n_leads, p=shares / shares.sum())

    # ---- opt-in time: volume grows over time, business-hours heavy, in New York time ------------
    created = np.empty(n_leads, dtype=object)
    for s in slugs:
        idx = np.where(proj == s)[0]
        start = pd.Timestamp(PROJECTS[s][1], tz=TZ)
        end = pd.Timestamp(PROJECTS[s][6], tz=TZ) if PROJECTS[s][6] else AS_OF
        span = (end - start).total_seconds() / 86400
        day = start.normalize() + pd.to_timedelta(np.floor(rng.beta(1.3, 1.0, len(idx)) * span), unit="D")
        hour = np.clip(rng.normal(14, 4.5, len(idx)), 0, 23.99)
        ts = day + pd.to_timedelta(hour * 3600, unit="s")
        created[idx] = list(ts)
    created = pd.DatetimeIndex(pd.to_datetime(list(created))).tz_convert(TZ)
    created = pd.DatetimeIndex(np.where(created > AS_OF, AS_OF - pd.Timedelta(minutes=5), created)).tz_convert(TZ)

    # ---- application answers (lead forms); ~40% of leads have none linked -----------------------
    has_app = rng.random(n_leads) < 0.6
    inv = np.where(has_app, rng.choice(list(INVEST), n_leads, p=[0.3, 0.35, 0.2, 0.15]), None)
    tl = np.where(has_app, rng.choice(list(TIMELINE), n_leads, p=[0.25, 0.3, 0.25, 0.2]), None)
    sit = np.where(has_app, rng.choice(list(SITUATION), n_leads, p=[0.4, 0.2, 0.2, 0.1, 0.1]), None)
    score = np.where(has_app, np.clip(rng.normal(55, 18, n_leads), 0, 100).round(), np.nan)

    # ---- Meta attribution: only from the project's first ad day, only for linked accounts -------
    camp = np.full(n_leads, None, dtype=object)
    adset = np.full(n_leads, None, dtype=object)
    ad = np.full(n_leads, None, dtype=object)
    fbclid = np.full(n_leads, None, dtype=object)
    fbp = np.full(n_leads, None, dtype=object)
    camp_eff = np.zeros(n_leads)
    for s in slugs:
        meta = PROJECTS[s][3]
        if not meta:
            continue
        first = max(pd.Timestamp(meta[0], tz=TZ), META_START)
        last = pd.Timestamp(meta[1], tz=TZ) + pd.Timedelta(days=1) if meta[1] else AS_OF
        idx = np.where((proj == s) & (created >= first) & (created <= last) & (rng.random(n_leads) < 0.85))[0]
        ct = rng.choice(list(CAMPAIGN_TYPES), len(idx), p=[0.3, 0.2, 0.15, 0.25, 0.1])
        tag = s[:3].upper()
        camp[idx] = [f"{tag} | {c}" for c in ct]
        adset[idx] = [f"{tag} | {c} | AS{rng.integers(1, 4)}" for c in ct]
        ad[idx] = [f"{a} | AD{rng.integers(1, 6)}" for a in adset[idx]]
        camp_eff[idx] = [CAMPAIGN_TYPES[c] for c in ct]
        ms = (created[idx].astype("int64") // 10**6).astype(str)
        fbclid[idx] = ["IwAR" + h for h in _hex(rng, len(idx), 40)]
        fbp[idx] = [f"fb.1.{m}.{rng.integers(10**9, 10**10)}" for m in ms]

    # ---- latent quality -> funnel progression ---------------------------------------------------
    lane = np.array([PROJECTS[s][0] for s in proj])
    poff = np.array([PROJECTS[s][4] or 0.0 for s in proj])
    hour_l = created.hour
    q = (poff + camp_eff
         + np.array([INVEST[x][0] if x else -0.3 for x in inv])
         + np.array([TIMELINE[x] if x else -0.2 for x in tl])
         + np.array([SITUATION[x] if x else 0.0 for x in sit])
         + np.where(has_app, (np.nan_to_num(score, nan=50) - 55) / 40, 0)
         + np.where((hour_l >= 9) & (hour_l <= 18), 0.15, -0.1)
         + rng.normal(0, 0.5, n_leads))

    def step(base: float, slope: float) -> np.ndarray:
        return rng.random(n_leads) < 1 / (1 + np.exp(-(base + slope * q)))

    reach = np.zeros(n_leads, dtype=int)                          # index into LADDER
    for k, (b, sl) in enumerate([(0.9, 0.6), (-0.6, 0.9), (0.1, 0.7), (-0.3, 0.6), (-0.6, 0.5), (-0.4, 0.7)], start=1):
        reach = np.where((reach == k - 1) & step(b, sl), k, reach)

    # lead-gen: there are no calls - a lead is either sold/placed (won) or not
    lg = lane == "lead_gen"
    lg_rate = np.where(proj == "insuracove-aus", 0.68, 0.012)
    reach = np.where(lg, np.where(rng.random(n_leads) < lg_rate, 6, 0), reach)

    # timing: days from opt-in to each stage, then censor at AS_OF
    age = (AS_OF - created).total_seconds().to_numpy() / 86400
    t_won = np.where(lg, rng.gamma(1.5, 4, n_leads), rng.gamma(2.0, 9, n_leads) + np.where(tl == "1-3 months", 20, 0))
    t_stage = np.vstack([np.zeros(n_leads)] + [t_won * f for f in (0.05, 0.2, 0.35, 0.5, 0.75)] + [t_won]).T
    observed = np.zeros(n_leads, dtype=int)
    for k in range(1, 7):
        observed = np.where((reach >= k) & (t_stage[:, k] <= age), k, observed)
    stage = np.array(LADDER, dtype=object)[observed]
    # canonical vocabulary quirks: no-shows sit at "no_show", some wins have moved on to onboarding
    no_show = (observed == 2) & (reach == 2) & (age > 10) & (rng.random(n_leads) < 0.7)
    stage[no_show] = "no_show"
    stage[(observed == 5) & (rng.random(n_leads) < 0.1)] = "deposit"
    stage[(observed == 6) & (age - t_won > 30) & (rng.random(n_leads) < 0.3)] = "onboarding"
    stage_at = created + pd.to_timedelta(t_stage[np.arange(n_leads), observed], unit="D")

    won = observed == 6
    lost = (~won) & (reach < 6) & (age > np.where(reach >= 2, 25, 14)) & (rng.random(n_leads) < 0.55)
    won_day = (created + pd.to_timedelta(t_won, unit="D")).normalize()     # calendar date in NY

    med = np.array([PROJECTS[s][5] for s in proj], float)
    mult = np.array([INVEST[x][1] if x else 1.0 for x in inv]) * np.where(sit == "Business owner", 1.2, 1.0)
    value = np.round(med * mult * rng.lognormal(0, 0.4, n_leads), 2)
    has_value = rng.random(n_leads) > 0.35                              # ~35% of won deals carry no value

    person = _uuid(rng, n_leads)
    lead_id = np.array([f"{s}:{p}" for s, p in zip(proj, person)])
    leads = pd.DataFrame({
        "lead_id": lead_id, "person_id": person, "project": proj,
        "opt_in_at": created.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "email_hash": _hex(rng, n_leads), "phone_hash": np.where(rng.random(n_leads) < 0.9, _hex(rng, n_leads), None),
        "utm_source": np.where(camp != None, "facebook", None), "utm_medium": np.where(camp != None, "paid", None),  # noqa: E711
        "utm_campaign": camp, "utm_term": adset, "utm_content": ad, "fbclid": fbclid, "fbp": fbp,
        "has_application": has_app.astype(int), "typeform_score": score,
        "q_investment": inv, "q_timeline": tl, "q_situation": sit,
    })
    if with_leaks:
        status = np.where(won, "won", np.where(lost, "lost", "open"))
        lq = np.where(observed >= 3, "hot", np.where(observed >= 1, "warm", "cold"))
        appt = np.where(observed >= 2, (created + pd.to_timedelta(t_stage[:, 2], unit="D")).strftime("%Y-%m-%dT%H:%M:%S+00:00"), None)
        call = np.where(observed >= 3, (created + pd.to_timedelta(t_stage[:, 3], unit="D")).strftime("%Y-%m-%dT%H:%M:%S+00:00"), None)
        leads = leads.assign(funnel_stage=stage, status=status, lead_quality=lq, appointment_at=appt, call_at=call)

    outcomes = pd.DataFrame({
        "lead_id": lead_id, "stage": stage,
        "stage_at": stage_at.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "is_lost": lost.astype(int),
        "won_date": np.where(won, won_day.strftime("%Y-%m-%d"), ""),
        "deal_value": np.where(won & has_value, value.astype(str), ""),
        "currency": np.where(proj == "insuracove-aus", "AUD", "USD"),
    })

    # ad spend: project x day (ads.v_project_daily), from each project's first ad day
    rows = []
    for s in slugs:
        meta = PROJECTS[s][3]
        if not meta:
            continue
        first = max(pd.Timestamp(meta[0]), META_START.tz_localize(None))
        last = pd.Timestamp(meta[1]) if meta[1] else AS_OF.tz_localize(None).normalize()
        days = pd.date_range(first, last, freq="D")
        n_day = pd.Series(created[proj == s].tz_localize(None).normalize()).value_counts()
        leads_d = n_day.reindex(days, fill_value=0).to_numpy()
        cpl = {"high_ticket": 95, "lead_gen": 22}[PROJECTS[s][0]]
        spend = np.round((leads_d + rng.poisson(2, len(days))) * cpl * rng.lognormal(0, 0.2, len(days)), 2)
        imp = (spend * rng.uniform(40, 70, len(days))).astype(int)
        rows.append(pd.DataFrame({"date": days.strftime("%Y-%m-%d"), "project": s, "spend": spend,
                                  "impressions": imp, "link_clicks": (imp * rng.uniform(0.008, 0.015, len(days))).astype(int),
                                  "leads": leads_d}))
    spend = pd.concat(rows, ignore_index=True)

    leads.sort_values("opt_in_at").to_csv(out / "leads.csv", index=False)
    outcomes.to_csv(out / "outcomes.csv", index=False)
    spend.to_csv(out / "ad_spend.csv", index=False)
    summary = {"leads": n_leads, "won": int(won.sum()), "won_with_value": int((won & has_value).sum()),
               "with_leaks": with_leaks}
    print(f"[sandbox] wrote {n_leads:,} leads ({won.sum():,} won, {(won & has_value).sum():,} with a value) "
          f"across {len(slugs)} projects -> {out}/" + ("  [WITH LEAK COLUMNS]" if with_leaks else ""))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-leads", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--with-leaks", action="store_true", help="add post-contact CRM columns to leads.csv")
    a = ap.parse_args()
    generate(a.out, a.n_leads, a.seed, a.with_leaks)


if __name__ == "__main__":
    main()
