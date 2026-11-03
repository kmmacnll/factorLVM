"""Generate realistic fake Factor data so the whole pipeline runs before real data arrives.

Ground truth is baked in (budget, timeline, placement and campaign really do change close
rates and deal sizes) so you can check the models recover it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

STAGES = ["new", "contacted", "qualified", "proposal", "won"]


def generate(out_dir: str | Path = "data", n_leads: int = 12000, seed: int = 7,
             end_date: str = "2026-10-15") -> None:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp(end_date)
    start = end - pd.Timedelta(days=540)

    clients = {"c_roofing": 1.15, "c_solar": 1.0, "c_reno": 0.85, "c_hvac": 0.7}       # value multiplier
    client_rate = {"c_roofing": 0.0, "c_solar": -0.2, "c_reno": 0.15, "c_hvac": 0.3}    # close-rate offset
    campaigns = {
        "Prospecting_Broad": -0.35, "Prospecting_LAL1%": 0.10, "Prospecting_Interest": -0.15,
        "Retargeting_30d": 0.55, "Retargeting_VideoViewers": 0.30, "Promo_FallOffer": -0.05,
        "Prospecting_Advantage+": 0.0, "Brand_Awareness_LeadGen": -0.6,
    }
    budget = {"<5k": (-0.6, 3500), "5k-15k": (0.0, 9500), "15k-40k": (0.35, 24000), "40k+": (0.55, 58000)}
    timeline = {"ASAP": 0.7, "1-3 months": 0.3, "3-6 months": -0.2, "Just researching": -1.1}
    ptype = {"Detached": (0.15, 1.2), "Townhouse": (0.0, 0.9), "Condo": (-0.3, 0.6), "Commercial": (0.1, 2.2)}
    placement = {"feed": 0.15, "stories": -0.1, "reels": -0.25, "marketplace": -0.35}

    n = n_leads
    # lead volume grows over time
    u = rng.beta(1.4, 1.0, n)
    created = start + pd.to_timedelta(u * 540, unit="D") + pd.to_timedelta(rng.integers(0, 86400, n), unit="s")
    hour = created.hour
    cl = rng.choice(list(clients), n, p=[0.3, 0.3, 0.25, 0.15])
    camp = rng.choice(list(campaigns), n, p=[0.22, 0.15, 0.12, 0.10, 0.08, 0.10, 0.15, 0.08])
    adset = np.array([f"{c}__AS{rng.integers(1, 4)}" for c in camp])
    ad = np.array([f"{a}__AD{rng.integers(1, 6)}" for a in adset])
    plat = rng.choice(["facebook", "instagram"], n, p=[0.58, 0.42])
    plc = rng.choice(list(placement), n, p=[0.45, 0.22, 0.25, 0.08])
    bud = rng.choice(list(budget), n, p=[0.35, 0.33, 0.22, 0.10])
    tl = rng.choice(list(timeline), n, p=[0.18, 0.32, 0.25, 0.25])
    pt = rng.choice(list(ptype), n, p=[0.45, 0.22, 0.25, 0.08])

    logit = (-2.4 + np.vectorize(client_rate.get)(cl) + np.vectorize(campaigns.get)(camp)
             + np.array([budget[b][0] for b in bud]) + np.vectorize(timeline.get)(tl)
             + np.array([ptype[p][0] for p in pt]) + np.vectorize(placement.get)(plc)
             + np.where(plat == "facebook", 0.1, 0.0)
             + np.where((hour >= 9) & (hour <= 17), 0.12, -0.05)
             + rng.normal(0, 0.35, n))
    p_win = 1 / (1 + np.exp(-logit))
    will_win = rng.random(n) < p_win

    # funnel progression for non-winners: drop out at some stage
    stage_idx = np.where(will_win, 4, 0)
    loser = ~will_win
    # probability of reaching each stage, correlated with latent quality
    q = 1 / (1 + np.exp(-(logit + 2.0)))
    reach = np.zeros(n, dtype=int)
    for k in range(1, 4):
        go = rng.random(n) < (0.35 + 0.55 * q) * (0.95 if k == 1 else 0.8)
        reach = np.where((reach == k - 1) & go, k, reach)
    stage_idx = np.where(loser, reach, stage_idx)

    # time to close (days) and deal value
    ttc = rng.gamma(shape=2.2, scale=14, size=n) + np.where(tl == "3-6 months", 45, 0) + np.where(tl == "1-3 months", 15, 0)
    base_val = np.array([budget[b][1] for b in bud]) * np.array([ptype[p][1] for p in pt]) * np.vectorize(clients.get)(cl)
    value = base_val * rng.lognormal(0, 0.45, n)

    closed_at = created + pd.to_timedelta(ttc, unit="D")
    as_of = end
    age_days = np.clip((as_of - created).days, 0, None)
    # censoring: deals that haven't closed yet by "today" look open
    won_obs = will_win & (closed_at <= as_of)
    # open leads still in pipeline: winners not yet closed sit at a mid stage; young losers not yet marked lost
    progress = np.clip(age_days / np.maximum(ttc, 1), 0, 1)
    obs_stage = np.where(won_obs, 4, np.where(will_win, np.floor(progress * 3.99).astype(int), stage_idx))
    lost_obs = (~will_win) & (age_days > 21)
    obs_stage = np.where(~will_win & (age_days <= 21), np.minimum(stage_idx, (age_days // 7).astype(int)), obs_stage)

    leads = pd.DataFrame({
        "lead_id": [f"L{i:06d}" for i in range(n)],
        "created_at": created.strftime("%Y-%m-%d %H:%M:%S"),
        "client_id": cl, "campaign_name": camp, "adset_name": adset, "ad_name": ad,
        "platform": plat, "placement": plc,
        "q_budget": bud, "q_timeline": tl, "q_property_type": pt,
        "email": [f"lead{i}@example.com" for i in range(n)],
        "phone": [f"+1604555{i % 10000:04d}" for i in range(n)],
    }).sort_values("created_at")

    outcomes = pd.DataFrame({
        "lead_id": [f"L{i:06d}" for i in range(n)],
        "stage": np.array(STAGES)[obs_stage],
        "is_lost": lost_obs.astype(int),
        "closed_at": np.where(won_obs, closed_at.strftime("%Y-%m-%d"), ""),
        "deal_value": np.where(won_obs, np.round(value, 2).astype(str), ""),
    })

    # daily spend per campaign, roughly proportional to lead volume with differing CPLs
    cpl = {"Prospecting_Broad": 185, "Prospecting_LAL1%": 210, "Prospecting_Interest": 240, "Retargeting_30d": 290,
           "Retargeting_VideoViewers": 175, "Promo_FallOffer": 260, "Prospecting_Advantage+": 150,
           "Brand_Awareness_LeadGen": 120}
    lead_days = leads.assign(date=pd.to_datetime(leads.created_at).dt.date).groupby(["date", "campaign_name"]).size()
    spend = (lead_days.rename("leads").reset_index()
             .assign(spend=lambda d: (d.leads * d.campaign_name.map(cpl) * rng.lognormal(0, 0.15, len(d))).round(2)))
    spend[["date", "campaign_name", "spend"]].to_csv(out / "ad_spend.csv", index=False)
    leads.to_csv(out / "leads.csv", index=False)
    outcomes.to_csv(out / "outcomes.csv", index=False)
    print(f"[synthetic] wrote {n} leads, {won_obs.sum()} observed wins -> {out}/")


if __name__ == "__main__":
    generate()

