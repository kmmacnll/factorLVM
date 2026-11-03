"""Score every lead and Monte Carlo the revenue each campaign is likely to produce."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .close_model import CloseModel
from .data import Dataset
from .funnel import FunnelModel
from .value_model import ValueModel


def score(ds: Dataset, cm: CloseModel, vm: ValueModel, fm: FunnelModel, cfg: dict) -> pd.DataFrame:
    df = ds.df
    sim = cfg["simulation"]
    out = df[["lead_id", "created_at"] + [c for c in ["client_id", "campaign", "adset", "ad", "platform", "placement",
                                                       "email", "phone", "email_hash", "phone_hash", "fbp"]
                                           if c in df.columns]
             + ["stage", "is_lost", "won", "deal_value", "age_days", "mature", "label"]].copy()
    out["p_close_at_submit"] = cm.predict(df)                    # what Meta should see on day 0
    out["p_close_now"] = fm.update(out["p_close_at_submit"].to_numpy(), df)
    ev, lo, hi = vm.expected_value(df)
    out["value_if_won"], out["value_if_won_lo90"], out["value_if_won_hi90"] = ev, lo, hi
    m = sim["gross_margin"]
    out["expected_value"] = out["p_close_at_submit"] * ev * m       # signal to send for value-based bidding
    out["expected_value_now"] = out["p_close_now"] * ev * m         # for pipeline / sales prioritisation
    out["max_cpl_at_target_roas"] = out["expected_value"] / sim["target_roas"]
    out["value_tier"] = pd.qcut(out["expected_value"].rank(method="first"), 5,
                                labels=["E (lowest)", "D", "C", "B", "A (highest)"])
    return out


def monte_carlo(scored: pd.DataFrame, vm: ValueModel, ds: Dataset, cfg: dict,
                group: str = "campaign", window_days: int | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Simulate total revenue per `group` for leads created in the last `window_days`.

    Each simulation: draw a posterior sample of the value model, decide close/no-close for every lead
    with its P(close), draw each closer's deal size, sum by group. Returns summary + total-revenue sims.
    """
    sim = cfg["simulation"]
    window_days = window_days or cfg["model"]["horizon_days"]
    rng = np.random.default_rng(sim["seed"])
    cut = ds.as_of - pd.Timedelta(days=window_days)
    w = scored[scored["created_at"] > cut]
    src = ds.df.loc[w.index]
    groups = w[group].astype(str).to_numpy() if group in w.columns else np.array(["all"] * len(w))
    gnames, gidx = np.unique(groups, return_inverse=True)
    # use the submit-time probability: we want the value these leads were worth when bought
    p = w["p_close_at_submit"].to_numpy()

    n_sims, chunk = sim["n_sims"], 250
    totals = np.zeros((n_sims, len(gnames)))
    mu_all, s_all = vm.draw_log_mu(src, n_draws=min(1000, n_sims), seed=sim["seed"])
    for start in range(0, n_sims, chunk):
        b = min(chunk, n_sims - start)
        d = rng.integers(0, len(s_all), b)
        closed = rng.random((b, len(w))) < p[None, :]
        val = np.exp(mu_all[d] + s_all[d][:, None] * rng.standard_normal((b, len(w)))) * sim["gross_margin"]
        rev = closed * val
        for g in range(len(gnames)):
            totals[start:start + b, g] = rev[:, gidx == g].sum(1)

    summ = pd.DataFrame({group: gnames, "leads": np.bincount(gidx, minlength=len(gnames)),
                         "rev_mean": totals.mean(0), "rev_p10": np.percentile(totals, 10, 0),
                         "rev_p50": np.percentile(totals, 50, 0), "rev_p90": np.percentile(totals, 90, 0),
                         "exp_closes": np.bincount(gidx, weights=p, minlength=len(gnames))})
    summ["value_per_lead"] = summ["rev_mean"] / summ["leads"]
    # spend joins on the group column when the spend table has it: ads.v_project_daily is per project
    # (client_id), so campaign-level runs carry value only and project-level runs carry ROAS too.
    if ds.spend is not None and group in ds.spend.columns:
        sp = ds.spend[ds.spend["date"] > cut.normalize()].groupby(ds.spend[group].astype(str))["spend"].sum()
        summ["spend"] = summ[group].map(sp).fillna(0.0)
        summ["cpl"] = summ["spend"] / summ["leads"]
        summ["roas_mean"] = summ["rev_mean"] / summ["spend"].replace(0, np.nan)
        prob = (totals / summ["spend"].replace(0, np.nan).to_numpy()[None, :]) >= sim["target_roas"]
        summ["p_beats_target_roas"] = prob.mean(0)
    summ = summ.sort_values("rev_mean", ascending=False).reset_index(drop=True)
    print(f"[sim] by {group}: {n_sims:,} sims over {len(w):,} leads from last {window_days}d | "
          f"expected value ${totals.sum(1).mean():,.0f} (P10 ${np.percentile(totals.sum(1), 10):,.0f} - "
          f"P90 ${np.percentile(totals.sum(1), 90):,.0f})")
    return summ, totals.sum(1)


def backtest(scored: pd.DataFrame, cm: CloseModel, group: str = "campaign", margin: float = 1.0) -> pd.DataFrame:
    """On the held-out test period: predicted value vs revenue that actually came in, per group."""
    n_test = cm.metrics["n_test"]
    t = scored[scored["mature"]].sort_values("created_at").tail(n_test)
    # most won deals in factor-data carry no value: impute those with the model's E[value | won] so a
    # missing value doesn't read as zero revenue (count reported as actual_value_imputed)
    imputed = (t["label"] == 1) & ~(t["deal_value"] > 0)
    won_value = t["deal_value"].where(~imputed, t["value_if_won"])
    t = t.assign(actual=np.where(t["label"] == 1, won_value * margin, 0.0), imputed=imputed.astype(int))
    g = t.groupby(t[group].astype(str)).agg(leads=("lead_id", "size"), predicted=("expected_value", "sum"),
                                            actual=("actual", "sum"), pred_closes=("p_close_at_submit", "sum"),
                                            actual_closes=("label", "sum"),
                                            actual_value_imputed=("imputed", "sum")).reset_index()
    return g.sort_values("predicted", ascending=False)


def backtest_summary(bt: pd.DataFrame) -> dict:
    a, p = bt["actual"].sum(), bt["predicted"].sum()
    ac, pc = bt["actual_closes"].sum(), bt["pred_closes"].sum()
    return {"backtest_value_error": float((p - a) / a) if a else float("nan"),
            "backtest_closes_error": float((pc - ac) / ac) if ac else float("nan"),
            "backtest_pred_closes": float(pc), "backtest_actual_closes": int(ac),
            "backtest_value_imputed_wins": int(bt["actual_value_imputed"].sum())}

