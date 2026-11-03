"""Where is the lead in the funnel, and what does that tell us?

1. Survival curve (Kaplan-Meier): how long closers take to close. Lets us fairly score young leads
   that simply haven't had time yet (right-censoring), instead of counting them as losses.
2. Bayesian funnel: Beta-Binomial posterior for every stage-to-stage conversion rate, and the
   likelihood ratio each stage gives for "this lead will win". Used to update a lead's P(close)
   as it moves down the CRM funnel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from .data import Dataset


@dataclass
class FunnelModel:
    km: KaplanMeierFitter
    horizon: float
    stage_names: list[str]
    transition_draws: np.ndarray       # [draws, n_stages-1] P(reach k+1 | reached k)
    reach_given_loss: np.ndarray       # [n_stages] P(reach k | did not win), posterior mean
    counts: pd.DataFrame

    def G(self, t) -> np.ndarray:
        """Share of eventual closers (within horizon) who have closed by day t."""
        F = 1 - self.km.survival_function_at_times(np.minimum(t, self.horizon)).to_numpy()
        FH = 1 - self.km.survival_function_at_times([self.horizon]).to_numpy()[0]
        return np.clip(F / max(FH, 1e-9), 0, 1)

    def update(self, p_submit: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        """P(close within horizon | features, age so far, current stage)."""
        p = np.asarray(p_submit, float)
        g = self.G(df["age_days"].to_numpy())
        p_age = p * (1 - g) / np.clip(1 - p * g, 1e-9, None)          # survived this long without closing
        lr = 1.0 / np.clip(self.reach_given_loss[df["stage_idx"].to_numpy()], 1e-3, 1)  # P(reach|win)=1
        odds = p_age / np.clip(1 - p_age, 1e-9, None) * lr
        p_now = odds / (1 + odds)
        p_now = np.where(df["won"].to_numpy() == 1, 1.0, p_now)
        p_now = np.where(df["is_lost"].to_numpy() == 1, 0.0, p_now)
        return p_now


def fit(ds: Dataset, cfg: dict, n_draws: int = 4000) -> FunnelModel:
    df, K, H = ds.df, len(ds.stages), cfg["model"]["horizon_days"]
    km = KaplanMeierFitter().fit(df["surv_T"], df["surv_E"], label="time to close")

    m = df[df["mature"]]
    rng = np.random.default_rng(cfg["simulation"]["seed"])
    rows, draws = [], []
    for k in range(K - 1):
        n_k = int((m.stage_idx >= k).sum())
        x_k = int((m.stage_idx >= k + 1).sum())
        draws.append(rng.beta(1 + x_k, 1 + n_k - x_k, n_draws))
        rows.append({"from": ds.stages[k], "to": ds.stages[k + 1], "n": n_k, "converted": x_k})
    counts = pd.DataFrame(rows)
    T = np.column_stack(draws)
    counts["rate_mean"] = T.mean(0)
    counts["rate_lo"], counts["rate_hi"] = np.percentile(T, 3, 0), np.percentile(T, 97, 0)

    nl = m[m["won"] == 0]
    reach = np.array([(1 + (nl.stage_idx >= k).sum()) / (2 + len(nl)) for k in range(K)])
    reach[0] = 1.0
    reach[-1] = 1.0  # won stage is handled separately
    med = df.loc[df.won == 1, "days_to_close"].median()
    print(f"[funnel] median days-to-close (winners) {med:.0f} | new->won {T.prod(1).mean():.1%} "
          f"| share of closers done by day {H}: {1 - km.survival_function_at_times([H]).iloc[0]:.1%} of all leads")
    return FunnelModel(km, H, ds.stages, T, reach, counts)

