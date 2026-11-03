"""Minimal Kaplan-Meier estimator (numpy only).

Replaces lifelines, whose autograd-gamma dependency fails to build on recent Pythons.
Exposes the small part of the lifelines API the engine uses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class KaplanMeier:
    def fit(self, durations, events, label: str = "KM_estimate") -> "KaplanMeier":
        T = np.asarray(durations, float)
        E = np.asarray(events, int).astype(bool)
        times, d = np.unique(T[E], return_counts=True)
        d = d.astype(float)
        Ts = np.sort(T)
        at_risk = (len(Ts) - np.searchsorted(Ts, times, side="left")).astype(float)   # #{T >= t}
        S = np.cumprod(1 - d / at_risk)
        # Greenwood variance, exponential ("log-log") confidence interval as in lifelines
        with np.errstate(divide="ignore", invalid="ignore"):
            gw = np.cumsum(d / (at_risk * (at_risk - d)))
            z = 1.959963984540054   # 95% interval
            v = gw / np.log(S) ** 2
            lo = np.exp(-np.exp(np.log(-np.log(S)) + z * np.sqrt(v)))
            hi = np.exp(-np.exp(np.log(-np.log(S)) - z * np.sqrt(v)))
        lo, hi = np.nan_to_num(lo, nan=0.0), np.nan_to_num(hi, nan=1.0)
        idx = np.r_[0.0, times]
        self._t, self._S = idx, np.r_[1.0, S]
        self.survival_function_ = pd.DataFrame({label: self._S}, index=pd.Index(idx, name="timeline"))
        self.confidence_interval_ = pd.DataFrame({f"{label}_lower_0.95": np.r_[1.0, lo], f"{label}_upper_0.95": np.r_[1.0, hi]},
                                                 index=pd.Index(idx, name="timeline"))
        return self

    def survival_function_at_times(self, t) -> pd.Series:
        t = np.atleast_1d(np.asarray(t, float))
        pos = np.searchsorted(self._t, t, side="right") - 1
        return pd.Series(self._S[np.clip(pos, 0, None)], index=t)
