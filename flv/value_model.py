"""E[deal value | won] - hierarchical Bayesian log-normal regression in PyMC.

log(value) = intercept + client_effect[client] + X @ beta + noise
client effects are partially pooled, so a client with 15 closed deals borrows strength from the others
instead of getting a wild estimate of its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import Dataset


@dataclass
class ValueModel:
    feat_cols: list[str]               # one-hot design columns
    raw_feats: list[str]
    clients: list[str]
    post: dict                         # numpy arrays of posterior draws
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    ppc: dict = field(default_factory=dict)

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        X = pd.get_dummies(df[self.raw_feats].astype(str), dtype=float)
        return X.reindex(columns=self.feat_cols, fill_value=0.0).to_numpy()

    def _client_idx(self, df: pd.DataFrame) -> np.ndarray:
        if "client_id" not in df.columns:
            return np.full(len(df), -1)
        lookup = {c: i for i, c in enumerate(self.clients)}
        return df["client_id"].astype(str).map(lookup).fillna(-1).astype(int).to_numpy()

    def draw_log_mu(self, df: pd.DataFrame, n_draws: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Returns (mu draws [n_draws, n_leads], sigma draws [n_draws])."""
        rng = np.random.default_rng(seed)
        S = len(self.post["intercept"])
        idx = rng.choice(S, size=min(n_draws, S), replace=False)
        X, ci = self._design(df), self._client_idx(df)
        a = self.post["intercept"][idx][:, None]
        b = self.post["beta"][idx]                                     # [d, k]
        ce = self.post["client_effect"][idx]                           # [d, n_clients]
        ce_i = np.where(ci[None, :] >= 0, ce[:, np.clip(ci, 0, None)], 0.0)
        mu = a + ce_i + ((X @ b.T).T if b.size else 0.0)
        return mu, self.post["sigma"][idx]

    def expected_value(self, df: pd.DataFrame, n_draws: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Posterior mean and 90% interval of E[value | won] per lead."""
        mu, s = self.draw_log_mu(df, n_draws)
        ev = np.exp(mu + 0.5 * s[:, None] ** 2)
        return ev.mean(0), np.percentile(ev, 5, 0), np.percentile(ev, 95, 0)


def train(ds: Dataset, cfg: dict) -> ValueModel:
    import pymc as pm

    vc = cfg["model"]["value_model"]
    all_won = ds.df[ds.df["won"] == 1]
    # ~85% of factor-data deals carry no value: train only on won deals that have one
    won = all_won[all_won["deal_value"] > 0].copy()
    print(f"[value] using {len(won):,} of {len(all_won):,} won deals that have a deal value "
          f"({len(all_won) - len(won):,} won without a value are left out)")
    if len(won) < 20:
        raise ValueError(f"Only {len(won)} won deals with a deal_value - need at least ~20 for the value model.")
    raw = [c for c in vc["features"] if c in won.columns]
    X = pd.get_dummies(won[raw].astype(str), drop_first=True, dtype=float)
    feat_cols = list(X.columns)
    y = np.log(won["deal_value"].to_numpy())
    has_client = "client_id" in won.columns
    clients = sorted(won["client_id"].astype(str).unique()) if has_client else ["_all"]
    ci = won["client_id"].astype(str).map({c: i for i, c in enumerate(clients)}).to_numpy() if has_client \
        else np.zeros(len(won), dtype=int)

    with pm.Model(coords={"feat": feat_cols, "client": clients}) as model:
        intercept = pm.Normal("intercept", mu=y.mean(), sigma=2.0)
        beta = pm.Normal("beta", 0.0, 1.0, dims="feat")
        tau = pm.HalfNormal("client_sd", 0.5)
        z = pm.Normal("z_client", 0.0, 1.0, dims="client")
        client_effect = pm.Deterministic("client_effect", z * tau, dims="client")
        sigma = pm.HalfNormal("sigma", 1.0)
        mu = intercept + pm.math.dot(X.to_numpy(), beta) + client_effect[ci]
        pm.Normal("log_value", mu=mu, sigma=sigma, observed=y)

        seed = cfg["simulation"]["seed"]
        if vc.get("method", "advi") == "nuts":
            idata = pm.sample(draws=vc["draws"], tune=vc["tune"], chains=vc["chains"], random_seed=seed,
                              progressbar=False, target_accept=0.9)
        else:
            approx = pm.fit(n=vc["advi_iterations"], method="advi", random_seed=seed, progressbar=False)
            idata = approx.sample(vc["draws"], random_seed=seed)

    p = idata.posterior
    flat = lambda v: p[v].stack(s=("chain", "draw")).transpose("s", ...).to_numpy()
    post = {"intercept": flat("intercept"), "beta": flat("beta"), "client_effect": flat("client_effect"),
            "sigma": flat("sigma"), "client_sd": flat("client_sd")}
    if post["beta"].ndim == 1:
        post["beta"] = post["beta"][:, None]

    # coefficient summary as % change in deal size (exp(beta) - 1)
    rows = []
    for j, f in enumerate(feat_cols):
        e = np.exp(post["beta"][:, j]) - 1
        rows.append({"term": f, "kind": "feature", "mean": e.mean(), "lo": np.percentile(e, 3), "hi": np.percentile(e, 97)})
    for j, c in enumerate(clients):
        e = np.exp(post["client_effect"][:, j]) - 1
        rows.append({"term": f"project: {c}", "kind": "client", "mean": e.mean(), "lo": np.percentile(e, 3),
                     "hi": np.percentile(e, 97)})
    vm = ValueModel(feat_cols, raw, clients if has_client else [], post, pd.DataFrame(rows))

    # posterior predictive check on the log scale
    mu_d, s_d = vm.draw_log_mu(won, n_draws=50)
    rng = np.random.default_rng(1)
    rep = mu_d + s_d[:, None] * rng.standard_normal(mu_d.shape)
    vm.ppc = {"observed": y, "replicated": rep[:20]}

    if vc.get("method") == "nuts":
        import arviz as az
        rh = az.rhat(idata, var_names=["intercept", "beta", "sigma"])
        r = max(float(np.nanmax(np.asarray(rh[v]))) for v in ["intercept", "beta", "sigma"])
        print(f"[value] NUTS max R-hat {r:.3f} (want < 1.01)")
    print(f"[value] fit on {len(won):,} won deals | median deal ${np.exp(np.median(y)):,.0f} "
          f"| noise sd (log) {post['sigma'].mean():.2f}")
    return vm
