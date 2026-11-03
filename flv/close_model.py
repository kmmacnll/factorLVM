"""P(close within horizon | what we know at submission) - XGBoost + Platt calibration, time-split validation."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve

from .data import Dataset


@dataclass
class CloseModel:
    booster: xgb.XGBClassifier
    calibrator: LogisticRegression
    features: list[str]
    metrics: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    def _raw(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(self.booster.predict_proba(df[self.features])[:, 1], 1e-6, 1 - 1e-6)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raw = self._raw(df)
        z = np.log(raw / (1 - raw)).reshape(-1, 1)
        return np.clip(self.calibrator.predict_proba(z)[:, 1], 1e-4, 1 - 1e-4)

    def contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """SHAP values (log-odds contributions) per feature, computed by XGBoost."""
        dm = xgb.DMatrix(df[self.features], enable_categorical=True)
        sv = self.booster.get_booster().predict(dm, pred_contribs=True)
        return pd.DataFrame(sv[:, :-1], columns=self.features, index=df.index)   # last column = bias


def train(ds: Dataset, cfg: dict) -> CloseModel:
    feats = ds.cat_cols + ds.num_cols
    m = ds.df[ds.df["mature"]].sort_values("created_at")
    n = len(m)
    if n < 200:
        raise ValueError(f"Only {n} mature leads - need at least a few hundred to train. "
                         f"Lower model.horizon_days or add more history.")

    tf = cfg["model"]["test_fraction"]
    n_test = int(n * tf)
    n_cal = int((n - n_test) * 0.2)
    fit = m.iloc[: n - n_test - n_cal]
    cal = m.iloc[n - n_test - n_cal: n - n_test]
    test = m.iloc[n - n_test:]

    pos = fit["label"].mean()
    clf = xgb.XGBClassifier(
        **cfg["model"]["xgboost"], tree_method="hist", enable_categorical=True, max_cat_to_onehot=8,
        eval_metric="logloss", random_state=cfg["simulation"]["seed"], n_jobs=-1,
    )
    clf.fit(fit[feats], fit["label"], eval_set=[(cal[feats], cal["label"])], verbose=False)
    # Platt scaling: 2-parameter recalibration, stable on small calibration sets
    rc = np.clip(clf.predict_proba(cal[feats])[:, 1], 1e-6, 1 - 1e-6)
    platt = LogisticRegression(C=1e6).fit(np.log(rc / (1 - rc)).reshape(-1, 1), cal["label"])
    model = CloseModel(clf, platt, feats)

    p = model.predict(test)
    y = test["label"].to_numpy()
    base = np.full_like(p, pos)
    model.metrics = {
        "n_train": len(fit), "n_calib": len(cal), "n_test": len(test),
        "test_period": f"{test.created_at.min().date()} -> {test.created_at.max().date()}",
        "base_rate_train": float(pos), "base_rate_test": float(y.mean()),
        "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
        "brier": float(brier_score_loss(y, p)), "brier_baseline": float(brier_score_loss(y, base)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
    }
    model.metrics["brier_skill"] = 1 - model.metrics["brier"] / model.metrics["brier_baseline"]

    # diagnostics for charts
    nb = 10 if len(test) > 1000 else 5
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=nb, strategy="quantile")
    fpr, tpr, _ = roc_curve(y, p)
    dec = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False) + 1
    lift = (pd.DataFrame({"decile": dec.to_numpy(), "y": y, "p": p}).groupby("decile")
            .agg(actual=("y", "mean"), predicted=("p", "mean"), n=("y", "size")).reset_index())
    shap = model.contributions(test).abs().mean().sort_values(ascending=False)
    model.diagnostics = {"calib": (mean_pred, frac_pos), "roc": (fpr, tpr), "lift": lift, "importance": shap}
    print(f"[close] AUC {model.metrics['auc']:.3f} | Brier skill {model.metrics['brier_skill']:.1%} "
          f"| test base rate {y.mean():.1%} on {len(test):,} leads")
    return model
