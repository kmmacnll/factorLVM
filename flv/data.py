"""Load the CSVs, map column names, validate, and build the modeling table."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _rename(df: pd.DataFrame, mapping: dict, required: list[str], name: str) -> pd.DataFrame:
    inv = {v: k for k, v in (mapping or {}).items() if v}
    df = df.rename(columns=inv)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name}: missing required columns {missing}. Fix the `columns.{name}` mapping in config.yaml. "
            f"Columns found: {list(df.columns)}")
    return df


def to_local_naive(s: pd.Series, tz: str) -> pd.Series:
    """Timestamps (UTC or offset-aware strings) -> naive wall-clock time in the business timezone."""
    t = pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
    return t.dt.tz_convert(tz).dt.tz_localize(None)


def to_business_date(s: pd.Series, tz: str) -> pd.Series:
    """Deal won dates -> calendar dates.

    Close stores won dates as *dates* at midnight UTC. Converting those to New York would move them
    to the previous day, so a value that is exactly midnight UTC (or a bare YYYY-MM-DD) is kept as that
    calendar date. Anything with a real time of day (GHL) is converted to the business timezone first.
    """
    raw = s.astype("string").str.strip().replace("", pd.NA)
    t = pd.to_datetime(raw, errors="coerce", utc=True, format="mixed")
    midnight_utc = (t.dt.hour == 0) & (t.dt.minute == 0) & (t.dt.second == 0)
    local = t.dt.tz_convert(tz)
    out = local.dt.tz_localize(None).dt.normalize()
    out = out.where(~midnight_utc, t.dt.tz_localize(None).dt.normalize())
    return out


@dataclass
class Dataset:
    df: pd.DataFrame            # one row per lead, all features + labels
    spend: pd.DataFrame | None  # daily spend (per project, or per campaign) - may be None
    as_of: pd.Timestamp
    cat_cols: list[str]
    num_cols: list[str]
    stages: list[str]


def _filter_projects(df: pd.DataFrame, d: dict, name: str) -> pd.DataFrame:
    if "client_id" not in df.columns:
        return df
    keep, drop = d.get("projects"), set(d.get("exclude_projects") or [])
    n0 = len(df)
    if drop:
        df = df[~df["client_id"].astype(str).isin(drop)]
    if keep:
        df = df[df["client_id"].astype(str).isin(set(keep))]
    if len(df) < n0:
        print(f"[data] {name}: kept {len(df):,} of {n0:,} rows after project filter "
              f"(projects={keep or 'all'}, excluded={sorted(drop) or 'none'})")
    return df


def load(cfg: dict) -> Dataset:
    d, cols = cfg["data"], cfg["columns"]
    stages = cfg["funnel"]["stages"]
    aliases = {k.lower(): v.lower() for k, v in (cfg["funnel"].get("stage_aliases") or {}).items()}
    tz = d.get("timezone", "America/New_York")

    leads = _rename(pd.read_csv(d["leads_csv"]), cols["leads"], ["lead_id", "created_at"], "leads")
    outc = _rename(pd.read_csv(d["outcomes_csv"]), cols["outcomes"], ["lead_id", "stage"], "outcomes")
    leads = _filter_projects(leads, d, "leads")

    # opt-in time in the business timezone (all projects are America/New_York today)
    leads["created_at"] = to_local_naive(leads["created_at"], tz)
    bad = leads["created_at"].isna().sum()
    if bad:
        print(f"[data] warning: dropping {bad} leads with unparseable created_at")
        leads = leads.dropna(subset=["created_at"])
    dups = leads["lead_id"].duplicated().sum()
    if dups:
        print(f"[data] warning: {dups} duplicate lead_ids in leads - keeping first")
        leads = leads.sort_values("created_at").drop_duplicates("lead_id")
    outc = outc.drop_duplicates("lead_id", keep="last")

    df = leads.merge(outc, on="lead_id", how="left")
    df["stage"] = df["stage"].fillna(stages[0]).astype(str).str.strip().str.lower().replace(aliases)
    unknown = set(df["stage"]) - set(stages)
    if unknown:
        raise ValueError(f"outcomes: stages {unknown} not in funnel.stages {stages} - add them to "
                         f"funnel.stages or funnel.stage_aliases in config.yaml")
    df["stage_idx"] = df["stage"].map({s: i for i, s in enumerate(stages)}).astype(int)
    df["is_lost"] = pd.to_numeric(df.get("is_lost", 0), errors="coerce").fillna(0).astype(int)
    df["closed_at"] = to_business_date(df["closed_at"], tz) if "closed_at" in df else pd.NaT
    df["deal_value"] = pd.to_numeric(df.get("deal_value"), errors="coerce")
    df["won"] = (df["stage_idx"] == len(stages) - 1).astype(int)
    df.loc[df["won"] == 1, "is_lost"] = 0

    as_of = pd.Timestamp(d["as_of_date"]) if d.get("as_of_date") else max(
        df["created_at"].max(), df["closed_at"].max() if df["closed_at"].notna().any() else df["created_at"].max())
    df = df[df["created_at"] <= as_of].copy()

    H = cfg["model"]["horizon_days"]
    df["age_days"] = (as_of - df["created_at"]).dt.total_seconds() / 86400
    # won dates are calendar dates: a deal won on the opt-in day can come out slightly negative -> clip to 0
    df["days_to_close"] = ((df["closed_at"] - df["created_at"]).dt.total_seconds() / 86400).clip(lower=0)
    df["mature"] = df["age_days"] >= H                      # old enough to know the label
    df["label"] = ((df["won"] == 1) & (df["days_to_close"].fillna(0) <= H)).astype(int)
    df["is_open"] = (df["won"] == 0) & (df["is_lost"] == 0)

    # survival columns: duration until close, or until today if not (yet) closed
    df["surv_T"] = np.where(df["won"] == 1, df["days_to_close"].fillna(df["age_days"]), df["age_days"]).clip(min=0.01)
    df["surv_E"] = df["won"]

    feats = cfg["features"]
    cat_cols = [c for c in feats["categorical"] if c in df.columns]
    skipped = set(feats["categorical"]) - set(cat_cols)
    if skipped:
        print(f"[data] note: categorical features not found, skipping: {sorted(skipped)}")
    num_cols = [c for c in feats.get("numeric") or [] if c in df.columns]
    if feats.get("time_features", True):
        df["hour"] = df["created_at"].dt.hour               # local (business-timezone) hour
        df["dow"] = df["created_at"].dt.dayofweek
        df["month"] = df["created_at"].dt.month
        num_cols += ["hour", "dow", "month"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")       # NaN is fine: XGBoost learns a missing branch
    for c in cat_cols:
        # null ad-level fields (pre-2026-07-01, no ad account, organic) become their own "(missing)" level
        df[c] = df[c].astype("string").fillna("(missing)").astype("category")

    spend = None
    sp = d.get("ad_spend_csv")
    if sp and Path(sp).exists():
        spend = _rename(pd.read_csv(sp), cols.get("ad_spend", {}), ["date", "spend"], "ad_spend")
        spend = _filter_projects(spend, d, "ad_spend")
        spend["date"] = pd.to_datetime(spend["date"])       # already a business-timezone day
        spend["spend"] = pd.to_numeric(spend["spend"], errors="coerce").fillna(0)

    print(f"[data] {len(df):,} leads | {df['won'].sum():,} won | {df['mature'].sum():,} mature (>= {H}d old) "
          f"| as of {as_of.date()}")
    if df.loc[df.mature, "label"].sum() < 30:
        print("[data] WARNING: fewer than 30 wins among mature leads - models will be very noisy.")
    return Dataset(df=df, spend=spend, as_of=as_of, cat_cols=cat_cols, num_cols=num_cols, stages=stages)
