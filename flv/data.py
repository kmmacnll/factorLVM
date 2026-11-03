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


@dataclass
class Dataset:
    df: pd.DataFrame            # one row per lead, all features + labels
    spend: pd.DataFrame | None  # daily spend per campaign (may be None)
    as_of: pd.Timestamp
    cat_cols: list[str]
    num_cols: list[str]
    stages: list[str]


def load(cfg: dict) -> Dataset:
    d, cols = cfg["data"], cfg["columns"]
    stages = cfg["funnel"]["stages"]

    leads = _rename(pd.read_csv(d["leads_csv"]), cols["leads"], ["lead_id", "created_at"], "leads")
    outc = _rename(pd.read_csv(d["outcomes_csv"]), cols["outcomes"], ["lead_id", "stage"], "outcomes")

    leads["created_at"] = pd.to_datetime(leads["created_at"], errors="coerce")
    bad = leads["created_at"].isna().sum()
    if bad:
        print(f"[data] warning: dropping {bad} leads with unparseable created_at")
        leads = leads.dropna(subset=["created_at"])
    dups = leads["lead_id"].duplicated().sum()
    if dups:
        print(f"[data] warning: {dups} duplicate lead_ids in leads - keeping first")
        leads = leads.drop_duplicates("lead_id")
    outc = outc.drop_duplicates("lead_id", keep="last")

    df = leads.merge(outc, on="lead_id", how="left")
    df["stage"] = df["stage"].fillna(stages[0]).astype(str).str.strip().str.lower()
    unknown = set(df["stage"]) - set(stages)
    if unknown:
        raise ValueError(f"outcomes: stages {unknown} not in funnel.stages {stages} - add or map them in config.yaml")
    df["stage_idx"] = df["stage"].map({s: i for i, s in enumerate(stages)}).astype(int)
    df["is_lost"] = pd.to_numeric(df.get("is_lost", 0), errors="coerce").fillna(0).astype(int)
    df["closed_at"] = pd.to_datetime(df.get("closed_at"), errors="coerce")
    df["deal_value"] = pd.to_numeric(df.get("deal_value"), errors="coerce")
    df["won"] = (df["stage_idx"] == len(stages) - 1).astype(int)
    # a won deal with no close date: assume closed at as-of (still counts as won)

    as_of = pd.Timestamp(d["as_of_date"]) if d.get("as_of_date") else max(
        df["created_at"].max(), df["closed_at"].max() if df["closed_at"].notna().any() else df["created_at"].max())
    df = df[df["created_at"] <= as_of].copy()

    H = cfg["model"]["horizon_days"]
    df["age_days"] = (as_of - df["created_at"]).dt.total_seconds() / 86400
    df["days_to_close"] = (df["closed_at"] - df["created_at"]).dt.total_seconds() / 86400
    df["mature"] = df["age_days"] >= H                      # old enough to know the label
    df["label"] = ((df["won"] == 1) & (df["days_to_close"].fillna(0) <= H)).astype(int)
    df["is_open"] = (df["won"] == 0) & (df["is_lost"] == 0)

    # survival columns: duration until close, or until today if not (yet) closed
    df["surv_T"] = np.where(df["won"] == 1, df["days_to_close"].fillna(df["age_days"]), df["age_days"]).clip(min=0.01)
    df["surv_E"] = df["won"]

    cat_cols = [c for c in cfg["features"]["categorical"] if c in df.columns]
    skipped = set(cfg["features"]["categorical"]) - set(cat_cols)
    if skipped:
        print(f"[data] note: categorical features not found, skipping: {sorted(skipped)}")
    num_cols = [c for c in cfg["features"].get("numeric", []) if c in df.columns]
    if cfg["features"].get("time_features", True):
        df["hour"] = df["created_at"].dt.hour
        df["dow"] = df["created_at"].dt.dayofweek
        df["month"] = df["created_at"].dt.month
        num_cols += ["hour", "dow", "month"]
    for c in cat_cols:
        df[c] = df[c].astype("string").fillna("(missing)").astype("category")

    spend = None
    sp = d.get("ad_spend_csv")
    if sp and Path(sp).exists():
        spend = _rename(pd.read_csv(sp), cols.get("ad_spend", {}), ["date", "campaign", "spend"], "ad_spend")
        spend["date"] = pd.to_datetime(spend["date"])
        spend["spend"] = pd.to_numeric(spend["spend"], errors="coerce").fillna(0)

    print(f"[data] {len(df):,} leads | {df['won'].sum():,} won | {df['mature'].sum():,} mature (>= {H}d old) "
          f"| as of {as_of.date()}")
    if df.loc[df.mature, "label"].sum() < 30:
        print("[data] WARNING: fewer than 30 wins among mature leads - models will be very noisy.")
    return Dataset(df=df, spend=spend, as_of=as_of, cat_cols=cat_cols, num_cols=num_cols, stages=stages)

