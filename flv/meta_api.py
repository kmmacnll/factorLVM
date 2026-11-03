"""Meta Marketing API (pull spend) + Conversions API (push predicted lead values back for value-based bidding).

Credentials come from .env (see .env.example). Nothing is sent while `meta.dry_run: true` in config.yaml -
payloads are written to outputs/capi_payloads.json so you can inspect them first.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

GRAPH = "https://graph.facebook.com"


def _env() -> dict:
    load_dotenv()
    return {k: os.getenv(k, "").strip() for k in
            ["META_ACCESS_TOKEN", "META_AD_ACCOUNT_ID", "META_PIXEL_ID", "META_API_VERSION", "META_TEST_EVENT_CODE"]}


def _require(env: dict, keys: list[str]) -> None:
    missing = [k for k in keys if not env.get(k)]
    if missing:
        raise RuntimeError(f"Missing {missing} in .env - copy .env.example to .env and fill them in.")


def pull_spend(cfg: dict, since: str, until: str) -> pd.DataFrame:
    """Daily spend per campaign from the Marketing API -> writes data.ad_spend_csv in your column names."""
    env = _env()
    _require(env, ["META_ACCESS_TOKEN", "META_AD_ACCOUNT_ID"])
    ver = env["META_API_VERSION"] or "v23.0"
    url = f"{GRAPH}/{ver}/{env['META_AD_ACCOUNT_ID']}/insights"
    params = {"level": "campaign", "fields": "campaign_name,spend,impressions,clicks", "time_increment": 1,
              "time_range": json.dumps({"since": since, "until": until}), "limit": 500,
              "access_token": env["META_ACCESS_TOKEN"]}
    rows = []
    while url:
        r = requests.get(url, params=params, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"Marketing API error {r.status_code}: {r.text[:500]}")
        j = r.json()
        rows += j.get("data", [])
        url, params = j.get("paging", {}).get("next"), None
    df = pd.DataFrame(rows)
    if df.empty:
        print("[meta] no insights returned for that range")
        return df
    m = cfg["columns"]["ad_spend"]
    out = pd.DataFrame({m["date"]: df["date_start"], m["campaign"]: df["campaign_name"],
                        m["spend"]: pd.to_numeric(df["spend"])})
    for extra in ["impressions", "clicks"]:
        if extra in df:
            out[extra] = pd.to_numeric(df[extra])
    out.to_csv(cfg["data"]["ad_spend_csv"], index=False)
    print(f"[meta] pulled {len(out):,} campaign-days of spend -> {cfg['data']['ad_spend_csv']}")
    return out


def _sha(x) -> str | None:
    if x is None or (isinstance(x, float) and pd.isna(x)) or str(x).strip() == "":
        return None
    return hashlib.sha256(str(x).strip().lower().encode()).hexdigest()


def _clean(x) -> str | None:
    if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x) or str(x).strip() in ("", "(missing)"):
        return None
    return str(x).strip()


def _norm_phone(x) -> str | None:
    if x is None or pd.isna(x):
        return None
    d = "".join(ch for ch in str(x) if ch.isdigit())
    return d or None


def build_events(scored: pd.DataFrame, cfg: dict, as_of: pd.Timestamp) -> list[dict]:
    mc = cfg["meta"]
    recent = scored[scored["created_at"] >= as_of - pd.Timedelta(days=mc["send_only_new_leads_days"])]
    events = []
    for r in recent.itertuples(index=False):
        # factor-data exports sha256 hashes of the normalised email/phone (track.people.email_hash /
        # phone_hash), never the raw values. Raw email/phone are still hashed here if a CSV has them.
        # TODO verify: factor-data's normalisation matches Meta's (email trimmed + lowercased; phone digits
        # only, with country code) - otherwise CAPI match rates will be poor.
        em = _clean(getattr(r, "email_hash", None)) or _sha(getattr(r, "email", None))
        ph = _clean(getattr(r, "phone_hash", None)) or _sha(_norm_phone(getattr(r, "phone", None)))
        ud = {k: v for k, v in {"em": [em], "ph": [ph]}.items() if v[0]}
        fbp = _clean(getattr(r, "fbp", None))
        if fbp:
            ud["fbp"] = fbp                                     # browser id: sent unhashed, per Meta's spec
        if not ud:
            continue  # Meta can't match an event with no identifiers
        events.append({
            "event_name": mc["event_name"],
            "event_time": int(pd.Timestamp(r.created_at).timestamp()),
            "event_id": f"flv_{r.lead_id}",                    # dedup key - re-sending the same lead is safe
            "action_source": "system_generated",
            "user_data": ud,
            "custom_data": {"value": round(float(getattr(r, mc["value_field"], r.expected_value)), 2),
                            "currency": mc["currency"],
                            "lead_event_source": "factor_lead_value_engine",
                            "predicted_close_prob": round(float(r.p_close_at_submit), 4)},
        })
    return events


def push_values(scored: pd.DataFrame, cfg: dict, as_of: pd.Timestamp, out_dir: str = "outputs") -> dict:
    events = build_events(scored, cfg, as_of)
    Path(out_dir).mkdir(exist_ok=True)
    Path(out_dir, "capi_payloads.json").write_text(json.dumps(events[:2000], indent=1))
    if cfg["meta"]["dry_run"]:
        print(f"[meta] DRY RUN: built {len(events):,} CAPI events -> {out_dir}/capi_payloads.json (nothing sent)")
        return {"sent": 0, "built": len(events), "dry_run": True}
    env = _env()
    _require(env, ["META_ACCESS_TOKEN", "META_PIXEL_ID"])
    ver = env["META_API_VERSION"] or "v23.0"
    url = f"{GRAPH}/{ver}/{env['META_PIXEL_ID']}/events"
    sent, errors = 0, []
    for i in range(0, len(events), 1000):                     # CAPI accepts up to 1000 events per call
        body = {"data": json.dumps(events[i:i + 1000]), "access_token": env["META_ACCESS_TOKEN"]}
        if env["META_TEST_EVENT_CODE"]:
            body["test_event_code"] = env["META_TEST_EVENT_CODE"]
        r = requests.post(url, data=body, timeout=60)
        if r.status_code == 200:
            sent += r.json().get("events_received", 0)
        else:
            errors.append(r.text[:300])
        time.sleep(0.2)
    print(f"[meta] sent {sent:,}/{len(events):,} events" + (f" | {len(errors)} batch errors: {errors[0]}" if errors else ""))
    return {"sent": sent, "built": len(events), "errors": errors, "dry_run": False}

