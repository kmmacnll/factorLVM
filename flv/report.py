"""Build a self-contained interactive HTML dashboard (Plotly) from the fitted models."""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# palette (validated categorical order; text never wears series color)
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e9e8e4", "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
FONT = "Inter, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"


def _base(fig: go.Figure, title: str, sub: str = "", h: int = 380, **kw) -> go.Figure:
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>" + (f"<br><span style='font-size:12px;color:{INK2}'>{sub}</span>" if sub else ""),
                   x=0.01, xanchor="left", font=dict(size=15, color=INK)),
        font=dict(family=FONT, size=12, color=INK2), paper_bgcolor=SURF, plot_bgcolor=SURF, height=h,
        margin=dict(l=60, r=24, t=78 if sub else 56, b=48), hoverlabel=dict(bgcolor="white", font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, ticks="", automargin=True)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, ticks="", automargin=True)
    return fig


NO_ATTRIBUTION = "(no ad attribution)"


def _label(s: pd.Series) -> pd.Series:
    """Campaign / ad set "(missing)" = opt-ins before Meta history (2026-07-01), organic or unlinked accounts."""
    return s.astype(str).replace("(missing)", NO_ATTRIBUTION)


def _money(x):
    return f"${x:,.0f}"


def fig_campaign_mc(summ: pd.DataFrame, window: int) -> go.Figure:
    d = summ.sort_values("rev_mean")
    fig = go.Figure(go.Bar(
        x=d.rev_mean, y=_label(d.iloc[:, 0]), orientation="h", marker=dict(color=S1, cornerradius=4),
        error_x=dict(type="data", symmetric=False, array=d.rev_p90 - d.rev_mean, arrayminus=d.rev_mean - d.rev_p10,
                     color=INK2, thickness=1.5, width=4),
        customdata=np.c_[d.rev_p10, d.rev_p90, d.leads, d.value_per_lead],
        hovertemplate="<b>%{y}</b><br>Expected value %{x:$,.0f}<br>80% range %{customdata[0]:$,.0f} – "
                      "%{customdata[1]:$,.0f}<br>%{customdata[2]} leads · %{customdata[3]:$,.0f}/lead<extra></extra>"))
    return _base(fig, "Expected value by campaign (Monte Carlo)",
                 f"Leads from the last {window} days · bar = mean, whisker = P10–P90 of simulations",
                 h=max(340, 46 * len(d) + 110), xaxis_tickprefix="$", bargap=0.35)


def fig_value_vs_cpl(summ: pd.DataFrame, target: float) -> go.Figure | None:
    if summ is None or "cpl" not in summ or not (summ.spend > 0).any():
        return None
    d = summ[summ.spend > 0]
    name = d.columns[0]
    mx = max(d.cpl.max(), d.value_per_lead.max() / target) * 1.15
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx * target], mode="lines", line=dict(color=MUTED, dash="dot", width=1.5),
                             name=f"Break-even at {target:g}x ROAS", hoverinfo="skip"))
    good = d.roas_mean >= target
    for mask, col, nm in [(good, S3, f"Beats {target:g}x"), (~good, S2, f"Below {target:g}x")]:
        x = d[mask].sort_values("value_per_lead")
        pos = ["bottom center" if i % 2 else "top center" for i in range(len(x))]
        fig.add_trace(go.Scatter(
            x=x.cpl, y=x.value_per_lead, mode="markers+text", name=nm, text=x[name].astype(str), textposition=pos,
            textfont=dict(size=10, color=INK2),
            marker=dict(size=np.sqrt(x.leads) * 1.3 + 8, color=col, line=dict(color=SURF, width=2), opacity=0.9),
            customdata=np.c_[x.roas_mean, x.p_beats_target_roas, x.leads],
            hovertemplate="<b>%{text}</b><br>CPL %{x:$,.0f}<br>Predicted value/lead %{y:$,.0f}<br>"
                          "Predicted ROAS %{customdata[0]:.1f}x<br>P(beats target) %{customdata[1]:.0%}<br>"
                          "%{customdata[2]} leads<extra></extra>"))
    return _base(fig, "What a lead is worth vs. what it costs",
                 f"Per {'project' if name == 'client_id' else name} · above the dotted line = clears target ROAS · bubble size = lead volume",
                 h=460, xaxis_title="Cost per lead", yaxis_title="Predicted value per lead",
                 xaxis_tickprefix="$", yaxis_tickprefix="$", xaxis_range=[0, mx])


def fig_calibration(cm) -> go.Figure:
    mp, fp = cm.diagnostics["calib"]
    m = max(mp.max(), fp.max()) * 1.1
    fig = go.Figure([
        go.Scatter(x=[0, m], y=[0, m], mode="lines", line=dict(color=MUTED, dash="dot", width=1.5), name="Perfect",
                   hoverinfo="skip"),
        go.Scatter(x=mp, y=fp, mode="lines+markers", line=dict(color=S1, width=2), marker=dict(size=9, line=dict(color=SURF, width=2)),
                   name="Model", hovertemplate="Predicted %{x:.1%}<br>Actual %{y:.1%}<extra></extra>")])
    return _base(fig, "Calibration — does 20% mean 20%?", "Held-out test period, quantile bins",
                 xaxis_title="Predicted P(close)", yaxis_title="Actual close rate",
                 xaxis_tickformat=".0%", yaxis_tickformat=".0%")


def fig_lift(cm) -> go.Figure:
    l = cm.diagnostics["lift"]
    avg = cm.metrics["base_rate_test"]
    fig = go.Figure(go.Bar(x=l.decile, y=l.actual, marker=dict(color=[SEQ[min(5, int(i // 2))] for i in range(10)], cornerradius=4),
                           customdata=np.c_[l.predicted, l.actual / max(avg, 1e-9), l.n],
                           hovertemplate="Decile %{x}<br>Actual %{y:.1%} · predicted %{customdata[0]:.1%}<br>"
                                         "%{customdata[1]:.1f}x average · %{customdata[2]} leads<extra></extra>"))
    fig.add_hline(y=avg, line=dict(color=INK2, dash="dot", width=1.5),
                  annotation_text=f"average {avg:.1%}", annotation_font_color=INK2, annotation_position="top left")
    top = l.actual.iloc[-1] / max(avg, 1e-9)
    return _base(fig, "Lift by predicted-score decile", f"Top 10% of leads close at {top:.1f}× the average rate",
                 xaxis_title="Decile (1 = lowest score)", yaxis_title="Actual close rate", yaxis_tickformat=".0%",
                 xaxis_dtick=1, bargap=0.25)


def fig_roc(cm) -> go.Figure:
    fpr, tpr = cm.diagnostics["roc"]
    fig = go.Figure([
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, dash="dot", width=1.5), name="Random", hoverinfo="skip"),
        go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=S1, width=2), fill="tozeroy", fillcolor="rgba(42,120,214,0.08)",
                   name=f"AUC {cm.metrics['auc']:.3f}", hovertemplate="FPR %{x:.2f}<br>TPR %{y:.2f}<extra></extra>")])
    return _base(fig, "ROC curve", "Ranking quality on the held-out period",
                 xaxis_title="False positive rate", yaxis_title="True positive rate")


def fig_importance(cm) -> go.Figure:
    imp = cm.diagnostics["importance"].head(12).sort_values()
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker=dict(color=S1, cornerradius=4),
                           hovertemplate="%{y}<br>mean |SHAP| %{x:.3f}<extra></extra>"))
    return _base(fig, "What drives P(close)", "Mean absolute SHAP value (log-odds) on test leads",
                 h=max(320, 30 * len(imp) + 110), bargap=0.35)


def fig_value_effects(vm) -> go.Figure:
    s = vm.summary.copy().sort_values(["kind", "mean"])
    s["label"] = s.term.str.replace("_", " ", regex=False)
    cols = np.where(s.kind == "client", S2, S1)
    fig = go.Figure(go.Scatter(
        x=s["mean"], y=s.label, mode="markers", marker=dict(size=10, color=cols, line=dict(color=SURF, width=2)),
        error_x=dict(type="data", symmetric=False, array=s.hi - s["mean"], arrayminus=s["mean"] - s.lo, color=INK2, thickness=1.5, width=0),
        customdata=np.c_[s.lo, s.hi],
        hovertemplate="<b>%{y}</b><br>%{x:+.0%} deal size<br>94% CrI %{customdata[0]:+.0%} to %{customdata[1]:+.0%}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    return _base(fig, "Bayesian deal-size model — effect of each attribute",
                 "% change in deal value vs. baseline · 94% credible intervals · orange = project (partially pooled)",
                 h=max(360, 26 * len(s) + 120), xaxis_tickformat="+.0%", showlegend=False)


def fig_ppc(vm) -> go.Figure:
    obs, rep = vm.ppc["observed"], vm.ppc["replicated"]
    bins = np.linspace(min(obs.min(), rep.min()), max(obs.max(), rep.max()), 40)
    ctr = np.exp((bins[:-1] + bins[1:]) / 2)
    fig = go.Figure()
    for i, r in enumerate(rep):
        h, _ = np.histogram(r, bins, density=True)
        fig.add_trace(go.Scatter(x=ctr, y=h, mode="lines", line=dict(color="rgba(42,120,214,0.18)", width=1),
                                 name="Simulated from posterior", showlegend=i == 0, legendgroup="r", hoverinfo="skip"))
    h, _ = np.histogram(obs, bins, density=True)
    fig.add_trace(go.Scatter(x=ctr, y=h, mode="lines", line=dict(color=INK, width=2), name="Actual deals",
                             hovertemplate="~%{x:$,.0f}<extra></extra>"))
    return _base(fig, "Posterior predictive check", "If the model is right, the black line sits inside the blue band",
                 xaxis_type="log", xaxis_title="Deal value (log scale)", yaxis_title="Density", xaxis_tickprefix="$")


def fig_funnel(fm, ds) -> go.Figure:
    c = fm.counts
    names = list(c["from"]) + [c["to"].iloc[-1]]
    n = list(c["n"]) + [int(c["converted"].iloc[-1])]
    fig = go.Figure(go.Funnel(y=names, x=n, textinfo="value+percent initial",
                              marker=dict(color=[SEQ[min(5, i + 1)] for i in range(len(n))]),
                              connector=dict(fillcolor="#f0efec", line=dict(color=GRID, width=0)),
                              hovertemplate="%{y}: %{x:,} leads<extra></extra>", textfont=dict(color=INK)))
    return _base(fig, "Funnel (mature leads)", f"Leads ≥ {fm.horizon:g} days old, by furthest stage reached", h=360)


def fig_stage_rates(fm) -> go.Figure:
    c = fm.counts
    lab = c["from"] + " → " + c["to"]
    fig = go.Figure(go.Scatter(
        x=c.rate_mean, y=lab, mode="markers", marker=dict(size=11, color=S1, line=dict(color=SURF, width=2)),
        error_x=dict(type="data", symmetric=False, array=c.rate_hi - c.rate_mean, arrayminus=c.rate_mean - c.rate_lo,
                     color=INK2, thickness=2, width=6),
        customdata=np.c_[c.converted, c.n],
        hovertemplate="<b>%{y}</b><br>%{x:.1%}<br>%{customdata[0]:,} of %{customdata[1]:,}<extra></extra>"))
    return _base(fig, "Stage conversion rates (Beta-Binomial posterior)", "Dot = posterior mean · bar = 94% credible interval",
                 h=320, xaxis_tickformat=".0%", yaxis_autorange="reversed")


def fig_survival(fm, ds) -> go.Figure:
    sf = fm.km.survival_function_
    ci = fm.km.confidence_interval_
    t = sf.index.to_numpy()
    keep = t <= max(fm.horizon * 2, 30)
    t, F = t[keep], 1 - sf.iloc[:, 0].to_numpy()[keep]
    lo, hi = 1 - ci.iloc[:, 1].to_numpy()[keep], 1 - ci.iloc[:, 0].to_numpy()[keep]
    fig = go.Figure([
        go.Scatter(x=np.r_[t, t[::-1]], y=np.r_[hi, lo[::-1]], fill="toself", fillcolor="rgba(42,120,214,0.12)",
                   line=dict(width=0), hoverinfo="skip", showlegend=False),
        go.Scatter(x=t, y=F, mode="lines", line=dict(color=S1, width=2, shape="hv"), name="Cumulative close rate",
                   hovertemplate="Day %{x:.0f}: %{y:.1%} closed<extra></extra>")])
    fig.add_vline(x=fm.horizon, line=dict(color=INK2, dash="dot", width=1.5),
                  annotation_text=f"horizon {fm.horizon:g}d", annotation_font_color=INK2)
    return _base(fig, "Time to close (Kaplan–Meier, censoring-aware)",
                 "Young leads aren't counted as losses — they're censored at their current age",
                 xaxis_title="Days since lead created", yaxis_title="Share of leads closed", yaxis_tickformat=".1%", showlegend=False)


def fig_mc_total(total: np.ndarray, window: int) -> go.Figure:
    p10, p50, p90 = np.percentile(total, [10, 50, 90])
    fig = go.Figure(go.Histogram(x=total, nbinsx=60, marker=dict(color=S1, line=dict(color=SURF, width=1)),
                                 hovertemplate="%{x:$,.0f}<br>%{y} sims<extra></extra>"))
    for v, nm, pos in [(p10, "P10", "top left"), (p50, "P50", "top"), (p90, "P90", "top right")]:
        fig.add_vline(x=v, line=dict(color=INK, width=1.5 if nm == "P50" else 1, dash="solid" if nm == "P50" else "dot"),
                      annotation_text=nm, annotation_font_color=INK2, annotation_position=pos)
    return _base(fig, "Simulated total value from recent leads",
                 f"{len(total):,} Monte Carlo runs over every lead from the last {window} days · "
                 f"P10 {_money(p10)} · P50 {_money(p50)} · P90 {_money(p90)}",
                 xaxis_title="Total value (revenue × margin)", yaxis_title="Simulations", xaxis_tickprefix="$", bargap=0.02, showlegend=False)


def fig_backtest(bt: pd.DataFrame) -> go.Figure:
    bt = bt.sort_values("predicted")
    m = max(bt.predicted.max(), bt.actual.max()) * 1.1
    fig = go.Figure([
        go.Scatter(x=[0, m], y=[0, m], mode="lines", line=dict(color=MUTED, dash="dot", width=1.5), name="Perfect", hoverinfo="skip"),
        go.Scatter(x=bt.predicted, y=bt.actual, mode="markers+text", text=_label(bt.iloc[:, 0]), cliponaxis=False,
                   textposition=["bottom center" if i % 2 else "top center" for i in range(len(bt))],
                   textfont=dict(size=10, color=INK2), name="Campaign",
                   marker=dict(size=12, color=S1, line=dict(color=SURF, width=2)),
                   customdata=np.c_[bt.leads, bt.pred_closes, bt.actual_closes],
                   hovertemplate="<b>%{text}</b><br>Predicted %{x:$,.0f}<br>Actual %{y:$,.0f}<br>%{customdata[0]} leads · "
                                 "%{customdata[1]:.1f} predicted vs %{customdata[2]} actual closes<extra></extra>")])
    err = (bt.predicted.sum() - bt.actual.sum()) / max(bt.actual.sum(), 1)
    fig = _base(fig, "Backtest: predicted vs. actual value", f"Held-out period, per campaign · total error {err:+.1%}",
                 h=400, xaxis_title="Predicted value", yaxis_title="Actual value", xaxis_tickprefix="$",
                 yaxis_tickprefix="$", showlegend=False)
    fig.update_layout(margin=dict(r=110))                     # room for labels of the right-most campaigns
    return fig


def fig_value_dist(scored: pd.DataFrame) -> go.Figure:
    v = scored.expected_value.clip(lower=max(scored.expected_value.quantile(0.01), 1))
    bins = np.logspace(np.log10(v.min()), np.log10(v.max()), 50)
    tiers = ["E (lowest)", "D", "C", "B", "A (highest)"]
    fig = go.Figure()
    for i, t in enumerate(tiers):
        x = v[scored.value_tier == t]
        h, e = np.histogram(x, bins)
        fig.add_trace(go.Bar(x=np.sqrt(e[:-1] * e[1:]), y=h, width=np.diff(e) * 0.92, name=t, marker_color=SEQ[i + 1],
                             hovertemplate=f"Tier {t}<br>~%{{x:$,.0f}}: %{{y}} leads<extra></extra>"))
    fig = _base(fig, "Predicted value per lead (the number sent to Meta)", "Expected value = P(close) × E[deal | won] × margin · coloured by quintile tier",
                barmode="stack", xaxis_type="log", xaxis_title="Expected value per lead (log scale)", yaxis_title="Leads",
                xaxis_tickprefix="$")
    fig.update_layout(legend=dict(orientation="h", y=-0.28, yanchor="top", x=0, xanchor="left", traceorder="reversed"),
                      margin=dict(b=90), height=420)
    return fig


def _table(df: pd.DataFrame) -> str:
    cols = ["lead_id", "created_at", "campaign", "stage", "p_close_at_submit", "p_close_now", "value_if_won", "expected_value_now"]
    cols = [c for c in cols if c in df.columns]
    fmt = {"p_close_at_submit": "{:.1%}", "p_close_now": "{:.1%}", "value_if_won": "${:,.0f}", "expected_value_now": "${:,.0f}"}
    head = "".join(f"<th>{html.escape(c.replace('_', ' '))}</th>" for c in cols)
    rows = []
    df = df.assign(campaign=_label(df["campaign"])) if "campaign" in df.columns else df
    for _, r in df[cols].iterrows():
        cells = []
        for c in cols:
            v = r[c]
            s = fmt[c].format(v) if c in fmt else (v.strftime("%Y-%m-%d") if hasattr(v, "strftime") else str(v))
            cells.append(f"<td class='{'num' if c in fmt else ''}'>{html.escape(s)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def build(path, ds, cm, vm, fm, scored, summ, total, bt, cfg, proj_summ=None) -> str:
    window = cfg["model"]["horizon_days"]
    target = cfg["simulation"]["target_roas"]
    p10, p90 = np.percentile(total, [10, 90])
    won_vals = ds.df.loc[ds.df.won == 1, "deal_value"].dropna()
    kpis = [
        ("Leads analysed", f"{len(ds.df):,}", f"{ds.df.won.sum():,} won"),
        ("Close rate (mature)", f"{ds.df.loc[ds.df.mature, 'label'].mean():.1%}", f"within {window} days"),
        ("Model AUC", f"{cm.metrics['auc']:.3f}", f"Brier skill {cm.metrics['brier_skill']:+.1%}"),
        ("Median deal", _money(won_vals.median()) if len(won_vals) else "—", f"{len(won_vals):,} won deals"),
        (f"Value of last {window}d leads", _money(total.mean()), f"80% range {_money(p10)} – {_money(p90)}"),
    ]
    roas_src = proj_summ if proj_summ is not None and "spend" in proj_summ else summ
    if "spend" in roas_src:
        sp = roas_src.spend.sum()
        kpis.append(("Predicted ROAS", f"{total.mean() / sp:.2f}×" if sp else "—", f"on {_money(sp)} spend · target {target:g}×"))

    figs = [
        ("Where the money is", [fig_campaign_mc(summ, window), fig_value_vs_cpl(summ, target)
                                or fig_value_vs_cpl(proj_summ, target)]),
        ("Lead scoring model (XGBoost, calibrated)", [fig_calibration(cm), fig_lift(cm), fig_roc(cm), fig_importance(cm)]),
        ("Deal value model (hierarchical Bayesian)", [fig_value_effects(vm), fig_ppc(vm)]),
        ("Funnel & timing", [fig_funnel(fm, ds), fig_stage_rates(fm), fig_survival(fm, ds)]),
        ("Simulation & validation", [fig_mc_total(total, window), fig_backtest(bt), fig_value_dist(scored)]),
    ]
    first = True
    sections = []
    for title, fl in figs:
        cards = []
        fl = [f for f in fl if f is not None]
        for i, f in enumerate(fl):
            narrow = [j for j, g in enumerate(fl) if not (g.layout.height and g.layout.height > 420)]
            wide = "wide" if (f.layout.height and f.layout.height > 420) or (len(narrow) % 2 and i == narrow[-1]) else ""
            cards.append(f"<div class='card {wide}'>" + f.to_html(full_html=False, include_plotlyjs=True if first else False,
                                                               config={"displaylogo": False, "responsive": True}, default_width="100%") + "</div>")
            first = False
        sections.append(f"<h2>{html.escape(title)}</h2><div class='grid'>{''.join(cards)}</div>")

    top = scored[(scored.won == 0) & (scored.is_lost == 0) & (scored.age_days < 120)].nlargest(25, "expected_value_now")
    m = cm.metrics
    kpi_html = "".join(f"<div class='kpi'><div class='k'>{a}</div><div class='v'>{b}</div><div class='s'>{c}</div></div>" for a, b, c in kpis)
    page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Factor · Lead Value Engine</title>
<style>
:root{{--ink:{INK};--ink2:{INK2};--muted:{MUTED};--grid:{GRID};--surf:{SURF};--bg:#f4f3ef;--acc:{S1}}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);font-family:{FONT};color:var(--ink)}}
header{{background:#0f1720;color:#f0efec;padding:28px 36px 24px}}
header h1{{margin:0;font-size:24px;letter-spacing:-.01em}} header p{{margin:6px 0 0;color:#c3c2b7;font-size:13px}}
.badge{{display:inline-block;background:#1c5cab;color:#fff;border-radius:999px;padding:2px 10px;font-size:11px;margin-left:8px;vertical-align:middle}}
main{{padding:24px 36px 48px;max-width:1500px;margin:0 auto}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin-bottom:8px}}
.kpi{{background:var(--surf);border:1px solid var(--grid);border-radius:12px;padding:14px 16px}}
.kpi .k{{font-size:12px;color:var(--ink2)}} .kpi .v{{font-size:26px;font-weight:650;margin:4px 0 2px;letter-spacing:-.02em}}
.kpi .s{{font-size:12px;color:var(--muted)}}
h2{{font-size:16px;margin:30px 0 12px;color:var(--ink)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(560px,1fr));gap:14px}}
.card{{background:var(--surf);border:1px solid var(--grid);border-radius:12px;padding:6px;overflow:hidden}}
.card.wide{{grid-column:1/-1}}
table{{width:100%;border-collapse:collapse;font-size:12.5px;background:var(--surf)}}
th{{text-align:left;color:var(--ink2);font-weight:600;border-bottom:1px solid var(--grid);padding:8px 10px;text-transform:capitalize}}
td{{padding:7px 10px;border-bottom:1px solid #f0efec}} td.num{{text-align:right;font-variant-numeric:tabular-nums}}
.tablewrap{{background:var(--surf);border:1px solid var(--grid);border-radius:12px;overflow:auto;max-height:560px}}
.note{{font-size:12px;color:var(--ink2);line-height:1.55;background:var(--surf);border:1px solid var(--grid);border-radius:12px;padding:14px 18px}}
code{{background:#f0efec;padding:1px 5px;border-radius:4px}}
</style></head><body>
<header><h1>Factor · Lead Value Engine <span class="badge">as of {ds.as_of.date()}</span></h1>
<p>Predicts what each Meta lead is worth: P(close) from calibrated XGBoost × E[deal value] from a hierarchical Bayesian model,
updated by funnel stage and lead age, then Monte Carlo-simulated by campaign. Generated {datetime.now():%Y-%m-%d %H:%M}.</p></header>
<main>
<div class="kpis">{kpi_html}</div>
{''.join(sections)}
<h2>Hottest open leads right now</h2>
<div class="tablewrap">{_table(top)}</div>
<h2>Model notes</h2>
<div class="note">
Close model trained on {m['n_train']:,} leads, calibrated on {m['n_calib']:,}, tested on {m['n_test']:,} newer leads ({m['test_period']}).
Test base rate {m['base_rate_test']:.1%} · AUC {m['auc']:.3f} · Brier {m['brier']:.4f} vs {m['brier_baseline']:.4f} baseline.
<br>Only leads ≥ {window} days old are used to train (younger ones haven't had time to close). <code>p_close_at_submit</code> uses only
information known when the lead arrives — that's the one sent to Meta. <code>p_close_now</code> also uses lead age and CRM stage — use it for sales prioritisation, never for training.
<br>Deal values are only observed for won deals, so the value model describes deal size <i>given</i> a close. Monte Carlo treats P(close) as known
and propagates deal-size uncertainty through posterior draws.
<br>Real proof of lift is an A/B test: one campaign on value-based bidding with these values vs. a matched control.
</div>
</main>
<script>window.addEventListener('load',()=>document.querySelectorAll('.js-plotly-plot').forEach(p=>Plotly.Plots.resize(p)));</script>
</body></html>"""
    Path(path).write_text(page, encoding="utf-8")
    print(f"[report] dashboard -> {path}")
    return str(path)

