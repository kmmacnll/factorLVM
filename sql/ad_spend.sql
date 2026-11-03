-- ad_spend.sql -> data/ad_spend.csv : one row per project x day (ads.v_project_daily).
--
-- Binds %(projects)s / %(exclude)s like leads.sql.
-- Meta history starts 2026-07-01 (the backfill window); Vector, ScaleLogix and Follow The Hill have no
-- linked ad account, so they have no rows. Money is integer minor units -> divide by 100.
-- Spend is in the ad account's currency (ads.accounts.currency): USD for every high-ticket project;
-- Insuracove is AUD but is excluded by default (convert with ads.fx_rates if you include it).

select
    to_char(v.day, 'YYYY-MM-DD')     as date,        -- TODO verify: v.day is already the project-timezone day
    p.slug                           as project,
    (v.spend_minor / 100.0)::float8  as spend,
    v.impressions,
    v.link_clicks,
    v.leads
from ads.v_project_daily v
join core.projects p on p.id = v.project_id
where (%(projects)s::text[] is null or p.slug = any(%(projects)s::text[]))
  and not (p.slug = any(coalesce(%(exclude)s::text[], '{}'::text[])))
order by v.day, p.slug;
