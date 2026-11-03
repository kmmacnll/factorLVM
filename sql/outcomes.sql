-- outcomes.sql -> data/outcomes.csv : one row per person x project (same lead_id as leads.sql).
--
-- Binds %(projects)s / %(exclude)s like leads.sql.
--
-- Rules from the factor-data guide:
--   * furthest funnel stage comes from crm.stage_history rows with inferred = false only (exact event
--     times; inferred rows are ported / snapshot-diffed and only "roughly by then"), at or after opt-in
--   * won date and deal value come from crm.deals: status = 'won', value_corrected_minor / 100
--     (85 percent of deals have no value: value_source = 'none' -> deal_value left blank)
--   * Close won dates are DATES stored as midnight UTC (won_at_precision = 'date'): take the UTC date,
--     don't convert to New York (that moves them to the previous day). GHL won times are real timestamps:
--     convert to the project timezone. (The DB helper crm.deal_day(at, precision, tz) applies this rule.)
--   * do NOT use decide.v_lead_outcomes dates - they are not point-in-time
--   * stage output uses the canonical vocabulary; config.yaml funnel.stage_aliases folds
--     no_show -> booked, deposit -> proposal, onboarding -> won

with proj as (
    select id, slug, timezone
    from core.projects
    where (%(projects)s::text[] is null or slug = any(%(projects)s::text[]))
      and not (slug = any(coalesce(%(exclude)s::text[], '{}'::text[])))
),
optin as (                                  -- must match leads.sql
    select c.project_id,
           coalesce(pe.merged_into, pe.id) as person_id,
           min(c.date_created)             as opt_in_at
    from crm.contacts c
    join proj p on p.id = c.project_id
    join track.people pe on pe.id = c.person_id
    where c.deleted_at is null
    group by 1, 2
),
moves as (                                  -- time-exact stage moves, person resolved
    select h.project_id,
           coalesce(pc.merged_into, pc.id, pd.merged_into, pd.id) as person_id,
           h.to_funnel,
           h.at
    from crm.stage_history h
    join proj p on p.id = h.project_id
    left join crm.contacts c  on c.id = h.contact_id
    left join crm.deals d     on d.id = h.deal_id
    left join track.people pc on pc.id = c.person_id
    left join track.people pd on pd.id = d.person_id
    where not h.inferred
),
furthest as (
    select o.project_id, o.person_id,
           max(case m.to_funnel
                 when 'lead' then 0 when 'contacted' then 1 when 'booked' then 2 when 'no_show' then 2
                 when 'showed' then 3 when 'qualified' then 4 when 'proposal' then 5 when 'deposit' then 5
                 when 'won' then 6 when 'onboarding' then 6 end)     as stage_rank,
           max(m.at)                                                 as stage_at,
           bool_or(m.to_funnel in ('disqualified', 'lost'))          as moved_to_lost
    from optin o
    left join moves m on m.project_id = o.project_id and m.person_id = o.person_id
                     and m.at >= o.opt_in_at - interval '1 hour'
    group by 1, 2
),
deal as (                                   -- deals, person resolved, won date as a business-calendar date
    select d.project_id,
           coalesce(pe.merged_into, pe.id) as person_id,
           d.status,
           case when d.won_at_precision = 'date' then (d.won_at at time zone 'UTC')::date
                else (coalesce(d.won_at, d.status_changed_at) at time zone p.timezone)::date
           end                             as won_date,
           case when d.value_source <> 'none' and d.value_corrected_minor > 0
                then (d.value_corrected_minor / 100.0)::float8 end as deal_value,
           d.currency
    from crm.deals d
    join proj p on p.id = d.project_id
    join track.people pe on pe.id = d.person_id
    -- TODO verify: whether crm.deals has a deleted_at to filter ("76,773 live" suggests soft deletes)
),
first_win as (                              -- first won deal on/after the opt-in day
    select distinct on (o.project_id, o.person_id)
           o.project_id, o.person_id, d.won_date, d.deal_value, d.currency
    from optin o
    join proj p on p.id = o.project_id
    join deal d on d.project_id = o.project_id and d.person_id = o.person_id
    where d.status = 'won'
      and d.won_date >= (o.opt_in_at at time zone p.timezone)::date
    order by o.project_id, o.person_id, d.won_date
),
any_lost_deal as (
    select distinct project_id, person_id from deal where status = 'lost'
)
select
    concat(p.slug, ':', o.person_id)                                       as lead_id,
    case when w.person_id is not null then 'won'
         else (array['lead','contacted','booked','showed','qualified','proposal','won'])
              [coalesce(least(f.stage_rank, 5), 0) + 1]
    end                                                                    as stage,
    to_char(f.stage_at at time zone 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"+00:00"') as stage_at,
    -- TODO verify: refunds (to_funnel = 'refund') are left as won here; decide whether they count as lost
    (w.person_id is null and (coalesce(f.moved_to_lost, false) or l.person_id is not null))::int as is_lost,
    to_char(w.won_date, 'YYYY-MM-DD')                                      as won_date,
    w.deal_value,
    w.currency
from optin o
join proj p on p.id = o.project_id
left join furthest f      on f.project_id = o.project_id and f.person_id = o.person_id
left join first_win w     on w.project_id = o.project_id and w.person_id = o.person_id
left join any_lost_deal l on l.project_id = o.project_id and l.person_id = o.person_id
order by o.opt_in_at;
