-- leads.sql -> data/leads.csv : one row per person x project, with ONLY what is known at opt-in.
--
-- Run through `python run.py --extract` (flv/extract.py), which binds:
--   %(projects)s  text[] of core.projects.slug to include, or NULL for all
--   %(exclude)s   text[] of slugs to exclude (lead-gen projects by default)
--
-- Rules from the factor-data guide:
--   * person = coalesce(track.people.merged_into, track.people.id) - always resolve merges first
--   * opt-in = the person's first crm.contacts.date_created in the project (CRM creation time, not synced_at)
--   * application answers come from LEAD forms only (track.forms.kind = 'lead'); sale / customer forms are
--     post-sale records and would leak. Only responses submitted by opt-in (+1h grace for the Typeform ->
--     CRM sync) are used.
--   * PII: never select raw email / phone / names / free-text answers. Only person_id and the sha256 hashes.
--   * Survivorship: 36 percent of Typeform responses have no person_id (mostly applicants the form disqualified
--     before the contact questions). They cannot be joined here, so has_application = 0 does not mean
--     "did not apply" - see README.

with proj as (
    select id, slug
    from core.projects
    where (%(projects)s::text[] is null or slug = any(%(projects)s::text[]))
      and not (slug = any(coalesce(%(exclude)s::text[], '{}'::text[])))
),
contact as (
    select c.project_id,
           coalesce(pe.merged_into, pe.id) as person_id,
           c.date_created,
           c.attribution
    from crm.contacts c
    join proj p on p.id = c.project_id
    join track.people pe on pe.id = c.person_id
    where c.deleted_at is null
),
first_contact as (                          -- opt-in = first CRM contact of the person in the project
    select distinct on (project_id, person_id)
           project_id, person_id, date_created as opt_in_at, attribution
    from contact
    order by project_id, person_id, date_created
),
resp as (                                   -- lead-form (application) responses, person resolved
    select f.project_id,
           coalesce(pe.merged_into, pe.id) as person_id,
           r.submitted_at,
           r.score,
           r.answers
    from track.form_responses r
    join track.forms f on f.form_id = r.form_id
    join proj p on p.id = f.project_id
    join track.people pe on pe.id = r.person_id
    where f.kind = 'lead'
),
app as (                                    -- latest application known at opt-in
    select distinct on (fc.project_id, fc.person_id)
           fc.project_id, fc.person_id, r.score, r.answers
    from first_contact fc
    join resp r on r.project_id = fc.project_id and r.person_id = fc.person_id
    where r.submitted_at <= fc.opt_in_at + interval '1 hour'   -- TODO verify: grace for Typeform -> CRM lag
    order by fc.project_id, fc.person_id, r.submitted_at desc
)
select
    concat(p.slug, ':', fc.person_id)                                     as lead_id,
    fc.person_id,
    p.slug                                                                as project,
    to_char(fc.opt_in_at at time zone 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"+00:00"') as opt_in_at,
    s.email_hash,                                                         -- sha256 of normalised email (survivor)
    s.phone_hash,
    -- UTMs / click ids as the CRM recorded them on the first contact. Keys confirmed in the guide:
    -- utm_source/medium/campaign/content/term, fbp, fbclid (also nested first/last touch objects).
    -- Meta ad history (ads.*) only starts 2026-07-01, so ad-level fields are mostly null before then.
    fc.attribution ->> 'utm_source'                                       as utm_source,
    fc.attribution ->> 'utm_medium'                                       as utm_medium,
    fc.attribution ->> 'utm_campaign'                                     as utm_campaign,
    fc.attribution ->> 'utm_term'                                         as utm_term,     -- TODO verify: ad set name lives in utm_term
    fc.attribution ->> 'utm_content'                                      as utm_content,  -- TODO verify: ad name lives in utm_content
    fc.attribution ->> 'fbclid'                                           as fbclid,
    fc.attribution ->> 'fbp'                                              as fbp,
    (a.person_id is not null)::int                                        as has_application,
    a.score::float8                                                       as typeform_score, -- computed at submit: safe
    -- Answers are keyed by QUESTION TITLE and differ per form. List each project's lead-form titles with:
    --   select f.project_id, k, count(*) from track.form_responses r join track.forms f using (form_id),
    --          jsonb_object_keys(r.answers) k where f.kind = 'lead' group by 1, 2 order by 3 desc;
    -- then map the multiple-choice ones here. Never export free-text answers (may contain contact details).
    a.answers ->> 'TODO verify: investment question title'                 as q_investment,
    a.answers ->> 'TODO verify: timeline question title'                   as q_timeline,
    a.answers ->> 'TODO verify: current situation question title'          as q_situation
from first_contact fc
join proj p on p.id = fc.project_id
join track.people s on s.id = fc.person_id
left join app a on a.project_id = fc.project_id and a.person_id = fc.person_id
order by fc.opt_in_at;
