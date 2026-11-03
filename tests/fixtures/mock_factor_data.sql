-- Minimal mock of the factor-data tables/columns that sql/*.sql reads, with a handful of FAKE rows
-- covering the guide's traps. Used by tests/test_extract_sql.py against a throwaway Postgres.
drop schema if exists core, track, crm, ads cascade;
create schema core; create schema track; create schema crm; create schema ads;

create table core.projects (id text primary key, slug text, lane text, timezone text);
create table track.people (id text primary key, merged_into text, email_hash text, phone_hash text,
                           email text, phone text);
create table crm.contacts (id text primary key, project_id text, person_id text, date_created timestamptz,
                           deleted_at timestamptz, attribution jsonb, email text, phone text);
create table track.forms (form_id text primary key, project_id text, kind text);
create table track.form_responses (response_token text, form_id text, person_id text, submitted_at timestamptz,
                                   score numeric, answers jsonb);
create table crm.deals (id text primary key, project_id text, person_id text, status text, won_at timestamptz,
                        won_at_precision text, status_changed_at timestamptz, value_corrected_minor bigint,
                        value_source text, currency text);
create table crm.stage_history (project_id text, contact_id text, deal_id text, to_funnel text,
                                at timestamptz, inferred boolean);
create table ads.v_project_daily (project_id text, day date, spend_minor bigint, impressions int,
                                  link_clicks int, leads int);

insert into core.projects values
  ('pr_nir', 'nirvana', 'high_ticket', 'America/New_York'),
  ('pr_sen', 'senior-care-leads', 'lead_gen', 'America/New_York');

insert into track.people values
  ('p1',  null, 'h_email_p1', 'h_phone_p1', null, null),
  ('p1b', 'p1', 'h_email_p1b', null, null, null),   -- merged into p1
  ('p2',  null, 'h_email_p2', null, null, null),
  ('p3',  null, 'h_email_p3', null, null, null),
  ('p4',  null, 'h_email_p4', null, null, null),
  ('p5',  null, 'h_email_p5', null, null, null);

insert into crm.contacts values
  -- p1: first contact is under the merged-away id p1b
  ('c1b', 'pr_nir', 'p1b', '2026-08-10 14:00+00', null,
   '{"utm_source":"facebook","utm_campaign":"NIR | Prospecting","utm_term":"AS1","utm_content":"AD1","fbp":"fb.1.1.2","fbclid":"x"}', null, null),
  ('c1',  'pr_nir', 'p1',  '2026-08-12 09:00+00', null, '{}', null, null),
  ('c2',  'pr_nir', 'p2',  '2026-09-01 15:00+00', null, '{}', null, null),
  ('c3',  'pr_nir', 'p3',  '2026-09-05 18:00+00', null, null, null, null),
  ('c4',  'pr_sen', 'p4',  '2026-09-02 12:00+00', null, null, null, null),          -- excluded project
  ('c5',  'pr_nir', 'p5',  '2026-09-03 12:00+00', '2026-09-04 00:00+00', null, null, null);  -- deleted

insert into track.forms values ('f_lead', 'pr_nir', 'lead'), ('f_sale', 'pr_nir', 'sale');
insert into track.form_responses values
  ('r1', 'f_lead', 'p1b', '2026-08-10 13:50+00', 72, '{"Q":"A"}'),       -- before opt-in: used
  ('r2', 'f_sale', 'p1',  '2026-08-21 10:00+00', 99, '{"Q":"B"}'),       -- sale form: never used
  ('r3', 'f_lead', 'p3',  '2026-09-07 10:00+00', 40, '{"Q":"C"}');       -- 2 days after opt-in: not used

insert into crm.deals values
  ('d1', 'pr_nir', 'p1', 'won', '2026-08-20 00:00+00', 'date', null, 750000, 'opportunity', 'USD'),   -- Close date
  ('d2', 'pr_nir', 'p2', 'lost', null, null, null, null, 'none', 'USD'),
  ('d3', 'pr_nir', 'p3', 'won', '2026-09-20 02:30+00', 'timestamp', null, null, 'none', 'USD');     -- GHL time, no value

insert into crm.stage_history values
  ('pr_nir', 'c1b', null, 'booked',    '2026-08-11 10:00+00', false),
  ('pr_nir', 'c1',  null, 'showed',    '2026-08-13 10:00+00', false),
  ('pr_nir', null,  'd1', 'proposal',  '2026-08-14 10:00+00', true),    -- inferred: ignored
  ('pr_nir', 'c2',  null, 'contacted', '2026-09-02 10:00+00', false),
  ('pr_nir', 'c2',  null, 'lost',      '2026-09-10 10:00+00', false),
  ('pr_nir', 'c2',  null, 'qualified', '2026-09-05 10:00+00', true);    -- inferred: ignored

insert into ads.v_project_daily values
  ('pr_nir', '2026-08-10', 123456, 1000, 20, 3),
  ('pr_sen', '2026-08-10', 5000, 100, 2, 1);
