-- 01_events.sql
-- Load the raw event log (one row per ticket update) into typed tables.
--
-- Input : the CSV path in the DuckDB variable `csv_path`.
-- Output: events        - the update log, restricted to columns that are known at the time of the update
--         final_fields  - columns that the export back-filled with their FINAL value on every row
--
-- Why two tables: resolved_at, closed_at, closed_code and resolved_by carry the same (final) value on
-- every row of a ticket, including its very first row. They describe the future from the point of view
-- of any earlier row, so they are kept out of `events` entirely. Nothing that reads `events` can leak.

create or replace table raw_log as
select *
from read_csv(getvariable('csv_path'), header = true, all_varchar = true, nullstr = '?');

create or replace table events as
with typed as (
    select
        number,
        strptime(opened_at, '%d/%m/%Y %H:%M')      as opened_at,
        strptime(sys_updated_at, '%d/%m/%Y %H:%M') as updated_at,
        cast(sys_mod_count as integer)             as mod_count,
        incident_state                             as state,
        cast(reassignment_count as integer)        as reassignment_count,
        cast(reopen_count as integer)              as reopen_count,
        made_sla = 'true'                          as made_sla,
        impact,
        urgency,
        priority,
        u_priority_confirmation = 'true'           as priority_confirmed,
        contact_type,
        category,
        subcategory,
        u_symptom                                  as symptom,
        location,
        assignment_group,
        assigned_to,
        caller_id,
        opened_by,
        sys_updated_by                             as updated_by
    from raw_log
)
select
    *,
    -- A ticket is "done" from the support desk's point of view once it is Resolved or Closed.
    state in ('Resolved', 'Closed') as is_done,
    -- Position of the update within its ticket. Timestamps have minute resolution,
    -- so the system's own modification counter breaks ties.
    row_number() over (partition by number order by updated_at, mod_count) as seq
from typed
-- Five rows (five tickets, all opened on 2016-02-29) are stamped before their own ticket was opened.
where updated_at >= opened_at;

create or replace table final_fields as
select
    number,
    any_value(strptime(resolved_at, '%d/%m/%Y %H:%M')) as resolved_at_field,
    any_value(strptime(closed_at, '%d/%m/%Y %H:%M'))   as closed_at_field,
    any_value(closed_code)                             as closed_code,
    any_value(resolved_by)                             as resolved_by,
    count(distinct resolved_at)                        as n_distinct_resolved_at,
    count(distinct closed_at)                          as n_distinct_closed_at
from raw_log
group by number;
