-- 02_incidents.sql
-- One row per ticket, using the complete log (hindsight). This is the "what really happened" table:
-- it is used for descriptive statistics and to score predictions, never as model input.
--
-- Resolution time (`resolved_time`) = the moment the ticket entered Resolved/Closed for the last time,
-- i.e. the first update after its last "open" state. A ticket that is resolved, reopened and resolved
-- again counts from its final resolution.
--   * 24,904 tickets: the final entry into state Resolved.
--   * 14 tickets never show a final Resolved row and go straight to Closed; their closing time is used.
-- The export's own resolved_at field is missing for 1,556 tickets, and where present it agrees with
-- this definition for 98.7% of them (it is later for the rest: it is overwritten when a ticket is
-- edited while already Resolved). See tests/test_data_facts.py.

create or replace table incidents as
with last_open as (
    select number, max(seq) as seq
    from events
    where not is_done
    group by number
),
resolution as (
    select e.number, min(e.updated_at) as resolved_time
    from events e
    left join last_open lo using (number)
    where e.seq > coalesce(lo.seq, 0)
    group by e.number
),
first_row as (
    select * from events where seq = 1
),
last_row as (
    select * from events qualify seq = max(seq) over (partition by number)
),
counts as (
    select
        number,
        count(*) as n_updates,
        count(*) filter (where state = 'Resolved' and (prev_state is null or prev_state <> 'Resolved'))
            as n_resolved_entries
    from (select *, lag(state) over (partition by number order by seq) as prev_state from events)
    group by number
)
select
    f.number,
    f.opened_at,
    f.updated_at                                                  as first_seen_at,
    r.resolved_time,
    date_diff('minute', f.opened_at, r.resolved_time) / 60.0      as duration_h,
    l.updated_at                                                  as last_update_at,
    l.state                                                       as final_state,
    c.n_updates,
    c.n_resolved_entries,
    l.reassignment_count                                          as final_reassignment_count,
    l.reopen_count                                                as final_reopen_count,
    l.made_sla                                                    as final_made_sla,
    f.priority                                                    as first_priority,
    l.priority                                                    as final_priority,
    f.contact_type,
    ff.resolved_at_field,
    ff.closed_at_field
from first_row f
join last_row l using (number)
join resolution r using (number)
join counts c using (number)
join final_fields ff using (number);
