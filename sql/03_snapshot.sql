-- 03_snapshot.sql
-- "What did the ticket system show at time t?"
--
-- Every ticket in the file is closed by the end, so open tickets are recreated by stopping the clock
-- at a cutoff time t and hiding every update after it. These macros read only rows with
-- updated_at <= t, so nothing from after the cutoff can reach their output
-- (tests/test_no_leakage.py rewrites the future rows and checks the output does not change).
--
--   select * from snapshot_at(timestamp '2016-04-04');   -- every ticket visible at t
--   select * from open_at(timestamp '2016-04-04');       -- only those still open at t
--
-- For a ticket that looks done at t, duration_h is its time to resolution as known at t.
-- For a ticket still open at t, duration_h is its age at t (a censored observation).

create or replace macro snapshot_at(t) as table
with seen as (
    select * from events where updated_at <= t
),
cur as (
    select * from seen qualify seq = max(seq) over (partition by number)
),
last_open as (
    select number, max(seq) as seq
    from seen
    where not is_done
    group by number
),
done_since as (
    select s.number, min(s.updated_at) as resolved_time
    from seen s
    left join last_open lo using (number)
    where s.seq > coalesce(lo.seq, 0)
    group by s.number
),
history as (
    select
        number,
        count(*)                                            as n_updates,
        count(distinct assignment_group)                    as n_groups_seen,
        count(*) filter (where state like 'Awaiting%')      as n_awaiting_updates
    from seen
    group by number
)
select
    cur.number,
    cur.opened_at,
    cast(t as timestamp)                                                    as cutoff,
    cur.is_done                                                             as resolved,
    d.resolved_time,
    date_diff('minute', cur.opened_at, coalesce(d.resolved_time, t)) / 60.0 as duration_h,
    cur.state,
    cur.updated_at                                                          as last_update_at,
    date_diff('minute', cur.updated_at, t) / 60.0                           as hours_since_update,
    h.n_updates,
    h.n_groups_seen,
    h.n_awaiting_updates,
    cur.reassignment_count,
    cur.reopen_count,
    cur.made_sla                                                            as made_sla_so_far,
    cur.impact,
    cur.urgency,
    cur.priority,
    cur.priority_confirmed,
    cur.contact_type,
    cur.category,
    cur.subcategory,
    cur.symptom,
    cur.location,
    cur.assignment_group,
    cur.assigned_to is not null                                             as has_assignee,
    cur.caller_id,
    cur.opened_by
from cur
join history h using (number)
left join done_since d using (number);

create or replace macro open_at(t) as table
select * from snapshot_at(t) where not resolved;
