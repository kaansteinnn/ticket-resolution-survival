"""A snapshot at time t must depend only on what was recorded up to t.

The check is direct: rewrite or delete everything after the cutoff and require an identical snapshot.
"""

import duckdb
import pandas as pd
import pytest

from src import db

CUTOFFS = ["2016-03-14 00:00:00", "2016-04-04 09:30:00", "2016-05-16 00:00:00"]


def snap(con, cutoff):
    return db.snapshot(con, cutoff)


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_snapshot_ignores_rewritten_future(scratch_con, cutoff):
    before = snap(scratch_con, cutoff)
    scratch_con.execute(
        f"""
        update events
        set state = 'Closed', is_done = true, reassignment_count = 99, reopen_count = 99,
            made_sla = false, priority = '1 - Critical', assignment_group = 'Group X',
            category = 'Category X', assigned_to = 'Nobody'
        where updated_at > timestamp '{cutoff}'
        """
    )
    after = snap(scratch_con, cutoff)
    pd.testing.assert_frame_equal(before, after)


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_snapshot_ignores_deleted_future(scratch_con, cutoff):
    before = snap(scratch_con, cutoff)
    scratch_con.execute(f"delete from events where updated_at > timestamp '{cutoff}'")
    after = snap(scratch_con, cutoff)
    pd.testing.assert_frame_equal(before, after)


def test_snapshot_changes_when_the_past_changes(scratch_con):
    """Guard against a test that passes only because the comparison is blind."""
    cutoff = CUTOFFS[1]
    before = snap(scratch_con, cutoff)
    scratch_con.execute(
        f"update events set reassignment_count = 99 where updated_at <= timestamp '{cutoff}'"
    )
    after = snap(scratch_con, cutoff)
    assert not before.equals(after)


def test_open_tickets_at_cutoff_resolve_after_it(con):
    """Every ticket shown as open at t has its true resolution after t."""
    cutoff = pd.Timestamp(CUTOFFS[1])
    open_now = db.snapshot(con, cutoff, only_open=True)
    truth = db.incidents(con).set_index("number")
    assert len(open_now) > 1000
    assert (truth.loc[open_now["number"], "resolved_time"] > cutoff).all()
    assert open_now["resolved_time"].isna().all()
    assert (open_now["opened_at"] <= cutoff).all()
    assert (open_now["last_update_at"] <= cutoff).all()


def _toy_events(rows):
    """Hand-written log: (number, opened_at, updated_at, mod_count, state)."""
    con = duckdb.connect()
    con.execute(
        """
        create table events (
            number varchar, opened_at timestamp, updated_at timestamp, mod_count integer,
            state varchar, reassignment_count integer, reopen_count integer, made_sla boolean,
            impact varchar, urgency varchar, priority varchar, priority_confirmed boolean,
            contact_type varchar, category varchar, subcategory varchar, symptom varchar,
            location varchar, assignment_group varchar, assigned_to varchar, caller_id varchar,
            opened_by varchar, updated_by varchar, is_done boolean, seq integer
        )
        """
    )
    for seq, (number, opened, updated, mod, state) in enumerate(rows, start=1):
        con.execute(
            """
            insert into events (number, opened_at, updated_at, mod_count, state, reassignment_count,
                                reopen_count, made_sla, is_done, seq)
            values (?, ?, ?, ?, ?, 0, 0, true, ?, ?)
            """,
            [number, opened, updated, mod, state, state in ("Resolved", "Closed"), seq],
        )
    db.run_sql_files(con, ["03_snapshot.sql"])
    return con


def test_reopened_ticket_is_open_again_and_counts_from_final_resolution():
    con = _toy_events(
        [
            ("A", "2016-03-01 08:00", "2016-03-01 08:00", 0, "New"),
            ("A", "2016-03-01 08:00", "2016-03-01 10:00", 1, "Resolved"),
            ("A", "2016-03-01 08:00", "2016-03-02 08:00", 2, "Active"),  # reopened
            ("A", "2016-03-01 08:00", "2016-03-03 08:00", 3, "Resolved"),
            ("A", "2016-03-01 08:00", "2016-03-03 09:00", 4, "Resolved"),  # edit while resolved
            ("A", "2016-03-01 08:00", "2016-03-08 09:00", 5, "Closed"),
        ]
    )
    row = lambda t: db.snapshot(con, t).iloc[0]

    assert db.snapshot(con, "2016-03-01 07:00").empty  # not opened yet

    first = row("2016-03-01 12:00")  # looks resolved after 2 hours
    assert first["resolved"] and first["duration_h"] == 2.0

    reopened = row("2016-03-02 12:00")  # open again, 28 hours old
    assert not reopened["resolved"] and reopened["duration_h"] == 28.0

    final = row("2016-03-10 00:00")  # final resolution, not the later edit or the closing
    assert final["resolved"] and final["duration_h"] == 48.0
