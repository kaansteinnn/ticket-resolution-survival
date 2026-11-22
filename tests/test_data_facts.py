"""The facts about the data that the README states, re-measured from the built tables."""

import pytest


def one(con, sql):
    return con.execute(sql).fetchall()[0]


def test_row_and_ticket_counts(con):
    assert one(con, "select count(*), count(distinct number) from raw_log") == (141_712, 24_918)
    # Five rows stamped before their own ticket was opened are dropped.
    assert one(con, "select count(*), count(distinct number) from events") == (141_707, 24_918)
    assert one(con, "select count(*) from incidents") == (24_918,)


def test_every_ticket_ends_closed(con):
    assert one(con, "select count(*) from incidents where final_state <> 'Closed'") == (0,)


def test_opening_dates_are_a_three_month_cohort(con):
    first, last = one(con, "select min(opened_at)::date::varchar, max(opened_at)::date::varchar from incidents")
    assert (first, last) == ("2016-02-29", "2017-02-16")
    share = one(
        con,
        "select avg((opened_at < timestamp '2016-06-01')::int) from incidents",
    )[0]
    assert share > 0.98


def test_sla_breach_share(con):
    breached, share = one(
        con,
        "select count(*) filter (where not final_made_sla), avg((not final_made_sla)::int) from incidents",
    )
    assert breached == 9_115
    assert share == pytest.approx(0.366, abs=0.001)


def test_back_filled_columns_never_change_within_a_ticket(con):
    """resolved_at / closed_at hold the final value on every row: they are hindsight, not history."""
    assert one(
        con,
        "select max(n_distinct_resolved_at), max(n_distinct_closed_at) from final_fields",
    ) == (1, 1)
    assert one(con, "select count(*) from final_fields where resolved_at_field is null") == (1_556,)


def test_events_table_has_no_hindsight_columns(con):
    cols = {r[0] for r in con.execute("describe events").fetchall()}
    assert not cols & {"resolved_at", "closed_at", "closed_code", "resolved_by", "active"}


def test_resolution_time_agrees_with_export_field(con):
    """Our resolution time vs the export's resolved_at, where the export has one."""
    n, equal, earlier, later = one(
        con,
        """
        select count(*),
               count(*) filter (where resolved_time = resolved_at_field),
               count(*) filter (where resolved_time < resolved_at_field),
               count(*) filter (where resolved_time > resolved_at_field)
        from incidents where resolved_at_field is not null
        """,
    )
    assert (n, equal, earlier, later) == (23_362, 23_066, 295, 1)


def test_resolution_time_is_never_before_opening(con):
    assert one(con, "select count(*) from incidents where duration_h < 0 or duration_h is null") == (0,)


def test_duration_distribution(con):
    median, mean, q90 = one(
        con,
        "select median(duration_h), avg(duration_h), quantile_cont(duration_h, 0.9) from incidents",
    )
    assert median == pytest.approx(15.1, abs=0.1)
    assert mean == pytest.approx(165.3, abs=0.1)
    assert q90 == pytest.approx(357.5, abs=0.1)


def test_closing_follows_resolution_by_five_days(con):
    """Closing is an automatic step about 120 hours after resolution, so resolution is the real event."""
    median_gap = one(
        con,
        "select median(date_diff('minute', resolved_time, closed_at_field) / 60.0) from incidents",
    )[0]
    assert median_gap == pytest.approx(120.6, abs=0.1)
