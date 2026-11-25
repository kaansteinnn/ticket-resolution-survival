"""The two descriptive findings, checked on the numbers the report quotes."""

import numpy as np
import pandas as pd
import pytest

from src import config, db
from src.analysis import mean_trap, waiting

APRIL_4 = pd.Timestamp("2016-04-04").date()


@pytest.fixture(scope="module")
def weekly(con):
    return mean_trap.weekly_table(con).set_index("cutoff")


def test_resolved_only_mean_understates_the_truth_every_week(weekly):
    assert (weekly["naive_mean"] < weekly["true_mean"]).all()
    row = weekly.loc[pd.Timestamp("2016-04-04").date()]
    assert row["naive_mean"] == pytest.approx(3.69, abs=0.01)
    assert row["true_mean"] == pytest.approx(9.05, abs=0.01)


def test_kaplan_meier_median_tracks_the_truth(weekly):
    """From April on (where the median is estimable) KM is far closer to the truth than the naive median."""
    late = weekly[weekly.index >= pd.Timestamp("2016-04-04").date()]
    km_error = (late["km_median"] - late["true_median"]).abs()
    naive_error = (late["naive_median"] - late["true_median"]).abs()
    assert km_error.max() < 0.05
    assert (naive_error > km_error).all()


def test_kaplan_meier_is_not_usable_in_the_first_two_weeks(weekly):
    """Reported, not hidden: with half the tickets still open the median cannot be estimated well."""
    assert np.isnan(weekly.loc[pd.Timestamp("2016-03-07").date(), "km_median"])
    second = weekly.loc[pd.Timestamp("2016-03-14").date()]
    assert (second["km_median"], second["true_median"]) == pytest.approx((3.72, 3.26), abs=0.01)
    from_third_week = weekly[weekly.index >= pd.Timestamp("2016-03-21").date()]
    assert (from_third_week["km_median"] - from_third_week["true_median"]).abs().max() < 0.06
    from_april = weekly[weekly.index >= APRIL_4]
    assert (from_april["km_median"] - from_april["true_median"]).abs().max() < 0.005


def test_share_resolved_within_a_week_three_ways(weekly):
    row = weekly.loc[APRIL_4]
    assert row[["naive_within_7d", "km_within_7d", "true_within_7d"]].tolist() == pytest.approx(
        [0.790, 0.681, 0.698], abs=0.001
    )


def test_mean_trap_is_not_the_effect_of_one_team(con):
    others = db.never_handled_by(con, config.BIGGEST_LATE_TEAM)
    assert 24918 - len(others) == 454
    table = mean_trap.weekly_table(con, only=others).set_index("cutoff")
    assert (table["naive_mean"] < table["true_mean"]).all()
    row = table.loc[APRIL_4]
    assert (row["naive_mean"], row["true_mean"]) == pytest.approx((3.65, 7.72), abs=0.01)
    assert (row["km_median"], row["true_median"]) == pytest.approx((1.97, 1.97), abs=0.01)


def test_waiting_pattern_is_not_one_team_and_not_batch_entry(con):
    tickets = db.incidents(con)
    subsets = waiting.subsets_table(tickets, db.never_handled_by(con, config.BIGGEST_LATE_TEAM)).set_index("tickets")
    medians = ["median_remaining_1d", "median_remaining_7d", "median_remaining_28d"]
    assert subsets.loc["All tickets", medians].tolist() == pytest.approx([5.61, 7.11, 26.98], abs=0.01)
    assert subsets.loc[f"Never handled by {config.BIGGEST_LATE_TEAM}", medians].tolist() == pytest.approx(
        [5.16, 6.80, 18.13], abs=0.01
    )
    batch = subsets.index[2]
    assert subsets.loc[batch, "n"] == 24814
    assert subsets.loc[batch, medians].tolist() == pytest.approx([5.63, 7.10, 26.99], abs=0.01)

    calendar = waiting.calendar_table(tickets).set_index("measure")["value"]
    assert calendar["share resolved in a minute with 5+ resolutions"] == pytest.approx(0.0042, abs=0.0001)
    assert calendar["most tickets resolved in one minute"] == 6
    assert calendar["share resolved within 1 hour of a whole number of weeks after opening"] == pytest.approx(
        0.011, abs=0.001
    )


def test_resolution_rate_does_not_fall_at_every_age(con):
    """The remaining time grows at the ages quoted, but the pattern is not a smooth law."""
    durations = (db.incidents(con)["duration_h"] / 24.0).to_numpy()
    next_day = lambda age: (waiting.remaining(durations, age) <= 1).mean()
    assert next_day(4) < next_day(6)
    assert next_day(10) < next_day(14)


def test_remaining_time_grows_with_waiting(con):
    durations = (db.incidents(con)["duration_h"] / 24.0).to_numpy()
    med = lambda age: np.median(waiting.remaining(durations, age))
    assert med(0) == pytest.approx(0.63, abs=0.01)
    assert med(1) == pytest.approx(5.61, abs=0.01)
    assert med(7) == pytest.approx(7.11, abs=0.01)
    assert med(28) == pytest.approx(26.98, abs=0.01)
    assert med(0) < med(1) < med(7) < med(14) < med(28)
