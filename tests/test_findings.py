"""The two descriptive findings, checked on the numbers the report quotes."""

import numpy as np
import pandas as pd
import pytest

from src import db
from src.analysis import mean_trap, waiting


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


def test_remaining_time_grows_with_waiting(con):
    durations = (db.incidents(con)["duration_h"] / 24.0).to_numpy()
    med = lambda age: np.median(waiting.remaining(durations, age))
    assert med(0) == pytest.approx(0.63, abs=0.01)
    assert med(1) == pytest.approx(5.61, abs=0.01)
    assert med(7) == pytest.approx(7.11, abs=0.01)
    assert med(28) == pytest.approx(26.98, abs=0.01)
    assert med(0) < med(1) < med(7) < med(14) < med(28)
