"""The model numbers quoted in the README, checked against the tables written by `python run_all.py`.

Refitting the forest takes minutes, so these tests read the result tables instead of recomputing them.
"""

from pathlib import Path

import pandas as pd
import pytest

TABLES = Path(__file__).resolve().parent.parent / "reports" / "tables"
FOREST = "Random survival forest"
OLDEST = "Oldest first"
TEAM = "Team's past late rate, then oldest"
REGRESSION = "Regression on resolved tickets"

pytestmark = pytest.mark.skipif(not (TABLES / "model_summary.csv").exists(), reason="run `python run_all.py` first")


def table(name):
    return pd.read_csv(TABLES / name).set_index("method")


def test_headline_numbers():
    s = table("model_summary.csv")
    assert (s["days"] == 31).all()
    assert s.loc[FOREST, ["c_index_mean", "top10_mean", "brier_mean"]].tolist() == [0.705, 0.986, 0.173]
    assert s.loc[OLDEST, ["c_index_mean", "top10_mean"]].tolist() == [0.674, 0.876]
    assert s.loc[TEAM, ["c_index_mean", "top10_mean"]].tolist() == [0.686, 0.972]
    assert s.loc[REGRESSION, "brier_mean"] == 0.247
    assert s.loc["Cox", ["c_index_mean", "top10_mean", "brier_mean"]].tolist() == [0.690, 0.982, 0.180]


def test_forest_beats_oldest_first_every_day_and_is_best_overall():
    s = table("model_summary.csv")
    assert s.loc[FOREST, "days_c_index_above_reference"] == 31
    assert s.loc[FOREST, "days_top10_above_reference"] == 31
    assert s["c_index_mean"].idxmax() == FOREST
    assert s["top10_mean"].idxmax() == FOREST
    assert s["brier_mean"].idxmin() == FOREST


def test_without_the_biggest_late_team_the_edge_nearly_disappears():
    """Reported, not hidden: outside one team the forest barely beats the rule of thumb, and Cox loses to it."""
    s = table("model_summary_without_biggest_late_team.csv")
    assert s.loc[FOREST, ["c_index_mean", "top10_mean"]].tolist() == [0.665, 0.853]
    assert s.loc[OLDEST, ["c_index_mean", "top10_mean"]].tolist() == [0.659, 0.848]
    assert s.loc[FOREST, "days_c_index_above_reference"] == 24
    assert s.loc[FOREST, "days_top10_above_reference"] == 19
    assert s.loc["Cox", "c_index_mean"] < s.loc[OLDEST, "c_index_mean"]


def test_resolved_only_regression_is_too_optimistic():
    daily = pd.read_csv(TABLES / "model_daily.csv")
    late = daily.drop_duplicates("cutoff")["late_share"]
    predicted = daily.groupby("method")["mean_p_late"].mean()
    assert late.mean() == pytest.approx(0.659, abs=0.001)
    assert predicted[FOREST] == pytest.approx(0.649, abs=0.001)
    assert predicted[REGRESSION] == pytest.approx(0.422, abs=0.001)
    base_rate_brier = (late * (1 - late)).mean()
    assert base_rate_brier == pytest.approx(0.222, abs=0.001)
    assert table("model_summary.csv").loc[REGRESSION, "brier_mean"] > base_rate_brier


def test_leaf_size_was_chosen_on_april_data():
    selection = pd.read_csv(TABLES / "model_selection.csv")
    assert selection.loc[selection["brier"].idxmin(), "min_samples_leaf"] == 120


def test_forecast_of_how_many_tickets_will_be_late():
    """Finding 4: survival methods miss by about 40 tickets a day; resolved-only regression by 282."""
    w = pd.read_csv(TABLES / "model_workload.csv").set_index("method")
    assert (w["days"] == 31).all()
    assert w["actual_late_mean"].round().eq(788).all()
    forecast = w["forecast_late_mean"].round().astype(int).to_dict()
    miss = w["abs_error_mean"].round().astype(int).to_dict()
    assert forecast == {
        "Late share of all past queues": 662,
        "Late share of the latest past queue": 719,
        REGRESSION: 506,
        "Kaplan–Meier by age": 778,
        "Cox": 767,
        FOREST: 777,
    }
    assert miss == {
        "Late share of all past queues": 126,
        "Late share of the latest past queue": 69,
        REGRESSION: 282,
        "Kaplan–Meier by age": 40,
        "Cox": 44,
        FOREST: 43,
    }
    assert w.loc[REGRESSION, "days_forecast_too_low"] == 31
    assert w.loc["Late share of all past queues", "fixed_late_share"] == 0.552
    assert w.loc["Late share of the latest past queue", "fixed_late_share"] == 0.6
    # Reported, not hidden: the forest is no better at this than Kaplan–Meier without any ticket attribute.
    assert w.loc[FOREST, "abs_error_mean"] > w.loc["Kaplan–Meier by age", "abs_error_mean"]


def test_one_team_holds_a_fifth_of_the_queue_and_is_nearly_always_late():
    path = TABLES.parent.parent / "app" / "data" / "predictions.parquet"
    if not path.exists():
        pytest.skip("run `python run_all.py` first")
    df = pd.read_parquet(path)
    team = df["assignment_group"] == "Group 9"
    assert team.mean() == pytest.approx(0.189, abs=0.001)
    assert df.loc[team, "actual_late"].mean() == pytest.approx(0.974, abs=0.001)
    assert (team & df["actual_late"]).sum() / df["actual_late"].sum() == pytest.approx(0.281, abs=0.001)
    assert df.loc[team, "number"].nunique() == 268 and df["number"].nunique() == 4078
