"""The model numbers quoted in the README, checked against the tables written by `python run_all.py`.

Refitting the forest takes minutes, so these tests read the result tables instead of recomputing them.
"""

from pathlib import Path

import pandas as pd
import pytest

from src import config

TABLES = Path(__file__).resolve().parent.parent / "reports" / "tables"
FOREST = "Random survival forest"
OLDEST = "Oldest first"
TEAM = "Team's past late rate, then oldest"
AGE_BAND = "Past late rate by age band"
REGRESSION = "Regression on resolved tickets"
KM = "Kaplan–Meier by age"
CLASSIFIER = "Gradient boosting classifier"

pytestmark = pytest.mark.skipif(not (TABLES / "check_intervals.csv").exists(), reason="run `python run_all.py` first")


def table(name, index="method"):
    return pd.read_csv(TABLES / name).set_index(index)


def interval(scope, quantity, metric):
    t = pd.read_csv(TABLES / "check_intervals.csv")
    return t[(t["scope"] == scope) & (t["quantity"] == quantity) & (t["metric"] == metric)].iloc[0]


def test_headline_numbers():
    s = table("model_summary.csv")
    assert (s["days"] == 31).all()
    assert s.loc[FOREST, ["c_index_mean", "top10_mean", "brier_mean"]].tolist() == [0.705, 0.986, 0.173]
    assert s.loc[CLASSIFIER, ["c_index_mean", "top10_mean", "brier_mean"]].tolist() == [0.705, 0.985, 0.173]
    assert s.loc["Cox", ["c_index_mean", "top10_mean", "brier_mean"]].tolist() == [0.690, 0.982, 0.180]
    assert s.loc[KM, ["c_index_mean", "top10_mean", "brier_mean"]].tolist() == [0.673, 0.878, 0.189]
    assert s.loc[REGRESSION, ["c_index_mean", "top10_mean", "brier_mean"]].tolist() == [0.666, 0.945, 0.247]
    assert s.loc[AGE_BAND, ["c_index_mean", "top10_mean", "brier_mean"]].tolist() == [0.660, 0.910, 0.203]
    assert s.loc[OLDEST, ["c_index_mean", "top10_mean"]].tolist() == [0.674, 0.876]
    assert s.loc[TEAM, ["c_index_mean", "top10_mean"]].tolist() == [0.686, 0.972]


def test_forest_beats_oldest_first_on_the_whole_queue():
    s = table("model_summary.csv")
    assert s.loc[FOREST, "days_c_index_above_reference"] == 31
    assert s.loc[FOREST, "days_top10_above_reference"] == 31
    gap = interval("All tickets", f"{FOREST} minus {OLDEST}", "c_index")
    assert gap["value"] == pytest.approx(0.031, abs=0.001)
    assert gap["lo"] > 0


def test_an_ordinary_classifier_does_as_well_as_the_survival_forest():
    """Reported, not hidden: a survival model is not required for the 7-day question."""
    for metric in ("c_index", "brier", "top10", "count_error"):
        gap = interval("All tickets", f"{FOREST} minus {CLASSIFIER}", metric)
        assert gap["lo"] < 0 < gap["hi"]


def test_dropping_the_open_tickets_is_what_breaks_the_forecast():
    t = pd.read_csv(TABLES / "check_open_tickets.csv")
    everything, resolved, random = t.iloc[0], t.iloc[1], t.iloc[2:]
    assert (everything["rows"], resolved["rows"]) == (25640, 19747)
    assert (random["rows"] == 19747).all() and len(random) == 3
    assert (everything["brier"], everything["count_error"]) == pytest.approx((0.173, 45.2), abs=0.05)
    assert everything["brier"] == pytest.approx(0.173, abs=0.0005)
    assert (resolved["brier"], resolved["count_error"]) == pytest.approx((0.264, 266.3), abs=0.05)
    assert resolved["brier"] == pytest.approx(0.264, abs=0.001)
    assert random["brier"].between(0.172, 0.174).all()
    assert random["count_error"].between(44, 46).all()
    # The resolved-only rows are a biased sample: far fewer late tickets than in the rows it replaces.
    assert (everything["late_share_in_training"], resolved["late_share_in_training"]) == pytest.approx((0.552, 0.418), abs=0.001)


def test_without_the_biggest_late_team_no_model_beats_oldest_first():
    """Reported, not hidden: refitted without one team, the forest ties with the rule of thumb and Cox loses to it."""
    s = table("check_without_team.csv")
    assert (s["training_rows"] == 26039).all() and (s["test_tickets"] == 3840).all()
    assert s.loc[FOREST, ["c_index", "top10"]].round(3).tolist() == [0.662, 0.866]
    assert s.loc[OLDEST, ["c_index", "top10"]].round(3).tolist() == [0.659, 0.848]
    assert s.loc["Cox", "c_index"] < s.loc[OLDEST, "c_index"]
    scope = f"Without {config.BIGGEST_LATE_TEAM}"
    for metric in ("c_index", "top10"):
        gap = interval(scope, f"{FOREST} minus {OLDEST}", metric)
        assert gap["lo"] < 0 < gap["hi"]
    # Without that team the spreadsheet calculation forecasts the count as well as the models do.
    assert s.loc[[AGE_BAND, KM, "Cox", FOREST, CLASSIFIER], "count_error"].between(46, 48).all()
    assert s.loc[REGRESSION, "count_error"] == pytest.approx(205.2, abs=0.1)


def test_tickets_never_seen_in_training_are_scored_separately():
    t = pd.read_csv(TABLES / "check_new_tickets.csv")
    new = t[t["tickets"] == "Never in a training queue"].set_index("method")
    seen = t[t["tickets"] == "Also in a training queue"].set_index("method")
    assert (new["n_tickets"] == 3039).all() and (seen["n_tickets"] == 1039).all()
    assert new["rows"].iloc[0] + seen["rows"].iloc[0] == 37168
    assert seen["rows"].iloc[0] / 37168 == pytest.approx(0.484, abs=0.001)
    assert new.loc[FOREST, "c_index"] == pytest.approx(0.633, abs=0.001)
    assert new.loc[OLDEST, "c_index"] == pytest.approx(0.593, abs=0.001)
    assert new.loc[TEAM, "top10"] > new.loc[FOREST, "top10"]  # the one-variable rule is ahead here


def test_best_method_depends_on_the_measure():
    t = table("check_top_share.csv")
    assert t["top10"].idxmax() == FOREST
    assert t["top20"].idxmax() == "Cox"
    assert t.loc[["Cox", FOREST], "top20"].round(3).tolist() == [0.973, 0.962]


def test_resolved_only_regression_is_too_optimistic():
    daily = pd.read_csv(TABLES / "model_daily.csv")
    late = daily.drop_duplicates("cutoff")["late_share"]
    predicted = daily.groupby("method")["mean_p_late"].mean()
    assert late.mean() == pytest.approx(0.659, abs=0.001)
    assert predicted[FOREST] == pytest.approx(0.649, abs=0.001)
    assert predicted[REGRESSION] == pytest.approx(0.422, abs=0.001)
    # Giving every ticket the same 66% (the month's late share): Brier 0.225.
    same_for_all = (late * (1 - late.mean()) ** 2 + (1 - late) * late.mean() ** 2).mean()
    assert same_for_all == pytest.approx(0.225, abs=0.001)
    assert table("model_summary.csv").loc[REGRESSION, "brier_mean"] > same_for_all


def test_settings_were_chosen_on_april_data():
    forest = pd.read_csv(TABLES / "model_selection.csv")
    assert forest.loc[forest["brier"].idxmin(), "min_samples_leaf"] == 120
    classifier = pd.read_csv(TABLES / "model_selection_classifier.csv")
    best = classifier.loc[classifier["brier"].idxmin()]
    assert (best["max_leaf_nodes"], best["min_samples_leaf"]) == (7, 240)


def test_forecast_of_how_many_tickets_will_be_late():
    """Methods that keep the open tickets miss by 40-45 tickets a day; resolved-only regression by 282."""
    w = pd.read_csv(TABLES / "model_workload.csv").set_index("method")
    assert (w["days"] == 31).all()
    assert w["actual_late_mean"].round().eq(788).all()
    forecast = w["forecast_late_mean"].round().astype(int).to_dict()
    miss = w["abs_error_mean"].round().astype(int).to_dict()
    assert forecast == {
        "Late share of all past queues": 662,
        "Late share of the latest past queue": 719,
        AGE_BAND: 715,
        REGRESSION: 506,
        KM: 778,
        "Cox": 767,
        FOREST: 777,
        CLASSIFIER: 788,
    }
    assert miss == {
        "Late share of all past queues": 126,
        "Late share of the latest past queue": 69,
        AGE_BAND: 74,
        REGRESSION: 282,
        KM: 40,
        "Cox": 44,
        FOREST: 43,
        CLASSIFIER: 45,
    }
    assert w.loc[REGRESSION, "days_forecast_too_low"] == 31
    assert w.loc["Late share of all past queues", "fixed_late_share"] == 0.552
    assert w.loc["Late share of the latest past queue", "fixed_late_share"] == 0.6
    # Reported, not hidden: the forest is no better at this than Kaplan–Meier without any ticket attribute.
    gap = interval("All tickets", f"{FOREST} minus {KM}", "count_error")
    assert gap["value"] > 0 and gap["lo"] < 0 < gap["hi"]


def test_one_team_holds_a_fifth_of_the_queue_and_is_nearly_always_late():
    path = TABLES.parent.parent / "app" / "data" / "predictions.parquet"
    if not path.exists():
        pytest.skip("run `python run_all.py` first")
    df = pd.read_parquet(path)
    team = df["assignment_group"] == config.BIGGEST_LATE_TEAM
    assert team.mean() == pytest.approx(0.189, abs=0.001)
    assert df.loc[team, "actual_late"].mean() == pytest.approx(0.974, abs=0.001)
    assert (team & df["actual_late"]).sum() / df["actual_late"].sum() == pytest.approx(0.281, abs=0.001)
    assert df.loc[team, "number"].nunique() == 268 and df["number"].nunique() == 4078
    # Every ticket the forest puts in its riskiest 10% belongs to that team.
    picked = []
    for _, day in df.groupby("cutoff"):
        k = max(1, round(0.10 * len(day)))
        picked.append(day.sort_values("p_late", ascending=False, kind="stable").head(k))
    picked = pd.concat(picked)
    assert (picked["assignment_group"] == config.BIGGEST_LATE_TEAM).all() and len(picked) == 3714
    # The team named in config is the one the training data singles out (src/checks.py writes the scope name).
    scopes = pd.read_csv(TABLES / "check_intervals.csv")["scope"].unique().tolist()
    assert scopes == ["All tickets", f"Without {config.BIGGEST_LATE_TEAM}"]


def test_weekly_refitting_helps_a_little_and_does_not_remove_the_drift():
    t = pd.read_csv(TABLES / "check_retraining.csv")
    forest = t[t["method"] == FOREST].set_index(["period", "training"])["count_error"]
    assert forest["1-31 May", "Once, on 1 May"] == pytest.approx(42.7, abs=0.1)
    assert forest["1-31 May", "Every week"] == pytest.approx(41.2, abs=0.1)
    assert forest["21-26 May", "Once, on 1 May"] == pytest.approx(94.4, abs=0.1)
    assert forest["21-26 May", "Every week"] == pytest.approx(88.0, abs=0.1)


def test_forest_results_do_not_hinge_on_the_random_seed():
    t = pd.read_csv(TABLES / "check_seeds.csv")
    assert t["seed"].tolist() == [0, 1, 2, 3, 4]
    assert t["c_index"].between(0.703, 0.706).all()
    assert t["brier"].between(0.173, 0.174).all()
    assert t["count_error"].between(42, 43).all()
    # Every seed stays ahead of Cox and of "oldest first" on ranking, and behind Kaplan–Meier on the count.
    s = table("model_summary.csv")
    w = table("model_workload.csv")
    assert t["c_index"].min() > s.loc["Cox", "c_index_mean"] > s.loc[OLDEST, "c_index_mean"]
    assert t["count_error"].min() > w.loc[KM, "abs_error_mean"]


def test_probabilities_are_right_on_average_but_not_for_every_team():
    t = pd.read_csv(TABLES / "check_calibration.csv")
    teams = t[t["by"] == "team"].set_index("group")
    assert teams.index[:2].tolist() == [config.BIGGEST_LATE_TEAM, "Group 20"]
    assert (teams.loc["Group 20", "predicted_late"], teams.loc["Group 20", "actual_late"]) == pytest.approx((0.54, 0.75), abs=0.005)
    assert (teams.loc["Group 70", "predicted_late"], teams.loc["Group 70", "actual_late"]) == pytest.approx((0.39, 0.20), abs=0.005)
    ages = t[t["by"] == "age_days"]
    assert len(ages) == 7 and ages["rows"].sum() == 37168
