"""The demo runs end to end on the committed predictions file."""

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app" / "app.py"
PREDICTIONS = ROOT / "app" / "data" / "predictions.parquet"

pytestmark = pytest.mark.skipif(not PREDICTIONS.exists(), reason="run `python run_all.py` first")


def test_predictions_file_is_consistent():
    df = pd.read_parquet(PREDICTIONS)
    curve = df[[f"p_resolved_d{d:02d}" for d in range(1, 31)]].to_numpy()
    assert df["cutoff"].nunique() == 31 and df["cutoff"].min() == "2016-05-01" and df["cutoff"].max() == "2016-05-31"
    assert ((curve >= 0) & (curve <= 1)).all()
    assert (curve[:, 1:] >= curve[:, :-1] - 1e-6).all()
    assert ((df["p_late"] - (1 - curve[:, 6])).abs() < 1e-3).all()
    assert (df["actual_late"] == (df["actual_remaining_days"] > 7)).all()
    assert (df["actual_remaining_days"] >= 0).all()  # stored rounded to 0.01 day


def test_app_renders_and_reveals_outcomes():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    assert app.metric[0].label == "Açık kayıt"

    app.toggle[0].set_value(True).run()
    assert not app.exception
    assert "Gerçekte çözülmeyen" in [m.label for m in app.metric]


def test_demo_walkthrough_numbers_for_16_may():
    """The demo opens on 16 May 2016; these are the numbers a walkthrough of that day shows."""
    df = pd.read_parquet(PREDICTIONS)
    day = df[df["cutoff"] == "2016-05-16"].sort_values("p_late", ascending=False, kind="stable")
    k = round(0.10 * len(day))
    assert (len(day), k) == (1239, 124)
    assert round(day["p_late"].sum()) == 811 and day["actual_late"].sum() == 756
    assert (day["p_late"] > 0.8).sum() == 294
    assert day.head(k)["actual_late"].sum() == 124
    assert day.sort_values("age_days", ascending=False, kind="stable").head(k)["actual_late"].sum() == 100
    assert (day["assignment_group"] == "Group 9").sum() == 232
    top = day.iloc[0]
    assert top["assignment_group"] == "Group 9" and round(top["p_late"], 2) == 0.97
    assert round(float(top["p_resolved_d30"]), 2) == 0.08
