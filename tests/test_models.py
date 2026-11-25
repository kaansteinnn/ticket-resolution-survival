"""The methods and the scoring, on small hand-made data."""

import numpy as np
import pandas as pd
import pytest

from src import evaluate, models


def toy(n=400, seed=0):
    """Tickets whose remaining time depends on one feature; a third are still open (censored)."""
    rng = np.random.default_rng(seed)
    slow = rng.integers(0, 2, n)
    true_h = rng.exponential(np.where(slow == 1, 400.0, 40.0))
    follow_up = rng.uniform(100, 600, n)
    frame = pd.DataFrame(
        {
            "number": [f"T{i}" for i in range(n)],
            "duration_h": rng.uniform(1, 500, n),
            "remaining_h": np.minimum(true_h, follow_up),
            "event": true_h <= follow_up,
            "followup_h": follow_up,
            "assignment_group": np.where(slow == 1, "slow team", "fast team"),
        }
    )
    x = pd.DataFrame({"slow": slow.astype(float), "noise": rng.normal(size=n)})
    return frame, x, slow


@pytest.mark.parametrize(
    "model", [models.ResolvedOnlyRegression(), models.Cox(), models.SurvivalForest(min_samples_leaf=20, n_estimators=30)]
)
def test_probabilities_are_valid_and_follow_the_signal(model):
    frame, x, slow = toy()
    slowness, prob = model.fit(frame, x).predict(frame, x)
    assert prob.shape == (len(frame), len(models.GRID_H))
    assert ((prob >= 0) & (prob <= 1)).all()
    assert (np.diff(prob, axis=1) >= -1e-9).all()  # P(resolved by h) never falls as h grows
    assert slowness[slow == 1].mean() > slowness[slow == 0].mean()
    assert prob[slow == 1, 6].mean() < prob[slow == 0, 6].mean()


def test_kaplan_meier_by_age_conditions_on_survival_so_far():
    # Ten tickets resolved at 1..10 days: S(5d) = 0.5 and S(7d) = 0.3, so 2 of the 5 still open resolve within 2 days.
    km = models.KaplanMeierByAge().fit_tickets(24.0 * np.arange(1, 11), np.ones(10, dtype=bool))
    frame = pd.DataFrame({"duration_h": [24.0 * 5]})
    _, prob = km.predict(frame, None)
    assert prob[0, 1] == pytest.approx(0.4)
    assert prob[0, 4] == pytest.approx(1.0)


def test_team_history_ranks_slow_team_first_then_oldest():
    frame, x, slow = toy()
    rule = models.TeamHistory().fit(frame, x)
    assert rule.biggest_late_team_ == "slow team"
    slowness, prob = rule.predict(frame, x)
    assert prob is None
    assert slowness[slow == 1].min() > slowness[slow == 0].max()
    fast = frame[slow == 0]
    assert slowness[slow == 0][fast["duration_h"].argmax()] == slowness[slow == 0].max()


def test_seven_day_outcome_is_known_only_after_seven_days_of_follow_up():
    """A ticket resolved early in a recently opened window is not "known": using it would favour quick tickets."""
    frame = pd.DataFrame(
        {
            "remaining_h": [50.0, 100.0, 200.0, 168.0],
            "event": [True, False, True, False],
            "followup_h": [100.0, 100.0, 400.0, 168.0],
        }
    )
    known, late = models.known_late(frame)
    assert known.tolist() == [False, False, True, True]
    assert late[known].tolist() == [True, True]  # resolved after 200 h; still open after the full 7 days


def test_classifier_and_age_bands_answer_the_seven_day_question_only():
    frame, x, slow = toy()
    for model in (models.LateClassifier(min_samples_leaf=20), models.LateRateByAge()):
        p_late, prob = model.fit(frame, x).predict(frame, x)
        assert ((p_late >= 0) & (p_late <= 1)).all()
        assert np.allclose(prob[:, models.LATE_IDX], 1 - p_late)
        assert np.isnan(np.delete(prob, models.LATE_IDX, axis=1)).all()
    p_late, _ = models.LateClassifier(min_samples_leaf=20).fit(frame, x).predict(frame, x)
    assert p_late[slow == 1].mean() > p_late[slow == 0].mean()


def test_classifier_that_drops_open_tickets_is_too_optimistic():
    frame, x, _ = toy(n=2000)
    known, late = models.known_late(frame)
    everything = models.LateClassifier(min_samples_leaf=20).fit(frame, x)
    resolved_only = models.LateClassifier(min_samples_leaf=20, rows="resolved").fit(frame, x)
    assert resolved_only.n_used_ < everything.n_used_ == known.sum()
    assert resolved_only.predict(frame, x)[0].mean() < everything.predict(frame, x)[0].mean() - 0.05
    assert everything.predict(frame, x)[0][known].mean() == pytest.approx(late[known].mean(), abs=0.02)


def test_queue_metrics_on_a_perfect_and_a_reversed_ranking():
    remaining = 24.0 * np.arange(1, 21)  # 1..20 days; 13 of 20 are late (> 7 days)
    frame = pd.DataFrame({"remaining_h": remaining, "event": True})
    perfect = evaluate.queue_metrics(frame, remaining, None)
    reversed_ = evaluate.queue_metrics(frame, -remaining, None)
    assert perfect["late_share"] == pytest.approx(13 / 20)
    assert (perfect["c_index"], perfect["top10"]) == (1.0, 1.0)
    assert (reversed_["c_index"], reversed_["top10"]) == (0.0, 0.0)

    certain = np.tile((remaining <= 168)[:, None].astype(float), (1, len(models.GRID_H)))
    assert evaluate.queue_metrics(frame, remaining, certain)["brier"] == 0.0
    assert evaluate.queue_metrics(frame, remaining, 1 - certain)["brier"] == 1.0


def test_late_label_refuses_incomplete_follow_up():
    frame = pd.DataFrame({"remaining_h": [50.0], "event": [False]})
    with pytest.raises(ValueError):
        evaluate.is_late(frame)
