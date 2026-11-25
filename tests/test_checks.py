"""The resampling used for the intervals computes the same measures as the main scoring."""

import numpy as np
import pandas as pd
import pytest

from src import checks, evaluate, models


def queue(n=300, seed=1):
    rng = np.random.default_rng(seed)
    remaining = np.round(rng.exponential(200.0, n), 0) + 1  # whole hours, so some remaining times are tied
    slowness = np.round(np.log(remaining) + rng.normal(0, 1.0, n), 1)  # a noisy ranking with tied scores
    p_resolved = np.tile(rng.uniform(0, 1, n)[:, None], (1, len(models.GRID_H)))
    frame = pd.DataFrame({"number": [f"T{i}" for i in range(n)], "remaining_h": remaining, "event": True})
    return frame, slowness, p_resolved


@pytest.mark.parametrize("with_probability", [True, False])
def test_weights_of_one_reproduce_the_ordinary_measures(with_probability):
    frame, slowness, p_resolved = queue()
    prob = p_resolved if with_probability else None
    expected = evaluate.queue_metrics(frame, slowness, prob)
    got = checks.weighted_day(frame["remaining_h"].to_numpy(), slowness, prob, np.ones((1, len(frame))))
    assert got["c_index"][0] == pytest.approx(expected["c_index"], abs=1e-12)
    assert got["top10"][0] == pytest.approx(expected["top10"], abs=1e-12)
    if with_probability:
        late = evaluate.is_late(frame)
        assert got["brier"][0] == pytest.approx(expected["brier"], abs=1e-12)
        assert got["count_error"][0] == pytest.approx(abs((1 - prob[:, 6]).sum() - late.sum()), abs=1e-9)
    else:
        assert np.isnan(got["brier"][0]) and np.isnan(got["count_error"][0])


def test_a_ticket_drawn_twice_counts_twice():
    frame, slowness, p_resolved = queue(n=40)
    w = np.ones((1, len(frame)))
    w[0, :5] = 2.0
    doubled = pd.concat([frame, frame.iloc[:5]], ignore_index=True)
    order = np.r_[np.arange(len(frame)), np.arange(5)]
    expected = evaluate.queue_metrics(doubled, slowness[order], p_resolved[order])
    got = checks.weighted_day(frame["remaining_h"].to_numpy(), slowness, p_resolved, w)
    assert got["c_index"][0] == pytest.approx(expected["c_index"], abs=1e-12)
    assert got["brier"][0] == pytest.approx(expected["brier"], abs=1e-12)


def test_bootstrap_keeps_all_days_of_a_ticket_together():
    """Two identical days: every replicate must give the same value on both, so the mean equals one day's value."""
    frame, slowness, p_resolved = queue(n=60)
    two_days = pd.concat([frame.assign(cutoff=1), frame.assign(cutoff=2)], ignore_index=True)
    predictions = {"m": (np.r_[slowness, slowness], np.r_[p_resolved, p_resolved])}
    both = checks.bootstrap(two_days, predictions, n_rep=20, seed=3)
    one = checks.bootstrap(frame.assign(cutoff=1), {"m": (slowness, p_resolved)}, n_rep=20, seed=3)
    for metric in checks.METRICS:
        assert np.allclose(both["m", metric], one["m", metric])
