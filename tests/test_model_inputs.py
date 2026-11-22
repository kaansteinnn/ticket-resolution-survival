"""Model inputs and labels must respect time: features from <= cutoff, labels from <= label_time."""

import numpy as np
import pandas as pd
import pytest

from src import evaluate, features

CUTOFFS = pd.to_datetime(["2016-04-04", "2016-04-11"])
LABEL_TIME = pd.Timestamp("2016-04-17")
FORBIDDEN = ("resolved", "closed", "remaining", "event", "made_sla", "sla")


def rewrite_after(con, when):
    con.execute(
        f"""
        update events
        set state = 'Closed', is_done = true, reassignment_count = 99, reopen_count = 99,
            priority = '1 - Critical', assignment_group = 'Group X', category = 'Category X'
        where updated_at > timestamp '{when}'
        """
    )


def test_features_ignore_everything_after_the_cutoff(scratch_con):
    encoder = features.Encoder().fit(features.landmark_frame(scratch_con, CUTOFFS[:1]))
    before = encoder.transform(features.landmark_frame(scratch_con, CUTOFFS[:1]))
    rewrite_after(scratch_con, CUTOFFS[0])
    after = encoder.transform(features.landmark_frame(scratch_con, CUTOFFS[:1]))
    assert len(before) > 1000
    pd.testing.assert_frame_equal(before, after)


def test_training_labels_ignore_everything_after_label_time(scratch_con):
    before = features.landmark_frame(scratch_con, CUTOFFS, LABEL_TIME)
    rewrite_after(scratch_con, LABEL_TIME)
    after = features.landmark_frame(scratch_con, CUTOFFS, LABEL_TIME)
    pd.testing.assert_frame_equal(before, after)
    assert (~before["event"]).sum() > 100  # the still-open tickets are kept as censored rows
    censored = before[~before["event"]]
    hours_to_label_time = (LABEL_TIME - censored["cutoff"]) / pd.Timedelta(hours=1)
    assert np.allclose(censored["remaining_h"], hours_to_label_time)
    assert (before["remaining_h"] > 0).all()


def test_no_feature_is_built_from_an_outcome_column(con):
    frame = features.landmark_frame(con, CUTOFFS[:1])
    columns = features.Encoder().fit(frame).columns_
    assert not [c for c in columns if any(word in c.lower() for word in FORBIDDEN)]


def test_cutoff_must_precede_label_time(con):
    with pytest.raises(ValueError):
        features.landmark_frame(con, [LABEL_TIME], LABEL_TIME)


def test_protocol_keeps_training_before_test():
    assert evaluate.TRAIN_CUTOFFS.max() < evaluate.TRAIN_END <= evaluate.TEST_CUTOFFS.min()
    assert evaluate.SELECTION_FIT_CUTOFFS.max() < evaluate.SELECTION_FIT_END <= evaluate.SELECTION_CUTOFFS.min()
    # Selection uses 7-day outcomes that were complete by the end of the training period.
    assert evaluate.SELECTION_CUTOFFS.max() + pd.Timedelta(days=evaluate.LATE_DAYS) <= evaluate.TRAIN_END
