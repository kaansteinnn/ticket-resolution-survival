"""Turn point-in-time snapshots into model input.

`landmark_frame` stacks the tickets that are open at each cutoff. Every feature comes from
`open_at(cutoff)` (sql/03_snapshot.sql), which reads only updates up to the cutoff. The label
(how much longer the ticket stayed open) is read from a second snapshot at `label_time`, so a
training set never contains anything that was unknown at `label_time`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import db

# Far beyond the last update in the file: "as known at the very end" = the true outcome.
END_OF_TIME = pd.Timestamp("2100-01-01")

NUMERIC = [
    "log_age_h",
    "log_hours_since_update",
    "log_n_updates",
    "reassignment_count",
    "reopen_count",
    "n_groups_seen",
    "n_awaiting_updates",
    "priority_level",
    "impact_level",
    "urgency_level",
    "priority_confirmed",
    "has_assignee",
    "not_by_phone",
]
STATES = ["Active", "Awaiting User Info", "Awaiting other"]  # reference level: New
TOP_CATEGORIES = 8
TOP_GROUPS = 12


def landmark_frame(con, cutoffs, label_time=END_OF_TIME) -> pd.DataFrame:
    """Open tickets at each cutoff, with the remaining time as known at `label_time`.

    Columns added to the snapshot: `remaining_h` (hours from the cutoff to resolution, or to
    `label_time` if still open then) and `event` (True if resolved by `label_time`).
    """
    label_time = pd.Timestamp(label_time)
    known = db.snapshot(con, label_time).set_index("number")[["resolved", "resolved_time"]]
    frames = []
    for cutoff in pd.DatetimeIndex(cutoffs):
        if cutoff >= label_time:
            raise ValueError(f"cutoff {cutoff} is not before label_time {label_time}")
        snap = db.snapshot(con, cutoff, only_open=True).drop(columns=["resolved", "resolved_time"])
        outcome = known.loc[snap["number"]]
        end = outcome["resolved_time"].where(outcome["resolved"], label_time).to_numpy()
        snap["event"] = outcome["resolved"].to_numpy()
        snap["remaining_h"] = (end - np.datetime64(cutoff)) / np.timedelta64(1, "h")
        frames.append(snap)
    return pd.concat(frames, ignore_index=True)


def _level(series: pd.Series) -> pd.Series:
    """'3 - Moderate' -> 3."""
    return series.str.slice(0, 1).astype(int)


class Encoder:
    """Snapshot columns -> numeric matrix. Category levels are learned from the training rows only."""

    def fit(self, frame: pd.DataFrame) -> "Encoder":
        self.categories_ = frame["category"].value_counts().index[:TOP_CATEGORIES].tolist()
        self.groups_ = frame["assignment_group"].value_counts().index[:TOP_GROUPS].tolist()
        self.columns_ = self.transform(frame).columns.tolist()
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = pd.DataFrame(index=frame.index)
        x["log_age_h"] = np.log1p(frame["duration_h"])
        x["log_hours_since_update"] = np.log1p(frame["hours_since_update"])
        x["log_n_updates"] = np.log1p(frame["n_updates"])
        x["reassignment_count"] = frame["reassignment_count"]
        x["reopen_count"] = frame["reopen_count"]
        x["n_groups_seen"] = frame["n_groups_seen"]
        x["n_awaiting_updates"] = frame["n_awaiting_updates"]
        x["priority_level"] = _level(frame["priority"])
        x["impact_level"] = _level(frame["impact"])
        x["urgency_level"] = _level(frame["urgency"])
        x["priority_confirmed"] = frame["priority_confirmed"].astype(int)
        x["has_assignee"] = frame["has_assignee"].astype(int)
        x["not_by_phone"] = (frame["contact_type"] != "Phone").astype(int)

        state = frame["state"].where(frame["state"].isin(["New", "Active", "Awaiting User Info"]), "Awaiting other")
        for level in STATES:
            x[f"state={level}"] = (state == level).astype(int)
        for level in self.categories_:
            x[f"category={level}"] = (frame["category"] == level).astype(int)
        x["group=none"] = frame["assignment_group"].isna().astype(int)
        for level in self.groups_:
            x[f"group={level}"] = (frame["assignment_group"] == level).astype(int)
        return x.astype(float)
