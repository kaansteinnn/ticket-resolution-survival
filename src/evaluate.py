"""Fit the methods on what was known at the end of April and score them on every day of May.

Protocol:
  training  - tickets open at cutoffs every 3 days from 3 March to 29 April; how long each stayed
              open is taken as known on 1 May (still open then = censored)
  selection - the settings of the forest and of the classifier are chosen inside the training
              period: fit on cutoffs up to 9 April as known on 17 April, compare on the daily
              queues of 17-23 April
  test      - the queue of open tickets at 00:00 on each of the 31 days of May; the true outcome is
              read from the rest of the file
  "late"    - not resolved within 7 days of the cutoff (counted to the ticket's final resolution)

A ticket stays in the queue for several days, so the 31 daily queues are not 31 independent
samples; src/checks.py gives intervals that resample tickets.

Measures, computed per daily queue and then summarised over the 31 days:
  c_index   - of two tickets in the queue, how often the one ranked slower really took longer
  top10     - share of truly late tickets among the 10% ranked riskiest
  brier     - mean squared error of P(late); lower is better (methods that only rank have none)

Run: python -m src.evaluate
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored

from src import config, db, features, models

TRAIN_END = pd.Timestamp("2016-05-01")
TRAIN_CUTOFFS = pd.date_range("2016-03-03", "2016-04-29", freq="3D")
TEST_CUTOFFS = pd.date_range("2016-05-01", "2016-05-31", freq="D")

SELECTION_FIT_END = pd.Timestamp("2016-04-17")
SELECTION_FIT_CUTOFFS = pd.date_range("2016-03-03", "2016-04-09", freq="3D")
SELECTION_CUTOFFS = pd.date_range("2016-04-17", "2016-04-23", freq="D")
LEAF_SIZES = [60, 120, 240, 480]
CLASSIFIER_GRID = [(leaves, leaf) for leaves in (7, 15, 31) for leaf in (60, 120, 240)]

LATE_DAYS = 7
LATE_H = models.LATE_H
LATE_IDX = models.LATE_IDX
TOP_SHARE = 0.10
PAST_SHARE_ALL = "Late share of all past queues"
PAST_SHARE_LATEST = "Late share of the latest past queue"

PREDICTIONS_PATH = config.ROOT / "app" / "data" / "predictions.parquet"


def is_late(frame: pd.DataFrame) -> np.ndarray:
    """Not resolved within 7 days of the cutoff. Requires 7 days of follow-up for censored rows."""
    if ((~frame["event"]) & (frame["remaining_h"] < LATE_H)).any():
        raise ValueError("some tickets are censored before the 7-day horizon")
    return (frame["remaining_h"] > LATE_H).to_numpy()


def queue_metrics(frame: pd.DataFrame, slowness: np.ndarray, prob_resolved, top_share: float = TOP_SHARE) -> dict:
    """Measures for one day's queue."""
    late = is_late(frame)
    c_index = concordance_index_censored(
        frame["event"].to_numpy(), frame["remaining_h"].to_numpy(), -slowness
    )[0]
    # "Riskiest" = highest P(late) when the method gives one, otherwise its ranking score.
    # Ties keep row order (tickets are sorted by number, i.e. older first).
    risk = slowness if prob_resolved is None else 1.0 - prob_resolved[:, LATE_IDX]
    top = np.argsort(-risk, kind="stable")[: max(1, int(round(top_share * len(frame))))]
    out = {
        "tickets": len(frame),
        "late_share": late.mean(),
        "c_index": c_index,
        "top10": late[top].mean(),
        "brier": np.nan,
        "mean_p_late": np.nan,
    }
    if prob_resolved is not None:
        p_late = 1.0 - prob_resolved[:, LATE_IDX]
        out["brier"] = np.mean((p_late - late) ** 2)
        out["mean_p_late"] = p_late.mean()
    return out


def score_by_day(frame: pd.DataFrame, predictions: dict, top_share: float = TOP_SHARE) -> pd.DataFrame:
    rows = []
    for cutoff, idx in frame.groupby("cutoff").indices.items():
        day = frame.iloc[idx]
        for name, (slowness, prob) in predictions.items():
            m = queue_metrics(day, slowness[idx], None if prob is None else prob[idx], top_share)
            rows.append({"cutoff": pd.Timestamp(cutoff).date(), "method": name, **m})
    return pd.DataFrame(rows)


def subset(predictions: dict, mask: np.ndarray) -> dict:
    return {name: (s[mask], None if p is None else p[mask]) for name, (s, p) in predictions.items()}


def count_error(daily: pd.DataFrame) -> pd.Series:
    """Per method: mean over the days of |forecast number of late tickets - actual number|."""
    error = (daily["tickets"] * (daily["mean_p_late"] - daily["late_share"])).abs()
    return error.groupby(daily["method"], sort=False).mean()


def summarise(daily: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Mean and range over the daily queues, and on how many days each method beats the reference."""
    order = daily["method"].drop_duplicates().tolist()
    ref = daily[daily["method"] == reference].set_index("cutoff")
    rows = []
    for name in order:
        d = daily[daily["method"] == name].set_index("cutoff")
        row = {"method": name, "days": len(d)}
        for metric in ["c_index", "top10", "brier"]:
            row[f"{metric}_mean"] = d[metric].mean()
            row[f"{metric}_min"] = d[metric].min()
            row[f"{metric}_max"] = d[metric].max()
        row["days_c_index_above_reference"] = int((d["c_index"] > ref["c_index"]).sum())
        row["days_top10_above_reference"] = int((d["top10"] > ref["top10"]).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def past_late_shares(train: pd.DataFrame) -> dict:
    """The no-model forecast: the share of past queues that was late, applied to today's queue size.

    Only training queues whose 7-day outcome was already known on 1 May can be used.
    """
    known = train[train["cutoff"] <= TRAIN_END - pd.Timedelta(days=LATE_DAYS)]
    late = pd.Series(is_late(known), index=known["cutoff"].to_numpy())
    return {PAST_SHARE_ALL: late.mean(), PAST_SHARE_LATEST: late.groupby(level=0).mean().iloc[-1]}


def workload_table(daily: pd.DataFrame, past_shares: dict) -> pd.DataFrame:
    """How many tickets of each daily queue will be late: every forecast against the actual count.

    A model's forecast for a day is the sum of its late probabilities over that day's queue; the
    no-model forecasts multiply the queue size by a late share measured before May. Methods that
    only rank tickets give no forecast.
    """
    days = daily.drop_duplicates("cutoff")
    actual = (days["tickets"] * days["late_share"]).to_numpy()
    forecasts = {name: (share, days["tickets"].to_numpy() * share) for name, share in past_shares.items()}
    for name, d in daily.dropna(subset=["mean_p_late"]).groupby("method", sort=False):
        forecasts[name] = (np.nan, (d["tickets"] * d["mean_p_late"]).to_numpy())
    rows = []
    for name, (share, forecast) in forecasts.items():
        error = forecast - actual
        rows.append(
            {
                "method": name,
                "days": len(error),
                "fixed_late_share": share,
                "actual_late_mean": actual.mean(),
                "forecast_late_mean": forecast.mean(),
                "abs_error_mean": np.abs(error).mean(),
                "abs_error_min": np.abs(error).min(),
                "abs_error_max": np.abs(error).max(),
                "days_forecast_too_low": int((error < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def calibration_table(frame: pd.DataFrame, predictions: dict, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed share resolved within 7 days, in tenths of the predictions (all test days pooled)."""
    resolved_in_time = ~is_late(frame)
    rows = []
    for name, (_, prob) in predictions.items():
        if prob is None:
            continue
        p = prob[:, LATE_IDX]
        group = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False)
        for g in range(bins):
            mask = (group == g).to_numpy()
            rows.append(
                {
                    "method": name,
                    "bin": g + 1,
                    "rows": int(mask.sum()),
                    "predicted": p[mask].mean(),
                    "observed": resolved_in_time[mask].mean(),
                }
            )
    return pd.DataFrame(rows)


def fit_methods(con, frame: pd.DataFrame, known_at: pd.Timestamp, leaf_size: int, classifier: tuple[int, int],
                without_team: str | None = None):
    """Fit every method on `frame`, using only what was known at `known_at`.

    Kaplan-Meier has no ticket attributes and is fitted on one row per ticket visible at `known_at`.
    `without_team` leaves that team's tickets out of it (the caller removes them from `frame`).
    """
    encoder = features.Encoder().fit(frame)
    x = encoder.transform(frame)
    tickets = db.snapshot(con, known_at)
    if without_team is not None:
        tickets = tickets[tickets["assignment_group"] != without_team]
    fitted = [
        models.OldestFirst().fit(frame, x),
        models.TeamHistory().fit(frame, x),
        models.LateRateByAge().fit(frame, x),
        models.ResolvedOnlyRegression().fit(frame, x),
        models.KaplanMeierByAge().fit_tickets(tickets["duration_h"].to_numpy(), tickets["resolved"].to_numpy()),
        models.Cox().fit(frame, x),
        models.SurvivalForest(min_samples_leaf=leaf_size).fit(frame, x),
        models.LateClassifier(*classifier).fit(frame, x),
    ]
    return encoder, fitted


def select_settings(con) -> tuple[int, tuple[int, int], pd.DataFrame, pd.DataFrame]:
    """Choose the forest's leaf size and the classifier's tree size on April data only.

    Both are chosen by the lowest Brier score on the daily queues of 17-23 April.
    """
    fit_frame = features.landmark_frame(con, SELECTION_FIT_CUTOFFS, SELECTION_FIT_END)
    val_frame = features.landmark_frame(con, SELECTION_CUTOFFS, TRAIN_END)
    encoder = features.Encoder().fit(fit_frame)
    x_fit, x_val = encoder.transform(fit_frame), encoder.transform(val_frame)

    def score(model) -> dict:
        daily = score_by_day(val_frame, {"m": model.fit(fit_frame, x_fit).predict(val_frame, x_val)})
        return {"brier": daily["brier"].mean(), "c_index": daily["c_index"].mean(), "top10": daily["top10"].mean()}

    rows = []
    for leaf in LEAF_SIZES:
        start = time.time()
        rows.append({"min_samples_leaf": leaf, **score(models.SurvivalForest(min_samples_leaf=leaf))})
        print(f"  forest leaf {leaf:4d}: brier {rows[-1]['brier']:.4f}  c-index {rows[-1]['c_index']:.3f}  ({time.time() - start:.0f}s)")
    forest = pd.DataFrame(rows)
    rows = [
        {"max_leaf_nodes": leaves, "min_samples_leaf": leaf, **score(models.LateClassifier(leaves, leaf))}
        for leaves, leaf in CLASSIFIER_GRID
    ]
    classifier = pd.DataFrame(rows)
    best = classifier.loc[classifier["brier"].idxmin()]
    return (
        int(forest.loc[forest["brier"].idxmin(), "min_samples_leaf"]),
        (int(best["max_leaf_nodes"]), int(best["min_samples_leaf"])),
        forest,
        classifier,
    )


def demo_table(test: pd.DataFrame, prob_resolved: np.ndarray) -> pd.DataFrame:
    """What the demo shows: each May queue with the forest's predictions and the hidden outcome."""
    out = pd.DataFrame(
        {
            "cutoff": test["cutoff"].dt.date.astype(str),
            "number": test["number"],
            "opened_at": test["opened_at"],
            "age_days": (test["duration_h"] / 24).round(2),
            "days_since_update": (test["hours_since_update"] / 24).round(2),
            "state": test["state"],
            "priority": test["priority"],
            "category": test["category"],
            "assignment_group": test["assignment_group"].fillna("(none)"),
            "reassignment_count": test["reassignment_count"],
            "n_updates": test["n_updates"],
            "p_late": (1 - prob_resolved[:, LATE_IDX]).round(4),
            "actual_remaining_days": (test["remaining_h"] / 24).round(2),
            "actual_late": is_late(test),
        }
    )
    curve = pd.DataFrame(
        prob_resolved.round(4).astype("float32"),
        columns=[f"p_resolved_d{d:02d}" for d in range(1, prob_resolved.shape[1] + 1)],
        index=out.index,
    )
    return pd.concat([out, curve], axis=1)


@dataclass
class Run:
    """Everything the main run produced; src/checks.py starts from it instead of refitting."""

    leaf_size: int
    classifier: tuple[int, int]
    selection: pd.DataFrame
    classifier_selection: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    fitted: list
    predictions: dict
    daily: pd.DataFrame

    def method(self, kind):
        return next(m for m in self.fitted if isinstance(m, kind))


def run(con) -> Run:
    print(" selecting forest and classifier settings on April data")
    leaf_size, classifier, selection, classifier_selection = select_settings(con)
    print(f"  chosen: forest leaf {leaf_size}; classifier {classifier[0]} leaves, at least {classifier[1]} rows each")

    train = features.landmark_frame(con, TRAIN_CUTOFFS, TRAIN_END)
    test = features.landmark_frame(con, TEST_CUTOFFS)
    print(
        f" training rows {len(train):,} ({train['number'].nunique():,} tickets, "
        f"{(~train['event']).mean():.0%} still open on 1 May); "
        f"test rows {len(test):,} ({test['number'].nunique():,} tickets)"
    )
    start = time.time()
    encoder, fitted = fit_methods(con, train, TRAIN_END, leaf_size, classifier)
    x_test = encoder.transform(test)
    predictions = {m.name: m.predict(test, x_test) for m in fitted}
    print(f" fitted and predicted in {time.time() - start:.0f}s")
    return Run(leaf_size, classifier, selection, classifier_selection, train, test, encoder.transform(train),
               x_test, fitted, predictions, score_by_day(test, predictions))


def main() -> Run:
    print("Model: fit on data known at the end of April, test on every day of May")
    con = db.connect()
    r = run(con)
    con.close()
    test, predictions, daily = r.test, r.predictions, r.daily
    summary = summarise(daily, reference=models.OldestFirst.name)

    team_rule = r.method(models.TeamHistory)
    in_team = (test["assignment_group"] == team_rule.biggest_late_team_).to_numpy()
    calibration = calibration_table(test, predictions)
    regression = r.method(models.ResolvedOnlyRegression)

    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    r.selection.round(4).to_csv(config.TABLES_DIR / "model_selection.csv", index=False)
    r.classifier_selection.round(4).to_csv(config.TABLES_DIR / "model_selection_classifier.csv", index=False)
    daily.round(4).to_csv(config.TABLES_DIR / "model_daily.csv", index=False)
    summary.round(3).to_csv(config.TABLES_DIR / "model_summary.csv", index=False)
    calibration.round(4).to_csv(config.TABLES_DIR / "model_calibration.csv", index=False)
    workload = workload_table(daily, past_late_shares(r.train))
    decimals = {c: 3 if c == "fixed_late_share" else 1 for c in workload.columns if c not in ("method", "days")}
    workload.round(decimals).to_csv(config.TABLES_DIR / "model_workload.csv", index=False)
    r.method(models.Cox).coefficients().round(4).to_csv(config.TABLES_DIR / "cox_coefficients.csv", index=False)

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    demo_table(test, predictions[models.SurvivalForest.name][1]).to_parquet(PREDICTIONS_PATH, index=False)

    print(f" regression used {regression.n_used_:,} resolved rows and dropped {regression.n_dropped_:,} open ones")
    print(f" classifier used {r.method(models.LateClassifier).n_used_:,} rows whose 7-day outcome was known")
    print(f" share of late tickets in the May queues: {daily['late_share'].mean():.3f}")
    cols = ["method", "c_index_mean", "c_index_min", "c_index_max", "top10_mean", "top10_min", "top10_max",
            "brier_mean", "days_c_index_above_reference", "days_top10_above_reference"]
    print(summary[cols].round(3).to_string(index=False))
    print(workload.round(1).to_string(index=False))
    print(
        f" team with most late tickets in training: {team_rule.biggest_late_team_} "
        f"({team_rule.biggest_late_team_rate_:.0%} late); {in_team.mean():.1%} of the May queue rows, "
        f"{is_late(test)[in_team].mean():.1%} of them late"
    )
    return r


if __name__ == "__main__":
    main()
