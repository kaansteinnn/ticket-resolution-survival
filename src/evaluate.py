"""Fit the methods on what was known at the end of April and score them on every day of May.

Protocol (fixed before looking at May):
  training  - tickets open at cutoffs every 3 days from 3 March to 29 April; how long each stayed
              open is taken as known on 1 May (still open then = censored)
  selection - the forest's leaf size is chosen inside the training period: fit on cutoffs up to
              9 April as known on 17 April, compare on the daily queues of 17-23 April
  test      - the queue of open tickets at 00:00 on each of the 31 days of May; the true outcome is
              read from the rest of the file
  "late"    - not resolved within 7 days of the cutoff

Measures, computed per daily queue and then summarised over the 31 days:
  c_index   - of two tickets in the queue, how often the one ranked slower really took longer
  top10     - share of truly late tickets among the 10% ranked riskiest
  brier     - mean squared error of P(late); lower is better (methods that only rank have none)

Run: python -m src.evaluate
"""

from __future__ import annotations

import time

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

LATE_DAYS = 7
LATE_H = 24.0 * LATE_DAYS
LATE_IDX = LATE_DAYS - 1  # position of the 7-day horizon in models.GRID_H
TOP_SHARE = 0.10

PREDICTIONS_PATH = config.ROOT / "app" / "data" / "predictions.parquet"


def is_late(frame: pd.DataFrame) -> np.ndarray:
    """Not resolved within 7 days of the cutoff. Requires 7 days of follow-up for censored rows."""
    if ((~frame["event"]) & (frame["remaining_h"] < LATE_H)).any():
        raise ValueError("some tickets are censored before the 7-day horizon")
    return (frame["remaining_h"] > LATE_H).to_numpy()


def queue_metrics(frame: pd.DataFrame, slowness: np.ndarray, prob_resolved) -> dict:
    """Measures for one day's queue."""
    late = is_late(frame)
    c_index = concordance_index_censored(
        frame["event"].to_numpy(), frame["remaining_h"].to_numpy(), -slowness
    )[0]
    # "Riskiest" = highest P(late) when the method gives one, otherwise its ranking score.
    # Ties keep row order (tickets are sorted by number, i.e. older first).
    risk = slowness if prob_resolved is None else 1.0 - prob_resolved[:, LATE_IDX]
    top = np.argsort(-risk, kind="stable")[: max(1, int(round(TOP_SHARE * len(frame))))]
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


def score_by_day(frame: pd.DataFrame, predictions: dict) -> pd.DataFrame:
    rows = []
    for cutoff, idx in frame.groupby("cutoff").indices.items():
        day = frame.iloc[idx]
        for name, (slowness, prob) in predictions.items():
            m = queue_metrics(day, slowness[idx], None if prob is None else prob[idx])
            rows.append({"cutoff": pd.Timestamp(cutoff).date(), "method": name, **m})
    return pd.DataFrame(rows)


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


def fit_methods(con, frame: pd.DataFrame, known_at: pd.Timestamp, leaf_size: int):
    """Fit every method on `frame`, using only what was known at `known_at`."""
    encoder = features.Encoder().fit(frame)
    x = encoder.transform(frame)
    tickets = db.snapshot(con, known_at)
    fitted = [
        models.OldestFirst().fit(frame, x),
        models.TeamHistory().fit(frame, x),
        models.ResolvedOnlyRegression().fit(frame, x),
        models.KaplanMeierByAge().fit_tickets(tickets["duration_h"].to_numpy(), tickets["resolved"].to_numpy()),
        models.Cox().fit(frame, x),
        models.SurvivalForest(min_samples_leaf=leaf_size).fit(frame, x),
    ]
    return encoder, fitted


def select_leaf_size(con) -> tuple[int, pd.DataFrame]:
    """Choose the forest's leaf size on April data only (lowest Brier score on the daily queues)."""
    fit_frame = features.landmark_frame(con, SELECTION_FIT_CUTOFFS, SELECTION_FIT_END)
    val_frame = features.landmark_frame(con, SELECTION_CUTOFFS, TRAIN_END)
    encoder = features.Encoder().fit(fit_frame)
    x_fit, x_val = encoder.transform(fit_frame), encoder.transform(val_frame)
    rows = []
    for leaf in LEAF_SIZES:
        start = time.time()
        forest = models.SurvivalForest(min_samples_leaf=leaf).fit(fit_frame, x_fit)
        daily = score_by_day(val_frame, {"forest": forest.predict(val_frame, x_val)})
        rows.append(
            {
                "min_samples_leaf": leaf,
                "brier": daily["brier"].mean(),
                "c_index": daily["c_index"].mean(),
                "top10": daily["top10"].mean(),
                "fit_seconds": round(time.time() - start),
            }
        )
        print(f"  leaf {leaf:4d}: brier {rows[-1]['brier']:.4f}  c-index {rows[-1]['c_index']:.3f}  ({rows[-1]['fit_seconds']}s)")
    table = pd.DataFrame(rows)
    best = int(table.loc[table["brier"].idxmin(), "min_samples_leaf"])
    return best, table


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


def main() -> None:
    print("Model: fit on data known at the end of April, test on every day of May")
    con = db.connect()

    print(" selecting forest leaf size on April data")
    leaf_size, selection = select_leaf_size(con)
    print(f"  chosen: {leaf_size}")

    train = features.landmark_frame(con, TRAIN_CUTOFFS, TRAIN_END)
    test = features.landmark_frame(con, TEST_CUTOFFS)
    print(
        f" training rows {len(train):,} ({train['number'].nunique():,} tickets, "
        f"{(~train['event']).mean():.0%} still open on 1 May); "
        f"test rows {len(test):,} ({test['number'].nunique():,} tickets)"
    )

    start = time.time()
    encoder, fitted = fit_methods(con, train, TRAIN_END, leaf_size)
    x_test = encoder.transform(test)
    predictions = {m.name: m.predict(test, x_test) for m in fitted}
    print(f" fitted and predicted in {time.time() - start:.0f}s")
    con.close()

    daily = score_by_day(test, predictions)
    summary = summarise(daily, reference=models.OldestFirst.name)

    # How much of the result is one team? Repeat the scoring without the team that held the
    # largest backlog of late tickets in the training period.
    team_rule = next(m for m in fitted if isinstance(m, models.TeamHistory))
    rest = (test["assignment_group"] != team_rule.biggest_late_team_).to_numpy()
    daily_rest = score_by_day(
        test[rest].reset_index(drop=True),
        {name: (s[rest], None if p is None else p[rest]) for name, (s, p) in predictions.items()},
    )
    summary_rest = summarise(daily_rest, reference=models.OldestFirst.name)
    calibration = calibration_table(test, predictions)
    cox = next(m for m in fitted if isinstance(m, models.Cox))
    regression = next(m for m in fitted if isinstance(m, models.ResolvedOnlyRegression))

    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    selection.round(4).to_csv(config.TABLES_DIR / "model_selection.csv", index=False)
    daily.round(4).to_csv(config.TABLES_DIR / "model_daily.csv", index=False)
    summary.round(4).to_csv(config.TABLES_DIR / "model_summary.csv", index=False)
    summary_rest.round(4).to_csv(config.TABLES_DIR / "model_summary_without_biggest_late_team.csv", index=False)
    calibration.round(4).to_csv(config.TABLES_DIR / "model_calibration.csv", index=False)
    cox.coefficients().round(4).to_csv(config.TABLES_DIR / "cox_coefficients.csv", index=False)

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    demo_table(test, predictions[models.SurvivalForest.name][1]).to_parquet(PREDICTIONS_PATH, index=False)

    print(f" regression used {regression.n_used_:,} resolved rows and dropped {regression.n_dropped_:,} open ones")
    print(f" share of late tickets in the May queues: {daily['late_share'].mean():.3f}")
    cols = ["method", "c_index_mean", "c_index_min", "c_index_max", "top10_mean", "top10_min", "top10_max",
            "brier_mean", "days_c_index_above_reference", "days_top10_above_reference"]
    print(summary[cols].round(3).to_string(index=False))
    print(
        f" team with most late tickets in training: {team_rule.biggest_late_team_} "
        f"({team_rule.biggest_late_team_rate_:.0%} late); "
        f"{1 - rest.mean():.1%} of the May queue rows. Without it "
        f"(late share {daily_rest['late_share'].mean():.3f}):"
    )
    print(summary_rest[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
