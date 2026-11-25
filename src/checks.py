"""Checks on the model results: each one changes something a sound conclusion should survive.

Every check writes one table to reports/tables:

  check_open_tickets.csv   what throwing away the still-open tickets costs, with a same-size control
  check_without_team.csv   every method refitted and scored without the team that holds most late tickets
  check_new_tickets.csv    tickets that were never in a training queue, scored separately
  check_top_share.csv      precision among the riskiest 5%, 10% and 20%
  check_calibration.csv    predicted vs actual late share by team and by age
  check_retraining.csv     refitting every week of May instead of once on 1 May
  check_seeds.csv          the forest with other random seeds
  check_intervals.csv      95% intervals that resample tickets (all days of a ticket stay together)

Run: python -m src.checks   (fits the models first; `python run_all.py` reuses the main run)
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src import config, db, evaluate, features, models

OLDEST = models.OldestFirst.name
TEAM = models.TeamHistory.name
AGE_BAND = models.LateRateByAge.name
REGRESSION = models.ResolvedOnlyRegression.name
KM = models.KaplanMeierByAge.name
COX = models.Cox.name
FOREST = models.SurvivalForest.name
CLASSIFIER = models.LateClassifier.name

METRICS = ["c_index", "top10", "brier", "count_error"]
CONTROL_SEEDS = [1, 2, 3]
FOREST_SEEDS = [0, 1, 2, 3, 4]
RETRAIN_DATES = pd.date_range("2016-05-08", "2016-05-29", freq="7D")
SLIDE = (pd.Timestamp("2016-05-21"), pd.Timestamp("2016-05-26"))  # the days the forecasts drift most
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 20161125
# Differences reported with an interval: (method, compared with).
DIFFERENCES = [
    (FOREST, OLDEST), (FOREST, TEAM), (FOREST, COX), (FOREST, CLASSIFIER), (FOREST, KM),
    (KM, AGE_BAND), (KM, REGRESSION),
]


def month_means(test: pd.DataFrame, predictions: dict) -> pd.DataFrame:
    """Per method: the four measures, averaged over the daily queues."""
    daily = evaluate.score_by_day(test, predictions)
    out = daily.groupby("method", sort=False)[["c_index", "top10", "brier"]].mean()
    out["count_error"] = evaluate.count_error(daily)
    return out


def open_tickets(r: evaluate.Run) -> pd.DataFrame:
    """The same classifier, three training sets: what does throwing away the open tickets cost?

    "Only tickets already resolved" drops every training row whose ticket was still open on 1 May.
    The control drops the same number of rows at random, to separate "fewer rows" from "which rows".
    """
    reference = r.method(models.LateClassifier)
    _, late = models.known_late(r.train)
    known = reference.training_rows(r.train)
    resolved = models.LateClassifier(*r.classifier, rows="resolved").training_rows(r.train)
    variants = [("All rows with a known 7-day outcome", known), ("Only tickets already resolved", resolved)]
    for seed in CONTROL_SEEDS:
        keep = np.zeros(len(known), dtype=bool)
        keep[np.random.default_rng(seed).choice(np.flatnonzero(known), size=int(resolved.sum()), replace=False)] = True
        variants.append((f"Same number of rows, dropped at random (seed {seed})", keep))
    rows = []
    for label, keep in variants:
        model = models.LateClassifier(*r.classifier).fit(r.train, r.x_train, keep=keep)
        means = month_means(r.test, {"m": model.predict(r.test, r.x_test)}).loc["m"]
        rows.append({"training_rows": label, "rows": int(keep.sum()), "late_share_in_training": late[keep].mean(),
                     **means.to_dict()})
    return pd.DataFrame(rows)


def without_team(r: evaluate.Run, con) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Take the biggest late team out of training and test, refit everything, score again."""
    team = r.method(models.TeamHistory).biggest_late_team_
    train = r.train[r.train["assignment_group"] != team].reset_index(drop=True)
    test = r.test[r.test["assignment_group"] != team].reset_index(drop=True)
    encoder, fitted = evaluate.fit_methods(con, train, evaluate.TRAIN_END, r.leaf_size, r.classifier, without_team=team)
    x_test = encoder.transform(test)
    predictions = {m.name: m.predict(test, x_test) for m in fitted}
    table = month_means(test, predictions).reset_index()
    table.insert(1, "actual_late_mean", pd.Series(evaluate.is_late(test)).groupby(test["cutoff"]).sum().mean())
    table.insert(1, "test_tickets", test["number"].nunique())
    table.insert(1, "training_rows", len(train))
    return table, test, predictions


def new_tickets(r: evaluate.Run) -> pd.DataFrame:
    """A ticket open on 29 April and still open on 3 May is in both periods. Score the others separately."""
    seen = r.test["number"].isin(r.train["number"]).to_numpy()
    frames = []
    for label, mask in [("Never in a training queue", ~seen), ("Also in a training queue", seen)]:
        part = r.test[mask].reset_index(drop=True)
        table = month_means(part, evaluate.subset(r.predictions, mask)).reset_index()
        table.insert(0, "tickets", label)
        table.insert(2, "n_tickets", part["number"].nunique())
        table.insert(3, "rows", len(part))
        table.insert(4, "late_share", evaluate.is_late(part).mean())
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def top_share(r: evaluate.Run) -> pd.DataFrame:
    """Share of truly late tickets among the riskiest 5%, 10% and 20% of each daily queue."""
    columns = {}
    for share in (0.05, 0.10, 0.20):
        daily = evaluate.score_by_day(r.test, r.predictions, top_share=share)
        columns[f"top{round(share * 100)}"] = daily.groupby("method", sort=False)["top10"].mean()
    return pd.DataFrame(columns).reset_index()


def calibration(r: evaluate.Run, n_teams: int = 10) -> pd.DataFrame:
    """The forest's predicted late share against the actual one, by team and by age (all May rows pooled)."""
    frame = pd.DataFrame(
        {
            "number": r.test["number"],
            "team": r.test["assignment_group"].fillna("(none)"),
            "age_days": pd.cut(r.test["duration_h"] / 24, [0, 1, 3, 7, 14, 28, 60, np.inf], right=False),
            "predicted_late": 1 - r.predictions[FOREST][1][:, evaluate.LATE_IDX],
            "actual_late": evaluate.is_late(r.test),
        }
    )
    parts = []
    for by in ("team", "age_days"):
        g = frame.groupby(by, observed=True).agg(
            rows=("number", "size"), tickets=("number", "nunique"),
            predicted_late=("predicted_late", "mean"), actual_late=("actual_late", "mean"),
        )
        if by == "team":
            g = g.sort_values("rows", ascending=False).head(n_teams)
        g.index = g.index.astype(str)
        parts.append(g.reset_index(names="group").assign(by=by))
    out = pd.concat(parts, ignore_index=True)
    return out[["by", "group", "rows", "tickets", "predicted_late", "actual_late"]]


def retraining(r: evaluate.Run, con) -> pd.DataFrame:
    """Refit every method each Monday-to-Sunday week of May on what is known that day; compare with fitting once."""
    weekly = {name: (s.astype(float).copy(), None if p is None else p.copy()) for name, (s, p) in r.predictions.items()}
    for end in RETRAIN_DATES:
        start = time.time()
        cutoffs = pd.date_range(evaluate.TRAIN_CUTOFFS[0], end - pd.Timedelta(days=1), freq="3D")
        train = features.landmark_frame(con, cutoffs, end)
        encoder, fitted = evaluate.fit_methods(con, train, end, r.leaf_size, r.classifier)
        mask = ((r.test["cutoff"] >= end) & (r.test["cutoff"] < end + pd.Timedelta(days=7))).to_numpy()
        week = r.test[mask]
        x_week = encoder.transform(week)
        for m in fitted:
            slowness, prob = m.predict(week, x_week)
            weekly[m.name][0][mask] = slowness
            if prob is not None:
                weekly[m.name][1][mask] = prob
        print(f"  refit on {end.date()}: {len(train):,} rows ({time.time() - start:.0f}s)")
    in_slide = ((r.test["cutoff"] >= SLIDE[0]) & (r.test["cutoff"] <= SLIDE[1])).to_numpy()
    frames = []
    for period, mask in [("1-31 May", np.ones(len(r.test), dtype=bool)), ("21-26 May", in_slide)]:
        part = r.test[mask].reset_index(drop=True)
        for training, predictions in [("Once, on 1 May", r.predictions), ("Every week", weekly)]:
            table = month_means(part, evaluate.subset(predictions, mask)).reset_index()
            table.insert(0, "period", period)
            table.insert(1, "training", training)
            frames.append(table)
    return pd.concat(frames, ignore_index=True)


def seeds(r: evaluate.Run) -> pd.DataFrame:
    """The forest is random; how much do its results move with the seed?"""
    rows = []
    for seed in FOREST_SEEDS:
        if seed == 0:
            prediction = r.predictions[FOREST]
        else:
            forest = models.SurvivalForest(min_samples_leaf=r.leaf_size, random_state=seed).fit(r.train, r.x_train)
            prediction = forest.predict(r.test, r.x_test)
        rows.append({"seed": seed, **month_means(r.test, {FOREST: prediction}).loc[FOREST].to_dict()})
    return pd.DataFrame(rows)


def weighted_day(remaining_h: np.ndarray, slowness: np.ndarray, prob_resolved, w: np.ndarray) -> dict:
    """The measures of evaluate.queue_metrics for one queue in which ticket i counts w[r, i] times.

    `w` has one row per replicate; a row of ones gives the ordinary measures (tests/test_checks.py).
    Every outcome is observed, so two tickets are comparable whenever their remaining times differ.
    """
    late = (remaining_h > evaluate.LATE_H).astype(float)
    comparable = (remaining_h[:, None] < remaining_h[None, :]).astype(float)
    agree = comparable * ((slowness[:, None] < slowness[None, :]) + 0.5 * (slowness[:, None] == slowness[None, :]))
    size = w.sum(axis=1)
    out = {"c_index": ((w @ agree) * w).sum(axis=1) / ((w @ comparable) * w).sum(axis=1)}
    risk = slowness if prob_resolved is None else 1 - prob_resolved[:, evaluate.LATE_IDX]
    order = np.argsort(-risk, kind="stable")
    k = np.maximum(1.0, np.round(evaluate.TOP_SHARE * size))
    w_sorted = w[:, order]
    taken = np.clip(k[:, None] - (np.cumsum(w_sorted, axis=1) - w_sorted), 0, w_sorted)
    out["top10"] = (taken @ late[order]) / k
    if prob_resolved is None:
        out["brier"] = out["count_error"] = np.full(len(w), np.nan)
    else:
        p_late = 1 - prob_resolved[:, evaluate.LATE_IDX]
        out["brier"] = (w @ (p_late - late) ** 2) / size
        out["count_error"] = np.abs(w @ p_late - w @ late)
    return out


def bootstrap(test: pd.DataFrame, predictions: dict, n_rep: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> dict:
    """Resample tickets with replacement and recompute every measure; returns {(method, metric): n_rep values}.

    A ticket is in the queue for several days, so its days are drawn together: a ticket drawn twice
    counts twice on each of its days. The models are not refitted.
    """
    tickets, ticket_of_row = np.unique(test["number"].to_numpy(), return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(tickets), np.full(len(tickets), 1 / len(tickets)), size=n_rep).astype(float)
    remaining = test["remaining_h"].to_numpy()
    out = {(name, metric): np.zeros(n_rep) for name in predictions for metric in METRICS}
    days = test.groupby("cutoff").indices
    for idx in days.values():
        w = counts[:, ticket_of_row[idx]]
        for name, (slowness, prob) in predictions.items():
            day = weighted_day(remaining[idx], slowness[idx], None if prob is None else prob[idx], w)
            for metric in METRICS:
                out[name, metric] += day[metric]
    return {key: values / len(days) for key, values in out.items()}


def intervals(test: pd.DataFrame, predictions: dict, scope: str) -> pd.DataFrame:
    """Point value and 95% ticket-bootstrap interval for every method and for the listed differences."""
    point = month_means(test, predictions)
    boot = bootstrap(test, predictions)
    rows = []

    def add(label, metric, value, draws):
        if np.isnan(value):
            return
        lo, hi = np.percentile(draws, [2.5, 97.5])
        rows.append({"scope": scope, "quantity": label, "metric": metric, "value": value, "lo": lo, "hi": hi})

    for name in predictions:
        for metric in METRICS:
            add(name, metric, point.loc[name, metric], boot[name, metric])
    for a, b in DIFFERENCES:
        for metric in METRICS:
            add(f"{a} minus {b}", metric, point.loc[a, metric] - point.loc[b, metric], boot[a, metric] - boot[b, metric])
    return pd.DataFrame(rows)


def main(r: evaluate.Run | None = None) -> None:
    print("Checks on the model results")
    con = db.connect()
    if r is None:
        r = evaluate.run(con)
    out = config.TABLES_DIR
    out.mkdir(parents=True, exist_ok=True)

    def write(name: str, table: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
        table.round(digits).to_csv(out / f"check_{name}.csv", index=False)
        print(f" {name}")
        print(table.round(3).to_string(index=False))
        return table

    write("open_tickets", open_tickets(r), digits=6)
    team_table, team_test, team_predictions = without_team(r, con)
    write("without_team", team_table)
    write("new_tickets", new_tickets(r))
    write("top_share", top_share(r))
    write("calibration", calibration(r))
    write("retraining", retraining(r, con))
    write("seeds", seeds(r))
    con.close()
    start = time.time()
    team = r.method(models.TeamHistory).biggest_late_team_
    both = pd.concat(
        [intervals(r.test, r.predictions, "All tickets"), intervals(team_test, team_predictions, f"Without {team}")],
        ignore_index=True,
    )
    write("intervals", both)
    print(f" intervals took {time.time() - start:.0f}s")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    main()
