"""The methods that are compared, behind one interface.

Each method is fitted on a landmark frame (see src/features.py) and, for a set of open tickets, returns

  slowness       - a score; higher means "expected to stay open longer" (used for ranking)
  prob_resolved  - P(resolved within h hours of the cutoff) for each h in GRID_H, or None if the
                   method only ranks

Simple methods:
  OldestFirst         - the rule of thumb: the ticket that has been open longest is the riskiest
  TeamHistory         - one step smarter: rank by how often the ticket's team was late so far,
                        oldest first within a team
  LateRateByAge       - no model: how often past tickets of the same age were late
  ResolvedOnlyRegression - linear regression of log remaining time, fitted on the tickets that were
                        resolved by the training date (the still-open ones are thrown away)
  KaplanMeierByAge    - no ticket attributes at all: the overall Kaplan-Meier curve, conditioned on age

Survival models (use the still-open tickets as censored observations):
  Cox                 - proportional hazards, interpretable coefficients (lifelines)
  SurvivalForest      - random survival forest (scikit-survival)

An ordinary classifier, to see whether a survival model is needed at all:
  LateClassifier      - gradient boosting on the yes/no question "late or not", fitted on the rows
                        whose 7-day outcome was already known at training time

"Late" = not resolved within 7 days of the cutoff. A row's 7-day outcome is known at training time
when the ticket could be watched for 7 days (`followup_h` >= 168); an open ticket then counts as late.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

# Horizons at which every probabilistic method reports P(resolved): 1..30 days after the cutoff.
GRID_H = 24.0 * np.arange(1, 31)

LATE_H = 168.0  # "late" = not resolved within 7 days of the cutoff
LATE_IDX = 6  # position of the 7-day horizon in GRID_H

MIN_HOURS = 1 / 60  # timestamps have minute resolution


def known_late(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(rows whose 7-day outcome is known at training time, whether each row is late)."""
    known = (frame["followup_h"] >= LATE_H).to_numpy()
    late = ((~frame["event"]) | (frame["remaining_h"] > LATE_H)).to_numpy()
    return known, late


def only_late_horizon(p_late: np.ndarray) -> np.ndarray:
    """Probability table for a method that answers the 7-day question only (other horizons empty)."""
    prob = np.full((len(p_late), len(GRID_H)), np.nan)
    prob[:, LATE_IDX] = 1.0 - p_late
    return prob


class OldestFirst:
    name = "Oldest first"

    def fit(self, frame: pd.DataFrame, x: pd.DataFrame) -> "OldestFirst":
        return self

    def predict(self, frame: pd.DataFrame, x: pd.DataFrame):
        return frame["duration_h"].to_numpy(), None


class TeamHistory:
    """Rank by how often the ticket's team was late so far; oldest first within a team.

    A team's late rate is the share of its training rows not resolved within 7 days of the cutoff,
    using only rows whose 7-day outcome was already known at training time. Teams with fewer than
    MIN_ROWS such rows, and tickets without a team, get the overall rate.
    """

    name = "Team's past late rate, then oldest"
    MIN_ROWS = 50

    def fit(self, frame, x):
        known, late = known_late(frame)
        late = pd.Series(late[known])
        by_team = late.groupby(frame.loc[known, "assignment_group"].to_numpy()).agg(["mean", "size", "sum"])
        self.overall_ = float(late.mean())
        self.rates_ = by_team.loc[by_team["size"] >= self.MIN_ROWS, "mean"].to_dict()
        # The team holding the largest backlog of late tickets; used to ask "how much of the result is one team?"
        self.biggest_late_team_ = by_team["sum"].idxmax()
        self.biggest_late_team_rate_ = float(by_team.loc[self.biggest_late_team_, "mean"])
        return self

    def predict(self, frame, x):
        rate = frame["assignment_group"].map(self.rates_).fillna(self.overall_).to_numpy()
        order = np.lexsort((frame["duration_h"].to_numpy(), rate))  # by rate, then by age
        slowness = np.empty(len(frame))
        slowness[order] = np.arange(len(frame))
        return slowness, None


class LateRateByAge:
    """No model: the share of past tickets of the same age that were late.

    The calculation a team lead could do in a spreadsheet: split the past queues into age bands and
    count how many tickets of each band were still open a week later.
    """

    name = "Past late rate by age band"
    EDGES_H = 24.0 * np.array([0, 1, 3, 7, 14, 28, 56, np.inf])

    def _band(self, frame):
        return np.searchsorted(self.EDGES_H, frame["duration_h"].to_numpy(), side="right") - 1

    def fit(self, frame, x):
        known, late = known_late(frame)
        self.overall_ = float(late[known].mean())
        rates = pd.Series(late[known]).groupby(self._band(frame)[known]).mean()
        self.rates_ = rates.reindex(range(len(self.EDGES_H) - 1)).fillna(self.overall_).to_numpy()
        return self

    def predict(self, frame, x):
        p_late = self.rates_[self._band(frame)]
        return p_late, only_late_horizon(p_late)


class ResolvedOnlyRegression:
    """Log-normal regression that ignores censoring: only tickets already resolved are used."""

    name = "Regression on resolved tickets"

    def fit(self, frame, x):
        done = frame["event"].to_numpy()
        log_t = np.log(frame.loc[done, "remaining_h"].clip(lower=MIN_HOURS))
        self.model_ = LinearRegression().fit(x[done], log_t)
        self.sigma_ = float(np.std(log_t - self.model_.predict(x[done]), ddof=x.shape[1] + 1))
        self.n_used_, self.n_dropped_ = int(done.sum()), int((~done).sum())
        return self

    def predict(self, frame, x):
        mu = self.model_.predict(x)
        prob = norm.cdf((np.log(GRID_H)[None, :] - mu[:, None]) / self.sigma_)
        return mu, prob


class KaplanMeierByAge:
    """P(resolved by age + h | still open at age), from one Kaplan-Meier curve over all tickets."""

    name = "Kaplan–Meier by age"

    def fit_tickets(self, age_h: np.ndarray, resolved: np.ndarray) -> "KaplanMeierByAge":
        """Fit on one row per ticket: time since opening and whether it is resolved, as known at training time."""
        km = KaplanMeierFitter().fit(age_h, resolved)
        self.times_ = km.survival_function_.index.to_numpy()
        self.surv_ = km.survival_function_.iloc[:, 0].to_numpy()
        return self

    def _s(self, t: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.times_, t, side="right") - 1
        return self.surv_[np.clip(idx, 0, len(self.surv_) - 1)]

    def predict(self, frame, x):
        age = frame["duration_h"].to_numpy()
        s_now = self._s(age)
        s_later = self._s(age[:, None] + GRID_H[None, :])
        prob = 1.0 - np.divide(s_later, s_now[:, None], out=np.ones_like(s_later), where=s_now[:, None] > 0)
        return 1.0 - prob[:, 6], prob  # slowness: P(not resolved within 7 days)


class Cox:
    name = "Cox"

    def __init__(self, penalizer: float = 0.01):
        self.penalizer = penalizer

    def fit(self, frame, x):
        self.columns_ = x.columns[x.std() > 0].tolist()  # a column that never varies cannot be fitted
        data = x[self.columns_].copy()
        data["remaining_h"] = frame["remaining_h"].clip(lower=MIN_HOURS).to_numpy()
        data["event"] = frame["event"].to_numpy()
        data["number"] = frame["number"].to_numpy()
        # A ticket appears at several cutoffs; cluster_col gives standard errors that allow for that.
        self.model_ = CoxPHFitter(penalizer=self.penalizer).fit(
            data, duration_col="remaining_h", event_col="event", cluster_col="number"
        )
        return self

    def predict(self, frame, x):
        x = x[self.columns_]
        surv = self.model_.predict_survival_function(x, times=GRID_H).to_numpy().T
        return -self.model_.predict_log_partial_hazard(x).to_numpy(), 1.0 - surv

    def coefficients(self) -> pd.DataFrame:
        s = self.model_.summary
        return pd.DataFrame(
            {
                "feature": s.index,
                "hazard_ratio": s["exp(coef)"].to_numpy(),
                "ci_low": s["exp(coef) lower 95%"].to_numpy(),
                "ci_high": s["exp(coef) upper 95%"].to_numpy(),
                "p": s["p"].to_numpy(),
            }
        ).sort_values("hazard_ratio")


class SurvivalForest:
    name = "Random survival forest"

    # Remaining times are rounded up to 12-hour steps: the forest stores a curve per leaf at every
    # distinct time, and the horizons we report are whole days.
    STEP_H = 12.0

    def __init__(self, min_samples_leaf: int = 120, n_estimators: int = 200, random_state: int = 0):
        self.min_samples_leaf = min_samples_leaf
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, frame, x):
        steps = np.ceil(frame["remaining_h"].clip(lower=MIN_HOURS) / self.STEP_H) * self.STEP_H
        y = Surv.from_arrays(frame["event"].to_numpy(), steps.to_numpy())
        self.model_ = RandomSurvivalForest(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_features="sqrt",
            n_jobs=-1,
            random_state=self.random_state,
        ).fit(x, y)
        return self

    def predict(self, frame, x):
        surv = self.model_.predict_survival_function(x, return_array=True)
        times = self.model_.unique_times_
        # Step function: S(h) is the value at the last stored time <= h (1 before the first one).
        idx = np.searchsorted(times, GRID_H, side="right") - 1
        on_grid = np.where(idx[None, :] >= 0, surv[:, np.clip(idx, 0, None)], 1.0)
        return -self.model_.predict(x), 1.0 - on_grid


class LateClassifier:
    """Gradient boosting on the yes/no question itself: will this ticket be late?

    It is not a survival model: it answers one horizon only and learns from the rows whose 7-day
    outcome was already known at training time (a ticket still open after those 7 days is a "yes").

    `rows` chooses which of those rows are used. "known" (the default) uses all of them. "resolved"
    keeps only tickets already resolved at training time, i.e. it throws the still-open ones away;
    it exists to measure what that costs (src/checks.py).
    """

    name = "Gradient boosting classifier"

    def __init__(self, max_leaf_nodes: int = 7, min_samples_leaf: int = 240, rows: str = "known"):
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.rows = rows

    def training_rows(self, frame: pd.DataFrame) -> np.ndarray:
        known, _ = known_late(frame)
        return known & frame["event"].to_numpy() if self.rows == "resolved" else known

    def fit(self, frame, x, keep: np.ndarray | None = None):
        """`keep` optionally restricts the training rows further (used for the same-size control)."""
        rows = self.training_rows(frame) if keep is None else keep
        _, late = known_late(frame)
        self.n_used_ = int(rows.sum())
        self.model_ = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=10.0,
            early_stopping=False,
            random_state=0,
        ).fit(x[rows], late[rows])
        return self

    def predict(self, frame, x):
        p_late = self.model_.predict_proba(x)[:, 1]
        return p_late, only_late_horizon(p_late)
