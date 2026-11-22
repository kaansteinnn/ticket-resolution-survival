"""Finding 2: a ticket that has already waited long will, typically, wait even longer.

The intuitive rule is "tickets take 7 days on average, this one has been open for 5, so 2 to go".
Here we measure the opposite directly: among tickets still open after `a` days, how much longer did
they take? Every ticket in the file is eventually resolved, so this is a plain count on the complete
data (no model); intervals are 95% percentile bootstrap over tickets.

Run: python -m src.analysis.waiting
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, db, plotstyle

AGES_DAYS = [0, 0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 10, 14, 21, 28]
N_BOOTSTRAP = 1000
SEED = 20161125


def remaining(durations: np.ndarray, age: float) -> np.ndarray:
    """Remaining days for tickets still open at `age` (all tickets at age 0)."""
    at_risk = durations if age == 0 else durations[durations > age]
    return at_risk - age


def waiting_table(durations: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    boot = np.empty((N_BOOTSTRAP, len(AGES_DAYS)))
    for b in range(N_BOOTSTRAP):
        sample = rng.choice(durations, size=len(durations), replace=True)
        boot[b] = [np.median(remaining(sample, a)) for a in AGES_DAYS]
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)

    mean_all = durations.mean()
    rows = []
    for i, age in enumerate(AGES_DAYS):
        rem = remaining(durations, age)
        rows.append(
            {
                "days_open": age,
                "still_open": len(rem),
                "median_remaining": np.median(rem),
                "median_remaining_lo": lo[i],
                "median_remaining_hi": hi[i],
                "resolved_within_1d": (rem <= 1).mean(),
                "resolved_within_7d": (rem <= 7).mean(),
                "naive_remaining": max(mean_all - age, 0.0),
            }
        )
    return pd.DataFrame(rows)


TEXT = {
    "tr": {
        "title": "Kayıt ne kadar beklediyse, tipik olarak o kadar daha bekliyor",
        "x": "Kayıt kaç gündür açık",
        "y": "Kalan süre (gün)",
        "survival": "Ölçülen: kalan sürenin medyanı (bant: %95 aralık)",
        "naive": "Sezgisel: ortalama süre − bekleme",
    },
    "en": {
        "title": "The longer a ticket has waited, the longer it typically still has to wait",
        "x": "Days the ticket has been open",
        "y": "Remaining time (days)",
        "survival": "Measured: median remaining time (band: 95% interval)",
        "naive": "Intuitive: average time − time waited",
    },
}


def plot(table: pd.DataFrame, lang: str) -> None:
    t = TEXT[lang]
    fmt = (lambda v: f"{v:.1f}".replace(".", ",")) if lang == "tr" else (lambda v: f"{v:.1f}")
    x = table["days_open"]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.fill_between(
        x, table["median_remaining_lo"], table["median_remaining_hi"], color=plotstyle.SURVIVAL, alpha=0.12, lw=0
    )
    plotstyle.line(ax, x, table["median_remaining"], plotstyle.SURVIVAL, t["survival"])
    plotstyle.line(ax, x, table["naive_remaining"], plotstyle.NAIVE, t["naive"], markers=False)
    last = table.iloc[-1]
    plotstyle.end_label(ax, last["days_open"], last["median_remaining"], fmt(last["median_remaining"]))
    plotstyle.end_label(ax, last["days_open"], last["naive_remaining"], fmt(last["naive_remaining"]))
    ax.set_title(t["title"])
    ax.set_xlabel(t["x"])
    ax.set_ylabel(t["y"])
    ax.set_xticks([0, 7, 14, 21, 28])
    ax.set_xlim(-0.5, 28.5)
    ax.set_ylim(0, None)
    ax.legend(loc="upper left")
    fig.tight_layout()
    plotstyle.save(fig, "waiting", lang)


def main() -> None:
    print("Finding 2: waiting-time fallacy")
    plotstyle.apply()
    con = db.connect()
    durations = (db.incidents(con)["duration_h"] / 24.0).to_numpy()
    con.close()

    table = waiting_table(durations)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.round(3).to_csv(config.TABLES_DIR / "waiting.csv", index=False)
    for lang in plotstyle.LANGS:
        plot(table, lang)
    print(f"mean {durations.mean():.2f} d, median {np.median(durations):.2f} d, n {len(durations)}")
    print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
