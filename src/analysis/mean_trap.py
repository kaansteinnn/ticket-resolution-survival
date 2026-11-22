"""Finding 1: the average computed from resolved tickets understates how long work takes.

Every Monday we stop the clock and ask, for all tickets opened so far, "how long do tickets take?"
three ways:

  naive     - statistics of the tickets already resolved on that day (what a dashboard shows)
  survival  - Kaplan-Meier, which also counts the still-open tickets ("open for at least x days")
  truth     - what those same tickets turned out to take, read from the rest of the file

A second table looks at one day's dashboard broken down by opening week: recent weeks look fast
only because their slow tickets have not been resolved yet.

Run: python -m src.analysis.mean_trap
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times

from src import config, db, plotstyle

CUTOFFS = pd.date_range("2016-03-07", "2016-06-27", freq="7D")
DASHBOARD_DAY = pd.Timestamp("2016-05-02")
HOURS_PER_DAY = 24.0


def km_quantile(km: KaplanMeierFitter, q: float) -> float:
    """Duration by which a share q of tickets is resolved; inf if the curve has not got there."""
    return float(km.percentile(1 - q))


def weekly_table(con) -> pd.DataFrame:
    truth = db.incidents(con).set_index("number")["duration_h"] / HOURS_PER_DAY
    rows = []
    for cutoff in CUTOFFS:
        snap = db.snapshot(con, cutoff)
        days = snap["duration_h"] / HOURS_PER_DAY
        resolved = snap["resolved"]
        true_days = truth.loc[snap["number"]]

        km = KaplanMeierFitter().fit(days, resolved)
        ci = median_survival_times(km.confidence_interval_).iloc[0]

        rows.append(
            {
                "cutoff": cutoff.date(),
                "tickets": len(snap),
                "open": int((~resolved).sum()),
                "open_share": float((~resolved).mean()),
                "naive_mean": days[resolved].mean(),
                "true_mean": true_days.mean(),
                "naive_median": days[resolved].median(),
                "km_median": km_quantile(km, 0.5),
                "km_median_lo": float(ci.iloc[0]),
                "km_median_hi": float(ci.iloc[1]),
                "true_median": true_days.median(),
                "naive_p75": days[resolved].quantile(0.75),
                "km_p75": km_quantile(km, 0.75),
                "true_p75": true_days.quantile(0.75),
                "naive_p90": days[resolved].quantile(0.90),
                "km_p90": km_quantile(km, 0.90),
                "true_p90": true_days.quantile(0.90),
            }
        )
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)


def opening_week_table(con, day: pd.Timestamp = DASHBOARD_DAY) -> pd.DataFrame:
    """One day's dashboard, by the week the ticket was opened in."""
    truth = db.incidents(con).set_index("number")["duration_h"] / HOURS_PER_DAY
    snap = db.snapshot(con, day)
    snap["days"] = snap["duration_h"] / HOURS_PER_DAY
    snap["true_days"] = truth.loc[snap["number"]].to_numpy()
    snap["opening_week"] = snap["opened_at"].dt.to_period("W-SUN").dt.start_time.dt.date

    rows = []
    for week, grp in snap.groupby("opening_week"):
        done = grp[grp["resolved"]]
        rows.append(
            {
                "opening_week": week,
                "tickets": len(grp),
                "resolved_share": len(done) / len(grp),
                "naive_mean": done["days"].mean(),
                "true_mean": grp["true_days"].mean(),
                "naive_p90": done["days"].quantile(0.9),
                "true_p90": grp["true_days"].quantile(0.9),
            }
        )
    return pd.DataFrame(rows)


TEXT = {
    "tr": {
        "suptitle": "Her pazartesi: “işlerimiz ne kadar sürüyor?” sorusuna üç cevap",
        "mean": "Ortalama süre (gün)",
        "median": "Medyan süre (gün)",
        "p90": "10 işten en yavaş 1'i (%90'lık, gün)",
        "naive": "Yalnız çözülmüş kayıtlar",
        "km": "Kaplan–Meier (açık kayıtlar dahil)",
        "truth": "Sonradan gerçekleşen",
        "x": "Hesabın yapıldığı gün (2016)",
        "week_title": "2 Mayıs 2016 günü görünen ortalama süre, açılış haftasına göre (gün)",
        "week_naive": "O gün görünen (çözülmüş kayıtlar)",
        "week_truth": "Sonradan gerçekleşen",
        "week_x": "Kaydın açıldığı hafta (2016)",
        "months": ["Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem"],
    },
    "en": {
        "suptitle": "Every Monday: three answers to “how long do our tickets take?”",
        "mean": "Mean time (days)",
        "median": "Median time (days)",
        "p90": "Slowest 1 in 10 (90th percentile, days)",
        "naive": "Resolved tickets only",
        "km": "Kaplan–Meier (open tickets included)",
        "truth": "What actually happened",
        "x": "Day of calculation (2016)",
        "week_title": "Mean time by opening week, as seen on 2 May 2016 (days)",
        "week_naive": "Shown that day (resolved tickets)",
        "week_truth": "What actually happened",
        "week_x": "Week the ticket was opened (2016)",
        "months": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"],
    },
}


def _date_axis(ax, months: list[str]) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(lambda x, _: f"1 {months[mdates.num2date(x).month - 1]}")


def plot_weekly(table: pd.DataFrame, lang: str) -> None:
    t = TEXT[lang]
    x = pd.to_datetime(table["cutoff"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    panels = [
        ("mean", "naive_mean", None, "true_mean"),
        ("median", "naive_median", "km_median", "true_median"),
        ("p90", "naive_p90", "km_p90", "true_p90"),
    ]
    for ax, (key, naive, km, truth) in zip(axes, panels):
        plotstyle.line(ax, x, table[truth], plotstyle.TRUTH, t["truth"], markers=False)
        if km:
            if key == "median":
                ax.fill_between(
                    x, table["km_median_lo"], table["km_median_hi"], color=plotstyle.SURVIVAL, alpha=0.12, lw=0
                )
            plotstyle.line(ax, x, table[km], plotstyle.SURVIVAL, t["km"])
        plotstyle.line(ax, x, table[naive], plotstyle.NAIVE, t["naive"])
        ax.set_title(t[key])
        ax.set_ylim(0, None)
        ax.set_xlabel(t["x"])
        _date_axis(ax, t["months"])

    handles, labels = axes[1].get_legend_handles_labels()
    order = [2, 1, 0]
    fig.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper left",
        bbox_to_anchor=(0.045, 0.94),
        ncol=3,
        fontsize=10,
    )
    fig.suptitle(t["suptitle"], x=0.05, y=1.02, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    plotstyle.save(fig, "mean_trap", lang)


def plot_opening_week(table: pd.DataFrame, lang: str) -> None:
    t = TEXT[lang]
    x = pd.to_datetime(table["opening_week"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    plotstyle.line(ax, x, table["true_mean"], plotstyle.TRUTH, t["week_truth"])
    plotstyle.line(ax, x, table["naive_mean"], plotstyle.NAIVE, t["week_naive"])
    last = table.iloc[-1]
    plotstyle.end_label(ax, x.iloc[-1], last["true_mean"], f"{last['true_mean']:.1f}".replace(".", "," if lang == "tr" else "."))
    plotstyle.end_label(ax, x.iloc[-1], last["naive_mean"], f"{last['naive_mean']:.1f}".replace(".", "," if lang == "tr" else "."))
    ax.set_title(t["week_title"])
    ax.set_xlabel(t["week_x"])
    ax.set_ylim(0, None)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(
        lambda v, _: f"{mdates.num2date(v).day} {t['months'][mdates.num2date(v).month - 1]}"
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    plotstyle.save(fig, "mean_trap_by_opening_week", lang)


def main() -> None:
    print("Finding 1: average-time trap")
    plotstyle.apply()
    con = db.connect()
    weekly = weekly_table(con)
    by_week = opening_week_table(con)
    con.close()

    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    weekly.round(3).to_csv(config.TABLES_DIR / "mean_trap_weekly.csv", index=False)
    by_week.round(3).to_csv(config.TABLES_DIR / "mean_trap_by_opening_week.csv", index=False)
    for lang in plotstyle.LANGS:
        plot_weekly(weekly, lang)
        plot_opening_week(by_week, lang)

    cols = ["cutoff", "open_share", "naive_mean", "true_mean", "naive_median", "km_median", "true_median",
            "naive_p90", "km_p90", "true_p90"]
    print(weekly[cols].round(2).to_string(index=False))
    print(by_week.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
