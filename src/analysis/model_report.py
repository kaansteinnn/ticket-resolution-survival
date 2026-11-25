"""Figures for the model comparison, the late-count forecast and the open-ticket check.

Drawn from the tables written by src/evaluate.py and src/checks.py.

Run: python -m src.analysis.model_report   (after python -m src.evaluate and python -m src.checks)
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from src import config, plotstyle

FOREST = "Random survival forest"
REGRESSION = "Regression on resolved tickets"
AGE_BAND = "Past late rate by age band"
CLASSIFIER = "Gradient boosting classifier"
SIMPLE = ["Oldest first", "Team's past late rate, then oldest", AGE_BAND, REGRESSION]

TEXT = {
    "tr": {
        "methods": {
            "Oldest first": "En eski kayıt önce",
            "Team's past late rate, then oldest": "Ekibin geçmiş gecikme oranı",
            AGE_BAND: "Yaşa göre geçmiş gecikme oranı",
            REGRESSION: "Çözülmüş kayıtlarla regresyon",
            "Kaplan–Meier by age": "Yaşa göre Kaplan–Meier",
            "Cox": "Cox",
            FOREST: "Sağkalım ormanı",
            CLASSIFIER: "Sınıflandırıcı (gecikir / gecikmez)",
        },
        "suptitle": "Mayıs 2016'nın 31 günlük kuyruğunda yöntemler (nokta: ortalama, çizgi: en düşük–en yüksek gün)",
        "c_index": "Sıralama doğruluğu (C-index)\nyüksek iyi",
        "top10": "En riskli %10'da gerçekten geciken\nyüksek iyi",
        "brier": "Olasılık hatası (Brier)\ndüşük iyi",
        "simple": "Basit yöntem",
        "survival": "Sağkalım yöntemi",
        "other": "Sıradan sınıflandırıcı",
        "random": "rastgele",
        "base": "herkese aynı oran",
        "no_prob": "olasılık vermez",
        "cal_title": "“%X çözülür” dediklerimizin yüzde kaçı çözüldü?",
        "cal_x": "Tahmin: 7 günde çözülme olasılığı",
        "cal_y": "Gerçekleşen çözülme oranı",
        "perfect": "tam isabet",
        "day_title": "Gün gün: 7 günde çözülenlerin oranı",
        "day_x": "Kuyruğun tarihi (Mayıs 2016)",
        "observed": "Gerçekleşen",
        "months": "May",
        "decimal": ",",
        "work_title": "Her günün kuyruğunda 7 günde çözülmeyecek kayıt sayısı: tahminler ve gerçekleşen",
        "work_y": "Kayıt sayısı",
        "work_truth": "Gerçekleşen",
        "work_forest": "Sağkalım ormanı",
        "work_past": "Yaşa göre geçmiş gecikme oranı",
        "work_regression": "Çözülmüş kayıtlarla regresyon",
        "open_title": "Aynı sınıflandırıcı, üç eğitim kümesi: geciken kayıt sayısındaki günlük ortalama hata",
        "open_x": "Günlük ortalama hata (kayıt)",
        "open_all": "7 günlük sonucu bilinen bütün satırlar",
        "open_random": "Aynı sayıda satır, rastgele atılmış",
        "open_resolved": "Yalnız çözülmüş kayıtlar\n(açık olanlar atılmış)",
        "open_rows": "satır",
    },
    "en": {
        "methods": {
            "Oldest first": "Oldest first",
            "Team's past late rate, then oldest": "Team's past late rate",
            AGE_BAND: "Past late rate by age band",
            REGRESSION: "Regression on resolved tickets",
            "Kaplan–Meier by age": "Kaplan–Meier by age",
            "Cox": "Cox",
            FOREST: "Random survival forest",
            CLASSIFIER: "Classifier (late / not late)",
        },
        "suptitle": "Methods on the 31 daily queues of May 2016 (dot: mean, line: lowest–highest day)",
        "c_index": "Ranking accuracy (C-index)\nhigher is better",
        "top10": "Truly late among the riskiest 10%\nhigher is better",
        "brier": "Probability error (Brier)\nlower is better",
        "simple": "Simple method",
        "survival": "Survival method",
        "other": "Ordinary classifier",
        "random": "random",
        "base": "same rate for all",
        "no_prob": "no probability",
        "cal_title": "Predicted vs observed: resolved within 7 days",
        "cal_x": "Predicted probability of resolution within 7 days",
        "cal_y": "Observed share resolved",
        "perfect": "perfect",
        "day_title": "Day by day: share resolved within 7 days",
        "day_x": "Date of the queue (May 2016)",
        "observed": "Observed",
        "months": "May",
        "decimal": ".",
        "work_title": "Tickets in each day's queue that will not be resolved within 7 days: forecasts and outcome",
        "work_y": "Tickets",
        "work_truth": "What actually happened",
        "work_forest": "Random survival forest",
        "work_past": "Past late rate by age band",
        "work_regression": "Regression on resolved tickets",
        "open_title": "One classifier, three training sets: mean daily error in the number of late tickets",
        "open_x": "Mean daily error (tickets)",
        "open_all": "All rows with a known 7-day outcome",
        "open_random": "Same number of rows, dropped at random",
        "open_resolved": "Resolved tickets only\n(open ones thrown away)",
        "open_rows": "rows",
    },
}


def _num(value: float, lang: str, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", TEXT[lang]["decimal"])


def plot_comparison(summary: pd.DataFrame, daily: pd.DataFrame, lang: str) -> None:
    t = TEXT[lang]
    late_share = daily.drop_duplicates("cutoff")["late_share"]
    references = {
        "c_index": (0.5, t["random"]),
        "top10": (late_share.mean(), t["random"]),
        "brier": ((late_share * (1 - late_share)).mean(), t["base"]),
    }
    methods = summary["method"].tolist()[::-1]  # first method at the top
    colour = lambda name: plotstyle.NAIVE if name in SIMPLE else plotstyle.OTHER if name == CLASSIFIER else plotstyle.SURVIVAL
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.9), sharey=True)
    for ax, metric in zip(axes, ["c_index", "top10", "brier"]):
        ref, ref_label = references[metric]
        ax.axvline(ref, color=plotstyle.AXIS, linewidth=1)
        ax.annotate(ref_label, xy=(ref, len(methods) - 0.45), ha="center", va="bottom", fontsize=8.5,
                    color=plotstyle.INK_MUTED, annotation_clip=False)
        for y, name in enumerate(methods):
            row = summary[summary["method"] == name].iloc[0]
            color = colour(name)
            mean = row[f"{metric}_mean"]
            if pd.isna(mean):
                ax.annotate(t["no_prob"], xy=(ref, y), xytext=(8, 0), textcoords="offset points", va="center",
                            fontsize=9, color=plotstyle.INK_MUTED)
                continue
            ax.plot([row[f"{metric}_min"], row[f"{metric}_max"]], [y, y], color=color, linewidth=2, alpha=0.35,
                    solid_capstyle="round")
            ax.plot(mean, y, "o", color=color, markersize=8, markeredgecolor=plotstyle.SURFACE, markeredgewidth=1.5)
            ax.annotate(_num(mean, lang), xy=(row[f"{metric}_max"], y), xytext=(7, 0), textcoords="offset points",
                        va="center", fontsize=9, color=plotstyle.INK)
        ax.set_title(t[metric], fontsize=10)
        ax.set_yticks(range(len(methods)), [t["methods"][m] for m in methods])
        ax.set_ylim(-0.6, len(methods) - 0.2)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0)
        ax.margins(x=0.18)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=plotstyle.NAIVE, label=t["simple"]),
        plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=plotstyle.SURVIVAL, label=t["survival"]),
        plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=plotstyle.OTHER, label=t["other"]),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.935), ncol=3, fontsize=10)
    fig.suptitle(t["suptitle"], x=0.015, y=1.0, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    plotstyle.save(fig, "model_comparison", lang)


def plot_calibration(calibration: pd.DataFrame, daily: pd.DataFrame, lang: str) -> None:
    t = TEXT[lang]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.4))

    left.plot([0, 1], [0, 1], color=plotstyle.AXIS, linewidth=1)
    left.annotate(t["perfect"], xy=(0.82, 0.82), xytext=(8, -12), textcoords="offset points", ha="left",
                  fontsize=9, color=plotstyle.INK_MUTED)
    for name, color in [(REGRESSION, plotstyle.NAIVE), (FOREST, plotstyle.SURVIVAL)]:
        rows = calibration[calibration["method"] == name]
        plotstyle.line(left, rows["predicted"], rows["observed"], color, t["methods"][name])
    left.set_title(t["cal_title"])
    left.set_xlabel(t["cal_x"])
    left.set_ylabel(t["cal_y"])
    left.set_xlim(0, 1)
    left.set_ylim(0, 1)
    left.set_aspect("equal")
    left.grid(axis="x", visible=True)
    left.legend(loc="upper left")
    percent = lambda v, _: f"%{round(v * 100)}" if lang == "tr" else f"{round(v * 100)}%"
    left.xaxis.set_major_formatter(percent)
    left.yaxis.set_major_formatter(percent)

    days = daily.drop_duplicates("cutoff")
    x = pd.to_datetime(days["cutoff"])
    plotstyle.line(right, x, 1 - days["late_share"], plotstyle.TRUTH, t["observed"], markers=False)
    for name, color in [(FOREST, plotstyle.SURVIVAL), (REGRESSION, plotstyle.NAIVE)]:
        rows = daily[daily["method"] == name]
        plotstyle.line(right, pd.to_datetime(rows["cutoff"]), 1 - rows["mean_p_late"], color, t["methods"][name],
                       markers=False)
    right.set_title(t["day_title"])
    right.set_xlabel(t["day_x"])
    right.set_ylim(0, 0.7)
    right.set_xlim(x.min() - pd.Timedelta(days=1), x.max() + pd.Timedelta(hours=12))
    right.yaxis.set_major_formatter(percent)
    right.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 8, 15, 22, 29]))
    right.xaxis.set_major_formatter(lambda v, _: f"{mdates.num2date(v).day} {t['months']}")
    right.legend(loc="lower left", fontsize=9)

    fig.tight_layout()
    plotstyle.save(fig, "calibration", lang)


def plot_workload(daily: pd.DataFrame, workload: pd.DataFrame, lang: str) -> None:
    """Day by day: how many tickets of the queue turned out late, against three forecasts of that count."""
    t = TEXT[lang]
    days = daily.drop_duplicates("cutoff")
    x = pd.to_datetime(days["cutoff"])
    tickets = days["tickets"].to_numpy()
    forecast = lambda name: tickets * daily.loc[daily["method"] == name, "mean_p_late"].to_numpy()
    series = [
        (t["work_truth"], tickets * days["late_share"].to_numpy(), plotstyle.TRUTH, "-", 7),
        (t["work_forest"], forecast(FOREST), plotstyle.SURVIVAL, "-", -7),
        (t["work_past"], forecast(AGE_BAND), plotstyle.NAIVE, (0, (4, 3)), 0),
        (t["work_regression"], forecast(REGRESSION), plotstyle.NAIVE, "-", 0),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for label, y, color, style, nudge in series:  # nudge: the first two lines end close together
        plotstyle.line(ax, x, y, color, label, markers=False, linestyle=style)
        plotstyle.end_label(ax, x.iloc[-1], y[-1], label, dy=nudge)
    ax.set_title(t["work_title"])
    ax.set_ylabel(t["work_y"])
    ax.set_ylim(0, 1000)
    ax.set_xlim(x.min() - pd.Timedelta(hours=12), x.max() + pd.Timedelta(hours=12))
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 8, 15, 22, 29]))
    ax.xaxis.set_major_formatter(lambda v, _: f"{mdates.num2date(v).day} {t['months']}")
    fig.tight_layout()
    plotstyle.save(fig, "workload", lang)


def plot_open_tickets(check: pd.DataFrame, lang: str) -> None:
    """What throwing away the still-open tickets costs: one classifier, three training sets."""
    t = TEXT[lang]
    all_rows, resolved = check.iloc[0], check.iloc[1]
    random = check.iloc[2:]
    rows = lambda n: f"({int(n):,} {t['open_rows']})".replace(",", "." if lang == "tr" else ",")
    bars = [
        (f"{t['open_all']}\n{rows(all_rows['rows'])}", all_rows["count_error"], plotstyle.SURVIVAL),
        (f"{t['open_random']}\n{rows(random['rows'].iloc[0])}", random["count_error"].mean(), plotstyle.SURVIVAL),
        (f"{t['open_resolved']}\n{rows(resolved['rows'])}", resolved["count_error"], plotstyle.NAIVE),
    ]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for y, (label, value, color) in enumerate(bars[::-1]):
        ax.barh(y, value, color=color, height=0.55)
        ax.annotate(f"{value:.0f}", xy=(value, y), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=10, color=plotstyle.INK)
    ax.set_yticks(range(len(bars)), [b[0] for b in bars[::-1]])
    ax.set_title(t["open_title"])
    ax.set_xlabel(t["open_x"])
    ax.set_xlim(0, resolved["count_error"] * 1.12)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    fig.tight_layout()
    plotstyle.save(fig, "open_tickets", lang)


def main() -> None:
    print("Model figures")
    plotstyle.apply()
    summary = pd.read_csv(config.TABLES_DIR / "model_summary.csv")
    daily = pd.read_csv(config.TABLES_DIR / "model_daily.csv")
    calibration = pd.read_csv(config.TABLES_DIR / "model_calibration.csv")
    workload = pd.read_csv(config.TABLES_DIR / "model_workload.csv")
    open_check = pd.read_csv(config.TABLES_DIR / "check_open_tickets.csv")
    for lang in plotstyle.LANGS:
        plot_comparison(summary, daily, lang)
        plot_calibration(calibration, daily, lang)
        plot_workload(daily, workload, lang)
        plot_open_tickets(open_check, lang)


if __name__ == "__main__":
    main()
