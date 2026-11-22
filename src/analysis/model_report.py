"""Figures for the model comparison, drawn from the tables written by src/evaluate.py.

Run: python -m src.analysis.model_report   (after python -m src.evaluate)
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from src import config, plotstyle

FOREST = "Random survival forest"
REGRESSION = "Regression on resolved tickets"
SIMPLE = ["Oldest first", "Team's past late rate, then oldest", REGRESSION]

TEXT = {
    "tr": {
        "methods": {
            "Oldest first": "En eski kayıt önce",
            "Team's past late rate, then oldest": "Ekibin geçmiş gecikme oranı",
            REGRESSION: "Çözülmüş kayıtlarla regresyon",
            "Kaplan–Meier by age": "Yaşa göre Kaplan–Meier",
            "Cox": "Cox",
            FOREST: "Sağkalım ormanı",
        },
        "suptitle": "Mayıs 2016'nın 31 günlük kuyruğunda yöntemler (nokta: ortalama, çizgi: en düşük–en yüksek gün)",
        "c_index": "Sıralama doğruluğu (C-index)\nyüksek iyi",
        "top10": "En riskli %10'da gerçekten geciken\nyüksek iyi",
        "brier": "Olasılık hatası (Brier)\ndüşük iyi",
        "simple": "Sezgisel yöntem",
        "survival": "Sağkalım yöntemi",
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
    },
    "en": {
        "methods": {
            "Oldest first": "Oldest first",
            "Team's past late rate, then oldest": "Team's past late rate",
            REGRESSION: "Regression on resolved tickets",
            "Kaplan–Meier by age": "Kaplan–Meier by age",
            "Cox": "Cox",
            FOREST: "Random survival forest",
        },
        "suptitle": "Methods on the 31 daily queues of May 2016 (dot: mean, line: lowest–highest day)",
        "c_index": "Ranking accuracy (C-index)\nhigher is better",
        "top10": "Truly late among the riskiest 10%\nhigher is better",
        "brier": "Probability error (Brier)\nlower is better",
        "simple": "Simple method",
        "survival": "Survival method",
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
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
    for ax, metric in zip(axes, ["c_index", "top10", "brier"]):
        ref, ref_label = references[metric]
        ax.axvline(ref, color=plotstyle.AXIS, linewidth=1)
        ax.annotate(ref_label, xy=(ref, len(methods) - 0.45), ha="center", va="bottom", fontsize=8.5,
                    color=plotstyle.INK_MUTED, annotation_clip=False)
        for y, name in enumerate(methods):
            row = summary[summary["method"] == name].iloc[0]
            color = plotstyle.NAIVE if name in SIMPLE else plotstyle.SURVIVAL
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
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.925), ncol=2, fontsize=10)
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


def main() -> None:
    print("Model figures")
    plotstyle.apply()
    summary = pd.read_csv(config.TABLES_DIR / "model_summary.csv")
    daily = pd.read_csv(config.TABLES_DIR / "model_daily.csv")
    calibration = pd.read_csv(config.TABLES_DIR / "model_calibration.csv")
    for lang in plotstyle.LANGS:
        plot_comparison(summary, daily, lang)
        plot_calibration(calibration, daily, lang)


if __name__ == "__main__":
    main()
