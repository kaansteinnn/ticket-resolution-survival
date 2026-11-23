"""Demo: go back to a day in May 2016, see that day's open tickets ranked by the risk of being late.

Run: streamlit run app/app.py

The app only reads files that the pipeline wrote (python run_all.py):
  app/data/predictions.parquet   every May queue with the survival forest's predictions and the
                                 outcome that was hidden from the model
  reports/tables, reports/figures
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
PREDICTIONS = APP_DIR / "data" / "predictions.parquet"
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"

LATE_DAYS = 7
TOP_SHARE = 0.10
CURVE_COLUMNS = [f"p_resolved_d{d:02d}" for d in range(1, 31)]
SERIES_BLUE = "#2a78d6"
MUTED = "#898781"

STATE_TR = {
    "New": "Yeni",
    "Active": "İşlemde",
    "Awaiting User Info": "Kullanıcıdan bilgi bekleniyor",
    "Awaiting Problem": "Problem kaydı bekleniyor",
    "Awaiting Vendor": "Tedarikçi bekleniyor",
    "Awaiting Evidence": "Kanıt bekleniyor",
}
METHOD_TR = {
    "Oldest first": "En eski kayıt önce",
    "Team's past late rate, then oldest": "Ekibin geçmiş gecikme oranı, sonra en eski",
    "Regression on resolved tickets": "Çözülmüş kayıtlarla regresyon",
    "Kaplan–Meier by age": "Yaşa göre Kaplan–Meier",
    "Cox": "Cox",
    "Random survival forest": "Sağkalım ormanı (demodaki model)",
    "Late share of all past queues": "Model yok: geçmiş kuyrukların gecikme oranı × bugünkü kuyruk",
    "Late share of the latest past queue": "Model yok: sonucu bilinen en son kuyruğun oranı × bugünkü kuyruk",
}

st.set_page_config(page_title="Bu iş ne zaman biter?", page_icon="⏳", layout="wide")


@st.cache_data
def load_predictions() -> pd.DataFrame:
    df = pd.read_parquet(PREDICTIONS)
    df["cutoff"] = pd.to_datetime(df["cutoff"]).dt.date
    df["state_tr"] = df["state"].map(STATE_TR).fillna("Diğer")
    return df


@st.cache_data
def load_table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / name)


def percent(value: float) -> str:
    return f"%{value * 100:.0f}"


def top_hits(queue: pd.DataFrame, by: str) -> tuple[int, int]:
    """(tickets in the riskiest 10% by `by`, how many of them were truly late)."""
    k = max(1, round(TOP_SHARE * len(queue)))
    top = queue.sort_values(by, ascending=False, kind="stable").head(k)
    return k, int(top["actual_late"].sum())


def curve_figure(ticket: pd.Series, reveal: bool) -> go.Figure:
    days = list(range(0, 31))
    prob = [0.0] + [float(ticket[c]) for c in CURVE_COLUMNS]
    fig = go.Figure(
        go.Scatter(
            x=days,
            y=prob,
            mode="lines",
            line=dict(color=SERIES_BLUE, width=2, shape="hv"),
            hovertemplate="%{x}. güne kadar çözülme olasılığı: <b>%{y:.0%}</b><extra></extra>",
        )
    )
    fig.add_vline(x=LATE_DAYS, line_width=1, line_color=MUTED)
    fig.add_annotation(x=LATE_DAYS, y=1.04, yref="paper", text=f"gecikme sınırı: {LATE_DAYS} gün", showarrow=False,
                       font=dict(size=12, color=MUTED))
    actual = float(ticket["actual_remaining_days"])
    if reveal and actual <= 30:
        fig.add_vline(x=actual, line_width=2, line_color="#0b0b0b")
        fig.add_annotation(x=actual, y=0.04, yref="paper", xanchor="left", xshift=6,
                           text=f"gerçekte {actual:.1f} gün sonra çözüldü", showarrow=False, font=dict(size=12))
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
        hovermode="x",
        xaxis=dict(title="Seçilen günden sonra geçen gün", range=[0, 30], dtick=7, showgrid=False),
        yaxis=dict(title="Çözülmüş olma olasılığı", range=[0, 1], tickformat=".0%"),
    )
    return fig


predictions = load_predictions()
days_available = sorted(predictions["cutoff"].unique())

st.title("Bu iş ne zaman biter?")
st.caption(
    "Bir BT şirketinin destek kayıtları (UCI, 24.918 kayıt). Geçmişte bir güne gidin: o sabah açık olan "
    "kayıtlar, önümüzdeki 7 günde çözülmeme riskine göre sıralanır. Model yalnız 1 Mayıs 2016'ya kadar "
    "bilinenlerle kuruldu; Mayıs'ta olanları görmedi."
)
with st.expander("Bu ekran ne işe yarar?"):
    st.markdown(
        "Bir destek masasında her sabah yaklaşık 1.200 kayıt açık durur. Ekran, o sabah için üç soruya cevap verir:\n\n"
        "1. **Bu hafta kaç kayıt elimizde kalacak?** Üstteki “Tahmin: 7 günde çözülmeyecek” sayısı.\n"
        "2. **Hangi kayıtlar büyük ihtimalle gecikecek?** Gecikme riskine göre sıralı liste.\n"
        "3. **Tek bir kayıt ne zaman biter?** Satıra tıklayınca açılan eğri: 7, 14 ve 30 gün içinde çözülme olasılığı.\n\n"
        "“Gecikme” burada, seçilen günden sonraki 7 gün içinde çözülmemek demek. Soldaki "
        "“Sonradan ne olduğunu göster” düğmesi, modelden gizlenen gerçek sonucu açar; tahmin onunla karşılaştırılır."
    )

with st.sidebar:
    st.header("Ayarlar")
    day = st.date_input(
        "Gün (saat 00:00)", value=date(2016, 5, 16), min_value=days_available[0], max_value=days_available[-1]
    )
    reveal = st.toggle("Sonradan ne olduğunu göster", value=False,
                       help="Modelden gizlenen kısım: kayıt gerçekte ne zaman çözüldü?")
    queue_all = predictions[predictions["cutoff"] == day]
    team_counts = queue_all["assignment_group"].value_counts()
    teams = st.multiselect("Ekip", options=team_counts.index.tolist(),
                           format_func=lambda g: f"{g} ({team_counts[g]})", placeholder="Tüm ekipler")
    states = st.multiselect("Durum", options=sorted(queue_all["state_tr"].unique()), placeholder="Tüm durumlar")

queue = queue_all
if teams:
    queue = queue[queue["assignment_group"].isin(teams)]
if states:
    queue = queue[queue["state_tr"].isin(states)]
queue = queue.sort_values("p_late", ascending=False, kind="stable").reset_index(drop=True)

tab_queue, tab_quality, tab_findings = st.tabs(["Açık kayıtlar", "Model ne kadar iyi?", "İki yaygın yanılgı"])

with tab_queue:
    if queue.empty:
        st.info("Bu seçimde açık kayıt yok.")
    else:
        expected_late = queue["p_late"].sum()
        tiles = st.columns(4)
        tiles[0].metric("Açık kayıt", f"{len(queue):,}".replace(",", "."))
        tiles[1].metric("Tahmin: 7 günde çözülmeyecek", f"{expected_late:.0f}",
                        help="Kayıtların gecikme olasılıklarının toplamı.")
        if reveal:
            tiles[2].metric("Gerçekte çözülmeyen", f"{int(queue['actual_late'].sum())}")
            k, hits = top_hits(queue, "p_late")
            _, hits_oldest = top_hits(queue, "age_days")
            tiles[3].metric(f"En riskli {k} kayıtta geciken", f"{hits}",
                            help=f"“En eski {k} kayıt” kuralıyla bu sayı {hits_oldest} olurdu.")
        else:
            tiles[2].metric("Açık kayıtların ortanca yaşı", f"{queue['age_days'].median():.1f} gün")
            tiles[3].metric("Gecikme riski %80 üstü", f"{int((queue['p_late'] > 0.8).sum())}")

        table = pd.DataFrame(
            {
                "Kayıt": queue["number"],
                "Gecikme riski": (queue["p_late"] * 100).round(),
                "Ekip": queue["assignment_group"],
                "Durum": queue["state_tr"],
                "Öncelik": queue["priority"],
                "Yaş (gün)": queue["age_days"],
                "Son güncelleme (gün önce)": queue["days_since_update"],
                "El değiştirme": queue["reassignment_count"],
            }
        )
        if reveal:
            table["Gerçekte kalan (gün)"] = queue["actual_remaining_days"]
            table["Gecikti mi"] = queue["actual_late"].map({True: "Evet", False: "Hayır"})

        left, right = st.columns([3, 2], gap="large")
        with left:
            st.subheader("Gecikme riskine göre sıralı kuyruk")
            selection = st.dataframe(
                table,
                hide_index=True,
                height=460,
                on_select="rerun",
                selection_mode="single-row",
                column_config={
                    "Gecikme riski": st.column_config.ProgressColumn(
                        "Gecikme riski", format="%d%%", min_value=0, max_value=100,
                        help="Önümüzdeki 7 günde çözülmeme olasılığı",
                    ),
                    "Yaş (gün)": st.column_config.NumberColumn(format="%.1f"),
                    "Son güncelleme (gün önce)": st.column_config.NumberColumn(format="%.1f"),
                    "Gerçekte kalan (gün)": st.column_config.NumberColumn(format="%.1f"),
                },
            )
            st.caption("Bir satıra tıklayın: o kaydın çözülme olasılığı eğrisi sağda görünür.")
        with right:
            rows = selection.selection.rows if selection and selection.selection.rows else [0]
            ticket = queue.iloc[rows[0]]
            st.subheader(ticket["number"])
            st.markdown(
                f"**{ticket['assignment_group']}** · {ticket['state_tr']} · {ticket['age_days']:.1f} gündür açık  \n"
                f"7 günde çözülme olasılığı: **{percent(1 - ticket['p_late'])}** · "
                f"14 günde: **{percent(ticket['p_resolved_d14'])}** · 30 günde: **{percent(ticket['p_resolved_d30'])}**"
            )
            st.plotly_chart(curve_figure(ticket, reveal), width="stretch", config={"displayModeBar": False})
            if reveal and ticket["actual_remaining_days"] > 30:
                st.caption(f"Gerçekte {ticket['actual_remaining_days']:.0f} gün sonra çözüldü (grafiğin dışında).")

with tab_quality:
    summary = load_table("model_summary.csv")
    rest = load_table("model_summary_without_biggest_late_team.csv")
    daily = load_table("model_daily.csv")
    late_share = daily.drop_duplicates("cutoff")["late_share"].mean()

    def quality_table(source: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Yöntem": source["method"].map(METHOD_TR),
                "Sıralama doğruluğu (C-index)": source["c_index_mean"],
                "En riskli %10'da geciken": source["top10_mean"],
                "Olasılık hatası (Brier)": source["brier_mean"].map(lambda v: "olasılık vermez" if pd.isna(v) else f"{v:.3f}"),
            }
        )

    fmt = {
        "Sıralama doğruluğu (C-index)": st.column_config.NumberColumn(format="%.3f"),
        "En riskli %10'da geciken": st.column_config.NumberColumn(format="%.3f"),
    }
    workload = load_table("model_workload.csv").set_index("method")
    forest_row = workload.loc["Random survival forest"]
    regression_row = workload.loc["Regression on resolved tickets"]
    st.subheader("1. Bu hafta kaç kayıt elimizde kalacak?")
    st.markdown(
        f"Mayıs'ın her sabahı için “bu kuyruktaki kayıtların kaçı 7 günde çözülmeyecek?” tahmini, sonradan "
        f"gerçekleşen sayıyla karşılaştırıldı. Gerçekte günde ortalama **{forest_row['actual_late_mean']:.0f}** "
        f"kayıt çözülmedi. Demodaki model ortalama **{forest_row['abs_error_mean']:.0f}** kayıt şaştı. Yalnız "
        f"çözülmüş kayıtlardan öğrenen regresyon **{regression_row['abs_error_mean']:.0f}** kayıt şaştı ve 31 günün "
        f"{regression_row['days_forecast_too_low']:.0f}'inde eksik tahmin etti: açık kalan işleri hiç görmediği için."
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Yöntem": workload.index.map(METHOD_TR),
                "Tahmin (kayıt/gün)": workload["forecast_late_mean"],
                "Gerçekleşen (kayıt/gün)": workload["actual_late_mean"],
                "Ortalama şaşma (kayıt/gün)": workload["abs_error_mean"],
                "Eksik tahmin edilen gün (31'de)": workload["days_forecast_too_low"],
            }
        ),
        hide_index=True,
        column_config={
            c: st.column_config.NumberColumn(format="%.0f")
            for c in ["Tahmin (kayıt/gün)", "Gerçekleşen (kayıt/gün)", "Ortalama şaşma (kayıt/gün)"]
        },
    )
    st.image(str(FIGURES / "workload_tr.png"))
    st.caption(
        "Özelliklere hiç bakmayan en basit sağkalım yöntemi (yaşa göre Kaplan–Meier) bu soruda orman kadar iyi: "
        "kazanç modelin karmaşıklığından değil, açık kalan işlerin hesaba katılmasından geliyor."
    )

    st.subheader("2. Hangi kayıtlar gecikecek?")
    st.markdown(
        f"Her gün için: o sabah açık olan kayıtlar sıralandı, sonra gerçekte ne olduğuna bakıldı. "
        f"Kuyruktaki kayıtların ortalama **{percent(late_share)}**'si 7 günde çözülmedi. "
        f"Sıralama doğruluğu (C-index): rastgele seçilen iki kayıttan hangisinin daha geç çözüleceğini doğru "
        f"bilme oranı; 0,5 = yazı tura. Olasılık hatası (Brier): düşük olan iyi."
    )
    st.dataframe(quality_table(summary), hide_index=True, column_config=fmt)
    st.image(str(FIGURES / "model_comparison_tr.png"))

    st.subheader("3. Sonucun ne kadarı tek bir ekip?")
    st.markdown(
        "Modelin en riskli gördüğü %10'un tamamı tek bir ekibin (Group 9) kuyruğundan geliyor; bu ekip açık "
        "kayıtların yaklaşık beşte birini tutuyor ve kayıtlarının neredeyse hepsi gecikiyor. O ekip çıkarılınca "
        "model, “en eski kayıt önce” kuralına sıralamada ancak denk; farkı olasılık verebilmesi."
    )
    st.dataframe(quality_table(rest), hide_index=True, column_config=fmt)

    st.subheader("4. Olasılıklar tutuyor mu?")
    st.image(str(FIGURES / "calibration_tr.png"))
    st.markdown(
        "Yalnız çözülmüş kayıtlarla kurulan regresyon fazla iyimser: açık kalanları hiç görmediği için "
        "çözülme olasılığını olduğundan yüksek veriyor. Sağkalım ormanı ayın büyük kısmında gerçeğe yakın; "
        "Mayıs'ın son haftasında işler yavaşlayınca o da iyimser kalıyor."
    )

with tab_findings:
    st.subheader("1. Çözülmüş kayıtlardan hesaplanan ortalama, işlerin ne kadar sürdüğünü olduğundan kısa gösterir")
    st.image(str(FIGURES / "mean_trap_tr.png"))
    st.subheader("2. “Ortalama 7 gün, 5 gündür açık, 2 günü kaldı” mantığı yanlış")
    st.image(str(FIGURES / "waiting_tr.png"))

st.divider()
st.caption(
    "Veri: Amaral, C., Fantinato, M., Peres, S. (2018). Incident management process enriched event log. "
    "UCI Machine Learning Repository. DOI 10.24432/C57S4H. Lisans CC BY 4.0."
)
