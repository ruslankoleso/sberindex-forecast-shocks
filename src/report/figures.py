"""Графики для отчёта и презентации → reports/figures/*.png.
Запуск: python -m src.report.figures (после расчёта прогнозов, ансамбля и детекторов)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

OUT = Path("reports/figures")
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8b8a85", "#e7e6e1"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"   # проверенная палитра, слоты 1–4


def _style(ax, title, ylabel=None):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _dates(ax, step=3):
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=range(1, 13, step)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m.%y"))


def _fig(w=8, h=4.2, n=1):
    fig, axes = plt.subplots(1, n, figsize=(w, h), facecolor=SURFACE)
    return fig, axes


def mae_by_horizon():
    s = pd.read_csv("data/interim/metrics.csv")
    ens = pd.concat(pd.read_parquet(f"data/interim/forecasts/{m}.parquet") for m in ["ens_inv_mae"])
    from src.eval.metrics import summarize
    s = pd.concat([s[s.model != "ens_inv_mae"], summarize(ens)])
    t = s.pivot(index="model", columns="h", values="MAE")
    rows = [("prophet", "Prophet (базовая модель)", ORANGE), ("chronos2_ft", "Chronos-2 дообученный", YELLOW),
            ("snaive_natg", "Сезонная наивная с дрейфом", AQUA), ("ens_inv_mae", "Ансамбль (веса по обратной ошибке)", BLUE)]
    fig, ax = _fig(9, 4.6)
    hs = [1, 3, 6, 12]
    w = 0.2
    for k, (m, label, c) in enumerate(rows):
        vals = t.loc[m, hs].values
        x = np.arange(len(hs)) + (k - 1.5) * w
        ax.bar(x, vals, width=w - 0.02, color=c, label=label, edgecolor=SURFACE, linewidth=1)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 60, f"{v:,.0f}".replace(",", " "), ha="center", fontsize=7, color=INK2)
    ax.set_xticks(range(len(hs)), [f"{h} мес." for h in hs])
    _style(ax, "Ошибка прогноза (MAE) по горизонтам — меньше лучше", "руб. на жителя в месяц")
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper left")
    ax.set_ylim(0, t.loc[[r[0] for r in rows], hs].values.max() * 1.18)
    fig.tight_layout(); fig.savefig(OUT / "01_mae_by_horizon.png", dpi=160); plt.close(fig)


def example_forecast(tid=1, name="Майкоп"):
    from src.eval.cv import load_wide
    Y = load_wide(yaml.safe_load(Path("configs/eval.yaml").read_text(encoding="utf-8")))
    fig, ax = _fig(9, 4.2)
    ax.plot(Y.index, Y[tid], color=INK, linewidth=2, label="Факт")
    for m, label, c in [("prophet", "Prophet, прогноз на 1 мес.", ORANGE), ("ens_inv_mae", "Ансамбль, прогноз на 1 мес.", BLUE)]:
        f = pd.read_parquet(f"data/interim/forecasts/{m}.parquet")
        f = f[(f.h == 1) & (f.territory_id == tid)].sort_values("target")
        ax.plot(f.target, f.y_pred, color=c, linewidth=2, marker="o", markersize=4, label=label)
    _style(ax, f"{name}: факт и прогнозы на месяц вперёд (2024)", "руб. на жителя в месяц")
    _dates(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "02_example_forecast.png", dpi=160); plt.close(fig)


def detectors():
    s = pd.read_csv("data/interim/changepoint/benchmark_summary.csv", index_col=0)
    names = {"base_residual": "По остаткам прогноза", "pelt": "PELT", "forecast_residual": "По остаткам скользящего среднего",
             "bocpd": "BOCPD", "page_hinkley": "Пейдж–Хинкли", "cusum": "CUSUM"}
    s = s.loc[[k for k in names if k in s.index]].iloc[::-1]
    fig, (a1, a2) = _fig(11, 4.2, 2)
    lab = [names[k] for k in s.index]
    for ax, col, title, good in [(a1, "доля_найденных", "Находит шоков, % ↑", True),
                                 (a2, "реакция_на_выброс", "Реагирует на выброс, % ↓", False)]:
        cols = [BLUE if k == "base_residual" else MUTED for k in s.index]
        ax.barh(lab, s[col] * 100, color=cols, height=0.6)
        for i, v in enumerate(s[col] * 100):
            ax.text(v + 1, i, f"{np.floor(v + 0.5):.0f} %", va="center", fontsize=8, color=INK2)
        _style(ax, title)
        ax.grid(axis="x", color=GRID); ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 100)
    fig.tight_layout(); fig.savefig(OUT / "03_detectors.png", dpi=160); plt.close(fig)


def shock_examples(items=((1305, "Волгореченск"), (1673, "Орск"))):
    from src.eval.cv import load_wide
    from src.forecast.lgbm_model import load_national
    Y = load_wide(yaml.safe_load(Path("configs/eval.yaml").read_text(encoding="utf-8")))
    L = load_national(yaml.safe_load(Path("configs/forecast.yaml").read_text(encoding="utf-8")))
    sh = pd.read_parquet("data/interim/changepoint/panel_shocks.parquet")
    fig, axes = _fig(11, 4, len(items))
    for ax, (tid, name) in zip(np.atleast_1d(axes), items):
        y = Y[tid]
        base = [y.iloc[t - 12] * np.exp(L[Y.index[t - 1]] - L[Y.index[t - 1] - pd.DateOffset(years=1)]) for t in range(12, 24)]
        ax.plot(Y.index, y, color=INK, linewidth=2, label="Факт")
        ax.plot(Y.index[12:], base, color=BLUE, linewidth=2, linestyle="--", label="Прогноз сезонной наивной с дрейфом на 1 мес.")
        a = sh[(sh.territory_id == tid) & (sh.detector == "base_residual")]
        if len(a):
            ax.axvline(a.alarm.iloc[0], color=ORANGE, linewidth=2)
            ax.text(a.alarm.iloc[0], ax.get_ylim()[1], " тревога", color=INK2, fontsize=8, va="top")
        _style(ax, name, "руб. на жителя в месяц")
        _dates(ax)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "04_shock_examples.png", dpi=160); plt.close(fig)


def national_events():
    from src.changepoint.national_events import load_sa
    x = np.exp(load_sa())
    r = pd.read_parquet("data/interim/changepoint/national_scores.parquet")
    al = r[(r.detector == "bocpd") & r.alarm].month
    fig, ax = _fig(10, 4)
    ax.plot(x.index, x, color=INK, linewidth=2, label="Индекс трат (сезонно скорректированный)")
    for m in al:
        ax.axvline(m, color=ORANGE, linewidth=1.5, alpha=0.8)
    ax.plot([], [], color=ORANGE, linewidth=1.5, label="Тревога BOCPD")
    _style(ax, "Россия: реальные кризисы и тревоги детектора BOCPD", "индекс, дек. 2018 = 100")
    _dates(ax, 12)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "05_national_events.png", dpi=160); plt.close(fig)


def news_case(tid=1673, region="Оренбургской области", name="Орск"):
    """Пример согласования новостей и данных: всплеск новостей о паводке → ошибка прогноза МО."""
    from src.changepoint.benchmark import panel_sigma
    from src.eval.cv import load_wide
    from src.forecast.lgbm_model import load_national
    Y = load_wide(yaml.safe_load(Path("configs/eval.yaml").read_text(encoding="utf-8")))
    L = load_national(yaml.safe_load(Path("configs/forecast.yaml").read_text(encoding="utf-8")))
    sig, cen = panel_sigma(Y, L)
    ly = np.log(Y[tid].values)
    e = [(ly[t] - (ly[t - 12] + L[Y.index[t - 1]] - L[Y.index[t - 1] - pd.DateOffset(years=1)]) - cen[t]) / sig[t]
         for t in range(12, 24)]
    reg = pd.read_parquet("data/external/news/news_region_monthly.parquet")
    reg = reg[(reg.region == region) & (reg.month >= "2023-07-01")]
    k = reg.set_index("month")[["emergency", "social_pay", "industry_neg"]].sum(axis=1)
    thr = pd.read_csv("data/interim/changepoint/thresholds.csv", index_col=0)["threshold"]["base_residual"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.5), facecolor=SURFACE, sharex=True)
    a1.bar(k.index, k.values, width=20, color=ORANGE)
    _style(a1, f"Новости о ЧС и выплатах: {region.replace('ской области', 'ская область')}", "число заголовков")
    a2.plot(Y.index[12:], e, color=BLUE, linewidth=2, marker="o", markersize=4)
    a2.axhline(thr, color=MUTED, linestyle="--", linewidth=1)
    a2.text(Y.index[12], thr + 0.1, "порог тревоги", fontsize=8, color=INK2)
    _style(a2, f"{name}: ошибка прогноза относительно других МО (в сигмах)", "σ")
    _dates(a2, 2)
    fig.tight_layout(); fig.savefig(OUT / "06_news_case_orsk.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "DejaVu Sans"
    for f in (mae_by_horizon, example_forecast, detectors, shock_examples, national_events, news_case):
        f(); print("готово:", f.__name__)
