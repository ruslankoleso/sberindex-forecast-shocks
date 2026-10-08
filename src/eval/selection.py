"""Многокритериальный выбор модели прогноза и детектора шоков.

Для каждого кандидата — набор критериев (все приведены к виду «меньше — лучше»); кандидат
доминируется, если другой не хуже по всем критериям и строго лучше хотя бы по одному.
Недоминируемые (множество Парето) — разумные варианты выбора. Дополнительно — правила
агрегирования из теории выбора: Борда (сумма мест по критериям) и Коупленд (попарные сравнения
по большинству критериев). Если Парето, Борда и Коупленд согласны, выбор устойчив к способу
агрегирования. Сравнение на национальном ряде в рейтинг детекторов не входит: детектор по
остаткам прогноза работает по МО (на ряде России используется BOCPD).
Время расчёта — приблизительное, на одну точку прогноза для 2 016 МО на CPU ноутбука.
Запуск: python -m src.eval.selection
"""
import numpy as np
import pandas as pd

from src.eval.metrics import summarize

MODELS = {   # модель: (название, время расчёта на точку прогноза, с; интерпретируемость 1 — формула … 3 — «чёрный ящик»)
    "prophet": ("Prophet", 20, 2), "snaive_natg": ("Сезонная наивная с дрейфом", 0.1, 1),
    "lgbm_catmo": ("LightGBM", 15, 2), "lgbm_resid": ("Сез. наивная + LightGBM по остаткам", 15, 2),
    "autotheta": ("AutoTheta", 5, 1), "chronos2_ft_mix": ("Chronos-2 дообученный", 150, 3),
    "timesfm_lora": ("TimesFM дообученный (LoRA)", 360, 3), "tirex_bc_mix": ("TiRex", 240, 3),
    "ens_inv_mae": ("Ансамбль (веса по обратной ошибке)", 900, 3)}


def pareto(df, cols):
    X = df[cols].values
    dom = np.zeros(len(X), bool)
    for i in range(len(X)):
        for j in range(len(X)):
            if i != j and np.all(X[j] <= X[i]) and np.any(X[j] < X[i]):
                dom[i] = True
                break
    return ~dom


def borda_copeland(t, cols):
    """Правило Борда (сумма мест по критериям, меньше — лучше) и правило Коупленда
    (число побед минус поражений в попарных сравнениях по большинству критериев)."""
    R = t[cols].rank(method="average")
    borda = R.sum(axis=1)
    X = t[cols].values
    cop = np.zeros(len(X))
    for i in range(len(X)):
        for j in range(len(X)):
            if i == j:
                continue
            wins, losses = (X[i] < X[j]).sum(), (X[i] > X[j]).sum()
            cop[i] += 1 if wins > losses else (-1 if wins < losses else 0)
    return borda, pd.Series(cop, index=t.index)


def models():
    res = pd.concat(pd.read_parquet(f"data/interim/forecasts/{m}.parquet") for m in MODELS)
    s = summarize(res).pivot(index="model", columns="h", values="MAE")
    t = pd.DataFrame({f"MAE {h} мес.": s[h] for h in (1, 3, 6, 12)})
    # устойчивость: разброс MAE по точкам прогноза на 1 мес. (стандартное отклонение)
    stab = {}
    for m in MODELS:
        f = pd.read_parquet(f"data/interim/forecasts/{m}.parquet"); f = f[f.h == 1]
        stab[m] = (f.y_true - f.y_pred).abs().groupby(f.origin).mean().std()
    t["разброс MAE по месяцам"] = pd.Series(stab)
    t["время, с"] = pd.Series({m: v[1] for m, v in MODELS.items()})
    t["интерпретируемость"] = pd.Series({m: v[2] for m, v in MODELS.items()})
    t["Парето (точность)"] = pareto(t, [f"MAE {h} мес." for h in (1, 3, 6, 12)])
    t["Парето (все критерии)"] = pareto(t, list(t.columns[:7]))
    acc = [f"MAE {h} мес." for h in (1, 3, 6, 12)]
    t["Борда (точность)"], t["Коупленд (точность)"] = borda_copeland(t, acc)
    t.index = [MODELS[m][0] for m in t.index]
    return t.round(1).sort_values("Борда (точность)")


def detectors():
    b = pd.read_csv("data/interim/changepoint/benchmark_summary.csv", index_col=0)
    n = pd.read_parquet("data/interim/changepoint/national_scores.parquet")
    ev = pd.to_datetime(pd.Series(["2020-04-01", "2022-03-01"]))
    def extra(g):
        al = g[g.alarm].month
        near = np.any([(al >= e - pd.DateOffset(months=1)) & (al <= e + pd.DateOffset(months=6)) for e in ev], axis=0)
        return int((~near).sum())
    nat = n.groupby("detector").apply(extra)
    t = pd.DataFrame({"пропуск шоков": 1 - b["доля_найденных"], "задержка, мес.": b["задержка_мес"],
                      "тревога до шока": b["тревога_до_шока"], "реакция на выброс": b["реакция_на_выброс"],
                      "лишние тревоги по России": nat.reindex(b.index)})
    t = t.loc[[d for d in ["base_residual", "pelt", "forecast_residual", "bocpd", "page_hinkley", "cusum"] if d in t.index]]
    core = ["пропуск шоков", "задержка, мес.", "тревога до шока", "реакция на выброс"]
    t["Парето"] = pareto(t, core)
    t["Борда"], t["Коупленд"] = borda_copeland(t, core)
    t.index = t.index.map({"base_residual": "По остаткам прогноза", "pelt": "PELT", "forecast_residual": "По остаткам скользящего среднего",
                           "bocpd": "BOCPD", "page_hinkley": "Пейдж–Хинкли", "cusum": "CUSUM"})
    return t.round(3).sort_values("Борда")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    m, d = models(), detectors()
    print(m.to_string()); print(); print(d.to_string())
    m.to_csv("data/interim/selection_models.csv"); d.to_csv("data/interim/selection_detectors.csv")
