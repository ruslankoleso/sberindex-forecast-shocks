"""Детектор шоков, учитывающий новости (байесовская логика: новость о крупном событии
повышает априорную вероятность шока в МО региона).

Базовый детектор — по остаткам прогноза сезонной наивной модели с дрейфом (сигнал s_t —
минимальная |e| за 2 мес. подряд одного знака). Обычный порог — уровень 10 % ложных тревог.
Если о регионе МО в месяцы [t−2, t] вышли новости о крупном событии (configs/news.yaml →
major_events; только опубликованные до конца месяца t), порог снижается до уровня far_low
и не применяется фильтр собственной сезонности.

Оценка:
1. Реальные события: МО, названные в новостях о крупных событиях, и МО их регионов — нашёл ли
   детектор сдвиг и с какой задержкой от события (с оговоркой: метки и сниженный порог
   построены по одним новостям, оценка выигрыша оптимистична).
2. Цена: сколько тревог добавляется во всех МО регионов с событиями и в целом.
Запуск: python -m src.changepoint.news_informed
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.news.events import _cfg as news_cfg, major_events, residuals


def detector_signal(E):
    """s_t = min(|e_{t−1}|, |e_t|), если знаки совпадают, иначе 0 (как base_residual_scores)."""
    A = E.values
    S = np.zeros_like(A)
    for t in range(1, len(A)):
        same = np.sign(A[t - 1]) == np.sign(A[t])
        S[t] = np.where(same & ~np.isnan(A[t - 1]) & ~np.isnan(A[t]), np.minimum(np.abs(A[t - 1]), np.abs(A[t])), 0)
    return pd.DataFrame(S, index=E.index, columns=E.columns)


def seasonal_flags(Y, shift_month_idx, tid):
    """Повторяется ли скачок того же знака в тот же месяц прошлого года (≥ половины величины)."""
    x = np.log(Y[tid].values) - np.log(Y.median(axis=1).values)
    t = shift_month_idx
    j = lambda t: x[t:t + 2].mean() - x[t - 2:t].mean() if t - 2 >= 0 and t + 2 <= len(x) else np.nan
    a, b = j(t), j(t - 12)
    return bool(np.isfinite(b) and np.sign(a) == np.sign(b) and abs(b) >= 0.5 * abs(a))


def run():
    cfg = yaml.safe_load(Path("configs/changepoint.yaml").read_text(encoding="utf-8"))
    pc = cfg["news_prior"]
    E = residuals()
    S = detector_signal(E)
    months = E.index[E.index >= "2024-01-01"]
    thr = pd.read_csv("data/interim/changepoint/thresholds.csv", index_col=0)["threshold"]["base_residual"]
    # сниженный порог: тот же способ калибровки (максимум сигнала за 2024 г. по всем МО), уровень far_low
    mx = S.loc[months].max()
    thr_low = float(np.quantile(mx, 1 - pc["far_low"]))
    ncfg = news_cfg()
    ev, _ = major_events(ncfg, E)
    ref = pd.read_parquet("data/external/territory_reference.parquet").set_index("territory_id")
    region = ref["region"].reindex(E.columns)
    ev_reg = ev.drop_duplicates(["region", "month"])[["region", "month"]]
    prior = pd.DataFrame(False, index=E.index, columns=E.columns)
    for r in ev_reg.itertuples():
        for k in range(pc["window_months"]):
            t = r.month + pd.DateOffset(months=k)
            if t in prior.index:
                prior.loc[t, region == r.region] = True
    from src.eval.cv import load_wide
    Y = load_wide(yaml.safe_load(Path("configs/eval.yaml").read_text(encoding="utf-8")))
    idx = {m: i for i, m in enumerate(E.index)}
    rows = []
    for tid in E.columns:
        for mode in ("обычный", "с новостями"):
            alarm = None
            for t in months:
                use_low = mode == "с новостями" and prior.at[t, tid]
                if S.at[t, tid] > (thr_low if use_low else thr):
                    seas = seasonal_flags(Y, idx[t] - 1, tid)
                    if seas and not (use_low and pc["override_seasonal"]):
                        continue
                    alarm = t
                    break
            rows.append(dict(territory_id=tid, режим=mode, alarm=alarm))
    al = pd.DataFrame(rows)
    return al, ev, prior, thr, thr_low, region


if __name__ == "__main__":
    al, ev, prior, thr, thr_low, region = run()
    print(f"порог обычный {thr:.2f}, сниженный {thr_low:.2f}")
    piv = al.pivot(index="territory_id", columns="режим", values="alarm")
    print("тревог всего:", piv.notna().sum().to_dict())
    ev_regions = set(ev.region)
    in_ev = piv[region.reindex(piv.index).isin(ev_regions).values]
    print(f"МО в регионах с крупными событиями: {len(in_ev)}; тревог там:", in_ev.notna().sum().to_dict())
    # реальные события: названные в новостях МО и все МО региона события
    out = []
    for r in ev.drop_duplicates(["territory_id", "month"]).itertuples():
        for mode in ("обычный", "с новостями"):
            a = piv.at[r.territory_id, mode] if r.territory_id in piv.index else None
            ok = a is not None and pd.notna(a) and r.month <= a <= r.month + pd.DateOffset(months=4)
            lag = ((a.year - r.month.year) * 12 + a.month - r.month.month) if ok else np.nan
            out.append(dict(territory_id=r.territory_id, region=r.region, month=r.month, named=r.named,
                            режим=mode, найден=ok, задержка=lag))
    o = pd.DataFrame(out)
    print("\nНайден сдвиг в течение 4 мес. после крупного события (доля МО):")
    print(o.pivot_table(index="named", columns="режим", values="найден", aggfunc="mean").round(3).rename(
        index={True: "названные в новостях города", False: "все МО региона"}).to_string())
    print("средняя задержка, мес.:", o[o.найден].groupby("режим").задержка.mean().round(2).to_dict())
    ref = pd.read_parquet("data/external/territory_reference.parquet").set_index("territory_id")
    named = o[o.named].copy(); named["МО"] = named.territory_id.map(ref.mo_name).str[:35]
    print(named.pivot_table(index=["МО", "month"], columns="режим", values="задержка", aggfunc="first").to_string())
    Path("data/interim/changepoint").mkdir(parents=True, exist_ok=True)
    pd.to_pickle(dict(alarms=al, events_eval=o, thr=thr, thr_low=thr_low), "data/interim/changepoint/news_informed.pkl")
