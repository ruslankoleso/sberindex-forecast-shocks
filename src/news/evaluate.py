"""Польза новостей для обнаружения шоков (объяснение и раннее предупреждение).

1. Объяснение: у шоков, найденных лучшим детектором (по ошибке прогноза), ищем новости
   про это МО или его регион по «значимым» темам в окне [месяц сдвига − 1, месяц тревоги].
   Сравниваем с тем же показателем у случайных пар МО-месяц без шока (базовый уровень).
2. Раннее предупреждение: для всех МО и месяцев 2024 г. сравниваем вероятность тревоги
   детектора в ближайшие 1–3 месяца после месяца с «значимыми» новостями про МО/регион
   и без них (относительный риск). Новости берутся только за месяцы до прогнозного окна.
Запуск: python -m src.news.evaluate
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

TOPICS = ["emergency", "industry_neg", "industry_pos", "social_pay"]


def load(cfg):
    out = Path(cfg["monthly_file"]).parent
    reg = pd.read_parquet(out / "news_region_monthly.parquet")
    mo = pd.read_parquet(out / "news_mo_monthly.parquet")
    ref = pd.read_parquet("data/external/territory_reference.parquet").set_index("territory_id")
    return reg, mo, ref


def news_count(tid, months, reg, mo, ref, topics=TOPICS, level="both"):
    """Число новостей значимых тем про МО (и/или его регион) за заданные месяцы."""
    n = 0.0
    if level in ("both", "mo"):
        m = mo[(mo.territory_id == tid) & mo.month.isin(months)]
        n += m[topics].sum().sum()
    if level in ("both", "region"):
        r = ref.loc[tid, "region"] if tid in ref.index else None
        if isinstance(r, str):
            g = reg[(reg.region == r) & reg.month.isin(months)]
            n += g[topics].sum().sum()
    return n


def explanation(cfg, seed=42):
    reg, mo, ref = load(cfg)
    sh = pd.read_parquet("data/interim/changepoint/panel_shocks.parquet")
    sh = sh[(sh.detector == "base_residual") & ~sh.seasonal & (sh.shift_pct.abs() >= 10)]
    rows = []
    for r in sh.itertuples():
        months = pd.date_range(r.shift_month - pd.DateOffset(months=1), r.alarm, freq="MS")
        rows.append(dict(territory_id=r.territory_id, mo=r.mo_name, region=r.region, shift_month=r.shift_month,
                         shift_pct=r.shift_pct,
                         news_mo=news_count(r.territory_id, months, reg, mo, ref, level="mo"),
                         news_region=news_count(r.territory_id, months, reg, mo, ref, level="region")))
    ex = pd.DataFrame(rows)
    # базовый уровень: случайные МО и окна той же длины в 2024 г., без шока
    rng = np.random.default_rng(seed)
    terr = pd.read_parquet("data/interim/territories.parquet")
    full = terr[terr.months == 24].territory_id.values
    shocked = set(sh.territory_id)
    base = []
    for _ in range(1000):
        tid = rng.choice([t for t in full if t not in shocked])
        start = pd.Timestamp("2024-01-01") + pd.DateOffset(months=int(rng.integers(0, 10)))
        months = pd.date_range(start, start + pd.DateOffset(months=2), freq="MS")
        base.append(dict(news_mo=news_count(tid, months, reg, mo, ref, level="mo"),
                         news_region=news_count(tid, months, reg, mo, ref, level="region")))
    base = pd.DataFrame(base)
    return ex, base


def early_warning(cfg, topics=("emergency", "industry_neg"), horizon=3):
    """Относительный риск: P(тревога в месяцы m+1..m+horizon | были новости в месяце m)
    / P(тревога в m+1..m+horizon | новостей не было). Только МО с полной историей, m ∈ 2024."""
    reg, mo, ref = load(cfg)
    sh = pd.read_parquet("data/interim/changepoint/panel_shocks.parquet")
    al = sh[(sh.detector == "base_residual") & ~sh.seasonal][["territory_id", "alarm"]]
    first_alarm = al.groupby("territory_id").alarm.min()
    terr = pd.read_parquet("data/interim/territories.parquet")
    full = terr[terr.months == 24].territory_id.values
    regc = reg.assign(k=reg[list(topics)].sum(axis=1)).pivot_table(index="region", columns="month", values="k", aggfunc="sum")
    moc = mo.assign(k=mo[list(topics)].sum(axis=1)).pivot_table(index="territory_id", columns="month", values="k", aggfunc="sum")
    rows = []
    for m in pd.date_range("2024-01-01", "2024-09-01", freq="MS"):
        win = pd.date_range(m + pd.DateOffset(months=1), m + pd.DateOffset(months=horizon), freq="MS")
        for tid in full:
            fa = first_alarm.get(tid)
            if fa is not None and fa <= m:                # шок уже был раньше — не раннее предупреждение
                continue
            r = ref.loc[tid, "region"] if tid in ref.index else None
            n = (moc.loc[tid, m] if tid in moc.index and m in moc.columns else 0) or 0
            n += (regc.loc[r, m] if isinstance(r, str) and r in regc.index and m in regc.columns else 0) or 0
            rows.append(dict(territory_id=tid, month=m, news=n > 0, alarm=fa is not None and fa in win))
    d = pd.DataFrame(rows)
    p1, p0 = d[d.news].alarm.mean(), d[~d.news].alarm.mean()
    return dict(n_with_news=int(d.news.sum()), n_without=int((~d.news).sum()),
                p_alarm_with_news=p1, p_alarm_without=p0, relative_risk=p1 / p0 if p0 > 0 else np.nan)


if __name__ == "__main__":
    cfg = yaml.safe_load(Path("configs/news.yaml").read_text(encoding="utf-8"))
    ex, base = explanation(cfg)
    pd.set_option("display.width", 220)
    print("Шоки (лучший детектор):", len(ex))
    for lvl in ("news_mo", "news_region"):
        print(f"  доля с новостями ({lvl}): шоки {(ex[lvl] > 0).mean():.0%}, случайные МО {(base[lvl] > 0).mean():.0%};"
              f" среднее число: {ex[lvl].mean():.1f} против {base[lvl].mean():.1f}")
    print(ex.sort_values("news_mo", ascending=False).head(12).to_string(index=False))
    ew = early_warning(cfg)
    print("Раннее предупреждение (новости о ЧС/закрытии производств в месяце m → тревога в m+1..m+3):")
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in ew.items()})


def news_spikes(cfg, topics=("emergency", "industry_neg", "social_pay"), min_count=3, ratio=3.0):
    """Всплеск новостей о регионе: в месяце не меньше min_count значимых новостей и в ratio раз
    больше медианы этого региона по всем месяцам. Возвращает (регион, месяц, число)."""
    reg, _, _ = load(cfg)
    reg = reg.assign(k=reg[list(topics)].sum(axis=1))
    w = reg.pivot_table(index="region", columns="month", values="k", aggfunc="sum").fillna(0)
    med = w.median(axis=1).clip(lower=1)
    sp = w.stack().rename("k").reset_index()
    sp = sp[(sp.k >= min_count) & (sp.k >= ratio * sp.region.map(med))]
    return sp


def residual_response(cfg, lags=(0, 1, 2)):
    """Средняя |стандартизованная ошибка прогноза основы| у МО региона в месяцы m+lag
    после всплеска новостей о регионе в месяце m — против МО без всплеска в тот же месяц."""
    import yaml as _y
    from src.changepoint.benchmark import panel_sigma
    from src.eval.cv import load_wide
    from src.forecast.lgbm_model import load_national
    Y = load_wide(_y.safe_load(Path("configs/eval.yaml").read_text(encoding="utf-8")))
    L = load_national(_y.safe_load(Path("configs/forecast.yaml").read_text(encoding="utf-8")))
    sig, cen = panel_sigma(Y, L)
    lY = np.log(Y.values)
    E = np.full(lY.shape, np.nan)
    for t in range(12, len(Y)):
        g = L[Y.index[t - 1]] - L[Y.index[t - 1] - pd.DateOffset(years=1)]
        E[t] = (lY[t] - (lY[t - 12] + g) - cen[t]) / sig[t]
    E = pd.DataFrame(np.abs(E), index=Y.index, columns=Y.columns)
    ref = pd.read_parquet("data/external/territory_reference.parquet").set_index("territory_id")
    region = ref["region"].reindex(Y.columns)
    sp = news_spikes(cfg)
    rows = []
    for lag in lags:
        for r in sp.itertuples():
            t = r.month + pd.DateOffset(months=lag)
            if t not in E.index or t < pd.Timestamp("2024-01-01"):
                continue
            inreg = region == r.region
            if inreg.sum() == 0:
                continue
            rows.append(dict(lag=lag, region=r.region, month=r.month, n_mo=int(inreg.sum()),
                             err_region=E.loc[t, inreg.values].mean(), err_other=E.loc[t, ~inreg.values].mean()))
    return pd.DataFrame(rows), sp
