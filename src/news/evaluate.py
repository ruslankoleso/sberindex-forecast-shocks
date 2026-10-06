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


if __name__ == "__main__":
    cfg = yaml.safe_load(Path("configs/news.yaml").read_text(encoding="utf-8"))
    ex, base = explanation(cfg)
    pd.set_option("display.width", 220)
    print("Шоки (лучший детектор):", len(ex))
    for lvl in ("news_mo", "news_region"):
        print(f"  доля с новостями ({lvl}): шоки {(ex[lvl] > 0).mean():.0%}, случайные МО {(base[lvl] > 0).mean():.0%};"
              f" среднее число: {ex[lvl].mean():.1f} против {base[lvl].mean():.1f}")
    print(ex.sort_values("news_mo", ascending=False).head(12).to_string(index=False))
