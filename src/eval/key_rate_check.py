"""Проверка: улучшает ли поправка на ключевую ставку прогноз роста трат России.
Точки прогноза 2021-12…2026-07, горизонты 1/3/6/12; ошибка годового роста, п.п.
Запуск: python -m src.eval.key_rate_check"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.forecast.key_rate import adjusted_growth, growth, load_series

if __name__ == "__main__":
    cfg = yaml.safe_load(Path("configs/forecast.yaml").read_text(encoding="utf-8"))["key_rate"]
    L, r = load_series()
    rows = []
    for o in L.index[(L.index >= "2021-12-01")]:
        for h in (1, 3, 6, 12):
            t = o + pd.DateOffset(months=h)
            if t not in L.index:
                continue
            true = growth(L, t)
            rows.append(dict(origin=o, h=h, метод="рост сохранится", err=abs(growth(L, o) - true)))
            for k in [cfg["window"]] + cfg["sensitivity_windows"]:
                g, b, n = adjusted_growth(L, r, o, h, k, cfg["min_train"])
                rows.append(dict(origin=o, h=h, метод=f"+ ставка, окно {k} мес.", err=abs(g - true), b=b, n=n))
    d = pd.DataFrame(rows)
    d["err"] *= 100
    print("Ошибка прогноза годового роста трат России, п.п. (точек:", d.origin.nunique(), ")")
    print(d.pivot_table(index="метод", columns="h", values="err").round(2).to_string())
    d["год"] = d.origin.dt.year
    print("\nПо годам, h=6:")
    print(d[d.h == 6].pivot_table(index="метод", columns="год", values="err").round(1).to_string())
    k = cfg["window"]
    bb = d[(d.метод == f"+ ставка, окно {k} мес.")].groupby("h").b.agg(["first", "last"])
    print(f"\nКоэффициент b (окно {k}): изменение роста трат на 1 п.п. изменения ставки — в начале и в конце периода")
    print(bb.round(3).to_string())
    d.to_parquet("data/interim/key_rate_check.parquet", index=False)
