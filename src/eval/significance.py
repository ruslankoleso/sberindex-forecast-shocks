"""Значимость различий в точности прогноза (тест Диболда–Мариано и парный тест по МО).

Потери — абсолютные ошибки. Для каждой пары моделей и горизонта:
1. Диболд–Мариано (DM) по точкам прогноза: ряд разностей средних потерь по МО в каждой точке;
   HAC-оценка дисперсии (Ньюи–Уэст с весами Бартлетта, лаг h−1), поправка Харви–Лейбурна–Ньюболда на малую выборку.
2. Парный тест по МО: средняя по точкам прогноза разность потерь для каждого МО, знаковый
   тест Уилкоксона (2 016 пар) — устойчив к выбросам. МО не независимы (общие шоки страны),
   поэтому его p-значения занижены; основным считаем DM, а Уилкоксон и долю МО — дополнением.
Запуск: python -m src.eval.significance
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

FC = Path("data/interim/forecasts")
PAIRS = [("ens_inv_mae", "prophet"), ("snaive_natg", "prophet"), ("chronos2_ft_mix", "prophet"),
         ("ens_inv_mae", "snaive_natg"), ("chronos2_ft_mix", "snaive_natg")]


def dm_test(d, h):
    """d — ряд разностей потерь по точкам прогноза (A − B); H0: E[d] = 0."""
    T = len(d)
    if T < 3:
        return np.nan, np.nan
    dbar = d.mean()
    gamma = [np.mean((d[k:] - dbar) * (d[:T - k] - dbar)) for k in range(h)]
    var = (gamma[0] + 2 * sum((1 - k / h) * gamma[k] for k in range(1, h))) / T   # веса Бартлетта (Ньюи–Уэст)
    if var <= 0:
        return np.nan, np.nan
    dm = dbar / np.sqrt(var)
    k = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)          # поправка HLN
    stat = dm * k
    return stat, 2 * stats.t.sf(abs(stat), T - 1)


def main():
    F = {}
    for m in {x for p in PAIRS for x in p}:
        f = pd.read_parquet(FC / f"{m}.parquet")
        f["ae"] = (f.y_true - f.y_pred).abs()
        F[m] = f.set_index(["origin", "h", "territory_id"])["ae"]
    rows = []
    for a, b in PAIRS:
        diff = (F[a] - F[b]).dropna().rename("d").reset_index()
        for h, g in diff.groupby("h"):
            per_origin = g.groupby("origin").d.mean().values
            stat, p_dm = dm_test(per_origin, int(h))
            per_mo = g.groupby("territory_id").d.mean()
            p_w = stats.wilcoxon(per_mo).pvalue
            rows.append(dict(A=a, B=b, h=int(h), разница_MAE=round(g.d.mean(), 1), точек=len(per_origin),
                             DM=round(stat, 2) if np.isfinite(stat) else None,
                             p_DM=round(p_dm, 3) if np.isfinite(p_dm) else None,
                             доля_МО_где_A_лучше=round((per_mo < 0).mean(), 3), p_Уилкоксон=float(f"{p_w:.2g}")))
    r = pd.DataFrame(rows)
    r.to_csv("data/interim/significance.csv", index=False)
    print(r.to_string(index=False))


if __name__ == "__main__":
    main()
