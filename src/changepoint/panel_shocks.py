"""Поиск шоков в данных МО за 2023–2024 (главный анализ второй задачи).

Для каждого из 2 016 МО с полной историей:
  сигнал x_t = log(траты МО) − медиана log(трат) всех МО в месяц t;
  детекторы — из src/changepoint/detectors.py, пороги — из бенчмарка (10 % ложных тревог);
  первая тревога детектора = обнаруженный шок (месяц и направление).
Фильтр «своей сезонности»: если в тот же календарный месяц другого года сигнал
скачет так же (тот же знак, не меньше половины величины), это сезонность МО, а не шок.
Запуск: python -m src.changepoint.panel_shocks
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.changepoint.benchmark import relative_signal
from src.changepoint.detectors import DETECTORS
from src.eval.cv import load_wide

MAIN = ["forecast_residual", "pelt", "bocpd"]


def jump(x, t, k=2):
    """Изменение уровня в месяце t: среднее t..t+k−1 минус среднее t−k..t−1."""
    if t - k < 0 or t + k > len(x):
        return np.nan
    return x[t:t + k].mean() - x[t - k:t].mean()


def run(warmup=8):
    Y = load_wide(yaml.safe_load(Path("configs/eval.yaml").read_text(encoding="utf-8")))
    X = relative_signal(Y)
    thr = pd.read_csv("data/interim/changepoint/thresholds.csv", index_col=0)["threshold"]
    ref = pd.read_parquet("data/external/territory_reference.parquet").set_index("territory_id")
    rows = []
    for c in X.columns:
        x = X[c].values
        for d in MAIN:
            s = DETECTORS[d](x, warmup=warmup)
            idx = np.where(s[warmup:] > thr[d])[0]
            if not len(idx):
                continue
            a = warmup + idx[0]
            # дата начала сдвига: месяц наибольшего скачка в окне [a−3, a]
            cands = [t for t in range(max(warmup, a - 3), a + 1)]
            t0 = max(cands, key=lambda t: abs(np.nan_to_num(jump(x, t))))
            j = jump(x, t0)
            other = t0 - 12 if t0 >= 12 else t0 + 12
            jo = jump(x, other)
            seasonal = bool(np.isfinite(jo) and np.sign(jo) == np.sign(j) and abs(jo) >= 0.5 * abs(j))
            rows.append(dict(territory_id=c, detector=d, alarm=X.index[a], shift_month=X.index[t0],
                             shift_pct=100 * (np.exp(j) - 1), same_month_other_year_pct=100 * (np.exp(jo) - 1)
                             if np.isfinite(jo) else np.nan, seasonal=seasonal))
    r = pd.DataFrame(rows).merge(ref[["mo_name", "region"]], left_on="territory_id", right_index=True, how="left")
    return r, X.shape[1]


if __name__ == "__main__":
    r, n = run()
    out = Path("data/interim/changepoint"); out.mkdir(parents=True, exist_ok=True)
    r.to_parquet(out / "panel_shocks.parquet", index=False)
    print(f"МО проверено: {n}")
    for d, g in r.groupby("detector"):
        real = g[~g.seasonal]
        print(f"{d:18s} тревог {len(g):4d} ({100*len(g)/n:.0f} % МО), из них «своя сезонность» {g.seasonal.sum():4d}, "
              f"шоков после фильтра {len(real):4d} ({100*len(real)/n:.0f} % МО)")
