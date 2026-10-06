"""Проверка детекторов на реальных шоках по России (2018–2026).

Ряд: сезонно скорректированный индекс потребительских трат СберИндекса («Всего»).
Онлайн-режим: в месяце t детектор видит последние до 24 месяцев (окно растёт с 13); из окна вычитается
линейный тренд, оценённый по первым 12 месяцам окна (т. е. «отклонение от тренда
прошлого года»). Пороги — те же, что подобраны на бенчмарке (без подгонки под этот ряд).
Известные события задаются в configs/changepoint.yaml (national_events).
Запуск: python -m src.changepoint.national_events
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.changepoint.detectors import DETECTORS


def load_sa():
    d = pd.read_parquet("data/raw/consumper-spending-index-sa.parquet")
    d = d[d["type"] == "Всего"]
    return np.log(pd.Series(d["value"].values, index=pd.to_datetime(d["period"])).sort_index())


def run(cfg):
    x = load_sa()
    thr = pd.read_csv("data/interim/changepoint/thresholds.csv", index_col=0)["threshold"]
    win, warm = 24, 12
    rows = []
    for t in range(warm, len(x)):                       # окно растёт от 13 до 24 мес., дальше скользит
        w = x.iloc[max(0, t - win + 1):t + 1].values
        tt = np.arange(len(w))
        b, a = np.polyfit(tt[:warm], w[:warm], 1)
        r = w - (a + b * tt)
        for name, det in DETECTORS.items():
            sc = det(r, warmup=warm)[-1]
            rows.append(dict(month=x.index[t], detector=name, score=sc, alarm=sc > thr[name]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    cfg = yaml.safe_load(Path("configs/changepoint.yaml").read_text(encoding="utf-8"))
    r = run(cfg)
    r.to_parquet("data/interim/changepoint/national_scores.parquet", index=False)
    ev = pd.to_datetime(pd.Series(cfg["national_events"]))
    print("Известные события:", {k: v.strftime("%Y-%m") for k, v in ev.items()})
    rows = []
    for d, g in r.groupby("detector"):
        al = g[g.alarm].month
        res = {"детектор": d, "тревог_всего": len(al)}
        for k, e in ev.items():
            hit = al[(al >= e) & (al <= e + pd.DateOffset(months=3))]
            res[f"{k}: задержка, мес"] = (hit.iloc[0].year - e.year) * 12 + hit.iloc[0].month - e.month if len(hit) else "не найдено"
        far = al[~np.any([(al >= e - pd.DateOffset(months=1)) & (al <= e + pd.DateOffset(months=6)) for e in ev], axis=0)]
        res["тревоги вне событий"] = ", ".join(m.strftime("%Y-%m") for m in far) or "нет"
        rows.append(res)
    print(pd.DataFrame(rows).to_string(index=False))
