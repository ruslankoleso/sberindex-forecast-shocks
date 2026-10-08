"""Поправка роста трат страны на ключевую ставку.

Сезонная наивная модель с дрейфом считает, что годовой рост трат России сохранится:
ĝ(t+h) = g(t), где g(t) = log N(t) − log N(t−12). Ставка действует с задержкой, поэтому
проверяем: предсказывает ли изменение ставки за последние k месяцев (Δr_k) отклонение
будущего роста от текущего: g(t+h) − g(t) = b_h · Δr_k(t) + ε.
Коэффициент b_h оценивается методом наименьших квадратов только по прошлым точкам
(t′ + h ≤ t), заново на каждой точке прогноза — без утечки будущего.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_series():
    from src.forecast.lgbm_model import load_national
    L = load_national(yaml.safe_load(Path("configs/forecast.yaml").read_text(encoding="utf-8")))
    r = pd.read_parquet("data/external/macro_monthly.parquet")["key_rate_mean"]
    return L, r


def growth(L, t):
    return L[t] - L[t - pd.DateOffset(years=1)]


def fit_beta(L, r, origin, h, k, min_train):
    """b_h по прошлым точкам t′ ≤ origin − h (цель g(t′+h) известна к origin)."""
    xs, ys = [], []
    for t in L.index:
        if t + pd.DateOffset(months=h) > origin or t - pd.DateOffset(months=12 + k) < L.index[0]:
            continue
        t2 = t + pd.DateOffset(months=h)
        if t2 not in L.index or t - pd.DateOffset(months=k) not in r.index or t not in r.index:
            continue
        xs.append(r[t] - r[t - pd.DateOffset(months=k)])
        ys.append(growth(L, t2) - growth(L, t))
    if len(xs) < min_train:
        return 0.0, len(xs)
    xs, ys = np.array(xs), np.array(ys)
    return float((xs @ ys) / (xs @ xs)), len(xs)


def adjusted_growth(L, r, origin, h, k, min_train):
    """Прогноз годового роста страны для месяца origin+h с поправкой на ставку."""
    b, n = fit_beta(L, r, origin, h, k, min_train)
    return growth(L, origin) + b * (r[origin] - r[origin - pd.DateOffset(months=k)]), b, n
