"""Простые бейзлайны. Интерфейс: predict(train: DataFrame T×N, steps) -> array steps×N."""
import numpy as np


class Naive:
    name = "naive"

    def predict(self, train, steps):
        return np.tile(train.values[-1], (steps, 1))


class SeasonalNaive:
    """Значение того же месяца год назад (нужно ≥12 наблюдений)."""
    name = "seasonal_naive"

    def predict(self, train, steps):
        v = train.values
        return np.stack([v[len(v) - 12 + (s % 12)] for s in range(steps)])


class SeasonalNaiveGrowth:
    """Сезонный наивный × последний наблюдаемый рост г/г (если есть 13+ точек)."""
    name = "seasonal_naive_growth"

    def predict(self, train, steps):
        v = train.values
        base = SeasonalNaive().predict(train, steps)
        if len(v) < 13:
            return base
        g = v[-1] / v[-13]
        return base * g


REGISTRY = {c.name: c for c in (Naive, SeasonalNaive, SeasonalNaiveGrowth)}


class SeasonalNaiveNationalGrowth:
    """Тот же месяц год назад × рост г/г национального ряда «Всего» на дату origin.

    Годовой рост у территории оценить нельзя, пока история < 13 мес., а национальный
    ряд (2018–2026) его даёт. Используется только национальное значение не позже origin.
    """
    name = "snaive_natg"

    def __init__(self):
        from pathlib import Path
        import yaml
        from src.forecast.lgbm_model import load_national
        self.L = load_national(yaml.safe_load(Path("configs/forecast.yaml").read_text(encoding="utf-8")))

    def predict(self, train, steps):
        import pandas as pd
        t0 = train.index[-1]
        g = np.exp(self.L[t0] - self.L[t0 - pd.DateOffset(years=1)])
        return SeasonalNaive().predict(train, steps) * g


REGISTRY["snaive_natg"] = SeasonalNaiveNationalGrowth
