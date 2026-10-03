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
