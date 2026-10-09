"""Прогноз национального ряда «Всего» и основа на его основе.

Основа с постоянным ростом (snaive_natg) предполагает, что рост страны за год
в будущем будет таким же, как на дату прогноза. Здесь рост страны прогнозируется
моделью по длинному национальному ряду (с 2018 года, только данные до origin):
    base[o, h] = y_terr[o+h-12] * N̂[o+h] / N[o+h-12],
где N̂ — прогноз национального ряда, N[o+h-12] известен (h ≤ 12).

Методы прогноза национального ряда:
- ets      — экспоненциальное сглаживание Хольта–Уинтерса (мультипликативная сезонность,
             затухающий тренд) по логарифму ряда;
- chronos2 — фундаментальная модель Chronos-2 zero-shot (длинный ряд — её сильная сторона).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.forecast.baselines import SeasonalNaive
from src.hub import resolve


def _load_L():
    from src.forecast.lgbm_model import load_national
    return load_national(yaml.safe_load(Path("configs/forecast.yaml").read_text(encoding="utf-8")))


def forecast_national(L_hist: pd.Series, steps: int, method: str, pipe=None) -> np.ndarray:
    """Прогноз log N на steps месяцев вперёд по истории L_hist (лог-уровни)."""
    if method == "ets":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        m = ExponentialSmoothing(L_hist.values, trend="add", damped_trend=True,
                                 seasonal="add", seasonal_periods=12).fit()
        return m.forecast(steps)
    if method == "chronos2":
        import torch
        q, _ = pipe.predict_quantiles([torch.tensor(np.exp(L_hist.values), dtype=torch.float32)],
                                      prediction_length=steps, quantile_levels=[0.5])
        q = q[0] if isinstance(q, list) else q
        return np.log(q.reshape(-1).numpy()[:steps])
    raise ValueError(method)


class SNaiveNationalForecast:
    """Сезонный наивный × прогнозный рост страны (метод задаётся параметром)."""

    def __init__(self, method="ets"):
        self.method = method
        self.name = f"snaive_nat_{method}"
        self.L = _load_L()
        self.pipe = None
        if method == "chronos2":
            from chronos import BaseChronosPipeline
            cfg = yaml.safe_load(Path("configs/foundation.yaml").read_text(encoding="utf-8"))["chronos2"]
            self.pipe = BaseChronosPipeline.from_pretrained(resolve(cfg["model_id"]), device_map=cfg["device"])

    def predict(self, train, steps):
        t0 = train.index[-1]
        hist = self.L[self.L.index <= t0]                       # только известное к origin
        fut = forecast_national(hist, steps, self.method, self.pipe)
        base = SeasonalNaive().predict(train, steps)
        out = []
        for h in range(1, steps + 1):
            prev = self.L[t0 + pd.DateOffset(months=h - 12)]    # тот же месяц год назад — известен
            out.append(base[h - 1] * np.exp(fut[h - 1] - prev))
        return np.stack(out)
