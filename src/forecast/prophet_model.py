"""Prophet — обязательный бейзлайн конкурса. Обучается отдельно на каждом МО.

Два варианта:
- prophet: настройки по умолчанию (годовая сезонность включается автоматически
  только при ≥2 годах данных, у нас её нет — типичный «из коробки» результат);
- prophet_yearly: годовая сезонность включена принудительно (3 гармоники) —
  более сильный и справедливый бейзлайн, с ним и сравниваем наши модели.
"""
import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)


def _fit_one(y, idx, steps, yearly):
    logging.getLogger("cmdstanpy").disabled = True
    from prophet import Prophet
    m = Prophet(yearly_seasonality=3 if yearly else "auto", weekly_seasonality=False,
                daily_seasonality=False)
    m.fit(pd.DataFrame({"ds": idx, "y": y}))
    f = m.predict(m.make_future_dataframe(steps, freq="MS", include_history=False))
    return f["yhat"].values


class Prophet_:
    def __init__(self, yearly=False, n_jobs=8):
        self.yearly, self.n_jobs = yearly, n_jobs
        self.name = "prophet_yearly" if yearly else "prophet"

    def predict(self, train, steps):
        res = Parallel(n_jobs=self.n_jobs, batch_size=16)(
            delayed(_fit_one)(train[c].values, train.index, steps, self.yearly)
            for c in train.columns)
        return np.stack(res, axis=1)
