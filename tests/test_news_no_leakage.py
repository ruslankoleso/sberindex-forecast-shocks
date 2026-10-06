"""Новостные признаки прогноза на дату origin не должны зависеть от новостей после origin."""
import numpy as np
import pandas as pd

from src.forecast.resid_model import ResidLGBM


def _model():
    m = object.__new__(ResidLGBM)               # без загрузки файлов: проверяем только сборку признаков
    m.cats, m.news = False, True
    idx = pd.date_range("2022-01-01", "2024-12-01", freq="MS")
    m.L = pd.Series(np.linspace(8, 9, len(idx)), index=idx)
    return m


def test_future_news_do_not_change_features():
    m = _model()
    idx = pd.date_range("2023-01-01", periods=24, freq="MS")
    v = np.log(np.random.default_rng(0).uniform(10, 20, (24, 3)))
    base = {"emergency": np.random.default_rng(1).uniform(0, 1, (24, 3))}
    o = 15
    m.news_arr = {k: a.copy() for k, a in base.items()}
    f1 = m._features(v, idx, o, 3, {})
    future = {k: a.copy() for k, a in base.items()}
    for a in future.values():
        a[o + 1:] = 1e6                          # «новости из будущего»
    m.news_arr = future
    f2 = m._features(v, idx, o, 3, {})
    pd.testing.assert_frame_equal(f1, f2)
