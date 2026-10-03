import numpy as np
import pandas as pd

from src.eval.cv import run_cv
from src.forecast.baselines import REGISTRY

CFG = dict(min_train_months=12, horizons=[1, 3, 6, 12])


def _Y(seed=0):
    r = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=24, freq="MS")
    return pd.DataFrame(r.uniform(10, 20, (24, 5)), index=idx, columns=list("abcde"))


def test_future_does_not_change_forecast():
    """Меняем данные после origin — прогноз на этом origin не должен измениться."""
    Y1, Y2 = _Y(), _Y()
    Y2.iloc[18:] *= 100
    for name, cls in REGISTRY.items():
        r1, r2 = run_cv(Y1, cls(), CFG), run_cv(Y2, cls(), CFG)
        a = r1[r1.origin <= Y1.index[16]].y_pred.values
        b = r2[r2.origin <= Y1.index[16]].y_pred.values
        assert np.allclose(a, b), name


def test_fold_counts():
    r = run_cv(_Y(), REGISTRY["naive"](), CFG)
    n = r.groupby("h").origin.nunique().to_dict()
    assert n == {1: 12, 3: 10, 6: 7, 12: 1}
