"""Метрики качества прогноза. MAE — основная (обязательная по условиям конкурса)."""
import numpy as np
import pandas as pd


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    sst = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - p) ** 2) / sst) if sst > 0 else np.nan


def summarize(res: pd.DataFrame) -> pd.DataFrame:
    """res: model, h, y_true, y_pred, y_prev_year. Возвращает MAE и R² по моделям и горизонтам.

    R²_lvl — по уровню расходов (в основном отражает различия между МО);
    R²_yoy — по росту к тому же месяцу прошлого года (отражает динамику).
    """
    rows = []
    for (m, h), g in res.groupby(["model", "h"]):
        yoy_t = g.y_true / g.y_prev_year - 1
        yoy_p = g.y_pred / g.y_prev_year - 1
        rows.append(dict(model=m, h=h, n=len(g), MAE=mae(g.y_true, g.y_pred),
                         R2_lvl=r2(g.y_true, g.y_pred), R2_yoy=r2(yoy_t, yoy_p),
                         MAE_yoy_pp=100 * mae(yoy_t, yoy_p)))
    return pd.DataFrame(rows).sort_values(["h", "MAE"]).reset_index(drop=True)
