"""Rolling-origin оценка: модель видит только данные до origin включительно."""
import numpy as np
import pandas as pd


def load_wide(cfg):
    p = pd.read_parquet(cfg["panel"])
    p = p[p["category"] == cfg["category"]]
    Y = p.pivot(index="period", columns="territory_id", values="value").sort_index()
    if cfg.get("only_full_history", True):
        Y = Y.loc[:, Y.notna().all()]
    return Y


def origins(T, h, min_train):
    """Индексы последнего обучающего месяца: цель origin+h должна лежать в данных."""
    return range(min_train - 1, T - h)


def run_cv(Y: pd.DataFrame, model, cfg) -> pd.DataFrame:
    """Возвращает длинную таблицу прогнозов модели на всех (origin, h)."""
    T, H = len(Y), max(cfg["horizons"])
    vals = Y.values
    out = []
    # одно обучение на origin даёт прогнозы сразу на все горизонты
    for o in range(cfg["min_train_months"] - 1, T - 1):
        train = Y.iloc[: o + 1]                      # строго до origin включительно
        pred = model.predict(train, min(H, T - 1 - o))   # (steps, N)
        for h in cfg["horizons"]:
            if o + h >= T:
                continue
            t = o + h
            prev = vals[t - 12] if t - 12 >= 0 else np.full(vals.shape[1], np.nan)
            out.append(pd.DataFrame(dict(
                model=model.name, origin=Y.index[o], target=Y.index[t], h=h,
                territory_id=Y.columns, y_true=vals[t], y_pred=pred[h - 1], y_prev_year=prev)))
    return pd.concat(out, ignore_index=True)
