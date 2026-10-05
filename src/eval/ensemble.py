"""Честный ансамбль и выбор модели по прошлому.

Для горизонта h и точки прогноза o веса считаются только по прогнозам
на прошлые точки o' с известным фактом: o' + h <= o. Методы:
- best_past — берём одну модель с наименьшим MAE на прошлых точках;
- inv_mae   — среднее моделей с весами 1 / MAE на прошлых точках;
- nnls      — неотрицательные веса (сумма 1), минимизирующие ошибку на прошлых точках.
Если прошлых точек меньше min_past_origins (например, h=12 — всего одна точка),
используется запасная модель.
Запуск: python -m src.eval.ensemble
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import nnls

from src.eval.metrics import summarize

KEY = ["origin", "h", "territory_id"]


def load(cfg):
    d = Path(cfg["forecasts_dir"])
    F = None
    for m in cfg["candidates"]:
        f = pd.read_parquet(d / f"{m}.parquet").set_index(KEY)
        if F is None:
            F = f[["target", "y_true", "y_prev_year"]].copy()
        F[m] = f["y_pred"]
    return F.reset_index()


def weights(past, cands, method):
    y = past["y_true"].values
    if method == "best_past":
        mae = {m: np.abs(past[m].values - y).mean() for m in cands}
        best = min(mae, key=mae.get)
        return {m: float(m == best) for m in cands}
    if method == "inv_mae":
        inv = {m: 1 / np.abs(past[m].values - y).mean() for m in cands}
        s = sum(inv.values())
        return {m: v / s for m, v in inv.items()}
    if method == "nnls":
        # ошибки в рублях, нормируем на уровень, чтобы крупные МО не доминировали
        sc = past["y_prev_year"].values
        A = past[cands].values / sc[:, None]
        w, _ = nnls(A, y / sc)
        w = w / w.sum() if w.sum() > 0 else np.full(len(cands), 1 / len(cands))
        return dict(zip(cands, w))
    raise ValueError(method)


def run(cfg):
    F = load(cfg)
    cands = cfg["candidates"]
    out, wlog = [], []
    for h, Fh in F.groupby("h"):
        origins = sorted(Fh["origin"].unique())
        for o in origins:
            cur = Fh[Fh["origin"] == o]
            past = Fh[Fh["origin"] + pd.DateOffset(months=int(h)) <= o]
            n_past = past["origin"].nunique()
            for method in cfg["methods"]:
                if n_past < cfg["min_past_origins"]:
                    w = {m: float(m == cfg["fallback"]) for m in cands}
                else:
                    w = weights(past, cands, method)
                pred = sum(w[m] * cur[m].values for m in cands)
                out.append(cur[KEY + ["target", "y_true", "y_prev_year"]].assign(
                    model=f"ens_{method}", y_pred=pred))
                wlog.append(dict(h=h, origin=o, method=method, n_past=n_past, **w))
    res = pd.concat(out, ignore_index=True)
    return res, pd.DataFrame(wlog)


if __name__ == "__main__":
    cfg = yaml.safe_load(Path("configs/ensemble.yaml").read_text(encoding="utf-8"))
    res, wlog = run(cfg)
    d = Path(cfg["forecasts_dir"])
    for m, g in res.groupby("model"):
        g.to_parquet(d / f"{m}.parquet", index=False)
    wlog.to_csv(d.parent / "ensemble_weights.csv", index=False)
    singles = pd.concat(pd.read_parquet(d / f"{m}.parquet") for m in cfg["candidates"])
    s = summarize(pd.concat([singles, res]))
    t = s.pivot(index="model", columns="h", values="MAE").round(0)
    print(t.sort_values(1).to_string())
