"""LightGBM поверх национальной основы.

Основа (snaive_natg): base[o, h] = y[o+h-12] * G(o), где G(o) — рост национального
ряда «Всего» за год на дату origin. Модель учит только поправку территории:
    target = log y[o+h] - log base[o, h]
и итоговый прогноз = base * exp(поправка).

Пар для обучения поправки нужно, чтобы и цель, и «тот же месяц год назад» были
известны к origin: 12 <= o'+h <= o. На первом фолде (origin = 12-й месяц) таких
пар нет — тогда прогноз равен основе.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _lgb():
    """Ленивый импорт: lightgbm и torch не должны загружаться в одном процессе (macOS, libomp)."""
    import lightgbm
    return lightgbm

from src.forecast.lgbm_model import load_national


class ResidLGBM:
    name = "lgbm_resid"

    def __init__(self, cfg_path="configs/forecast.yaml", spatial=True, cats=True, name=None):
        self.cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
        self.L = load_national(self.cfg)
        self.spatial, self.cats = spatial, cats
        if name:
            self.name = name
        dcfg = yaml.safe_load(Path("configs/data.yaml").read_text(encoding="utf-8"))
        if spatial:
            ma = pd.read_parquet(Path(dcfg["organizers_dir"]) / dcfg["market_access_file"])
            self.ma = ma.set_index("territory_id")["market_access"]
            ref = pd.read_parquet("data/external/territory_reference.parquet").set_index("territory_id")
            self.ref = ref
        if cats:
            p = pd.read_parquet(Path(dcfg["interim_dir"]) / "panel.parquet")
            p = p[p["category"] != "Все категории"]
            self.cat_wide = {c: g.pivot(index="period", columns="territory_id", values="value")
                             for c, g in p.groupby("category")}

    def _g(self, t):
        return self.L[t] - self.L[t - pd.DateOffset(years=1)]

    def _features(self, v, idx, o, h, static):
        N = v.shape[1]
        nan = np.full(N, np.nan)
        cur = v[o]
        nat = lambda k: self.L[idx[o]] - self.L[idx[o] - pd.DateOffset(months=k)]
        f = {"h": np.full(N, h), "month_t": np.full(N, (idx[o].month - 1 + h) % 12 + 1),
             # собственный рост территории минус рост страны за тот же срок
             "rel_d1": cur - v[o - 1] - nat(1) if o >= 1 else nan,
             "rel_d3": cur - v[o - 3] - nat(3) if o >= 3 else nan,
             "rel_d6": cur - v[o - 6] - nat(6) if o >= 6 else nan,
             "rel_yoy": cur - v[o - 12] - nat(12) if o >= 12 else nan,
             # как территория отклонялась от основы в последние месяцы
             "last_resid": cur - (v[o - 12] + self._g(idx[o - 12])) if o >= 12 else nan,
             "lvl_rel": cur - np.median(cur),
             "nat_yoy": np.full(N, nat(12))}
        F = pd.DataFrame(f)
        if self.cats:
            for c, cv in self.cat_v.items():
                F[f"sh_{c[:4]}"] = cv[o] - cur
        for k, val in static.items():
            F[k] = val
        return F

    def predict(self, train, steps):
        v = np.log(train.values)
        idx = train.index
        o = len(v) - 1
        cols = train.columns
        static = {}
        if self.spatial:
            static["market_access"] = self.ma.reindex(cols).values
            static["is_moscow"] = self.ref["is_moscow"].reindex(cols).astype(float).values
            static["is_mosobl"] = self.ref["is_mosobl"].reindex(cols).astype(float).values
            static["is_spb"] = self.ref["is_spb"].reindex(cols).astype(float).values
        if self.cats:
            self.cat_v = {c: np.log(w.reindex(index=idx, columns=cols).values)
                          for c, w in self.cat_wide.items()}
        base = lambda o2, h: v[o2 + h - 12] + self._g(idx[o2])
        X, y, w = [], [], []
        for o2 in range(0, o):
            for h in range(1, min(12, o - o2) + 1):
                if o2 + h - 12 < 0:
                    continue
                X.append(self._features(v, idx, o2, h, static))
                y.append(v[o2 + h] - base(o2, h))
                w.append(np.exp(v[o2]))
        out = []
        model = None
        if X:
            X, y, w = pd.concat(X, ignore_index=True), np.concatenate(y), np.concatenate(w)
            model = _lgb().LGBMRegressor(**self.cfg["lgbm"]).fit(X, y, sample_weight=w)
        for h in range(1, steps + 1):
            b = base(o, h)
            corr = model.predict(self._features(v, idx, o, h, static)[X.columns]) if model else 0.0
            out.append(np.exp(b + corr))
        return np.stack(out)
