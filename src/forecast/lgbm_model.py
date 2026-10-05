"""Глобальный LightGBM: одна модель на все МО.

Идея: коротких рядов МО (12–23 мес.) мало, чтобы выучить сезонность, поэтому
сезонный профиль берём из длинного национального ряда (2018–2022), а общую
динамику и «характер» МО выучиваем сразу по всем 2 016 МО.

Цель: лог-изменение расходов за h месяцев, log(y[t+h]) - log(y[t]).
Прогноз = y[t] * exp(предсказание). Модель учится по L1 с весом y[t],
чтобы оптимизировать MAE в рублях.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _lgb():
    """Ленивый импорт: lightgbm и torch не должны загружаться в одном процессе (macOS, libomp)."""
    import lightgbm
    return lightgbm


def load_national(cfg):
    n = pd.read_parquet(cfg["national_file"])
    n = n[n["type"] == cfg["national_type"]]
    s = pd.Series(n["value"].values, index=pd.to_datetime(n["period"])).sort_index()
    return np.log(s)


def seasonal_table(L, until, max_h):
    """sp[m, h] = медиана по годам log(y[m+h]) - log(y[m]), m — календарный месяц origin."""
    L = L[L.index <= pd.Timestamp(until)]
    rows = {}
    for m in range(1, 13):
        for h in range(1, max_h + 1):
            vals = []
            for t in L.index[L.index.month == m]:
                t2 = t + pd.DateOffset(months=h)
                if t2 in L.index:
                    vals.append(L[t2] - L[t])
            rows[(m, h)] = np.median(vals) if vals else np.nan
    return rows


class GlobalLGBM:
    name = "lgbm"

    def __init__(self, cfg_path="configs/forecast.yaml", use_national_seasonality=True,
                 name=None):
        self.cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
        self.L = load_national(self.cfg)
        self.use_nat = use_national_seasonality
        self.sp = seasonal_table(self.L, self.cfg["seasonal_until"], self.cfg["max_h"])
        if name:
            self.name = name

    # --- признаки -----------------------------------------------------------
    def _features(self, v, idx, o, h):
        """v: log-значения T×N (до origin включительно), o — индекс origin, h — горизонт."""
        N = v.shape[1]
        cur = v[o]
        month_o = idx[o].month
        f = {"h": np.full(N, h), "month_o": np.full(N, month_o),
             "month_t": np.full(N, (month_o - 1 + h) % 12 + 1),
             "lvl": cur,
             "d1": cur - v[o - 1] if o >= 1 else np.full(N, np.nan),
             "d2": cur - v[o - 2] if o >= 2 else np.full(N, np.nan),
             "d3": cur - v[o - 3] if o >= 3 else np.full(N, np.nan),
             "mean3": cur - v[max(0, o - 2): o + 1].mean(0),
             "mean_all": cur - v[: o + 1].mean(0),
             "yoy": cur - v[o - 12] if o >= 12 else np.full(N, np.nan),
             # то же изменение год назад: из месяца origin в месяц цели, если цель - год назад ≤ origin
             "seas_own": (v[o + h - 12] - cur) if o + h - 12 >= 0 else np.full(N, np.nan),
             "rank_lvl": pd.Series(cur).rank(pct=True).values,
             # кросс-секционные показатели (общая динамика всех МО на дату origin)
             "x_d1": np.full(N, np.median(cur - v[o - 1]) if o >= 1 else np.nan),
             "x_yoy": np.full(N, np.median(cur - v[o - 12]) if o >= 12 else np.nan)}
        if self.use_nat:
            f["seas_nat"] = np.full(N, self.sp[(month_o, h)])
            t0 = idx[o]
            nat_now = self.L.get(t0, np.nan)
            nat_prev = self.L.get(t0 - pd.DateOffset(years=1), np.nan)
            f["nat_yoy"] = np.full(N, nat_now - nat_prev)
        return pd.DataFrame(f)

    def _dataset(self, v, idx, o_max):
        """Обучающие пары (o', h) с целью, известной к моменту o_max."""
        X, y, w = [], [], []
        for o in range(0, o_max):
            for h in range(1, min(self.cfg["max_h"], o_max - o) + 1):
                F = self._features(v, idx, o, h)
                X.append(F); y.append(v[o + h] - v[o]); w.append(np.exp(v[o]))
        return pd.concat(X, ignore_index=True), np.concatenate(y), np.concatenate(w)

    # --- интерфейс ----------------------------------------------------------
    def predict(self, train, steps):
        v = np.log(train.values)
        idx = train.index
        o = len(v) - 1
        X, y, w = self._dataset(v, idx, o)
        m = _lgb().LGBMRegressor(**self.cfg["lgbm"])
        m.fit(X, y, sample_weight=w)
        out = []
        for h in range(1, steps + 1):
            F = self._features(v, idx, o, h)[X.columns]
            out.append(np.exp(v[o] + m.predict(F)))
        return np.stack(out)


def seasonal_index(L, until):
    """Национальный сезонный индекс S[1..12] (сумма 0): медиана отклонения от центрированной
    скользящей средней 2x12 по годам до `until`."""
    ma = L.rolling(12).mean().rolling(2).mean().shift(-6)
    res = (L - ma)[lambda s: s.index <= pd.Timestamp(until)].dropna()
    S = res.groupby(res.index.month).median()
    S = S.reindex(range(1, 13))
    return (S - S.mean()).values  # S[m-1]


class GlobalLGBMDeseason:
    """Версия 2: убираем национальную сезонность и учим МЕСЯЧНЫЙ ТЕМП роста.

    log y[t+h] = log y[t] + (S[m_t] - S[m_o]) + h * rate,
    rate предсказывает LightGBM. Горизонт не входит в признаки и не требует
    экстраполяции, поэтому модель работает и на h=12 при 12 мес. обучения.
    """
    name = "lgbm_ds"

    def __init__(self, cfg_path="configs/forecast.yaml", name=None):
        self.cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
        self.L = load_national(self.cfg)
        self.S = seasonal_index(self.L, self.cfg["seasonal_until"])
        self.Ld = self.L - self.S[self.L.index.month - 1]
        if name:
            self.name = name

    def _features(self, v, vd, idx, o, h):
        N = v.shape[1]
        full = lambda x: np.full(N, x, dtype=float)
        nan = np.full(N, np.nan)
        cur = vd[o]
        m_o, m_t = idx[o].month, (idx[o].month - 1 + h) % 12 + 1
        r = lambda k: (cur - vd[o - k]) / k if o >= k else nan
        t0 = idx[o]
        nat = lambda k: ((self.Ld.get(t0, np.nan) - self.Ld.get(t0 - pd.DateOffset(months=k), np.nan)) / k)
        f = {"r1": r(1), "r2": r(2), "r3": r(3), "r6": r(6), "r_all": r(o) if o >= 1 else nan,
             "yoy_rate": r(12),
             "lvl_rel": cur - np.median(cur),
             "dev_mean": cur - vd[: o + 1].mean(0),
             "rank_lvl": pd.Series(cur).rank(pct=True).values,
             "own_seas_resid": ((v[o + h - 12] - v[o]) - (self.S[m_t - 1] - self.S[m_o - 1]))
             if o + h - 12 >= 0 else nan,
             "x_r1": full(np.nanmedian(r(1))) if o >= 1 else nan,
             "x_r3": full(np.nanmedian(r(3))) if o >= 3 else nan,
             "nat_r1": full(nat(1)), "nat_r3": full(nat(3)), "nat_yoy_rate": full(nat(12))}
        return pd.DataFrame(f)

    def _target(self, v, idx, o, h):
        m_o, m_t = idx[o].month, idx[o + h].month
        return ((v[o + h] - v[o]) - (self.S[m_t - 1] - self.S[m_o - 1])) / h

    def predict(self, train, steps):
        v = np.log(train.values)
        idx = train.index
        vd = v - self.S[idx.month - 1][:, None]
        o = len(v) - 1
        X, y, w = [], [], []
        for o2 in range(0, o):
            for h in range(1, min(self.cfg["max_h"], o - o2) + 1):
                X.append(self._features(v, vd, idx, o2, h))
                y.append(self._target(v, idx, o2, h))
                w.append(np.exp(v[o2]) * h)
        X, y, w = pd.concat(X, ignore_index=True), np.concatenate(y), np.concatenate(w)
        m = _lgb().LGBMRegressor(**self.cfg["lgbm"])
        m.fit(X, y, sample_weight=w)
        out = []
        for h in range(1, steps + 1):
            F = self._features(v, vd, idx, o, h)[X.columns]
            m_o, m_t = idx[o].month, (idx[o].month - 1 + h) % 12 + 1
            out.append(np.exp(v[o] + (self.S[m_t - 1] - self.S[m_o - 1]) + h * m.predict(F)))
        return np.stack(out)


NAT_CATS = ["Продовольственные товары", "Непродовольственные товары", "Услуги",
            "Общественное питание"]


class GlobalLGBMX(GlobalLGBM):
    """LightGBM v1 + признаки из других таблиц (для ablation).

    cat_mo  — доли категорий МО (питание, здоровье, маркетплейсы, ...) в общих расходах
              на дату origin и их изменение за 3 мес.;
    nat_cat — национальные ряды по категориям: г/г и рост за 3 мес. относительно «Всего».
    Берутся только значения не позже origin (без утечки).
    """

    def __init__(self, cat_mo=False, nat_cat=False, name=None, **kw):
        super().__init__(name=name, **kw)
        self.cat_mo, self.nat_cat = cat_mo, nat_cat
        if cat_mo:
            p = pd.read_parquet(self.cfg.get("panel", "data/interim/panel.parquet"))
            p = p[p["category"] != "Все категории"]
            self.cat_wide = {c: g.pivot(index="period", columns="territory_id", values="value")
                             for c, g in p.groupby("category")}
        if nat_cat:
            n = pd.read_parquet(self.cfg["national_file"])
            self.nat_wide = {c: np.log(pd.Series(g["value"].values,
                                                 index=pd.to_datetime(g["period"])).sort_index())
                             for c, g in n.groupby("type") if c in NAT_CATS}

    def predict(self, train, steps):
        if self.cat_mo:
            self.cat_v = {c: np.log(w.reindex(index=train.index, columns=train.columns).values)
                          for c, w in self.cat_wide.items()}
        return super().predict(train, steps)

    def _features(self, v, idx, o, h):
        F = super()._features(v, idx, o, h)
        N = v.shape[1]
        if self.cat_mo:
            for c, cv in self.cat_v.items():
                sh = cv[: o + 1] - v[: o + 1]
                F[f"sh_{c[:4]}"] = sh[o]
                F[f"dsh3_{c[:4]}"] = sh[o] - sh[o - 3] if o >= 3 else np.nan
        if self.nat_cat:
            t0 = idx[o]
            for c, s in self.nat_wide.items():
                yoy = s.get(t0, np.nan) - s.get(t0 - pd.DateOffset(years=1), np.nan)
                d3 = (s.get(t0, np.nan) - s.get(t0 - pd.DateOffset(months=3), np.nan)
                      - (self.L.get(t0, np.nan) - self.L.get(t0 - pd.DateOffset(months=3), np.nan)))
                F[f"n_yoy_{c[:4]}"] = np.full(N, yoy)
                F[f"n_d3_{c[:4]}"] = np.full(N, d3)
        return F
