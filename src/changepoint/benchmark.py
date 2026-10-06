"""Бенчмарк детекторов точек изменений на искусственных шоках.

Сигнал: x_t = log(траты МО) − медиана log(траты) по всем МО в месяц t, т. е. отклонение
МО от «типичного МО» (общая сезонность и общий рост страны убраны).
В x реальных МО встраиваем шок в известный месяц τ (сдвиг уровня, излом тренда или
разовый выброс). Для каждого детектора порог подбирается так, чтобы доля ложных
тревог на рядах без шока была одинаковой (target_false_alarm). Затем считаем:
  - доля найденных шоков (тревога в [τ, τ+tolerance]),
  - средняя задержка обнаружения (мес.),
  - реакция на разовые выбросы (чем меньше, тем лучше: выброс — не перелом).
Все детекторы «дежурят» в одном окне (с monitor_from), порог подбирается для этого окна.
Запуск: python -m src.changepoint.benchmark
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.changepoint.detectors import DETECTORS
from src.eval.cv import load_wide


def relative_signal(Y):
    L = np.log(Y)
    return L.sub(L.median(axis=1), axis=0)


def inject(x, kind, size, tau):
    x = x.copy()
    if kind == "level":
        x[tau:] += size
    elif kind == "trend":
        x[tau:] += size * np.arange(1, len(x) - tau + 1)
    elif kind == "spike":
        x[tau] += size
    return x


def base_residual_scores(logy, L, idx, sigma, warmup, consec=2, center=None):
    """Детектор по ошибке прогноза национальной основы на 1 мес.:
    прогноз месяца t = траты МО год назад × рост страны за год на t−1;
    ошибка e_t = log факт − log прогноз; из неё вычитается медианная ошибка всех МО в этом
    месяце (общий промах прогноза — это не шок конкретного МО; общероссийские шоки
    ловит детектор на национальном ряде) и результат делится на σ_t — устойчивый разброс
    ошибок всех МО в этом месяце (собственная история МО для оценки разброса не нужна).
    score_t — минимальная |e|/σ за последние `consec` мес., если ошибки одного знака."""
    T = len(logy)
    e = np.full(T, np.nan)
    for t in range(12, T):
        g = L[idx[t - 1]] - L[idx[t - 1] - pd.DateOffset(years=1)]
        e[t] = (logy[t] - (logy[t - 12] + g) - center[t]) / sigma[t]
    score = np.zeros(T)
    for t in range(max(warmup, 12 + consec - 1), T):
        last = e[t - consec + 1:t + 1]
        if np.all(np.sign(last) == np.sign(last[0])):
            score[t] = np.abs(last).min()
    return score


def panel_sigma(Y, L):
    """Медиана и устойчивый (по MAD) разброс ошибок прогноза основы по всем МО в каждом месяце."""
    lY = np.log(Y.values)
    sig, cen = np.full(len(Y), np.nan), np.full(len(Y), np.nan)
    for t in range(12, len(Y)):
        g = L[Y.index[t - 1]] - L[Y.index[t - 1] - pd.DateOffset(years=1)]
        e = lY[t] - (lY[t - 12] + g)
        cen[t] = np.median(e)
        sig[t] = np.median(np.abs(e - cen[t])) * 1.4826
    return sig, cen


def first_alarm(score, thr, start):
    idx = np.where(score[start:] > thr)[0]
    return start + idx[0] if len(idx) else None


THRESHOLDS = {}


def run(cfg):
    rng = np.random.default_rng(cfg["seed"])
    Y = load_wide(yaml.safe_load(Path("configs/eval.yaml").read_text(encoding="utf-8")))
    X = relative_signal(Y)
    cols = rng.choice(X.columns, cfg["n_series"], replace=False)
    w, tol, m0 = cfg["warmup"], cfg["tolerance"], cfg["monitor_from"]
    taus = rng.integers(cfg["tau_range"][0], cfg["tau_range"][1] + 1, len(cols))
    from src.forecast.lgbm_model import load_national
    L = load_national(yaml.safe_load(Path("configs/forecast.yaml").read_text(encoding="utf-8")))
    sigma, center = panel_sigma(Y, L)
    logY = np.log(Y)
    dets = dict(DETECTORS)
    # сигнал x = log y − медиана; сдвиг x на Δ ≡ сдвиг log y на Δ, поэтому для base_residual
    # восстанавливаем log y = x + медиана
    med = logY.median(axis=1).values
    dets["base_residual"] = lambda x, warmup: base_residual_scores(x + med, L, Y.index, sigma, warmup, center=center)

    rows = []
    for dname, det in dets.items():
        null_scores = [det(X[c].values, warmup=w) for c in cols]
        # порог: (1 − target)-квантиль максимального сигнала в окне дежурства на рядах без шока
        thr = np.quantile([s[m0:].max() for s in null_scores], 1 - cfg["target_false_alarm"])
        THRESHOLDS[dname] = float(thr)
        for sname, sc in cfg["scenarios"].items():
            for c, tau in zip(cols, taus):
                s = det(inject(X[c].values, sc["type"], sc["size"], tau), warmup=w)
                a = first_alarm(s, thr, m0)
                rows.append(dict(detector=dname, scenario=sname, kind=sc["type"], tau=tau,
                                 alarm=a, early=(a is not None and a < tau),
                                 hit=(a is not None and tau <= a <= tau + tol),
                                 delay=(a - tau) if (a is not None and tau <= a <= tau + tol) else np.nan))
    return pd.DataFrame(rows)


def summarize(r):
    ch = r[r.kind != "spike"]
    sp = r[r.kind == "spike"]
    out = ch.groupby("detector").agg(доля_найденных=("hit", "mean"), задержка_мес=("delay", "mean"),
                                     тревога_до_шока=("early", "mean"))
    out["реакция_на_выброс"] = sp.groupby("detector").hit.mean()
    by_s = r.pivot_table(index="detector", columns="scenario", values="hit", aggfunc="mean")
    return out.round(3), by_s.round(2)


if __name__ == "__main__":
    cfg = yaml.safe_load(Path("configs/changepoint.yaml").read_text(encoding="utf-8"))
    r = run(cfg)
    out = Path("data/interim/changepoint"); out.mkdir(parents=True, exist_ok=True)
    r.to_parquet(out / "benchmark.parquet", index=False)
    s, by_s = summarize(r)
    s.to_csv(out / "benchmark_summary.csv")
    pd.Series(THRESHOLDS, name="threshold").to_csv(out / "thresholds.csv")
    print(s.sort_values("доля_найденных", ascending=False).to_string())
    print(by_s.to_string())
