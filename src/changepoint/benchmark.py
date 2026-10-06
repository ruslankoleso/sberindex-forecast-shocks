"""Бенчмарк детекторов точек изменений на искусственных шоках.

Сигнал: x_t = log(траты МО) − медиана log(траты) по всем МО в месяц t, т. е. отклонение
МО от «типичного МО» (общая сезонность и общий рост страны убраны).
В x реальных МО встраиваем шок в известный месяц τ (сдвиг уровня, излом тренда или
разовый выброс). Для каждого детектора порог подбирается так, чтобы доля ложных
тревог на рядах без шока была одинаковой (target_false_alarm). Затем считаем:
  - доля найденных шоков (тревога в [τ, τ+tolerance]),
  - средняя задержка обнаружения (мес.),
  - реакция на разовые выбросы (чем меньше, тем лучше: выброс — не перелом).
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


def first_alarm(score, thr, start):
    idx = np.where(score[start:] > thr)[0]
    return start + idx[0] if len(idx) else None


THRESHOLDS = {}


def run(cfg):
    rng = np.random.default_rng(cfg["seed"])
    Y = load_wide(yaml.safe_load(Path("configs/eval.yaml").read_text(encoding="utf-8")))
    X = relative_signal(Y)
    cols = rng.choice(X.columns, cfg["n_series"], replace=False)
    w, tol = cfg["warmup"], cfg["tolerance"]
    taus = rng.integers(cfg["tau_range"][0], cfg["tau_range"][1] + 1, len(cols))

    rows = []
    for dname, det in DETECTORS.items():
        null_scores = [det(X[c].values, warmup=w) for c in cols]
        # порог: (1 − target)-квантиль максимального сигнала на рядах без шока
        thr = np.quantile([s[w:].max() for s in null_scores], 1 - cfg["target_false_alarm"])
        THRESHOLDS[dname] = float(thr)
        for sname, sc in cfg["scenarios"].items():
            for c, tau in zip(cols, taus):
                s = det(inject(X[c].values, sc["type"], sc["size"], tau), warmup=w)
                a = first_alarm(s, thr, w)
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
