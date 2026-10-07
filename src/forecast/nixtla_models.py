"""Модели Nixtla для сравнения: классические статистические (StatsForecast) и
глобальная нейросеть N-HiTS (NeuralForecast). Интерфейс: predict(train T×N, steps) → steps×N.

- AutoETS / AutoTheta — подбираются отдельно для каждого МО. При 12–23 точках годовую
  сезонность (12 мес.) они оценить не могут, поэтому, как и фундаментальные модели,
  проверяются ещё и на ряду, удлинённом национальным рядом назад (backcast=True).
- N-HiTS — одна нейросеть на все МО (глобальная), окно истории 12 мес.; обучается заново
  на каждой точке прогноза только на данных до неё.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _long(train):
    d = train.stack().rename("y").reset_index()
    d.columns = ["ds", "unique_id", "y"]
    return d[["unique_id", "ds", "y"]]


class StatsModel:
    def __init__(self, model="AutoETS", backcast=False):
        self.model_name, self.backcast = model, backcast
        self.name = model.lower() + ("_bc" if backcast else "")
        if backcast:
            from src.forecast.foundation import _nat_level
            self.nat = _nat_level()

    def predict(self, train, steps):
        from statsforecast import StatsForecast
        from statsforecast import models as M
        if self.backcast:
            from src.forecast.foundation import _extend
            train = _extend(train, self.nat)
        season = 12 if len(train) >= 24 else 1          # сезонность только если видно ≥ 2 лет
        mdl = getattr(M, self.model_name)(season_length=season)
        sf = StatsForecast(models=[mdl], freq="MS", n_jobs=1)   # параллельный пул на macOS падает
        fc = sf.forecast(df=_long(train), h=steps)
        col = [c for c in fc.columns if c not in ("unique_id", "ds")][0]
        w = fc.pivot(index="ds", columns="unique_id", values=col)
        return w[train.columns].values


class NHITS:
    name = "nhits"

    def __init__(self, cfg_path="configs/forecast.yaml"):
        self.cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8")).get("nhits", {})

    def predict(self, train, steps):
        import logging
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NHITS as _NHITS
        h = steps
        inp = min(self.cfg.get("input_size", 12), len(train) - h) if len(train) - h >= 4 else 4
        m = _NHITS(h=h, input_size=max(inp, 4), max_steps=self.cfg.get("max_steps", 300),
                   scaler_type="robust", random_seed=self.cfg.get("seed", 42),
                   batch_size=self.cfg.get("batch_size", 64), accelerator="cpu",
                   enable_progress_bar=False, enable_model_summary=False, logger=False,
                   start_padding_enabled=True)
        nf = NeuralForecast(models=[m], freq="MS")
        nf.fit(df=_long(train))
        fc = nf.predict()
        w = fc.pivot(index="ds", columns="unique_id", values="NHITS")
        return w[train.columns].values[:steps]
