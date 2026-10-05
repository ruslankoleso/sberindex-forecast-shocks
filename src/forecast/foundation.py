"""Фундаментальные модели временных рядов в режиме zero-shot.

Модель предобучена на большом корпусе чужих рядов и не обучается на наших данных:
получает историю территории до origin и выдаёт прогноз (медиану и квантили).
Это прямой ответ на короткую историю (12–23 мес.): знание о типичных формах рядов
и сезонности модель приносит из предобучения.

Интерфейс как у остальных моделей: predict(train T×N, steps) -> steps×N (медиана).
Квантили сохраняются в self.last_quantiles для раннего предупреждения о шоках.
"""
from pathlib import Path

import numpy as np
import torch
import yaml


class ChronosBolt:
    name = "chronos_bolt"

    def __init__(self, cfg_path="configs/foundation.yaml", key="chronos_bolt", name=None):
        cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))[key]
        from chronos import BaseChronosPipeline
        self.cfg = cfg
        self.pipe = BaseChronosPipeline.from_pretrained(cfg["model_id"], device_map=cfg["device"],
                                                        torch_dtype=torch.float32)
        if name:
            self.name = name
        self.last_quantiles = None

    def predict(self, train, steps):
        ctx = [torch.tensor(train[c].values, dtype=torch.float32) for c in train.columns]
        qs = self.cfg["quantiles"]
        out_q = []
        bs = self.cfg["batch_size"]
        for i in range(0, len(ctx), bs):
            q, _ = self.pipe.predict_quantiles(ctx[i:i + bs], prediction_length=steps,
                                               quantile_levels=qs)
            if isinstance(q, list):                     # Chronos-2 возвращает список (1, steps, Q)
                q = torch.cat([x[:1] for x in q])
            out_q.append(q.numpy())                     # (b, steps, len(qs))
        q = np.concatenate(out_q)                       # (N, steps, Q)
        self.last_quantiles = q
        return q[:, :, qs.index(0.5)].T                 # (steps, N)


class ChronosBoltBackcast(ChronosBolt):
    """Chronos-Bolt с удлинённым контекстом.

    Ряд территории достраивается назад национальным рядом «Всего» (2018-12…2022-12),
    масштабированным к уровню территории по 2023 году: y_i(t) = k_i * nat(t),
    k_i = mean(y_i, 2023) / mean(nat, 2023). Модель видит 4+ лет сезонности.
    Будущее не используется: 2023 год всегда раньше origin (обучение ≥ 12 мес.),
    национальные данные берутся только до начала ряда территории.
    """
    name = "chronos_bolt_bc"

    def __init__(self, national_cfg="configs/forecast.yaml", **kw):
        super().__init__(**kw)
        from src.forecast.lgbm_model import load_national
        ncfg = yaml.safe_load(Path(national_cfg).read_text(encoding="utf-8"))
        self.nat = np.exp(load_national(ncfg))

    def predict(self, train, steps):
        start = train.index[0]
        hist = self.nat[self.nat.index < start]
        ov = self.nat.reindex(train.index[:12])
        k = train.iloc[:12].mean(0).values / ov.mean()
        back = np.outer(hist.values, k)                          # (len(hist), N)
        ext = np.vstack([back, train.values])
        import pandas as pd
        idx = hist.index.append(train.index)
        return super().predict(pd.DataFrame(ext, index=idx, columns=train.columns), steps)


class Chronos2(ChronosBolt):
    """Chronos-2 (≈120 млн параметров) — более новая модель семейства, zero-shot."""
    name = "chronos2"

    def __init__(self, **kw):
        super().__init__(key="chronos2", **kw)


class Chronos2Backcast(ChronosBoltBackcast):
    """Chronos-2 с контекстом, удлинённым национальным рядом (см. ChronosBoltBackcast)."""
    name = "chronos2_bc"

    def __init__(self, **kw):
        super().__init__(key="chronos2", **kw)


def _extend(train, nat):
    """Удлинение истории МО национальным рядом (см. ChronosBoltBackcast)."""
    import pandas as pd
    hist = nat[nat.index < train.index[0]]
    k = train.iloc[:12].mean(0).values / nat.reindex(train.index[:12]).mean()
    ext = np.vstack([np.outer(hist.values, k), train.values])
    return pd.DataFrame(ext, index=hist.index.append(train.index), columns=train.columns)


def _nat_level():
    from src.forecast.lgbm_model import load_national
    cfg = yaml.safe_load(Path("configs/forecast.yaml").read_text(encoding="utf-8"))
    return np.exp(load_national(cfg))


class TimesFM:
    """Google TimesFM 2.5 (200 млн параметров), zero-shot. backcast=True — с удлинённой историей."""

    def __init__(self, backcast=False, cfg_path="configs/foundation.yaml"):
        import timesfm
        cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))["timesfm"]
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(cfg["model_id"])
        self.model.compile(timesfm.ForecastConfig(
            max_context=cfg["max_context"], max_horizon=cfg["max_horizon"], normalize_inputs=True,
            use_continuous_quantile_head=True, force_flip_invariance=True,
            infer_is_positive=True, fix_quantile_crossing=True))
        self.backcast = backcast
        self.nat = _nat_level() if backcast else None
        self.name = "timesfm_bc" if backcast else "timesfm"
        self.last_quantiles = None

    def predict(self, train, steps):
        if self.backcast:
            train = _extend(train, self.nat)
        point, q = self.model.forecast(horizon=steps,
                                       inputs=[train[c].values.astype(float) for c in train.columns])
        self.last_quantiles = q[:, :steps, [1, 5, 9]]          # 0,1 / 0,5 / 0,9
        return np.asarray(point)[:, :steps].T


class TiRex:
    """NX-AI TiRex (xLSTM, 35 млн параметров), zero-shot. backcast=True — с удлинённой историей."""

    def __init__(self, backcast=False, cfg_path="configs/foundation.yaml"):
        from tirex import load_model
        cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))["tirex"]
        self.model = load_model(cfg["model_id"], device=cfg["device"])
        self.backcast = backcast
        self.nat = _nat_level() if backcast else None
        self.name = "tirex_bc" if backcast else "tirex"
        self.last_quantiles = None

    def predict(self, train, steps):
        if self.backcast:
            train = _extend(train, self.nat)
        ctx = torch.tensor(train.values.T, dtype=torch.float32)
        q, mean = self.model.forecast(context=ctx, prediction_length=steps)
        q = q.numpy() if hasattr(q, "numpy") else np.asarray(q)
        self.last_quantiles = q[:, :steps, [0, 4, 8]]
        return q[:, :steps, 4].T                                  # медиана


class Chronos2FineTuned(Chronos2Backcast):
    """Chronos-2, дообученный на наших рядах перед каждой точкой прогноза.

    На каждой точке прогноза (origin) копия Chronos-2 дообучается на рядах МО,
    обрезанных по origin и удлинённых национальным рядом назад. Обучающие окна
    берутся только внутри этих рядов — будущее после origin модели недоступно.
    Затем дообученная модель прогнозирует продолжение тех же рядов.
    """
    name = "chronos2_ft"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.base_pipe = self.pipe
        self.ft = yaml.safe_load(Path("configs/foundation.yaml").read_text(encoding="utf-8"))["chronos2_ft"]

    def predict(self, train, steps):
        ext = _extend(train, self.nat)
        inputs = [ext[c].values.astype(np.float32) for c in ext.columns]
        torch.manual_seed(self.ft["seed"])
        self.pipe = self.base_pipe.fit(
            inputs, prediction_length=self.ft["prediction_length"], num_steps=self.ft["num_steps"],
            batch_size=self.ft["batch_size"], learning_rate=self.ft["learning_rate"],
            finetune_mode=self.ft["mode"], output_dir=self.ft["output_dir"],
            remove_printer_callback=True, report_to=[], seed=self.ft["seed"])
        # родительский predict снова удлинит историю, поэтому передаём исходный train
        return super().predict(train, steps)
