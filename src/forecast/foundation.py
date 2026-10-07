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

    def __init__(self, national_cfg="configs/forecast.yaml", bc_mode="national", bc_w=0.5, **kw):
        super().__init__(**kw)
        from src.forecast.lgbm_model import load_national
        ncfg = yaml.safe_load(Path(national_cfg).read_text(encoding="utf-8"))
        self.nat = np.exp(load_national(ncfg))
        self.bc_mode, self.bc_w = bc_mode, bc_w            # форма года при backcasting: national / own / mix

    def predict(self, train, steps):
        return super().predict(_extend(train, self.nat, self.bc_mode, self.bc_w), steps)


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


def _extend(train, nat, mode="national", w=0.5):
    """Backcasting: ретроспективное восстановление ряда МО до начала данных (2018-12…2022-12).

    mode="national" — ряд России в масштабе МО: форма года (сезонность) — как у страны;
    mode="own"      — уровень и тренд России × собственная форма года МО;
    mode="mix"      — форма года = смесь формы страны и формы МО: nat^(1−w) · own^w (shrinkage).
    Масштаб и форма года МО считаются по первым 12 месяцам ряда (2023 г.), которые всегда
    раньше точки прогноза; тренд России — 12-месячное скользящее среднее национального ряда.
    """
    import pandas as pd
    hist = nat[nat.index < train.index[0]]
    first = train.iloc[:12]
    nat_first = nat.reindex(first.index)
    k = first.mean(0).values / nat_first.mean()
    if mode == "national":
        back = np.outer(hist.values, k)
    else:
        trend = nat.rolling(12, center=True, min_periods=6).mean().reindex(hist.index).values
        own = (first / first.mean(0)).values                       # 12 × N, форма года МО
        nprof = (nat_first / nat_first.mean()).values[:, None]     # 12 × 1, форма года страны
        prof = own if mode == "own" else nprof ** (1 - w) * own ** w
        prof = prof / prof.mean(0)                                  # среднее за год = 1
        moy = {m: i for i, m in enumerate(first.index.month)}
        rows = np.stack([prof[moy[m]] for m in hist.index.month])   # len(hist) × N
        back = trend[:, None] * k[None, :] * rows
    ext = np.vstack([back, train.values])
    return pd.DataFrame(ext, index=hist.index.append(train.index), columns=train.columns)


def _nat_level():
    from src.forecast.lgbm_model import load_national
    cfg = yaml.safe_load(Path("configs/forecast.yaml").read_text(encoding="utf-8"))
    return np.exp(load_national(cfg))


class TimesFM:
    """Google TimesFM 2.5 (200 млн параметров), zero-shot. backcast=True — с удлинённой историей."""

    def __init__(self, backcast=False, cfg_path="configs/foundation.yaml", bc_mode="national", bc_w=0.5):
        import timesfm
        cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))["timesfm"]
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(cfg["model_id"])
        self.model.compile(timesfm.ForecastConfig(
            max_context=cfg["max_context"], max_horizon=cfg["max_horizon"], normalize_inputs=True,
            per_core_batch_size=cfg["batch_size"],
            use_continuous_quantile_head=True, force_flip_invariance=True,
            infer_is_positive=True, fix_quantile_crossing=True))
        self.backcast = backcast
        self.nat = _nat_level() if backcast else None
        self.bc_mode, self.bc_w = bc_mode, bc_w
        self.name = ("timesfm_bc" + ("" if bc_mode == "national" else f"_{bc_mode}")) if backcast else "timesfm"
        self.last_quantiles = None

    def predict(self, train, steps):
        if self.backcast:
            train = _extend(train, self.nat, getattr(self, "bc_mode", "national"), getattr(self, "bc_w", 0.5))
        point, q = self.model.forecast(horizon=steps,
                                       inputs=[train[c].values.astype(float) for c in train.columns])
        self.last_quantiles = q[:, :steps, [1, 5, 9]]          # 0,1 / 0,5 / 0,9
        return np.asarray(point)[:, :steps].T


class TiRex:
    """NX-AI TiRex (xLSTM, 35 млн параметров), zero-shot. backcast=True — с удлинённой историей."""

    def __init__(self, backcast=False, cfg_path="configs/foundation.yaml", bc_mode="national", bc_w=0.5):
        from tirex import load_model
        cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))["tirex"]
        self.model = load_model(cfg["model_id"], device=cfg["device"])
        self.backcast = backcast
        self.nat = _nat_level() if backcast else None
        self.bc_mode, self.bc_w = bc_mode, bc_w
        self.name = ("tirex_bc" + ("" if bc_mode == "national" else f"_{bc_mode}")) if backcast else "tirex"
        self.last_quantiles = None

    def predict(self, train, steps):
        if self.backcast:
            train = _extend(train, self.nat, getattr(self, "bc_mode", "national"), getattr(self, "bc_w", 0.5))
        ctx = torch.tensor(train.values.T, dtype=torch.float32)
        q, mean = self.model.forecast(context=ctx, prediction_length=steps)
        q = q.numpy() if hasattr(q, "numpy") else np.asarray(q)
        self.last_quantiles = q[:, :steps, [0, 4, 8]]
        return q[:, :steps, 4].T                                  # медиана


class Chronos2FineTuned(Chronos2Backcast):  # noqa: D101
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
        ext = _extend(train, self.nat, self.bc_mode, self.bc_w)
        inputs = [ext[c].values.astype(np.float32) for c in ext.columns]
        torch.manual_seed(self.ft["seed"])
        self.pipe = self.base_pipe.fit(
            inputs, prediction_length=self.ft["prediction_length"], num_steps=self.ft["num_steps"],
            batch_size=self.ft["batch_size"], learning_rate=self.ft["learning_rate"],
            finetune_mode=self.ft["mode"], output_dir=self.ft["output_dir"],
            remove_printer_callback=True, report_to=[], seed=self.ft["seed"])
        # родительский predict снова удлинит историю, поэтому передаём исходный train
        return super().predict(train, steps)


class TimesFMXReg(TimesFM):
    """TimesFM 2.5 с внешними факторами (XReg, режим «xreg + timesfm»).

    Ковариаты подаются на историю И горизонт, поэтому используются только величины,
    известные заранее (без утечки будущего): месяц года, национальный сезонный профиль
    (оценён по данным до 2022 г.), число рабочих дней (производственный календарь);
    статичные — индекс доступности рынков и регион МО. История удлинена национальным рядом
    (TimesFM нужно ≥ 32 точек контекста).
    """
    name = "timesfm_xreg"

    def __init__(self, cfg_path="configs/foundation.yaml"):
        import timesfm
        super().__init__(backcast=True, cfg_path=cfg_path)
        cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))["timesfm"]
        self.model.compile(timesfm.ForecastConfig(
            max_context=cfg["max_context"], max_horizon=cfg["max_horizon"], normalize_inputs=True,
            per_core_batch_size=cfg["batch_size"], use_continuous_quantile_head=True,
            force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True,
            return_backcast=True))
        self.name = "timesfm_xreg"
        import pandas as pd
        from src.forecast.lgbm_model import load_national, seasonal_index
        fcfg = yaml.safe_load(Path("configs/forecast.yaml").read_text(encoding="utf-8"))
        self.S = seasonal_index(load_national(fcfg), fcfg["seasonal_until"])
        self.wd = pd.read_parquet("data/external/macro_monthly.parquet")["working_days"]
        dcfg = yaml.safe_load(Path("configs/data.yaml").read_text(encoding="utf-8"))
        self.ma = pd.read_parquet(Path(dcfg["organizers_dir"]) / dcfg["market_access_file"]).set_index("territory_id")["market_access"]
        self.region = pd.read_parquet("data/external/territory_reference.parquet").set_index("territory_id")["region"]

    def predict(self, train, steps):
        import pandas as pd
        ext = _extend(train, self.nat)
        idx = ext.index.append(pd.date_range(ext.index[-1] + pd.DateOffset(months=1), periods=steps, freq="MS"))
        month = [int(m) for m in idx.month]
        seas = [float(self.S[m - 1]) for m in idx.month]
        wd = [float(v) for v in self.wd.reindex(idx).fillna(self.wd.median()).values]
        cols = list(train.columns)
        n = len(cols)
        ma = self.ma.reindex(cols)
        reg = self.region.reindex(cols).fillna("нет").astype("category").cat.codes
        point, _ = self.model.forecast_with_covariates(
            inputs=[ext[c].values.astype(float) for c in cols],
            dynamic_numerical_covariates={"seasonal": [seas] * n, "working_days": [wd] * n},
            dynamic_categorical_covariates={"month": [month] * n},
            static_numerical_covariates={"market_access": ma.fillna(ma.median()).astype(float).tolist()},
            static_categorical_covariates={"region": [int(r) for r in reg]},
            xreg_mode="xreg + timesfm")
        return np.stack([np.asarray(p)[:steps] for p in point], axis=1)


class TimesFMLoRA:
    """TimesFM 2.5 (версия transformers), дообученный LoRA перед каждой точкой прогноза.

    По примеру из навыка timesfm-forecasting (examples/finetuning): адаптер LoRA на все
    линейные слои, AdamW, косинусное расписание. Обучающие окна (контекст 32 мес. →
    следующие 12 мес.) нарезаются случайно из рядов МО, обрезанных по origin и удлинённых
    национальным рядом, — будущее после origin модели недоступно. Прогноз — медиана.
    """
    name = "timesfm_lora"

    def __init__(self, cfg_path="configs/foundation.yaml", bc_mode="national", bc_w=0.5):
        self.cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))["timesfm_lora"]
        self.nat = _nat_level()
        self.bc_mode, self.bc_w = bc_mode, bc_w
        if bc_mode != "national":
            self.name = f"timesfm_lora_{bc_mode}"

    def _windows(self, series, rng):
        c, h = self.cfg["context_len"], self.cfg["horizon_len"]
        ctx, tgt = [], []
        for _ in range(self.cfg["num_samples"]):
            s = series[rng.integers(len(series))]
            st = rng.integers(0, len(s) - c - h + 1)
            ctx.append(s[st:st + c]); tgt.append(s[st + c:st + c + h])
        return torch.tensor(np.array(ctx), dtype=torch.float32), torch.tensor(np.array(tgt), dtype=torch.float32)

    def predict(self, train, steps):
        from peft import LoraConfig, get_peft_model
        from transformers import TimesFm2_5ModelForPrediction
        torch.manual_seed(self.cfg["seed"])
        rng = np.random.default_rng(self.cfg["seed"])
        ext = _extend(train, self.nat, self.bc_mode, self.bc_w)
        series = [ext[c].values.astype(np.float32) for c in ext.columns]
        base = TimesFm2_5ModelForPrediction.from_pretrained(self.cfg["model_dir"], torch_dtype=torch.float32)
        model = get_peft_model(base, LoraConfig(r=self.cfg["lora_r"], lora_alpha=self.cfg["lora_alpha"],
                                                target_modules="all-linear", lora_dropout=0.05, bias="none"))
        X, T = self._windows(series, rng)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=self.cfg["lr"], weight_decay=0.01)
        bs, n_steps = self.cfg["batch_size"], self.cfg["num_steps"]
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
        model.train()
        for step in range(n_steps):
            i = rng.integers(0, len(X), bs)
            loss = model(past_values=X[i], future_values=T[i], forecast_context_len=self.cfg["context_len"]).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(); sched.step()
        model.eval()
        out = []
        with torch.no_grad():
            for k in range(0, len(series), 256):
                # без forecast_context_len модель дополняет ряд до 16 384 точек — очень медленно
                o = model(past_values=[torch.tensor(s) for s in series[k:k + 256]],
                          forecast_context_len=self.cfg["predict_context_len"])
                out.append(o.full_predictions[:, :steps, 5].numpy())      # индекс 5 — медиана
        return np.concatenate(out).T
