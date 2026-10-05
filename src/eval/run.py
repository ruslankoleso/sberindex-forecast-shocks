"""Запуск: python -m src.eval.run --config configs/eval.yaml"""
import argparse

import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.eval.cv import load_wide, run_cv
from src.eval.metrics import summarize
from src.forecast.baselines import REGISTRY
def _lazy(module, cls, **kw):
    """Модель создаётся с импортом модуля только в своём процессе."""
    def make():
        import importlib
        return getattr(importlib.import_module(module), cls)(**kw)
    return make


FC = "src.forecast."
REGISTRY = {
    **REGISTRY,
    "prophet": _lazy(FC + "prophet_model", "Prophet_", yearly=False),
    "prophet_yearly": _lazy(FC + "prophet_model", "Prophet_", yearly=True),
    "lgbm": _lazy(FC + "lgbm_model", "GlobalLGBM"),
    "lgbm_no_nat": _lazy(FC + "lgbm_model", "GlobalLGBM", use_national_seasonality=False, name="lgbm_no_nat"),
    "lgbm_ds": _lazy(FC + "lgbm_model", "GlobalLGBMDeseason"),
    "lgbm_catmo": _lazy(FC + "lgbm_model", "GlobalLGBMX", cat_mo=True, name="lgbm_catmo"),
    "lgbm_natcat": _lazy(FC + "lgbm_model", "GlobalLGBMX", nat_cat=True, name="lgbm_natcat"),
    "lgbm_all": _lazy(FC + "lgbm_model", "GlobalLGBMX", cat_mo=True, nat_cat=True, name="lgbm_all"),
    "lgbm_resid": _lazy(FC + "resid_model", "ResidLGBM"),
    "lgbm_resid_plain": _lazy(FC + "resid_model", "ResidLGBM", spatial=False, cats=False, name="lgbm_resid_plain"),
    "snaive_nat_ets": _lazy(FC + "national", "SNaiveNationalForecast", method="ets"),
    "snaive_nat_chronos2": _lazy(FC + "national", "SNaiveNationalForecast", method="chronos2"),
    "chronos_bolt": _lazy(FC + "foundation", "ChronosBolt"),
    "chronos_bolt_bc": _lazy(FC + "foundation", "ChronosBoltBackcast"),
    "chronos2": _lazy(FC + "foundation", "Chronos2"),
    "chronos2_bc": _lazy(FC + "foundation", "Chronos2Backcast"),
}


def _run_one(name, config):
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    np.random.seed(cfg["seed"])
    Y = load_wide(cfg)
    res = run_cv(Y, REGISTRY[name](), cfg)
    res.to_parquet(Path(cfg["out_dir"]) / f"{name}.parquet", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--models", nargs="*")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    np.random.seed(cfg["seed"])
    Y = load_wide(cfg)
    print(f"Ряды: {Y.shape[1]} МО × {Y.shape[0]} мес.")
    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    # Каждая модель — в отдельном процессе: torch (Chronos) и многопоточный LightGBM
    # в одном процессе на macOS падают с segfault из-за конфликта OpenMP-библиотек.
    ctx = mp.get_context("spawn")
    for name in a.models or cfg["models"]:
        p = ctx.Process(target=_run_one, args=(name, a.config))
        p.start(); p.join()
        if p.exitcode != 0:
            raise SystemExit(f"Модель {name} завершилась с кодом {p.exitcode}")
    all_res = pd.concat(pd.read_parquet(f) for f in out.glob("*.parquet"))
    s = summarize(all_res)
    s.to_csv(out.parent / "metrics.csv", index=False)
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
