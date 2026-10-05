"""Запуск: python -m src.eval.run --config configs/eval.yaml"""
import argparse

try:  # torch должен загрузиться раньше lightgbm: иначе конфликт libomp и segfault на macOS
    import torch  # noqa: F401
except ImportError:
    pass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.eval.cv import load_wide, run_cv
from src.eval.metrics import summarize
from src.forecast.baselines import REGISTRY
from src.forecast.prophet_model import Prophet_

from src.forecast.lgbm_model import GlobalLGBM, GlobalLGBMDeseason, GlobalLGBMX

def _chronos_bolt():
    from src.forecast.foundation import ChronosBolt
    return ChronosBolt()


def _chronos_bolt_bc():
    from src.forecast.foundation import ChronosBoltBackcast
    return ChronosBoltBackcast()


REGISTRY = {**REGISTRY, "chronos_bolt": _chronos_bolt, "chronos_bolt_bc": _chronos_bolt_bc, "lgbm_catmo": lambda: GlobalLGBMX(cat_mo=True, name="lgbm_catmo"), "lgbm_natcat": lambda: GlobalLGBMX(nat_cat=True, name="lgbm_natcat"), "lgbm_all": lambda: GlobalLGBMX(cat_mo=True, nat_cat=True, name="lgbm_all"), "lgbm_ds": lambda: GlobalLGBMDeseason(), "lgbm": lambda: GlobalLGBM(), "lgbm_no_nat": lambda: GlobalLGBM(use_national_seasonality=False, name="lgbm_no_nat"), "prophet": lambda: Prophet_(False), "prophet_yearly": lambda: Prophet_(True)}


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
    for name in a.models or cfg["models"]:
        res = run_cv(Y, REGISTRY[name](), cfg)
        res.to_parquet(out / f"{name}.parquet", index=False)
    all_res = pd.concat(pd.read_parquet(f) for f in out.glob("*.parquet"))
    s = summarize(all_res)
    s.to_csv(out.parent / "metrics.csv", index=False)
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
