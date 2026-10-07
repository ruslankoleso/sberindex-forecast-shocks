"""Сравнение вариантов backcasting (форма года страны / своя / смесь) на всех МО и на атипичных.

Атипичные МО — 10 % с наибольшим средним отличием формы года (2023 г.) от формы года России.
Форма года считается по 2023 г. — до всех точек прогноза.
Запуск: python -m src.eval.backcast_compare
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.eval.cv import load_wide
from src.eval.metrics import summarize
from src.forecast.foundation import _nat_level


def atypical(share=0.10):
    Y = load_wide(yaml.safe_load(Path("configs/eval.yaml").read_text(encoding="utf-8")))
    nat = _nat_level()
    y = Y.loc["2023"]
    own = np.log(y / y.mean())
    nprof = np.log(nat["2023"] / nat["2023"].mean()).values
    dev = own.sub(nprof, axis=0).abs().mean()
    return set(dev.sort_values(ascending=False).index[: int(len(dev) * share)]), dev


if __name__ == "__main__":
    at, dev = atypical()
    rows = []
    d = Path("data/interim/forecasts")
    for fam in ("timesfm", "tirex"):
        for mode in ("", "_own", "_mix"):
            m = f"{fam}_bc{mode}"
            if not (d / f"{m}.parquet").exists():
                continue
            f = pd.read_parquet(d / f"{m}.parquet")
            for grp, g in (("все МО", f), ("атипичные 10 %", f[f.territory_id.isin(at)])):
                s = summarize(g)
                for r in s.itertuples():
                    rows.append(dict(модель=fam, вариант={"": "форма страны", "_own": "своя форма", "_mix": "смесь 50/50"}[mode],
                                     группа=grp, h=r.h, MAE=round(r.MAE)))
    t = pd.DataFrame(rows).pivot_table(index=["группа", "модель", "вариант"], columns="h", values="MAE")
    print(t.astype(int).to_string())
    print(f"атипичных МО: {len(at)}; медианное отличие формы года: все {dev.median()*100:.1f} %, атипичные {dev[list(at)].median()*100:.1f} %")
