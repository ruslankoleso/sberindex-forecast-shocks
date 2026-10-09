"""Веса Chronos-2 и TimesFM, дообученные на одной точке прогноза.

При оценке модели дообучаются заново перед каждой из 12 точек прогноза. Чтобы не хранить
12 копий весов, сохраняем одну — для точки «конец 2023 года» (прогноз на весь 2024 год).
Ноутбук загружает её и считает прогноз без дообучения; для остальных точек модели
нужно дообучать заново (python -m src.eval.run --models chronos2_ft_mix timesfm_lora).

    python -m src.forecast.finetuned_weights export   — дообучить, сохранить, упаковать в архив
    python -m src.forecast.finetuned_weights fetch    — скачать архив с весами (релиз на GitHub)
"""
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CFG = yaml.safe_load(Path("configs/foundation.yaml").read_text(encoding="utf-8"))["finetuned_weights"]
OUT = Path(CFG["out_dir"])
CHRONOS, TIMESFM = OUT / "chronos2_ft_mix", OUT / "timesfm_lora"


def train_until_origin():
    from src.eval.cv import load_wide
    Y = load_wide(yaml.safe_load(Path("configs/eval.yaml").read_text(encoding="utf-8")))
    return Y, Y.loc[:CFG["origin"]]


def forecast(model, steps=12):
    """Прогноз на 1…12 месяцев после точки прогноза: таблица месяц × МО."""
    Y, train = train_until_origin()
    pred = model.predict(train, steps)
    return pd.DataFrame(pred, index=Y.index[len(train):len(train) + steps], columns=Y.columns)


def chronos():
    from src.forecast.foundation import Chronos2Pretrained
    return Chronos2Pretrained(CHRONOS)


def timesfm():
    from src.forecast.foundation import TimesFMLoRA
    return TimesFMLoRA(adapter_dir=TIMESFM)


def fetch():
    if CHRONOS.exists() and TIMESFM.exists():
        return
    OUT.mkdir(parents=True, exist_ok=True)
    arc = OUT / CFG["archive"]
    print("скачиваю веса:", CFG["url"])
    urllib.request.urlretrieve(CFG["url"], arc)
    with tarfile.open(arc) as t:
        t.extractall(OUT, filter="data")
    arc.unlink()


def export():
    from src.forecast.foundation import Chronos2FineTuned, TimesFMLoRA
    _, train = train_until_origin()
    stored = {m: pd.read_parquet(f"data/interim/forecasts/{m}.parquet") for m in ("chronos2_ft_mix", "timesfm_lora")}
    for name, model in (("chronos2_ft_mix", Chronos2FineTuned(bc_mode="mix", name="chronos2_ft_mix", save_dir=CHRONOS)),
                        ("timesfm_lora", TimesFMLoRA(save_dir=TIMESFM))):
        model.predict(train, 12)
        # проверка: модель из сохранённых весов даёт тот же прогноз, что и при оценке
        again = forecast(chronos() if name == "chronos2_ft_mix" else timesfm())
        s = stored[name]; s = s[s.origin == CFG["origin"]].pivot(index="target", columns="territory_id", values="y_pred")
        diff = np.abs(again.reindex(s.index)[s.columns].values - s.values) / s.values
        print(f"{name}: расхождение с прогнозом из оценки — медиана {np.median(diff):.2%}, максимум {diff.max():.2%}")
    with tarfile.open(OUT / CFG["archive"], "w:gz") as t:
        for d in (CHRONOS, TIMESFM):
            t.add(d, arcname=d.name)
    print("архив:", OUT / CFG["archive"], f"{(OUT / CFG['archive']).stat().st_size / 1e6:.0f} МБ")


if __name__ == "__main__":
    {"export": export, "fetch": fetch}[sys.argv[1]]()
