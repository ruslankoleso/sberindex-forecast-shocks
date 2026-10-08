"""Как меняется ошибка моделей по мере роста истории (точки прогноза 2023-12…2024-11).
Проверка гипотезы «фундаментальная модель выигрывает при большем объёме данных».
Запуск: python -m src.eval.history_check"""
import pandas as pd

MODELS = {"snaive_natg": "Сезонная наивная с дрейфом", "chronos2_ft_mix": "Chronos-2 дообученный (смесь форм)",
          "ens_inv_mae": "Ансамбль"}

if __name__ == "__main__":
    rows = []
    for m, lab in MODELS.items():
        f = pd.read_parquet(f"data/interim/forecasts/{m}.parquet")
        f["ae"] = (f.y_true - f.y_pred).abs()
        f["история"] = (f.origin.dt.year - 2023) * 12 + f.origin.dt.month
        f["период"] = pd.cut(f["история"], [11, 17, 23], labels=["12–17 мес.", "18–23 мес."])
        for (h, p), g in f[f.h.isin([1, 3])].groupby(["h", "период"], observed=True):
            rows.append(dict(модель=lab, h=h, история=p, MAE=round(g.ae.mean())))
    print(pd.DataFrame(rows).pivot_table(index=["h", "история"], columns="модель", values="MAE").to_string())
