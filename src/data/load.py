"""Загрузка главной панели расходов из набора организаторов.

consumption.parquet: date (YYYY-MM), territory_id (МО в постоянных границах),
category, value (руб. на человека в месяц). Панель приводится к виду
(territory_id, period, category, value) с датой начала месяца.
Запуск: python -m src.data.load
"""
from pathlib import Path

import pandas as pd
import yaml

ALL = "Все категории"


def _cfg(path="configs/data.yaml"):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build_panel(cfg=None):
    """Длинная панель и справочник территорий (число месяцев, первый и последний месяц)."""
    cfg = cfg or _cfg()
    d = pd.read_parquet(Path(cfg["organizers_dir"]) / cfg["consumption_file"])
    d["period"] = pd.to_datetime(d["date"] + "-01")
    d["value"] = d["value"].astype(float)
    panel = (d.drop(columns="date")[["territory_id", "period", "category", "value"]]
             .sort_values(["territory_id", "category", "period"]).reset_index(drop=True))
    assert not panel.duplicated(["territory_id", "category", "period"]).any()
    total = panel[panel["category"] == ALL]
    terr = (total.groupby("territory_id")
            .agg(months=("value", "size"), first=("period", "min"), last=("period", "max"))
            .reset_index())
    return panel, terr


if __name__ == "__main__":
    cfg = _cfg()
    panel, terr = build_panel(cfg)
    out = Path(cfg["interim_dir"])
    out.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out / "panel.parquet", index=False)
    terr.to_parquet(out / "territories.parquet", index=False)
    print(f"Территорий: {len(terr)}, строк панели: {len(panel)}")
    print(terr["months"].value_counts().sort_index().tail(3).to_string())
