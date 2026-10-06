"""Новости → помесячные признаки, согласованные с данными СберИндекса.

Правило согласования (главное для критерия «корректный способ согласования»):
1. ВРЕМЯ. Новость относится к месяцу публикации (московское время). Данные СберИндекса —
   траты за календарный месяц m, известные после конца месяца m. Поэтому прогноз,
   сделанный по итогам месяца o, может использовать только новости, опубликованные
   до конца месяца o включительно. Новости «будущих» месяцев никогда не попадают
   в признаки (проверяется тестом).
2. МЕСТО. Новость привязывается к МО (если в заголовке город-МО) и к региону
   (город → его регион, или регион назван явно). Новость без места — общероссийская.
   Признак МО = новости про это МО + новости про его регион (+ общероссийский фон).
3. ЧАСТОТА. Дневной поток новостей агрегируется до месяца: число новостей каждой
   темы, нормированное на общее число новостей месяца (иначе рост самого потока
   новостей выглядел бы как рост событий).
Запуск: python -m src.news.features
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.news.geo import build_gazetteer, tag


def load_headlines(cfg):
    d = pd.read_parquet(cfg["headlines_file"])
    d = d[(d.published >= cfg["start"]) & (d.published <= pd.Timestamp(cfg["end"]) + pd.Timedelta(days=1))]
    d["month"] = d.published.dt.to_period("M").dt.to_timestamp()
    return d


def add_topics(d, topics):
    low = d.title.str.lower()
    # граница слова задаётся явно: в pandas 3 строки обрабатывает RE2, где \b понимает только латиницу
    new = {name: low.str.contains(r"(?:^|[^а-яёa-z0-9])(?:" + "|".join(pats) + ")", regex=True)
           for name, pats in topics.items()}
    return pd.concat([d, pd.DataFrame(new, index=d.index)], axis=1)


def build(cfg):
    ref = pd.read_parquet("data/external/territory_reference.parquet").set_index("territory_id")
    d = load_headlines(cfg)
    d = add_topics(d, cfg["topics"])
    gaz = build_gazetteer(ref)
    d["mo_ids"], d["regions"] = tag(d.title, gaz)
    d["is_national"] = d.regions.str.len() == 0
    topics = list(cfg["topics"])
    total = d.groupby("month").size().rename("n_total")

    def agg(g):
        return pd.Series({"n": len(g), **{t: g[t].sum() for t in topics}})

    reg = d[~d.is_national].explode("regions").groupby(["regions", "month"]).apply(agg).reset_index()
    reg = reg.rename(columns={"regions": "region"})
    mo = d[d.mo_ids.str.len() > 0].explode("mo_ids").groupby(["mo_ids", "month"]).apply(agg).reset_index()
    mo = mo.rename(columns={"mo_ids": "territory_id"})
    nat = d.groupby("month").apply(agg).reset_index()
    for t in (reg, mo, nat):
        t[topics + ["n"]] = t[topics + ["n"]].astype(float)
    return d, reg, mo, nat, total


if __name__ == "__main__":
    cfg = yaml.safe_load(Path("configs/news.yaml").read_text(encoding="utf-8"))
    d, reg, mo, nat, total = build(cfg)
    out = Path(cfg["monthly_file"]).parent
    d.drop(columns=["info"], errors="ignore").to_parquet(out / "headlines_tagged.parquet", index=False)
    reg.to_parquet(out / "news_region_monthly.parquet", index=False)
    mo.to_parquet(out / "news_mo_monthly.parquet", index=False)
    nat.merge(total, on="month").to_parquet(out / "news_national_monthly.parquet", index=False)
    print("заголовков:", len(d), "| с привязкой к региону:", round((~d.is_national).mean() * 100, 1), "%",
          "| к МО:", round((d.mo_ids.str.len() > 0).mean() * 100, 1), "%")
    print(d[list(cfg["topics"])].mean().round(3).to_string())
