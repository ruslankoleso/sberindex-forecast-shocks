"""Ручная проверка качества разметки «крупных событий».

Стратифицированная выборка (seed 42) из заголовков с местом (МО или регион):
  kw    — отмечены словарём (60);
  nlp   — отмечены только классификатором rubert-tiny2, p ≥ 0,3 (60);
  rest  — не отмечены ни одним способом (80) — для оценки пропусков.
Колонка «крупное_событие» заполняется вручную (да/нет). Оценка точности и полноты способов —
с весами страт (доля страты во всех заголовках с местом / доля в выборке).
Запуск: python -m src.news.annotation sample | python -m src.news.annotation score
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/annotation/major_events_sample.csv")
SIZES = {"kw": 60, "nlp": 60, "rest": 80}


def population():
    d = pd.read_parquet("data/external/news/headlines_tagged.parquet")
    m = pd.read_parquet("data/external/news/major_nlp.parquet")
    d = d.merge(m, on="url", how="left")
    d = d[(d.mo_ids.str.len() > 0) | (d.regions.str.len() > 0)].copy()
    d["strat"] = np.where(d.major_kw.fillna(False), "kw", np.where(d.p_major.fillna(0) >= 0.3, "nlp", "rest"))
    return d


def sample():
    d = population()
    s = pd.concat([g.sample(SIZES[k], random_state=42) for k, g in d.groupby("strat")])
    s = s.sample(frac=1, random_state=7)                        # перемешиваем, чтобы страта не подсказывала ответ
    s = s[["url", "published", "source", "title", "strat"]].assign(крупное_событие="", комментарий="")
    s.to_csv(OUT, index=False, encoding="utf-8-sig")
    print("выборка:", len(s), "→", OUT, "| размеры страт в генеральной совокупности:", d.strat.value_counts().to_dict())


def score():
    d = population()
    pop = d.strat.value_counts()
    s = pd.read_csv(OUT, encoding="utf-8-sig")
    xlsx = OUT.parent / "ПРОВЕРКА_разметки_новостей.xlsx"
    if xlsx.exists():                       # ответы после ручной проверки (тот же порядок строк)
        x = pd.read_excel(xlsx, sheet_name="Заголовки")
        assert (x["Заголовок"].values == s["title"].values).all(), "порядок строк в Excel изменён"
        s["крупное_событие"] = x["Крупное событие (да/нет)"].astype(str).str.strip().str.lower().values
        s["проверено_человеком"] = x["Проверено (поставьте +)"].fillna("").astype(str).str.contains(r"\+").values
        print(f"проверено человеком: {int(s.проверено_человеком.sum())} из {len(s)}")
    s = s[s["крупное_событие"].astype(str).str.lower().isin(["да", "нет"])].copy()
    s["y"] = s["крупное_событие"].str.lower().eq("да")
    w = s.strat.map(pop) / s.strat.map(s.strat.value_counts())
    s["w"] = w
    llm_path = Path("data/external/news/major_llm.parquet")
    llm = pd.read_parquet(llm_path)[["url", "llm_major"]] if llm_path.exists() else pd.DataFrame(columns=["url", "llm_major"])
    s = s.merge(llm, on="url", how="left")
    methods = {"словарь": s.strat.eq("kw"),
               "словарь или rubert-tiny2": s.strat.isin(["kw", "nlp"]),
               "(словарь или rubert) и Qwen": s.strat.isin(["kw", "nlp"]) & s.llm_major.fillna(False).astype(bool)}
    rows = []
    for name, pred in methods.items():
        tp = (s.w * (pred & s.y)).sum(); fp = (s.w * (pred & ~s.y)).sum(); fn = (s.w * (~pred & s.y)).sum()
        rows.append(dict(способ=name, точность=tp / (tp + fp) if tp + fp else np.nan, полнота=tp / (tp + fn) if tp + fn else np.nan))
    r = pd.DataFrame(rows); r["F1"] = 2 * r.точность * r.полнота / (r.точность + r.полнота)
    print(f"размечено вручную: {len(s)}; из них крупных событий: {int(s.y.sum())}")
    print(r.round(3).to_string(index=False))
    r.to_csv("data/annotation/labeling_quality.csv", index=False)


if __name__ == "__main__":
    {"sample": sample, "score": score}[sys.argv[1]]()
