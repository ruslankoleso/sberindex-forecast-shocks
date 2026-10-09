"""Нейросетевая разметка заголовков: «крупное событие» (паводки, режим ЧС, эвакуации,
прорывы дамб, закрытие предприятий и т. п.).

1. Эмбеддинги заголовков — rubert-tiny2 (вектор [CLS], нормированный).
2. Слабое обучение: положительные примеры — заголовки, отмеченные узким словарём
   (configs/news.yaml → major_events.pattern), отрицательные — случайные остальные.
   Логистическая регрессия на эмбеддингах. Вероятности для оценки — вне обучающей выборки
   (кросс-валидация), чтобы классификатор не «узнавал» свои же примеры.
3. Классификатор находит заголовки, близкие по смыслу к крупным событиям, но без слов
   из словаря («вода затопила посёлок»). Новые срабатывания проверяются выборочно вручную.
Запуск: python -m src.news.nlp
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from src.hub import resolve


def embed(titles, cfg):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(resolve(cfg["model_dir"]))
    model = AutoModel.from_pretrained(resolve(cfg["model_dir"])).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(titles), cfg["batch_size"]):
            b = tok(list(titles[i:i + cfg["batch_size"]]), padding=True, truncation=True,
                    max_length=cfg["max_length"], return_tensors="pt")
            v = model(**b).last_hidden_state[:, 0, :]
            out.append(torch.nn.functional.normalize(v, dim=-1).numpy().astype(np.float16))
    return np.vstack(out)


def classify(d, X, pattern, cfg):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    y = d.title.str.lower().str.contains(pattern, regex=True).values
    rng = np.random.default_rng(cfg["seed"])
    pos = np.where(y)[0]
    neg = rng.choice(np.where(~y)[0], min(len(pos) * cfg["neg_per_pos"], (~y).sum()), replace=False)
    tr = np.concatenate([pos, neg])
    oof = np.zeros(len(tr))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=cfg["seed"]).split(tr, y[tr]):
        m = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        m.fit(X[tr[a]].astype(np.float32), y[tr[a]])
        oof[b] = m.predict_proba(X[tr[b]].astype(np.float32))[:, 1]
    m = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0).fit(X[tr].astype(np.float32), y[tr])
    p = m.predict_proba(X.astype(np.float32))[:, 1]
    p[tr] = oof                                   # для обучающих примеров — вероятности вне выборки
    from sklearn.metrics import roc_auc_score, precision_score, recall_score
    stats = dict(n_pos=int(len(pos)), auc_oof=float(roc_auc_score(y[tr], oof)),
                 recall_oof=float(recall_score(y[tr], oof >= cfg["threshold"])),
                 precision_oof=float(precision_score(y[tr], oof >= cfg["threshold"])))
    return p, y, stats


if __name__ == "__main__":
    ncfg = yaml.safe_load(Path("configs/news.yaml").read_text(encoding="utf-8"))
    cfg = ncfg["nlp"]
    d = pd.read_parquet("data/external/news/headlines_tagged.parquet")
    cache = Path(cfg["emb_cache"])
    if cache.exists() and np.load(cache, mmap_mode="r").shape[0] == len(d):
        X = np.load(cache)
    else:
        X = embed(d.title.tolist(), cfg)
        np.save(cache, X)
    p, y, stats = classify(d, X, ncfg["major_events"]["pattern"], cfg)
    d["p_major"] = p
    d["major_kw"] = y
    d["major_nlp"] = p >= cfg["threshold"]
    print("Классификатор (вне выборки, относительно разметки словарём):", {k: round(v, 3) for k, v in stats.items()})
    print("крупные события: словарь", int(y.sum()), "| классификатор", int(d.major_nlp.sum()),
          "| только классификатор (новые)", int((d.major_nlp & ~d.major_kw).sum()))
    new = d[d.major_nlp & ~d.major_kw].sort_values("p_major", ascending=False)
    pd.set_option("display.max_colwidth", 110)
    print("\nНовые срабатывания (самые уверенные):"); print(new[["published", "source", "title", "p_major"]].head(25).to_string(index=False))
    print("\nНовые срабатывания (случайные 25 для ручной проверки):")
    print(new.sample(min(25, len(new)), random_state=1)[["title", "p_major"]].to_string(index=False))
    d[["url", "p_major", "major_kw", "major_nlp"]].to_parquet("data/external/news/major_nlp.parquet", index=False)
