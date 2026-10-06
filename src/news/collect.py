"""Сбор заголовков новостей из открытого архива Lenta.ru по рубрикам и дням.

Для каждой рубрики и дня обходим страницы /rubrics/<rubric>/<YYYY>/<MM>/<DD>/page/<n>/,
сохраняем заголовок, время публикации, рубрику и ссылку. Каждый рубрика-день кэшируется
в отдельный json — сбор можно прервать и продолжить. Пауза между запросами — delay_sec.
Запуск: python -m src.news.collect
"""
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup

UA = {"User-Agent": "Mozilla/5.0 (research; SberIndex hackathon; contact via GitHub)"}
MONTHS = {"января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6, "июля": 7,
          "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12}


def parse(html):
    s = BeautifulSoup(html, "html.parser")
    out = []
    for a in s.select("a.card-full-news"):
        title = a.select_one(".card-full-news__title")
        info = a.select_one(".card-full-news__info, .card-full-news__date")
        title = title.get_text(" ", strip=True) if title else a.get_text(" ", strip=True)
        info_t = info.get_text(" ", strip=True) if info else ""
        out.append({"title": title, "info": info_t, "url": a.get("href")})
    return out


def published(info, day):
    """«12:33, 3 июня 2024» → Timestamp; если только время — берём дату страницы."""
    m = re.search(r"(\d{1,2}):(\d{2})(?:,\s*(\d{1,2})\s+([а-я]+)\s+(\d{4}))?", info)
    if not m:
        return pd.Timestamp(day)
    hh, mm, d, mon, y = m.groups()
    if d:
        return pd.Timestamp(int(y), MONTHS.get(mon, day.month), int(d), int(hh), int(mm))
    return pd.Timestamp(day.year, day.month, day.day, int(hh), int(mm))


def collect_day(rubric, day, cfg, sess):
    path = Path(cfg["raw_dir"]) / f"{rubric}_{day:%Y-%m-%d}.json"
    if path.exists():
        return
    rows, seen = [], set()
    for page in range(1, cfg["max_pages_per_day"] + 1):
        url = f"https://lenta.ru/rubrics/{rubric}/{day:%Y/%m/%d}/" + (f"page/{page}/" if page > 1 else "")
        r = sess.get(url, headers=UA, timeout=30)
        time.sleep(cfg["delay_sec"])
        if r.status_code != 200:
            break
        items = [i for i in parse(r.text) if i["url"] not in seen]
        if not items:
            break
        for i in items:
            seen.add(i["url"])
            i["published"] = str(published(i["info"], day))
            i["rubric"] = rubric
            i["page_day"] = f"{day:%Y-%m-%d}"
        rows += items
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


def build_table(cfg):
    rows = []
    for f in sorted(Path(cfg["raw_dir"]).glob("*.json")):
        rows += json.loads(f.read_text(encoding="utf-8"))
    d = pd.DataFrame(rows)
    d["published"] = pd.to_datetime(d["published"])
    d = d.drop_duplicates("url")
    d.to_parquet(cfg["headlines_file"], index=False)
    return d


def collect_rubric(rub, cfg, start=None, end=None):
    sess = requests.Session()
    for i, day in enumerate(pd.date_range(start or cfg["start"], end or cfg["end"], freq="D")):
        try:
            collect_day(rub, day, cfg, sess)
        except Exception as e:                           # сеть: пропускаем день, докачаем при повторе
            print("ошибка", rub, day.date(), e, flush=True)
        if i % 30 == 0:
            print(rub, "день", day.date(), flush=True)


if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor
    cfg = yaml.safe_load(Path("configs/news.yaml").read_text(encoding="utf-8"))
    Path(cfg["raw_dir"]).mkdir(parents=True, exist_ok=True)
    # потоки: рубрика × часть периода (n_parts); внутри потока пауза delay_sec между запросами
    edges = pd.date_range(cfg["start"], cfg["end"], periods=cfg.get("n_parts", 2) + 1).normalize()
    parts = [(edges[i] + pd.Timedelta(days=int(i > 0)), edges[i + 1]) for i in range(len(edges) - 1)]
    jobs = [(r, s, e) for r in cfg["rubrics"] for s, e in parts]
    with ThreadPoolExecutor(len(jobs)) as ex:
        list(ex.map(lambda j: collect_rubric(j[0], cfg, j[1], j[2]), jobs))
    d = build_table(cfg)
    print("заголовков:", len(d), d["published"].min(), d["published"].max())
