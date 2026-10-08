"""Сбор дополнительных источников новостей: ИА REGNUM (карта сайта) и МЧС России (лента).

REGNUM: месячные файлы карты сайта содержат заголовок и дату каждой новости — скачиваем
файлы за декабрь 2022 – декабрь 2024 (по одному на месяц).
МЧС: лента /deyatelnost/press-centr/novosti?page=N (разрешена robots.txt), на странице ~10
новостей с датами; идём по страницам, пока не дойдём до новостей раньше начала периода.
Кэш по файлам/страницам — сбор можно прервать и продолжить.
Запуск: python -m src.news.collect_more
"""
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup

UA = {"User-Agent": "Mozilla/5.0 (research; SberIndex hackathon)"}
MONTHS = {"января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6, "июля": 7,
          "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12}
NS = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9", "n": "http://www.google.com/schemas/sitemap-news/0.9"}


def regnum(cfg, sess):
    c = cfg["sources"]["regnum"]
    out = Path(c["raw_dir"]); out.mkdir(parents=True, exist_ok=True)
    for ym in pd.period_range(cfg["start"][:7], cfg["end"][:7], freq="M"):
        f = out / f"{ym}.xml"
        if f.exists():
            continue
        r = sess.get(c["sitemap"].format(ym=str(ym)), headers=UA, timeout=120)
        r.raise_for_status()
        f.write_bytes(r.content)
        time.sleep(cfg["delay_sec"])
        print("regnum", ym, flush=True)
    rows = []
    for f in sorted(out.glob("*.xml")):
        root = ET.fromstring(f.read_bytes())
        for u in root.findall("s:url", NS):
            t = u.find("n:news/n:title", NS)
            lm = u.find("s:lastmod", NS)
            d = u.find("n:news/n:publication_date", NS)
            if t is None:
                continue
            rows.append(dict(title=t.text or "", url=u.find("s:loc", NS).text,
                             published=(lm.text if lm is not None else d.text), source="regnum"))
    d = pd.DataFrame(rows)
    d["published"] = pd.to_datetime(d["published"], utc=True).dt.tz_convert("Europe/Moscow").dt.tz_localize(None)
    return d


def _mchs_page(html):
    s = BeautifulSoup(html, "html.parser")
    items = {}
    for a in s.select('a[href*="/deyatelnost/press-centr/novosti/"]'):
        href = a.get("href")
        if not re.search(r"/novosti/\d+$", href or ""):
            continue
        t = a.get_text(" ", strip=True)
        if t:
            items.setdefault(href, {"title": t, "url": href})
    # даты: ищем ближайший к ссылке текст «5 октября 2026»
    out = []
    for href, it in items.items():
        a = s.find("a", href=href, string=True) or s.find("a", href=href)
        node, date = a, None
        for _ in range(6):
            node = node.parent
            if node is None:
                break
            m = re.search(r"(\d{1,2}) ([а-я]+) (\d{4})", node.get_text(" "))
            if m and m.group(2) in MONTHS:
                date = pd.Timestamp(int(m.group(3)), MONTHS[m.group(2)], int(m.group(1)))
                break
        it["published"] = str(date) if date is not None else None
        out.append(it)
    return out


def mchs(cfg, sess):
    c = cfg["sources"]["mchs"]
    out = Path(c["raw_dir"]); out.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(cfg["start"])
    for page in range(1, c["max_pages"] + 1):
        f = out / f"page_{page:04d}.json"
        if f.exists():
            items = json.loads(f.read_text(encoding="utf-8"))
        else:
            r = sess.get(c["list_url"].format(page=page), headers=UA, timeout=60)
            if r.status_code != 200:
                break
            items = _mchs_page(r.text)
            f.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
            time.sleep(cfg["delay_sec"])
        dates = [pd.Timestamp(i["published"]) for i in items if i["published"]]
        if page % 25 == 0:
            print("mchs page", page, min(dates) if dates else None, flush=True)
        if dates and max(dates) < start:
            break
    rows = []
    for f in sorted(out.glob("page_*.json")):
        rows += json.loads(f.read_text(encoding="utf-8"))
    d = pd.DataFrame(rows).dropna(subset=["published"]).drop_duplicates("url")
    d["published"] = pd.to_datetime(d["published"])
    d["source"] = "mchs"
    return d


if __name__ == "__main__":
    cfg = yaml.safe_load(Path("configs/news.yaml").read_text(encoding="utf-8"))
    sess = requests.Session()
    parts = []
    for name, fn in (("regnum", regnum), ("mchs", mchs)):
        try:
            d = fn(cfg, sess)
            d = d[(d.published >= cfg["start"]) & (d.published <= pd.Timestamp(cfg["end"]) + pd.Timedelta(days=1))]
            d.to_parquet(f"data/external/news/headlines_{name}.parquet", index=False)
            print(name, "заголовков:", len(d), d.published.min(), d.published.max(), flush=True)
        except Exception as e:
            print("ошибка источника", name, e, flush=True)
