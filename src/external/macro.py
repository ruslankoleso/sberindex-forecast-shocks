"""Загрузка открытых внешних рядов: ключевая ставка и курс USD/RUB (ЦБ РФ),
производственный календарь (isdayoff.ru). Результат — месячная таблица data/external/macro_monthly.parquet.
Запуск: python -m src.external.macro
"""
import io
import xml.etree.ElementTree as ET
from calendar import monthrange
from pathlib import Path

import numpy as np
import pandas as pd
import requests

START, END = "2018-12-01", pd.Timestamp.today().strftime("%Y-%m-%d")
OUT = Path("data/external")
UA = {"User-Agent": "Mozilla/5.0"}


def key_rate():
    url = ("https://www.cbr.ru/hd_base/KeyRate/?UniDbQuery.Posted=True&UniDbQuery.From=01.12.2018"
           f"&UniDbQuery.To={pd.Timestamp(END):%d.%m.%Y}")
    t = pd.read_html(io.StringIO(requests.get(url, headers=UA, timeout=60).text), decimal=",", thousands=" ")[0]
    t.columns = ["date", "key_rate"]
    t["date"] = pd.to_datetime(t["date"], format="%d.%m.%Y")
    return t.set_index("date").key_rate.sort_index()


def usd_rub():
    url = ("https://www.cbr.ru/scripts/XML_dynamic.asp?date_req1=01/12/2018&"
           f"date_req2={pd.Timestamp(END):%d/%m/%Y}&VAL_NM_RQ=R01235")
    root = ET.fromstring(requests.get(url, headers=UA, timeout=60).content)
    rows = [(pd.to_datetime(r.attrib["Date"], format="%d.%m.%Y"),
             float(r.find("Value").text.replace(",", "."))) for r in root.findall("Record")]
    return pd.Series(dict(rows)).sort_index()


def calendar():
    rows = []
    for y in range(2018, pd.Timestamp(END).year + 1):
        s = requests.get(f"https://isdayoff.ru/api/getdata?year={y}&pre=1", headers=UA, timeout=60).text.strip()
        d = pd.date_range(f"{y}-01-01", periods=len(s), freq="D")
        rows.append(pd.DataFrame({"date": d, "code": [int(c) for c in s]}))
    c = pd.concat(rows)
    c["m"] = c.date.dt.to_period("M").dt.to_timestamp()
    # 0 — рабочий, 1 — выходной, 2 — сокращённый предпраздничный, 4 — рабочий день по переносу
    g = c.groupby("m")
    return pd.DataFrame({"working_days": g.code.apply(lambda s: int((s != 1).sum())),
                         "days_off": g.code.apply(lambda s: int((s == 1).sum())),
                         "short_days": g.code.apply(lambda s: int((s == 2).sum()))})


def build():
    kr, fx = key_rate(), usd_rub()
    idx = pd.date_range(START, END, freq="D")
    kr_d = kr.reindex(idx).ffill()           # ставка действует до следующего решения
    fx_d = fx.reindex(idx).ffill()
    m = pd.DataFrame({"key_rate_mean": kr_d.resample("MS").mean(),
                      "key_rate_end": kr_d.resample("MS").last(),
                      "usd_mean": fx_d.resample("MS").mean(),
                      "usd_end": fx_d.resample("MS").last()})
    m["usd_chg"] = np.log(m.usd_end).diff()
    m["key_rate_chg"] = m.key_rate_end.diff()
    m = m.join(calendar())
    m.index.name = "period"
    return m


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    m = build()
    m.to_parquet(OUT / "macro_monthly.parquet")
    print(m.shape, m.index.min(), m.index.max())
    print(m.round(2).iloc[[0, 12, 15, 40, 41, 60, -2, -1]].to_string())
