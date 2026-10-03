"""Справочник МО → регион из ОКТМО (koo2018/oktmoregions, mreg.json) и сопоставление
с названиями МО СберИндекса.

Уровни ОКТМО: XX000000 — регион; XXYYY000 — МО (район, городской округ, округ,
внутригородская территория); XXYYYZZZ — поселения внутри района (не наш уровень).
"""
import json
import re
from pathlib import Path

import pandas as pd

RAW = Path("data/external/mreg.json")


TYPE_WORDS = {"муниципальный", "муниципального", "городской", "городского", "округ", "округа",
              "район", "района", "город", "г", "внутригородская", "территория", "города",
              "федерального", "значения", "поселение", "городское", "муниципальное",
              "образование", "им", "имени", "закрытое", "административно-территориальное",
              "образования", "зато"}


def _core(s: str) -> str:
    """Название без служебных слов (тип МО), нормализованное для сопоставления."""
    s = s.lower().replace("ё", "е")
    s = re.sub(r"[^а-я0-9\- ]", " ", s)
    return " ".join(sorted(t for t in s.split() if t not in TYPE_WORDS))


def _kind(s: str) -> str:
    s = s.lower()
    if "внутригородск" in s or "муниципальный округ" in s and "города федерального" in s:
        return "vtgfz"
    if "муниципальный округ" in s:
        return "mo"
    if "городской округ" in s:
        return "go"
    if "муниципальный район" in s:
        return "mr"
    return "?"


_norm = _core


def load_oktmo():
    d = json.loads(RAW.read_text(encoding="utf-8"))
    df = pd.DataFrame([(k[:8], v.strip()) for k, v in d.items()], columns=["oktmo", "name"])
    df["reg_code"] = df.oktmo.str[:2]
    reg = df[df.oktmo.str.endswith("000000")].copy()
    reg["region"] = (reg.name.str.replace(r"^Муниципальные образования ", "", regex=True)
                     .str.replace(r" \(.*\)", "", regex=True))
    reg = reg.set_index("reg_code").region.to_dict()
    # уровень МО: код оканчивается на 000, не регион и не группировочный заголовок
    mo = df[df.oktmo.str.endswith("000") & ~df.oktmo.str.endswith("00000")].copy()
    mo = mo[~mo.name.str.contains(r"^(?:Муниципальные|Городские округа|Сельские|Городские|Внутригородские|Муниципальные округа)\b",
                                  regex=True)]
    mo["region"] = mo.reg_code.map(reg)
    mo["key"] = mo.name.map(_core)
    mo["kind"] = mo.name.map(_kind)
    return mo[["oktmo", "name", "reg_code", "region", "key", "kind"]].reset_index(drop=True)


def match(mo_dir: pd.DataFrame):
    ok = load_oktmo()
    rows = []
    for r in mo_dir.itertuples():
        k, kd = _core(r.mo_name), _kind(r.mo_name)
        c = ok[ok.key == k]
        if len(c) > 1 and kd != "?":
            c2 = c[(c.kind == kd) | (c.kind == "?")]   # ОКТМО часто не пишет тип
            c = c2 if len(c2) else c
        rows.append(dict(mo_id=r.mo_id, n_cand=len(c),
                         region=c.region.iloc[0] if len(c) == 1 else None,
                         reg_code=c.reg_code.iloc[0] if len(c) == 1 else None,
                         oktmo=c.oktmo.iloc[0] if len(c) == 1 else None,
                         cand_regions="; ".join(sorted(set(c.region))) if len(c) else ""))
    return pd.DataFrame(rows)


def build_reference(mo_dir: pd.DataFrame, manual: dict):
    """Итоговый справочник: mo_id → регион (точный, ручной или неоднозначный)."""
    r = match(mo_dir).merge(mo_dir[["mo_id", "mo_name"]], on="mo_id")
    r["source"] = r.n_cand.map(lambda n: "oktmo" if n == 1 else ("ambiguous" if n > 1 else "none"))
    for i, row in r[r.region.isna()].iterrows():
        if row.mo_name in manual and not row.mo_id.endswith(tuple(f"#{k}" for k in range(1, 20))):
            r.loc[i, ["region", "source"]] = [manual[row.mo_name], "manual"]
    r["is_moscow"] = r.region.fillna("").str.contains("Москвы")
    r["is_mosobl"] = r.region.fillna("").eq("Московской области")
    r["is_spb"] = r.region.fillna("").str.contains("Петербург")
    return r
