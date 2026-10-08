"""Привязка заголовка новости к МО и региону.

Словарь мест строится из справочника территорий (territory_reference.parquet):
- города — из названий МО «городской округ город X» / «городской округ X»;
- регионы — из названий регионов ОКТМО («Костромской области» → основа «Костромск»)
  и ручного словаря коротких форм республик и округов (configs/news_places.yaml).
Сопоставление по основе слова (без окончания), с заглавной буквы: «в Орске», «Орска» → Орск.
"""
import re
from pathlib import Path

import pandas as pd
import yaml

VOWEL_END = "аяоеёьйыиуюэ"


def stem(word):
    w = word.strip()
    return w[:-1] if w and w[-1].lower() in VOWEL_END else w


def city_name(mo_name):
    m = re.match(r"^городской округ (?:город |поселок |посёлок )?([А-ЯЁ][а-яё\-]+(?: [А-ЯЁ][а-яё\-]+)?)$", mo_name)
    if not m:
        return None
    name = m.group(1)
    # «городской округ Ивдельский» — прилагательное, это не название города
    if re.search(r"(ский|цкий|ный|ой|ий)$", name):
        return None
    return name


def build_gazetteer(ref: pd.DataFrame, places_cfg="configs/news_places.yaml"):
    rows = []
    for tid, r in ref.iterrows():
        c = city_name(r["mo_name"])
        if c and len(stem(c)) >= 4:                     # «Урай» → «Ура» совпало бы с «Ураган»
            rows.append(dict(pattern=stem(c), territory_id=tid, region=r["region"], kind="city"))
    for reg in ref["region"].dropna().unique():
        m = re.match(r"^([А-ЯЁ][а-яё\-]+)(ской|цкой|ого|ой) (области|края)$", reg)
        if m and m.group(1) not in ("Москов",):         # «московский» — чаще про Москву-город
            rows.append(dict(pattern=m.group(1) + ("ск" if m.group(2) == "ской" else "цк" if m.group(2) == "цкой" else ""),
                             territory_id=None, region=reg, kind="region"))
    extra = yaml.safe_load(Path(places_cfg).read_text(encoding="utf-8"))
    for reg, pats in extra["regions"].items():
        for p in pats:
            rows.append(dict(pattern=p, territory_id=None, region=reg, kind="region"))
    g = pd.DataFrame(rows)
    # неоднозначные города (одно название в нескольких регионах) не используем
    amb = g[g.kind == "city"].groupby("pattern").region.nunique()
    g = g[~((g.kind == "city") & g.pattern.isin(amb[amb > 1].index))]
    return g.drop_duplicates(["pattern", "territory_id", "region"]).reset_index(drop=True)


def tag(titles: pd.Series, gaz: pd.DataFrame):
    """Для каждого заголовка — найденные МО и регионы (списки)."""
    pats = sorted(gaz.pattern.unique(), key=len, reverse=True)
    rx = re.compile(r"(?<![А-ЯЁа-яё\-])(" + "|".join(map(re.escape, pats)) + r")[а-яё]{0,4}(?![а-яё])")
    by_pat = gaz.groupby("pattern")
    out_mo, out_reg = [], []
    for t in titles:
        mos, regs = set(), set()
        for m in rx.finditer(t):
            # «в Смоленской области» — это регион, а не город Смоленск: после прилагательного
            # идёт слово-тип территории — привязываем только к региону
            after = t[m.end():m.end() + 12].lower()
            region_form = bool(re.match(r"\s+(област|кра[йяю]|округ|район|республик)", after))
            for _, r in by_pat.get_group(m.group(1)).iterrows():
                regs.add(r.region)
                if r.territory_id is not None and pd.notna(r.territory_id) and not region_form:
                    mos.add(int(r.territory_id))
        out_mo.append(sorted(mos)); out_reg.append(sorted(regs))
    return out_mo, out_reg
