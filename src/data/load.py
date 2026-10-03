"""Загрузка и очистка панели МО из parquet СберИндекса.

В выгрузке нет id муниципалитета, только название. 49 названий принадлежат
нескольким МО, поэтому id собираем сами: каждый непрерывный блок строк
(mo, category) с растущим периодом — отдельный ряд. Для «Все категории»
блоки однозначно соответствуют разным МО; категории привязываем к ним
венгерским алгоритмом по устойчивости доли в общих расходах.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment

ALL = "Все категории"


def _cfg(path="configs/data.yaml"):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def read_raw(cfg):
    d = pd.read_parquet(Path(cfg["raw_dir"]) / cfg["mo_file"],
                        columns=["period", "value", "category_15", "mo"])
    d["period"] = pd.to_datetime(d["period"])
    d = d.rename(columns={"category_15": "category"})
    key = d["mo"] + "|" + d["category"]
    d["block"] = ((key != key.shift()) | (d["period"] < d["period"].shift())).cumsum()
    return d


def _match_blocks(total, other):
    """Сопоставляет блоки категории блокам «Все категории» одного названия."""
    cost = np.full((len(total), len(other)), 1e3)
    for i, (_, a) in enumerate(total):
        for j, (_, f) in enumerate(other):
            s = a.join(f, how="inner", lsuffix="_a", rsuffix="_f")
            if len(s) >= 12:
                cost[i, j] = np.std(np.log(s["value_f"] / s["value_a"]))
    rows, cols = linear_sum_assignment(cost)
    return {i: j for i, j in zip(rows, cols) if cost[i, j] < 1e3}


def build_panel(cfg=None):
    """Возвращает длинную панель (mo_id, period, category, value) и справочник МО."""
    cfg = cfg or _cfg()
    d = read_raw(cfg)
    blocks = {b: g.set_index("period")[["value"]] for b, g in d.groupby("block")}
    meta = d.groupby("block").agg(mo=("mo", "first"), category=("category", "first"))
    meta["occ"] = meta.groupby(["mo", "category"]).cumcount()

    tot = meta[meta["category"] == ALL]
    n_dup = tot.groupby("mo").size()
    ids = {}
    for b, r in tot.iterrows():
        ids[b] = r["mo"] if n_dup[r["mo"]] == 1 else f"{r['mo']} #{r['occ'] + 1}"

    rows = [blocks[b].assign(mo_id=ids[b], category=ALL) for b in ids]
    for mo, grp in meta.groupby("mo"):
        t = [(b, blocks[b]) for b in tot.index[tot["mo"] == mo]]
        for cat, cg in grp[grp["category"] != ALL].groupby("category"):
            o = [(b, blocks[b]) for b in cg.index]
            if len(t) == 1 and len(o) == 1:
                pairs = {0: 0}
            else:
                pairs = _match_blocks(t, o)
            for i, j in pairs.items():
                rows.append(o[j][1].assign(mo_id=ids[t[i][0]], category=cat))

    panel = pd.concat(rows).reset_index().sort_values(["mo_id", "category", "period"])
    mo_dir = (panel[panel["category"] == ALL].groupby("mo_id")
              .agg(months=("value", "size"), first=("period", "min"), last=("period", "max"))
              .reset_index())
    mo_dir["mo_name"] = mo_dir["mo_id"].str.replace(r" #\d+$", "", regex=True)
    mo_dir["is_dup_name"] = mo_dir["mo_id"].str.contains(r" #\d+$")
    return panel, mo_dir


if __name__ == "__main__":
    cfg = _cfg()
    panel, mo_dir = build_panel(cfg)
    out = Path(cfg["interim_dir"])
    out.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out / "panel.parquet", index=False)
    mo_dir.to_parquet(out / "mo_directory.parquet", index=False)
    print(f"МО: {len(mo_dir)}, строк панели: {len(panel)}")
    print(mo_dir["months"].value_counts().sort_index().tail(5).to_string())
