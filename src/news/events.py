"""Новости для обнаружения шоков: анализ событий (event study) и карточки шоков.

1. Анализ событий. Из новостей по зафиксированному заранее правилу выделяются события:
   - событие МО: в месяце ≥ mo_min_news значимых новостей, где названо МО (город);
   - событие региона: всплеск значимых новостей о регионе (≥ region_min_news и ≥ region_ratio ×
     медианы региона) — затрагивает все МО региона.
   Исход — стандартизованный остаток прогноза сезонной наивной модели с дрейфом
   (как у детектора шоков: факт против прогноза, за вычетом общего для всех МО промаха,
   в единицах разброса по МО). Для событий строится средний остаток по месяцам
   относительно события (−3…+3) и сравнивается со случайными МО-месяцами
   (перестановочный тест): если событие реально сдвигает траты, после него |остаток| выше.
2. Карточки шоков. Для каждого шока, найденного лучшим детектором, — новости о МО и его
   регионе в окне [начало сдвига − lookback, тревога + lookahead], и сколько месяцев
   первая значимая новость опередила тревогу детектора.
Запуск: python -m src.news.events
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _cfg():
    return yaml.safe_load(Path("configs/news.yaml").read_text(encoding="utf-8"))


def residuals():
    """Стандартизованный остаток прогноза на 1 мес. (МО × месяц, 2024 г.), как в детекторе."""
    from src.changepoint.benchmark import panel_sigma
    from src.eval.cv import load_wide
    from src.forecast.lgbm_model import load_national
    Y = load_wide(yaml.safe_load(Path("configs/eval.yaml").read_text(encoding="utf-8")))
    L = load_national(yaml.safe_load(Path("configs/forecast.yaml").read_text(encoding="utf-8")))
    sig, cen = panel_sigma(Y, L)
    lY = np.log(Y.values)
    E = np.full(lY.shape, np.nan)
    for t in range(12, len(Y)):
        g = L[Y.index[t - 1]] - L[Y.index[t - 1] - pd.DateOffset(years=1)]
        E[t] = (lY[t] - (lY[t - 12] + g) - cen[t]) / sig[t]
    return pd.DataFrame(E, index=Y.index, columns=Y.columns)


def extract_events(cfg):
    ev = cfg["events"]
    d = Path("data/external/news")
    mo = pd.read_parquet(d / "news_mo_monthly.parquet")
    reg = pd.read_parquet(d / "news_region_monthly.parquet")
    ref = pd.read_parquet("data/external/territory_reference.parquet").set_index("territory_id")
    t = ev["topics"]
    mo["k"] = mo[t].sum(axis=1)
    e_mo = mo[mo.k >= ev["mo_min_news"]][["territory_id", "month", "k"]].assign(level="МО")
    reg["k"] = reg[t].sum(axis=1)
    w = reg.pivot_table(index="region", columns="month", values="k", aggfunc="sum").fillna(0)
    med = w.median(axis=1).clip(lower=1)
    sp = w.stack().rename("k").reset_index()
    sp = sp[(sp.k >= ev["region_min_news"]) & (sp.k >= ev["region_ratio"] * sp.region.map(med))]
    rows = []
    for r in sp.itertuples():
        for tid in ref.index[ref.region == r.region]:
            rows.append(dict(territory_id=tid, month=r.month, k=r.k, level="регион", region=r.region))
    e_reg = pd.DataFrame(rows)
    out = pd.concat([e_mo, e_reg], ignore_index=True)
    # если у МО в тот же месяц есть и событие МО, и событие региона — оставляем событие МО
    return out.sort_values("level").drop_duplicates(["territory_id", "month"])


def event_study(cfg, E):
    ev = cfg["events"]
    lo, hi = ev["window"]
    events = extract_events(cfg)
    events = events[events.territory_id.isin(E.columns)]
    months = list(E.index)

    def profile(pairs):
        """Средний |остаток| по относительным месяцам для набора (МО, месяц события)."""
        acc = {k: [] for k in range(lo, hi + 1)}
        for tid, m in pairs:
            for k in range(lo, hi + 1):
                t = m + pd.DateOffset(months=k)
                if t in E.index and not np.isnan(E.at[t, tid]):
                    acc[k].append(abs(E.at[t, tid]))
        return {k: (np.mean(v) if v else np.nan, len(v)) for k, v in acc.items()}

    res = {}
    rng = np.random.default_rng(ev["seed"])
    for level, g in events.groupby("level"):
        g = g[(g.month >= "2023-11-01") & (g.month <= "2024-11-01")]
        pairs = list(zip(g.territory_id, g.month))
        if not pairs:
            continue
        obs = profile(pairs)
        post = np.nanmean([obs[k][0] for k in (0, 1, 2)])
        # перестановки: случайные МО и месяцы того же числа
        null = []
        cols, mons = E.columns.values, [m for m in months if "2023-11-01" <= str(m.date()) <= "2024-11-01"]
        for _ in range(ev["n_permutations"]):
            rp = list(zip(rng.choice(cols, len(pairs)), [mons[i] for i in rng.integers(0, len(mons), len(pairs))]))
            v = [abs(E.at[m + pd.DateOffset(months=k), tid]) for tid, m in rp for k in (0, 1, 2)
                 if (m + pd.DateOffset(months=k)) in E.index and not np.isnan(E.at[m + pd.DateOffset(months=k), tid])]
            null.append(np.mean(v) if v else np.nan)
        null = np.array(null)
        res[level] = dict(n_events=len(pairs), n_mo=g.territory_id.nunique(), profile=obs, post=post,
                          null_mean=np.nanmean(null), p_value=float(np.mean(null >= post)))
    return res, events


def shock_cards(cfg):
    c, ev = cfg["cards"], cfg["events"]
    sh = pd.read_parquet("data/interim/changepoint/panel_shocks.parquet")
    sh = sh[(sh.detector == "base_residual") & ~sh.seasonal & (sh.shift_pct.abs() >= c["min_shift_pct"])]
    h = pd.read_parquet("data/external/news/headlines_tagged.parquet")
    h = h[h[ev["topics"]].any(axis=1)]
    cards = []
    for r in sh.sort_values("shift_pct", key=abs, ascending=False).itertuples():
        start = r.shift_month - pd.DateOffset(months=c["lookback_months"])
        end = r.alarm + pd.DateOffset(months=c["lookahead_months"])
        win = h[(h.month >= start) & (h.month <= end)]
        about_mo = win[win.mo_ids.apply(lambda x: r.territory_id in list(x))]
        about_reg = win[win.regions.apply(lambda x: isinstance(r.region, str) and r.region in list(x))]
        news = pd.concat([about_mo.assign(уровень="МО"), about_reg.assign(уровень="регион")]).drop_duplicates("url")
        # сначала крупные события (узкий словарь), затем новости о самом МО, затем о регионе
        news["крупное"] = news.title.str.lower().str.contains(cfg["major_events"]["pattern"], regex=True)
        news = news.sort_values(["крупное", "уровень", "published"], ascending=[False, True, True])
        first = news.published.min() if len(news) else pd.NaT
        lead = ((r.alarm.year - first.year) * 12 + r.alarm.month - first.month) if pd.notna(first) else np.nan
        cards.append(dict(territory_id=r.territory_id, mo=r.mo_name, region=r.region, shift_month=r.shift_month,
                          alarm=r.alarm, shift_pct=r.shift_pct, n_news_mo=len(about_mo), n_news_region=len(about_reg),
                          lead_months=lead, n_major=int(news["крупное"].sum()) if len(news) else 0,
                          headlines=news.head(c["max_headlines"])[["published", "уровень", "крупное", "title"]]))
    return cards


def write_cards_md(cards, path="reports/shock_cards.md"):
    lines = ["# Карточки шоков: что говорили новости", "",
             "Шоки, найденные лучшим детектором (по остаткам прогноза) в 2024 году, сдвиг ≥ 5 %, без собственной сезонности МО. "
             "Для каждого — новости значимых тем (ЧС, закрытие/открытие производств, выплаты) о МО и его регионе "
             "за 2 месяца до начала сдвига и до месяца после тревоги. «Опережение» — на сколько месяцев первая такая новость "
             "вышла раньше тревоги детектора. «Крупное событие» — заголовок из узкого словаря (паводки, режим ЧС, эвакуации, закрытие предприятий); такие новости показаны первыми. Подбор по ключевым словам ошибается (например, «выплата» в новости о знаменитости), поэтому карточка — кандидаты в объяснение, а не доказательство. Новости из федерального источника (Lenta.ru); отсутствие новостей не значит, что причины не было.", ""]
    for c in cards:
        if c["n_news_mo"] == 0 and c["n_news_region"] == 0 and abs(c["shift_pct"]) < 10:
            continue
        name = str(c["mo"]).replace("внутригородская территория города федерального значения", "").strip()
        lead = "нет новостей" if np.isnan(c["lead_months"]) else (f"{int(c['lead_months'])} мес. до тревоги" if c["lead_months"] > 0 else ("в месяц тревоги" if c["lead_months"] == 0 else "после тревоги"))
        lines += [f"## {name} ({c['region'] if isinstance(c['region'], str) else 'регион не определён'})", "",
                  f"Сдвиг трат **{c['shift_pct']:+.0f} %** с {c['shift_month']:%m.%Y}, тревога детектора в {c['alarm']:%m.%Y}. "
                  f"Новостей о МО: {c['n_news_mo']}, о регионе: {c['n_news_region']}. Первая новость: {lead}.", ""]
        if len(c["headlines"]):
            lines += ["| Дата | О чём | Крупное событие | Заголовок |", "|---|---|---|---|"]
            for x in c["headlines"].itertuples():
                lines.append(f"| {x.published:%d.%m.%Y} | {x.уровень} | {'да' if x.крупное else ''} | {x.title} |")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def major_events(cfg, E):
    """Крупные события: регион-месяц с ≥ min_news новостями узкого словаря. Затронутые МО —
    названные в этих новостях города, а если городов нет — все МО региона."""
    mj = cfg["major_events"]
    h = pd.read_parquet("data/external/news/headlines_tagged.parquet")
    h = h[h.title.str.lower().str.contains(mj["pattern"], regex=True)]
    h = h[(h.mo_ids.str.len() > 0) | (h.regions.str.len() > 0)]
    ref = pd.read_parquet("data/external/territory_reference.parquet").set_index("territory_id")
    rows = []
    for (reg, mon), g in h.explode("regions").groupby(["regions", "month"]):
        if len(g) < mj["min_news"]:
            continue
        named = {t for x in g.mo_ids for t in x if t in E.columns}
        tids = list(named) or [t for t in ref.index[ref.region == reg] if t in E.columns]
        for t in tids:
            rows.append(dict(territory_id=t, month=mon, region=reg, n_news=len(g), named=bool(named),
                             example=g.title.iloc[0]))
    ev = pd.DataFrame(rows).drop_duplicates(["territory_id", "month"])
    return ev[(ev.month >= "2023-11-01") & (ev.month <= "2024-11-01")], h


def did_test(E, pairs, pre, post, n_perm, seed):
    """Изменение |остатка| «после − до» для событий против случайных МО-месяцев (перестановки)."""
    A = np.abs(E.values)
    idx = {m: i for i, m in enumerate(E.index)}
    cols = {c: j for j, c in enumerate(E.columns)}

    def delta(tid, m):
        i0, j = idx[m], cols[tid]
        a = [A[i0 + k, j] for k in pre if 0 <= i0 + k < len(A)]
        b = [A[i0 + k, j] for k in post if 0 <= i0 + k < len(A)]
        a, b = [x for x in a if not np.isnan(x)], [x for x in b if not np.isnan(x)]
        return np.mean(b) - np.mean(a) if a and b else np.nan

    obs = np.nanmean([delta(t, m) for t, m in pairs])
    rng = np.random.default_rng(seed)
    months = [m for m in E.index if m >= pd.Timestamp("2024-03-01") and m <= pd.Timestamp("2024-10-01")]
    allc = list(E.columns)
    null = []
    for _ in range(n_perm):
        rp = [(allc[rng.integers(len(allc))], months[rng.integers(len(months))]) for _ in pairs]
        null.append(np.nanmean([delta(t, m) for t, m in rp]))
    null = np.array(null)
    return obs, float(np.nanmean(null)), float(np.mean(null >= obs))


def event_profile(E, pairs, lo=-2, hi=3):
    out = {}
    for k in range(lo, hi + 1):
        v = [abs(E.at[m + pd.DateOffset(months=k), t]) for t, m in pairs
             if (m + pd.DateOffset(months=k)) in E.index and not np.isnan(E.at[m + pd.DateOffset(months=k), t])]
        out[k] = float(np.mean(v)) if v else np.nan
    return out


if __name__ == "__main__":
    cfg = _cfg()
    E = residuals()
    res, events = event_study(cfg, E)
    print("Событий (МО-месяцев) по уровням:", events.level.value_counts().to_dict())
    for lvl, r in res.items():
        print(f"\n[{lvl}] событий: {r['n_events']}, МО: {r['n_mo']}")
        print("  средний |остаток| по месяцам относительно события:",
              {k: (round(v[0], 2) if not np.isnan(v[0]) else None, v[1]) for k, v in r["profile"].items()})
        print(f"  после события (0…+2): {r['post']:.2f}; случайные МО-месяцы: {r['null_mean']:.2f}; p = {r['p_value']:.3f}")
    cards = shock_cards(cfg)
    write_cards_md(cards)
    with_news = [c for c in cards if c["n_news_mo"] + c["n_news_region"] > 0]
    leads = [c["lead_months"] for c in with_news if not np.isnan(c["lead_months"])]
    print(f"\nКарточек шоков: {len(cards)}; с новостями: {len(with_news)}; "
          f"новость раньше тревоги: {sum(l > 0 for l in leads)}; в месяц тревоги: {sum(l == 0 for l in leads)}")
    pd.to_pickle(dict(res=res, cards=cards), "data/interim/news_events.pkl")
    # уточнение: крупные события (узкий словарь, см. configs/news.yaml → major_events)
    mj = cfg["major_events"]
    ev, _ = major_events(cfg, E)
    major = {}
    for lab, sub in [("все крупные события", ev), ("названные в новостях города", ev[ev.named]),
                     ("все МО региона события", ev[~ev.named])]:
        pairs = [(t, m) for t, m in zip(sub.territory_id, sub.month) if m >= pd.Timestamp("2024-03-01")]
        obs, nm, p = did_test(E, pairs, mj["pre"], mj["post"], cfg["events"]["n_permutations"], cfg["events"]["seed"])
        major[lab] = dict(n=len(pairs), profile=event_profile(E, pairs), delta=obs, null=nm, p=p)
        print(f"{lab}: n={len(pairs)}, после−до {obs:+.2f} (случайные {nm:+.2f}), p = {p:.3f}")
    print(ev.groupby(["region", "month"]).agg(n_news=("n_news", "first"), пример=("example", "first")).to_string())
    pd.to_pickle(dict(events=ev, res=major), "data/interim/major_events.pkl")
