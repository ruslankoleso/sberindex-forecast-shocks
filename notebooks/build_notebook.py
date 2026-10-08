"""Собирает notebooks/solution.ipynb — рассказ о решении с кодом, статистикой и графиками.
Запуск: python notebooks/build_notebook.py, затем выполнение:
jupyter nbconvert --to notebook --execute --inplace notebooks/solution.ipynb"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""# Прогноз потребления в МО и раннее обнаружение шоков

**Кейс СберИндекса.** Безналичные траты жителя в 2 190 муниципальных образованиях (МО), январь 2023 – декабрь 2024.

1. **Прогноз** трат на 1, 3, 6 и 12 месяцев — превзойти Prophet по MAE.
2. **Шоки** — как можно раньше и точнее находить точки структурных изменений.
3. **Новости** — согласовать с данными СберИндекса и использовать для поиска шоков.

Ноутбук — «живой отчёт»: лёгкие расчёты выполняются прямо здесь, тяжёлые (дообучение нейросетей, десятки моделей на скользящем окне) — заранее командами из `README.md`, их результаты читаются из `data/interim/`.

**Содержание:** 1. Данные · 2. Главная находка · 3. Проверка без подглядывания · 4. Сравнение моделей · 5. Фундаментальные модели · 6. Шоки · 7. Новости · 8. Ключевая ставка · 9. Итоги""")

code("""import os, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, yaml, matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display
pd.set_option("display.max_colwidth", 90); pd.set_option("display.width", 160)
plt.rcParams.update({"figure.figsize": (10, 4), "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
FC = ROOT / "data/interim/forecasts"
need = ["data/interim/panel.parquet", "data/interim/forecasts/prophet.parquet", "data/interim/changepoint/panel_shocks.parquet"]
missing = [p for p in need if not (ROOT / p).exists()]
print("Проект:", ROOT); print("Нет предрассчитанных файлов:" if missing else "Все предрассчитанные результаты на месте.", missing or "")""")

md("""## 1. Данные
Главный файл — набор организаторов `consumption.parquet`: одно число в месяц на МО и категорию — сколько в среднем потратил житель по картам.""")
code("""from src.data.load import build_panel
panel, terr = build_panel()
print(f"Строк: {len(panel):,}; МО: {panel.territory_id.nunique():,}; месяцев: {panel.period.nunique()} ({panel.period.min():%Y-%m} … {panel.period.max():%Y-%m})")
print("Категории:", ", ".join(panel.category.unique()))
print("МО с полной историей (24 мес.):", int((terr.months == 24).sum()))
panel[panel.category == "Все категории"].value.describe().round(0).to_frame("руб. на жителя в месяц").T""")
code("""from src.eval.cv import load_wide
Y = load_wide(yaml.safe_load(open("configs/eval.yaml")))
ref = pd.read_parquet("data/external/territory_reference.parquet").set_index("territory_id")
med = Y.median(axis=1)
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(Y.index, med, lw=2, color="#2a78d6"); ax[0].set_title("Медианные траты жителя МО, руб./мес.")
mm = (Y.pct_change().median(axis=1) * 100).iloc[1:]
ax[1].bar(mm.index, mm.values, width=20, color=np.where(mm.values > 0, "#1baf7a", "#eb6834")); ax[1].set_title("Медианное изменение за месяц, % (сезонность)")
plt.tight_layout(); plt.show()""")

md("""## 2. Главная находка: траты МО движутся вместе со страной
Почти всё изменение — общая сезонность и общий рост ≈15 % в год. Отсюда **сезонная наивная модель с дрейфом**: прогноз = тот же месяц год назад × рост трат по России за последний известный год (рост берётся из ряда по России с 2018 года — элемент иерархического прогнозирования).""")
code("""from src.forecast.lgbm_model import load_national
L = load_national(yaml.safe_load(open("configs/forecast.yaml")))
g = np.exp(L["2023-12-01"] - L["2022-12-01"])
mo = 1  # Майкоп
pred, fact = Y.loc["2023-12-01", mo] * g, Y.loc["2024-12-01", mo]
display(pd.DataFrame({"значение": [f"{Y.loc['2023-12-01', mo]:,.0f}", f"×{g:.3f}", f"{pred:,.0f}", f"{fact:,.0f}", f"{(pred/fact-1)*100:+.1f} %"]},
                     index=["Майкоп, декабрь 2023", "рост трат России за год", "прогноз на декабрь 2024", "факт", "ошибка"]))
fig, ax = plt.subplots(); ax.plot(np.exp(L), color="#0b0b0b"); ax.set_title("Траты по России, млрд руб./мес. — длинная история для роста и сезонности"); plt.show()""")

md("""## 3. Как честно проверяем: скользящее окно
В каждой точке прогноза модель обучается только на прошлом и прогнозирует 1, 3, 6, 12 месяцев вперёд. Ниже — расчёт «вживую» для двух простых моделей и тест на отсутствие утечки будущего.""")
code("""from src.eval.cv import run_cv
from src.eval.metrics import summarize
from src.forecast.baselines import SeasonalNaive, SeasonalNaiveNationalGrowth
cfg = yaml.safe_load(open("configs/eval.yaml"))
res = pd.concat([run_cv(Y, SeasonalNaive(), cfg), run_cv(Y, SeasonalNaiveNationalGrowth(), cfg)])
print("Точек прогноза по горизонтам:", res[res.model == "seasonal_naive"].groupby("h").origin.nunique().to_dict())
summarize(res).pivot(index="model", columns="h", values="MAE").round(0)""")
code("""import subprocess
print(subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], capture_output=True, text=True).stdout[-200:])""")

md("""## 4. Сравнение моделей прогноза
Все модели посчитаны на одних и тех же точках (`python -m src.eval.run --models …`). Метрика — **MAE**, руб. на жителя в месяц; дополнительно R² по уровню и по росту за год.""")
code("""NAMES = {"prophet": "Prophet (базовая)", "seasonal_naive": "Сезонная наивная", "snaive_natg": "Сезонная наивная с дрейфом",
         "snaive_shrink_w25_k1": "+ shrinkage роста МО (w=0,25)*", "autoets": "AutoETS", "autotheta": "AutoTheta", "nhits": "N-HiTS",
         "lgbm_catmo": "LightGBM", "lgbm_resid": "+ LightGBM по остаткам", "chronos2_bc": "Chronos-2 zero-shot + backcasting",
         "timesfm_bc": "TimesFM zero-shot + backcasting", "tirex_bc_mix": "TiRex + форма года МО", "timesfm_lora": "TimesFM дообученный (LoRA)",
         "chronos2_ft_mix": "Chronos-2 дообученный + форма года МО", "ens_inv_mae": "Ансамбль (веса по обратной ошибке)"}
have = [m for m in NAMES if (FC / f"{m}.parquet").exists()]
allres = pd.concat(pd.read_parquet(FC / f"{m}.parquet") for m in have)
S = summarize(allres)
tab = S.pivot(index="model", columns="h", values="MAE").loc[have].rename(index=NAMES).round(0).astype(int)
tab.columns = [f"{h} мес." for h in tab.columns]
display(tab.style.highlight_min(axis=0, color="#d6e8fb"))
print("* вес w подобран на тех же данных — оценка оптимистичная")""")
code("""Image("report/figures/01_mae_by_horizon.png")""")
code("""r2 = S[S.model.isin(["prophet", "snaive_natg", "chronos2_ft_mix", "ens_inv_mae"])].pivot(index="model", columns="h", values="R2_yoy").rename(index=NAMES).round(2)
r2.columns = [f"R² роста, {h} мес." for h in r2.columns]; r2""")
code("""Image("report/figures/02_example_forecast.png")""")
md("""**Какую модель выбрать.** Лучшая по точности — ансамбль; на 3–12 мес. он вровень с сезонной наивной моделью с дрейфом, которая и рекомендуется для практики (секунды расчёта, одна формула). Дообученный Chronos-2 — лучшая фундаментальная модель и «страховка» при смене режима:""")
code("""rows = []
for m in ["snaive_natg", "chronos2_ft_mix", "ens_inv_mae"]:
    f = pd.read_parquet(FC / f"{m}.parquet"); f["ae"] = (f.y_true - f.y_pred).abs()
    f["история"] = np.where(((f.origin.dt.year - 2023) * 12 + f.origin.dt.month) <= 17, "12–17 мес.", "18–23 мес.")
    rows.append(f[f.h == 1].groupby("история").ae.mean().rename(NAMES[m]))
pd.concat(rows, axis=1).round(0).astype(int).T.rename_axis("MAE на 1 мес. при истории")""")

md("""## 5. Фундаментальные модели на коротких рядах
У МО 12–23 точки — нейросетям этого мало (TimesFM нужно ≥32). Три шага: **backcasting** (восстановление прошлого МО по ряду России), **форма года МО** (смесь сезонности страны и самого МО), **дообучение** перед каждой точкой прогноза.""")
code("""from src.forecast.foundation import _extend, _nat_level
nat = _nat_level(); tid = ref.index[ref.mo_name.str.contains("Соболевский")][0]
fig, ax = plt.subplots(figsize=(12, 4))
for mode, c in [("national", "#8b8a85"), ("mix", "#2a78d6")]:
    e = _extend(Y.iloc[:12][[tid]], nat, mode)
    ax.plot(e.index[-36:], e[tid].values[-36:], lw=2, color=c, label={"national": "backcasting: сезонность страны", "mix": "backcasting: смесь с формой года МО"}[mode])
ax.axvline(pd.Timestamp("2023-01-01"), color="k", ls="--", lw=1); ax.legend(); ax.set_title("Соболевский р-н Камчатки: восстановленное прошлое (до пунктира) и реальные данные 2023")
plt.show()""")
code("""fm = ["timesfm", "timesfm_bc", "timesfm_bc_mix", "timesfm_lora", "chronos2", "chronos2_bc", "chronos2_ft", "chronos2_ft_mix", "tirex_bc", "tirex_bc_mix", "prophet"]
fm = [m for m in fm if (FC / f"{m}.parquet").exists()]
t = summarize(pd.concat(pd.read_parquet(FC / f"{m}.parquet") for m in fm)).pivot(index="model", columns="h", values="MAE").loc[fm].round(0).astype(int)
t.columns = [f"{h} мес." for h in t.columns]; t""")

md("""## 6. Обнаружение шоков
Шесть онлайн-детекторов (решение — только по прошлому). Разметки шоков нет, поэтому: (1) искусственные шоки в 400 реальных рядах МО при равной доле ложных тревог (10 %); (2) реальные кризисы России 2020 и 2022; (3) реальные шоки МО в 2024 году.""")
code("""bench = pd.read_csv("data/interim/changepoint/benchmark_summary.csv", index_col=0)
bench.sort_values("доля_найденных", ascending=False).round(3)""")
code("""Image("report/figures/03_detectors.png")""")
code("""Image("report/figures/05_national_events.png")""")
code("""sh = pd.read_parquet("data/interim/changepoint/panel_shocks.parquet")
best = sh[(sh.detector == "base_residual") & ~sh.seasonal & (sh.shift_pct.abs() >= 10)].copy()
best["МО"] = best.mo_name.str.replace("внутригородская территория города федерального значения", "").str[:40]
print(f"Шоков 2024 г. (лучший детектор, сдвиг ≥10 %, не сезонность): {len(best)} МО из {Y.shape[1]}")
best.sort_values("shift_pct", key=abs, ascending=False)[["МО", "region", "shift_month", "alarm", "shift_pct"]].head(10).round(1)""")
code("""Image("report/figures/04_shock_examples.png")""")

md("""## 7. Новости
**Согласование с данными СберИндекса:** время (месяц публикации, только новости до момента прогноза), место (город → МО и регион), частота (месячные доли тем). Источники: Lenta.ru, ИА REGNUM, МЧС России.""")
code("""h = pd.read_parquet("data/external/news/headlines_tagged.parquet")
print(f"Заголовков: {len(h):,}"); display(h.groupby("source").agg(заголовков=("title", "size"), с_регионом=("is_national", lambda s: f"{(~s).mean():.0%}")))
from src.news.geo import build_gazetteer, tag
gaz = build_gazetteer(ref)
ex = pd.Series(["В Орске эвакуировали жителей из-за прорыва дамбы", "В Туве ввели режим ЧС", "Жители Волгореченска пожаловались на закрытие ГРЭС"])
mo_ids, regs = tag(ex, gaz)
pd.DataFrame({"заголовок": ex, "МО": [[ref.mo_name.get(i, i) for i in x] for x in mo_ids], "регион": regs})""")
md("""**Путь из трёх попыток:** (1) новости как признаки прогноза — эффекта нет; (2) анализ событий по широкому правилу — эффекта нет; (3) только **крупные события** (паводки, режим ЧС, эвакуации, прорывы дамб, закрытия предприятий) — после них траты затронутых МО сильнее расходятся с прогнозом, пик через 1–2 мес. Оговорка: третья попытка — уточнение после неудачи.""")
code("""me = pd.read_pickle("data/interim/major_events.pkl")
pd.DataFrame({k: {"событий": v["n"], "после − до": round(v["delta"], 2), "у случайных МО": round(v["null"], 2), "p": round(v["p"], 3)} for k, v in me["res"].items()}).T""")
code("""Image("report/figures/07_event_study.png")""")
code("""Image("report/figures/06_news_case_orsk.png")""")
code("""ni = pd.read_pickle("data/interim/changepoint/news_informed.pkl")
al = ni["alarms"].pivot(index="territory_id", columns="режим", values="alarm")
print(f"Порог обычный {ni['thr']:.2f}, после новости о крупном событии в регионе — {ni['thr_low']:.2f}")
print("Тревог всего:", al.notna().sum().to_dict())
o = ni["events_eval"]
o.pivot_table(index="named", columns="режим", values="найден", aggfunc="mean").rename(index={True: "МО, названные в новостях", False: "все МО региона события"}).round(3)""")
code("""print(open("report/shock_cards.md", encoding="utf-8").read()[:3000])""")

md("""## 8. Ключевая ставка
Ставка одна для всех МО — проверяем её там, где она может помочь: в прогнозе роста трат страны (дрейф модели).""")
code("""kr = pd.read_parquet("data/interim/key_rate_check.parquet")
kr.pivot_table(index="метод", columns="h", values="err").round(2).rename_axis("ошибка прогноза годового роста, п.п.")""")

md("""## 9. Итоги
- Прогноз: ансамбль с весами по обратной ошибке лучше Prophet на 36–51 % на всех горизонтах; на 3–12 мес. сезонная наивная модель с дрейфом даёт ту же точность.
- Фундаментальные модели работают на коротких рядах после backcasting, учёта формы года МО и дообучения; лучший — Chronos-2.
- Шоки: детектор по остаткам прогноза находит 61 % искусственных шоков и почти не реагирует на выбросы; для России — BOCPD.
- Новости: согласованы по времени и месту; крупные события из новостей опережают сдвиг трат на 1–2 мес. и делают детектор чувствительнее там, где шок вероятнее.

Подробности — `reports/REPORT.md`, объяснение простым языком — `docs/EXPLAINED.md`, журнал решений — `docs/DECISIONS.md`.""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "sber-venv", "display_name": "Python (сбер .venv)", "language": "python"}
nbf.write(nb, "notebooks/solution.ipynb")
print("записан notebooks/solution.ipynb, ячеек:", len(C))
