"""Собирает notebooks/solution.ipynb — рассказ о решении с кодом, статистикой и графиками.
Запуск: python notebooks/build_notebook.py, затем выполнение:
jupyter nbconvert --to notebook --execute --inplace notebooks/solution.ipynb"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""# Прогноз потребления в МО и раннее обнаружение шоков

В этом ноутбуке мы проходим по решению шаг за шагом: от данных до найденных шоков. Задача — прогнозировать безналичные траты жителя в 2 190 муниципальных образованиях на 1, 3, 6 и 12 месяцев, обогнать Prophet и как можно раньше замечать структурные сдвиги.

Лёгкие расчёты идут прямо здесь. Тяжёлые — десятки моделей на скользящем окне и дообучение нейросетей — мы запускали заранее (`./run_all.sh`), а здесь читаем их результаты из `data/interim/`.

**План:** 1. Данные · 2. Главная находка · 3. Как проверяем · 4. Модели · 5. Нейросети на коротких рядах · 6. Шоки · 7. Новости · 8. Ключевая ставка · 9. Итоги""")

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
Главный файл — набор организаторов `consumption.parquet`. На каждый муниципалитет, месяц и категорию — одно число: сколько в среднем потратил по картам один житель.""")
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
На графиках выше видно, что почти все муниципалитеты живут в одном ритме: в декабре траты взлетают, в январе проваливаются, а год к году растут примерно на 15 %. Своя динамика у МО — это проценты поверх общей.

Отсюда простая и сильная модель — **сезонная наивная модель с дрейфом**: берём тот же месяц год назад и умножаем на рост трат по России за последний известный год. Рост берём именно по стране: у муниципалитета 12–23 месяца истории, и свой годовой рост по ним не посчитать, а ряд России идёт с 2018 года.""")
code("""from src.forecast.lgbm_model import load_national
L = load_national(yaml.safe_load(open("configs/forecast.yaml")))
g = np.exp(L["2023-12-01"] - L["2022-12-01"])
mo = 1  # Майкоп
pred, fact = Y.loc["2023-12-01", mo] * g, Y.loc["2024-12-01", mo]
display(pd.DataFrame({"значение": [f"{Y.loc['2023-12-01', mo]:,.0f}", f"×{g:.3f}", f"{pred:,.0f}", f"{fact:,.0f}", f"{(pred/fact-1)*100:+.1f} %"]},
                     index=["Майкоп, декабрь 2023", "рост трат России за год", "прогноз на декабрь 2024", "факт", "ошибка"]))
fig, ax = plt.subplots(); ax.plot(np.exp(L), color="#0b0b0b"); ax.set_title("Траты по России, млрд руб./мес. — длинная история для роста и сезонности"); plt.show()""")

md("""## 3. Как проверяем: скользящее окно
Модель в каждой точке прогноза видит только прошлое и прогнозирует 1, 3, 6 и 12 месяцев вперёд, потом мы сдвигаемся на месяц и повторяем. Ниже — то же самое вживую для двух простых моделей, а затем тест, который проверяет, что будущее в прогноз не попадает.""")
code("""from src.eval.cv import run_cv
from src.eval.metrics import summarize
from src.forecast.baselines import SeasonalNaive, SeasonalNaiveNationalGrowth
cfg = yaml.safe_load(open("configs/eval.yaml"))
res = pd.concat([run_cv(Y, SeasonalNaive(), cfg), run_cv(Y, SeasonalNaiveNationalGrowth(), cfg)])
print("Точек прогноза по горизонтам:", res[res.model == "seasonal_naive"].groupby("h").origin.nunique().to_dict())
summarize(res).pivot(index="model", columns="h", values="MAE").round(0)""")
code("""import subprocess
print(subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], capture_output=True, text=True).stdout[-200:])""")

md("""## 4. Модели прогноза
Все модели посчитаны на одних и тех же точках прогноза и муниципалитетах. Главная метрика — **MAE**, рублей на жителя в месяц. Дополнительно — R² по росту за год: угадывает ли модель, вырастут ли траты МО быстрее или медленнее среднего.""")
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
md("""**Значимы ли различия?** Тест Диболда–Мариано по точкам прогноза и доля муниципалитетов, где одна модель лучше другой. Точек немного (7–12), поэтому тест осторожный.""")
code("""sig = pd.read_csv("data/interim/significance.csv")
sig["A"] = sig.A.map(NAMES).fillna(sig.A); sig["B"] = sig.B.map(NAMES).fillna(sig.B)
sig[sig.h < 12][["A", "B", "h", "разница_MAE", "p_DM", "доля_МО_где_A_лучше"]]""")
md("""**Какую модель выбрать.** Сравниваем по нескольким критериям сразу: точность на каждом горизонте, стабильность, время, понятность. Смотрим, кто недоминируем (Парето), и агрегируем точность правилами Борда и Коупленда из теории выбора.""")
code("""pd.read_csv("data/interim/selection_models.csv", index_col=0)""")
md("""Оба правила ставят первым ансамбль. Но на 3–12 месяцев он статистически не отличается от сезонной наивной модели с дрейфом, а та считается за доли секунды и объясняется одной фразой, — её мы и рекомендуем для практики. Дообученный Chronos-2 — лучшая нейросеть; интересно, где именно он полезен:""")
code("""rows = []
for m in ["snaive_natg", "chronos2_ft_mix", "ens_inv_mae"]:
    f = pd.read_parquet(FC / f"{m}.parquet"); f["ae"] = (f.y_true - f.y_pred).abs()
    f["история"] = np.where(((f.origin.dt.year - 2023) * 12 + f.origin.dt.month) <= 17, "12–17 мес.", "18–23 мес.")
    rows.append(f[f.h == 1].groupby("история").ae.mean().rename(NAMES[m]))
pd.concat(rows, axis=1).round(0).astype(int).T.rename_axis("MAE на 1 мес. при истории")""")

md("""Chronos-2 выигрывает в первой половине 2024 года — тогда рост трат страны резко замедлился, и правило «рост сохранится» ошиблось. Когда всё успокоилось, простая модель его догнала. Поэтому в ансамбле нейросеть — страховка на случай смены режима.

## 5. Нейросети на коротких рядах
Фундаментальным моделям 12–23 точек мало (TimesFM официально нужно не меньше 32). Мы восстановили прошлое каждого МО по ряду России (**backcasting**), учли собственную **форму года** муниципалитета и **дообучили** модели перед каждой точкой прогноза. Пример — Соболевский район Камчатки, где траты поднимаются в октябре, после лососёвой путины:""")
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
Мы сравнили шесть детекторов, все решают только по прошлому. Готовой разметки шоков нет, поэтому проверяли тремя способами: встраивали шоки известной величины в 400 реальных рядов МО (с одинаковой для всех долей ложных тревог — 10 %), смотрели на реальные кризисы России 2020 и 2022 годов и на реальные события в муниципалитетах 2024 года.""")
code("""pd.read_csv("data/interim/selection_detectors.csv", index_col=0)""")
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
Новость — это момент и место в тексте, а данные СберИндекса — месяц и код муниципалитета. Согласуем их в три шага: **время** (новость относится к месяцу публикации, прогноз видит только новости до своего момента), **место** (город в заголовке → МО и регион, с учётом падежей), **частота** (поток новостей → месячные доли тем). Источники — Lenta.ru, ИА REGNUM и МЧС России.""")
code("""h = pd.read_parquet("data/external/news/headlines_tagged.parquet")
print(f"Заголовков: {len(h):,}"); display(h.groupby("source").agg(заголовков=("title", "size"), с_регионом=("is_national", lambda s: f"{(~s).mean():.0%}")))
from src.news.geo import build_gazetteer, tag
gaz = build_gazetteer(ref)
ex = pd.Series(["В Орске эвакуировали жителей из-за прорыва дамбы", "В Туве ввели режим ЧС", "Жители Волгореченска пожаловались на закрытие ГРЭС"])
mo_ids, regs = tag(ex, gaz)
pd.DataFrame({"заголовок": ex, "МО": [[ref.mo_name.get(i, i) for i in x] for x in mo_ids], "регион": regs})""")
md("""**Как размечали «крупные события».** Сначала словарь ключевых слов, потом классификатор на эмбеддингах rubert-tiny2 (находит похожие по смыслу заголовки без слов из словаря), потом языковая модель Qwen3-1.7B проверяет кандидатов. Качество оценили на выборке из 200 заголовков, размеченных вручную:""")
code("""q = Path("data/annotation/labeling_quality.csv")
pd.read_csv(q) if q.exists() else print("оценка качества ещё не посчитана: python -m src.news.annotation score")""")
md("""**Помогают ли новости находить шоки?** Смотрим, как ведут себя траты муниципалитетов до и после крупного события — по сравнению со случайными МО:""")
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
Ставка одна для всей страны, поэтому различий между муниципалитетами она не объясняет. Мы проверили её там, где она может помочь, — в прогнозе роста трат России, на ряде с 2018 года:""")
code("""kr = pd.read_parquet("data/interim/key_rate_check.parquet")
kr.pivot_table(index="метод", columns="h", values="err").round(2).rename_axis("ошибка прогноза годового роста, п.п.")""")

md("""## 9. Итоги
Общая динамика страны объясняет почти всё движение трат в муниципалитетах, и на ней построен прогноз, который на 36–51 % точнее Prophet. Нейросети на коротких рядах заработали после backcasting, учёта формы года и дообучения. Шоки лучше всего находит детектор по остаткам прогноза — он почти не путает их с разовыми выбросами. Новости привязаны к данным по времени и месту и помогают объяснять найденные сдвиги.

Подробнее — в [отчёте](../report/REPORT.md) и [презентации](../presentation/presentation.pdf).""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "sber-venv", "display_name": "Python (сбер .venv)", "language": "python"}
nbf.write(nb, "notebooks/solution.ipynb")
print("записан notebooks/solution.ipynb, ячеек:", len(C))
