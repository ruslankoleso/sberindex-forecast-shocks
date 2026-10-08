#!/usr/bin/env bash
# Полное воспроизведение решения: данные → прогнозы → ансамбль → шоки → новости → графики → ноутбук.
# Использование:  ./run_all.sh            — всё, кроме тяжёлых шагов (дообучение нейросетей, сбор новостей)
#                 ./run_all.sh --heavy    — включая тяжёлые шаги (несколько часов на CPU)
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
HEAVY=${1:-}

echo "== 1. Данные";               $PY -m src.data.load;  $PY -m src.external.macro
echo "== 2. Тест на утечку";       $PY -m pytest -q tests

echo "== 3. Прогнозы (скользящее окно)"
$PY -m src.eval.run --models naive seasonal_naive seasonal_naive_growth snaive_natg \
    snaive_shrink_w25_k1 snaive_shrink_w50_k1 snaive_shrink_w75_k1 prophet lgbm lgbm_no_nat lgbm_catmo lgbm_resid \
    autoets autotheta nhits
if [ "$HEAVY" = "--heavy" ]; then
  $PY -m src.eval.run --models chronos_bolt_bc chronos2_bc timesfm_bc timesfm_bc_mix tirex_bc tirex_bc_mix \
      chronos2_ft chronos2_ft_mix timesfm_lora timesfm_lora_mix
fi
$PY -m src.eval.ensemble;  $PY -m src.eval.significance
$PY -m src.eval.national_growth_check;  $PY -m src.eval.key_rate_check;  $PY -m src.eval.history_check

echo "== 4. Шоки"
$PY -m src.changepoint.benchmark;  $PY -m src.changepoint.national_events;  $PY -m src.changepoint.panel_shocks
$PY -m src.eval.selection

echo "== 5. Новости"
if [ "$HEAVY" = "--heavy" ]; then
  $PY -m src.news.collect;  $PY -m src.news.collect_more
fi
if [ -f data/external/news/headlines.parquet ]; then
  $PY -m src.news.features;  $PY -m src.news.nlp
  [ -f data/external/news/major_llm.parquet ] || $PY -m src.news.llm_verify
  $PY -m src.news.events;  $PY -m src.changepoint.news_informed;  $PY -m src.news.annotation score
else
  echo "   нет сырых заголовков (не хранятся в git) — запустите ./run_all.sh --heavy для сбора; пропускаю шаги с новостями"
fi

echo "== 6. Графики и ноутбук"
$PY -m src.report.figures
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/solution.ipynb
echo "Готово."
