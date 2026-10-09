"""Подготовка к запуску ноутбука.

1. Собирает панель МО из данных организаторов (data/interim/panel.parquet).
2. Раскладывает готовые результаты тяжёлых шагов из results/ в data/interim/:
   прогнозы нейросетей и моделей, которые долго считаются, и итоги разметки новостей.
   Уже посчитанные файлы не перезаписываются — если вы пересчитали модель сами, возьмётся ваш.

    python -m src.prepare          — подготовить
    python -m src.prepare save     — обновить results/ из data/interim/ (после полного пересчёта)
"""
import shutil
import sys
from pathlib import Path

RESULTS = Path("results")
# модели, которые в ноутбуке не пересчитываются: дообучение нейросетей, zero-shot модели и N-HiTS
HEAVY = ["chronos2", "chronos2_bc", "chronos2_ft", "chronos2_ft_mix", "chronos_bolt_bc",
         "timesfm", "timesfm_bc", "timesfm_bc_mix", "timesfm_lora", "timesfm_lora_mix", "timesfm_xreg",
         "tirex_bc", "tirex_bc_mix", "autoets", "autotheta", "nhits"]
NEWS = {"major_events.pkl": "data/interim", "news_events.pkl": "data/interim", "major_llm_v1.parquet": "data/interim",
        "news_informed.pkl": "data/interim/changepoint", "recheck_examples.parquet": "data/interim"}


def prepare():
    import subprocess
    if not Path("data/interim/panel.parquet").exists():
        subprocess.run([sys.executable, "-m", "src.data.load"], check=True)
    fc = Path("data/interim/forecasts"); fc.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in (RESULTS / "forecasts").glob("*.parquet"):
        if not (fc / f.name).exists():
            shutil.copy(f, fc / f.name); n += 1
    for name, dst in NEWS.items():
        Path(dst).mkdir(parents=True, exist_ok=True)
        if (RESULTS / "news" / name).exists() and not (Path(dst) / name).exists():
            shutil.copy(RESULTS / "news" / name, Path(dst) / name); n += 1
    print(f"Панель МО готова; из results/ разложено файлов: {n}")


def save():
    (RESULTS / "forecasts").mkdir(parents=True, exist_ok=True); (RESULTS / "news").mkdir(parents=True, exist_ok=True)
    for m in HEAVY:
        shutil.copy(f"data/interim/forecasts/{m}.parquet", RESULTS / "forecasts" / f"{m}.parquet")
    for name, src in NEWS.items():
        if (Path(src) / name).exists():
            shutil.copy(Path(src) / name, RESULTS / "news" / name)


if __name__ == "__main__":
    save() if sys.argv[1:] == ["save"] else prepare()
