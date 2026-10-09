"""Где брать веса предобученных моделей.

В конфигах указаны идентификаторы Hugging Face (например, google/timesfm-2.5-200m-pytorch).
Если веса уже скачаны в models/<имя репозитория>, берём их оттуда, иначе модель
скачивается с Hugging Face при первом запуске.
"""
from pathlib import Path


def resolve(model_id: str) -> str:
    local = Path("models") / model_id.split("/")[-1]
    if not local.exists():
        return model_id
    ckpt = local / "model.ckpt"          # TiRex хранится одним файлом
    return str(ckpt if ckpt.exists() and not (local / "config.json").exists() else local)
