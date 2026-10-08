"""Второй этап разметки: языковая модель Qwen3-1.7B проверяет кандидатов в «крупные события».

Кандидаты — заголовки с местом, отмеченные словарём или классификатором на rubert-tiny2.
Модель получает заголовок и выбирает один вариант (A–D); вероятность варианта считается
по логитам следующего токена — один проход сети без генерации текста (быстро на CPU/MPS).
Режим «размышлений» Qwen3 отключён.
Запуск: python -m src.news.llm_verify
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROMPT = (
    "Ты аналитик региональной экономики. Определи, о чём заголовок новости.\n"
    "A — крупное стихийное бедствие или ЧС на территории (паводок, наводнение, прорыв дамбы, крупный пожар, "
    "землетрясение, режим ЧС, массовая эвакуация, масштабное отключение электричества или тепла).\n"
    "B — закрытие, остановка или банкротство предприятия, массовые увольнения.\n"
    "C — открытие или запуск крупного предприятия, крупные инвестиции в территорию.\n"
    "D — другое (рядовое происшествие, политика, криминал, учения, отдельные люди, новости других стран).\n"
    "Заголовок: «{title}»\nОтветь одной буквой."
)
LABELS = ["A", "B", "C", "D"]
NAMES = {"A": "крупная ЧС", "B": "закрытие производства", "C": "открытие производства", "D": "другое"}


def load_model(cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model_dir"])
    tok.padding_side = "left"
    dev = cfg["device"] if (cfg["device"] != "mps" or torch.backends.mps.is_available()) else "cpu"
    model = AutoModelForCausalLM.from_pretrained(cfg["model_dir"], torch_dtype=torch.float16 if dev == "mps" else torch.float32).to(dev).eval()
    ids = [tok.encode(x, add_special_tokens=False)[0] for x in LABELS]
    return tok, model, ids, dev


def classify(titles, cfg):
    import torch
    tok, model, ids, dev = load_model(cfg)
    probs = []
    for i in range(0, len(titles), cfg["batch_size"]):
        texts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(title=t)}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False) for t in titles[i:i + cfg["batch_size"]]]
        b = tok(texts, return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            logits = model(**b).logits[:, -1, :].float()
        probs.append(torch.softmax(logits[:, ids], dim=-1).cpu().numpy())
        if i % (cfg["batch_size"] * 100) == 0:
            print("  проверено", i, "из", len(titles), flush=True)
    return np.vstack(probs)


def candidates(ncfg):
    d = pd.read_parquet("data/external/news/headlines_tagged.parquet")
    m = pd.read_parquet("data/external/news/major_nlp.parquet")
    d = d.merge(m, on="url", how="left")
    c = ncfg["llm"]
    sel = (d.major_kw.fillna(False) | (d.p_major.fillna(0) >= c["candidate_p"]))
    if c["require_place"]:
        sel &= (d.mo_ids.str.len() > 0) | (d.regions.str.len() > 0)
    return d[sel].copy()


if __name__ == "__main__":
    ncfg = yaml.safe_load(Path("configs/news.yaml").read_text(encoding="utf-8"))
    cand = candidates(ncfg)
    print("кандидатов:", len(cand), "| из них по словарю:", int(cand.major_kw.sum()))
    P = classify(cand.title.tolist(), ncfg["llm"])
    cand["llm_label"] = [LABELS[i] for i in P.argmax(1)]
    cand["llm_p"] = P.max(1)
    cand["llm_major"] = cand.llm_label.isin(["A", "B"])
    print("ответы модели:", cand.llm_label.map(NAMES).value_counts().to_dict())
    print("согласие со словарём: из словарных кандидатов модель признала крупными",
          f"{cand[cand.major_kw].llm_major.mean():.0%}; из новых (только rubert) — {cand[~cand.major_kw].llm_major.mean():.0%}")
    pd.set_option("display.max_colwidth", 100)
    for lab in ("A", "B"):
        print(f"\n[{NAMES[lab]}] примеры:")
        print(cand[cand.llm_label == lab].sample(min(12, (cand.llm_label == lab).sum()), random_state=1)[["source", "title", "llm_p"]].to_string(index=False))
    print("\n[словарь сказал «крупное», модель — «другое»] примеры:")
    x = cand[cand.major_kw & ~cand.llm_major]
    print(x.sample(min(12, len(x)), random_state=1)[["title", "llm_label"]].to_string(index=False))
    cand[["url", "llm_label", "llm_p", "llm_major"]].to_parquet("data/external/news/major_llm.parquet", index=False)
