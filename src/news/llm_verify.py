"""Второй этап разметки: языковая модель Qwen3-1.7B проверяет кандидатов в «крупные события».

Кандидаты — заголовки с местом, отмеченные словарём или классификатором на rubert-tiny2.
Модель получает инструкцию, 7 примеров (few-shot) и заголовок и отвечает «Да»/«Нет»; вероятность
ответа — по логитам следующего токена (один проход сети без генерации), с контекстной калибровкой.
Режим «размышлений» Qwen3 отключён.
Запуск: python -m src.news.llm_verify
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SYSTEM = ("Ты аналитик региональной экономики России. По заголовку новости определи, сообщает ли он о КРУПНОМ событии "
          "на конкретной территории России, которое может заметно изменить траты её жителей: стихийное бедствие "
          "(паводок, наводнение, прорыв дамбы, крупный природный пожар, землетрясение), введённый режим ЧС, массовая "
          "эвакуация, масштабная авария с отключением света или тепла, закрытие или остановка крупного предприятия, "
          "массовые увольнения. НЕ считаются: рядовые происшествия (пожар в доме, ДТП, отдельные пострадавшие), "
          "совещания, учения, прогнозы и подготовка, события за пределами России, военные новости без последствий для жителей. "
          "Ответь одним словом: Да или Нет.")
SHOTS = [("В Орске прорвало дамбу, затоплены тысячи домов", "Да"),
         ("Пожар в частном доме в Твери, погиб мужчина", "Нет"),
         ("В Туве ввели режим ЧС после взрыва на ТЭЦ", "Да"),
         ("Губернатор провёл совещание по подготовке к паводку", "Нет"),
         ("В Челябинской области останавливают металлургический завод, сотни уволены", "Да"),
         ("На Филиппинах эвакуировали тысячи жителей из-за вулкана", "Нет"),
         ("Минобороны сообщило о перехвате снарядов РСЗО «Ураган»", "Нет")]
LABELS = ["Да", "Нет"]
NAMES = {"Да": "крупное событие", "Нет": "другое"}


def load_model(cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model_dir"])
    tok.padding_side = "left"
    dev = cfg["device"] if (cfg["device"] != "mps" or torch.backends.mps.is_available()) else "cpu"
    model = AutoModelForCausalLM.from_pretrained(cfg["model_dir"], dtype=torch.float16 if dev == "mps" else torch.float32).to(dev).eval()
    ids = [tok.encode(x, add_special_tokens=False)[0] for x in LABELS]
    return tok, model, ids, dev


def _messages(title):
    m = [{"role": "system", "content": SYSTEM}]
    for t, a in SHOTS:
        m += [{"role": "user", "content": f"Заголовок: «{t}»"}, {"role": "assistant", "content": a}]
    return m + [{"role": "user", "content": f"Заголовок: «{title}»"}]


def classify(titles, cfg, calibrate=True):
    """Вероятность «Да» с контекстной калибровкой (Zhao et al., 2021): перекос модели к одному из
    ответов измеряется на бессодержательном заголовке и делится из вероятностей."""
    import torch
    tok, model, ids, dev = load_model(cfg)

    def run(batch):
        texts = [tok.apply_chat_template(_messages(t), tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False) for t in batch]
        b = tok(texts, return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            logits = model(**b).logits[:, -1, :].float()
        return torch.softmax(logits[:, ids], dim=-1).cpu().numpy()

    prior = run(["N/A", "", "Новость"]).mean(0) if calibrate else np.ones(len(ids))
    probs = []
    for i in range(0, len(titles), cfg["batch_size"]):
        p = run(titles[i:i + cfg["batch_size"]]) / prior
        probs.append(p / p.sum(1, keepdims=True))
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
    cand["llm_p"] = P[:, 0]
    cand["llm_major"] = cand.llm_p >= 0.5
    cand["llm_label"] = np.where(cand.llm_major, "Да", "Нет")
    print("ответы модели:", cand.llm_label.map(NAMES).value_counts().to_dict())
    print("согласие со словарём: из словарных кандидатов модель признала крупными",
          f"{cand[cand.major_kw].llm_major.mean():.0%}; из новых (только rubert) — {cand[~cand.major_kw].llm_major.mean():.0%}")
    pd.set_option("display.max_colwidth", 100)
    for lab in ("Да",):
        print(f"\n[{NAMES[lab]}] примеры:")
        print(cand[cand.llm_label == lab].sample(min(12, (cand.llm_label == lab).sum()), random_state=1)[["source", "title", "llm_p"]].to_string(index=False))
    print("\n[словарь сказал «крупное», модель — «другое»] примеры:")
    x = cand[cand.major_kw & ~cand.llm_major]
    print(x.sample(min(12, len(x)), random_state=1)[["title", "llm_label"]].to_string(index=False))
    cand[["url", "llm_label", "llm_p", "llm_major"]].to_parquet("data/external/news/major_llm.parquet", index=False)
