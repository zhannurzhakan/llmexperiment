"""Опыт 3: большой набор тестов на свежих токенайзерах и токенайзерах Anthropic.

Токенайзеры:
  OpenAI (tiktoken):  cl100k_base (GPT-4), o200k_base (GPT-4o), o200k_harmony (gpt-oss)
  Открытые (Hugging Face tokenizer.json): Qwen3.8, DeepSeek-V4.1, Mistral-Large-4,
                                          Gemma-4, Llama-4, GLM-5.3
  Anthropic: API /v1/messages/count_tokens — нужен ключ в переменной ANTHROPIC_API_KEY.
             Без ключа этот блок пропускается (токенайзер Claude не публикуется).

Запуск:
  .venv/bin/python experiments/03_tokenizers.py
  ANTHROPIC_API_KEY=... .venv/bin/python experiments/03_tokenizers.py   # + Claude

Результаты: experiments/03_results.md (сводка) и experiments/03_counts.json (сырые числа).
"""
import json
import os
import sys
import time
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
import tiktoken  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from corpus import (KK_TO_LAT, MULTILANG, PARALLEL, RU_TO_LAT, SPECIAL,  # noqa: E402
                    translit)

HERE = Path(__file__).parent

TIKTOKEN = {
    "gpt4:cl100k": "cl100k_base",
    "gpt4o:o200k": "o200k_base",
    "gpt-oss:harmony": "o200k_harmony",
}
HF = {
    "Qwen3.8": "Qwen/Qwen3.8-27B",
    "DeepSeek-V4.1": "deepseek-ai/DeepSeek-V4.1-Flash",
    "Mistral-Large-4": "mistralai/Mistral-Large-4-1T-A52B",
    "Gemma-4": "google/gemma-4-12B-it",
    "Llama-4": "unsloth/Llama-4-Scout-17B-16E-Instruct",  # зеркало: оригинал gated
    "GLM-5.3": "zai-org/GLM-5.3",
}
ANTHROPIC_MODELS = ["claude-sonnet-5-5", "claude-opus-5-5"]


# --- счётчики -----------------------------------------------------------------
def build_counters():
    counters = {}
    for name, enc_name in TIKTOKEN.items():
        enc = tiktoken.get_encoding(enc_name)
        counters[name] = lambda t, e=enc: len(e.encode(t, disallowed_special=()))
    for name, repo in HF.items():
        tok = Tokenizer.from_pretrained(repo)
        counters[name] = lambda t, tk=tok: len(tk.encode(t, add_special_tokens=False).ids)
    return counters


def anthropic_counters(key):
    """count_tokens включает служебные токены сообщения; вычитаем накладные расходы.

    Накладные расходы = count("a") - 1 (считаем, что "a" — ровно один токен).
    """
    counters = {}
    cache_path = HERE / "03_anthropic_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    def raw(model, text):
        k = f"{model}|{text}"
        if k not in cache:
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages/count_tokens",
                data=json.dumps({"model": model,
                                 "messages": [{"role": "user", "content": text}]}).encode(),
                headers={"content-type": "application/json",
                         "anthropic-version": "2023-06-01", "x-api-key": key},
            )
            cache[k] = json.load(urllib.request.urlopen(req, timeout=30))["input_tokens"]
            cache_path.write_text(json.dumps(cache, ensure_ascii=False))
            time.sleep(0.2)
        return cache[k]

    for model in ANTHROPIC_MODELS:
        overhead = raw(model, "a") - 1
        counters[f"Claude:{model.replace('claude-', '')}"] = (
            lambda t, m=model, o=overhead: raw(m, t) - o
        )
    return counters


# --- построение таблиц --------------------------------------------------------
def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    counters = build_counters()
    note_anthropic = ""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        try:
            counters.update(anthropic_counters(key))
        except Exception as e:  # noqa: BLE001
            note_anthropic = f"Блок Anthropic не выполнен: {type(e).__name__}: {e}"
    else:
        note_anthropic = ("Токенайзеры Anthropic **не проверены**: токенайзер Claude не публикуется, "
                          "а API подсчёта токенов требует ключ (`ANTHROPIC_API_KEY`), "
                          "которого в этой среде нет (запрос без ключа → HTTP 401).")
    names = list(counters)
    raw = {}  # category -> test_id -> {text, counts{tokenizer: n}}

    # Группа A
    a_variants = {}
    for tid, en, ru, kk in PARALLEL:
        v = {"EN": en, "RU": ru, "RU-lat": translit(ru, RU_TO_LAT),
             "KK": kk, "KK-lat": translit(kk, KK_TO_LAT)}
        a_variants[tid] = v
        for lang, text in v.items():
            raw[f"A:{tid}:{lang}"] = {"text": text, "counts": {n: counters[n](text) for n in names}}

    # Группа B
    for lang, text in MULTILANG:
        raw[f"B:{lang}"] = {"text": text, "counts": {n: counters[n](text) for n in names}}

    # Группа C
    for cat, tid, text in SPECIAL:
        raw[f"C:{cat}:{tid}"] = {"text": text, "counts": {n: counters[n](text) for n in names}}

    (HERE / "03_counts.json").write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")

    out = ["# Опыт 3: большой набор тестов на свежих токенайзерах\n"]
    out.append(f"Токенайзеров: {len(names)} — {', '.join(names)}.\n")
    if note_anthropic:
        out.append(f"> {note_anthropic}\n")
    out.append("Сырые числа — в `03_counts.json`. `×EN` — отношение числа токенов к английскому "
               "варианту в **том же** токенайзере.\n")

    # --- A: сводка ×EN усреднённо по 8 фразам
    out.append("\n## A. Параллельные фразы: во сколько раз дороже английского\n")
    out.append(f"Среднее `×EN` по {len(PARALLEL)} фразам (токены суммируются по всем фразам и делятся "
               "на сумму английских).\n")
    langs = ["RU", "RU-lat", "KK", "KK-lat"]
    rows = []
    for n in names:
        en_total = sum(raw[f"A:{t}:EN"]["counts"][n] for t in a_variants)
        row = [n, en_total]
        for lang in langs:
            tot = sum(raw[f"A:{t}:{lang}"]["counts"][n] for t in a_variants)
            row.append(f"{tot/en_total:.2f}×")
        rows.append(row)
    out.append(md_table(["Токенайзер", "Токенов EN (сумма)"] + langs, rows))

    # A: подробно по фразам
    out.append("\n### Подробно: токены по каждой фразе\n")
    for tid, v in a_variants.items():
        out.append(f"\n**{tid}** — EN: `{v['EN']}`\n")
        rows = []
        for n in names:
            rows.append([n] + [raw[f"A:{tid}:{lang}"]["counts"][n] for lang in ["EN", "RU", "RU-lat", "KK", "KK-lat"]])
        out.append(md_table(["Токенайзер", "EN", "RU", "RU-lat", "KK", "KK-lat"], rows))

    # --- B
    out.append("\n## B. Одно предложение на 10 языках (токенов)\n")
    out.append("Предложение: «Large language models process text in pieces called tokens.» и его переводы.\n")
    rows = []
    for lang, text in MULTILANG:
        rows.append([lang, len(text)] + [raw[f"B:{lang}"]["counts"][n] for n in names])
    out.append(md_table(["Язык", "Символов"] + names, rows))
    out.append("\n`×EN` по языкам (среднее по токенайзерам):\n")
    rows = []
    for lang, _ in MULTILANG:
        r = []
        for n in names:
            r.append(raw[f"B:{lang}"]["counts"][n] / raw["B:EN"]["counts"][n])
        rows.append([lang, f"{sum(r)/len(r):.2f}×", f"{min(r):.2f}×", f"{max(r):.2f}×"])
    out.append(md_table(["Язык", "среднее", "мин", "макс"], rows))

    # --- C
    out.append("\n## C. Числа, код, спецсимволы, редкие слова (токенов)\n")
    rows = []
    for cat, tid, text in SPECIAL:
        shown = text.replace("\n", "⏎").replace("\t", "→").replace("|", "\\|")
        rows.append([cat, f"`{shown}`", len(text)] + [raw[f"C:{cat}:{tid}"]["counts"][n] for n in names])
    out.append(md_table(["Тип", "Текст", "Симв."] + names, rows))

    report = "\n".join(out) + "\n"
    (HERE / "03_results.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
