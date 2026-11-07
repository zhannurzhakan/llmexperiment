"""Опыт 1: одна и та же фраза на русском и английском через токенайзеры.

Запуск:  .venv/bin/python experiments/01_ru_vs_en_tokens.py
Результат печатается в консоль и сохраняется в experiments/01_results.md
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import tiktoken  # noqa: E402

PHRASES = [
    ("Приветствие", "Привет, мир!", "Hello, world!"),
    ("Простая фраза", "Я люблю программировать.", "I love programming."),
    (
        "Длинная фраза",
        "Большие языковые модели обрабатывают текст по кусочкам, которые называются токенами.",
        "Large language models process text in pieces called tokens.",
    ),
]

# cl100k_base — GPT-4/3.5, o200k_base — GPT-4o
ENCODINGS = ["cl100k_base", "o200k_base"]


def piece(b):
    """Токен как текст; если он обрывает UTF-8 символ посередине — показать байты."""
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return "".join(f"\\x{x:02x}" for x in b)


def show(enc, text):
    ids = enc.encode(text)
    pieces = [piece(enc.decode_single_token_bytes(i)) for i in ids]
    return ids, pieces


def main():
    out = ["# Опыт 1: русский vs английский в токенайзере\n"]
    for enc_name in ENCODINGS:
        enc = tiktoken.get_encoding(enc_name)
        out.append(f"\n## Токенайзер `{enc_name}`\n")
        out.append("| Фраза | Язык | Символов | Токенов | Символов/токен |")
        out.append("|---|---|---|---|---|")
        details = []
        for title, ru, en in PHRASES:
            for lang, text in (("RU", ru), ("EN", en)):
                ids, pieces = show(enc, text)
                out.append(
                    f"| {title} | {lang} | {len(text)} | {len(ids)} | {len(text)/len(ids):.2f} |"
                )
                details.append((title, lang, text, pieces))
            n_ru = len(enc.encode(ru))
            n_en = len(enc.encode(en))
            details.append((title, "ratio", f"RU/EN = {n_ru/n_en:.2f}x", None))
        out.append("\n### Разбиение на токены\n")
        for title, lang, text, pieces in details:
            if pieces is None:
                out.append(f"\n**{title}: {text}**\n")
                continue
            shown = " ".join(f"[{p}]" for p in pieces)
            out.append(f"- **{title} / {lang}** — `{text}`  \n  {shown}")
    report = "\n".join(out) + "\n"
    print(report)
    Path(__file__).with_name("01_results.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
