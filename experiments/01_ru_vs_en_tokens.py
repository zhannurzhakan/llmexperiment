"""Опыт 1: одна и та же фраза на разных языках/письменностях через токенайзеры.

Варианты текста:
  EN        — английский (база для сравнения)
  RU        — русский, кириллица
  RU-lat    — русский латиницей (транслитерация по схеме RU_TO_LAT)
  KK        — казахский, кириллица
  KK-lat    — казахский латиницей (транслитерация по схеме KK_TO_LAT)

Запуск:  .venv/bin/python experiments/01_ru_vs_en_tokens.py
Результат печатается в консоль и сохраняется в experiments/01_results.md
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import tiktoken  # noqa: E402

# Схемы транслитерации. Строчные буквы; заглавные получаются автоматически.
RU_TO_LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}

# Условная схема казахской латиницы (диакритика в духе проектов 2021-2025 гг.)
KK_TO_LAT = {
    "а": "a", "ә": "ä", "б": "b", "в": "v", "г": "g", "ғ": "ğ", "д": "d",
    "е": "e", "ё": "yo", "ж": "j", "з": "z", "и": "i", "й": "y", "к": "k",
    "қ": "q", "л": "l", "м": "m", "н": "n", "ң": "ñ", "о": "o", "ө": "ö",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ұ": "ū", "ү": "ü",
    "ф": "f", "х": "h", "һ": "h", "ц": "ts", "ч": "ç", "ш": "ş", "щ": "şç",
    "ъ": "", "ы": "ı", "і": "i", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


def translit(text, table):
    out = []
    for ch in text:
        low = ch.lower()
        if low in table:
            lat = table[low]
            out.append(lat.capitalize() if ch != low else lat)
        else:
            out.append(ch)
    return "".join(out)


# (название, EN, RU, KK)
PHRASES = [
    ("Приветствие", "Hello, world!", "Привет, мир!", "Сәлем, әлем!"),
    (
        "Простая фраза",
        "I love programming.",
        "Я люблю программировать.",
        "Мен бағдарламалауды жақсы көремін.",
    ),
    (
        "Длинная фраза",
        "Large language models process text in pieces called tokens.",
        "Большие языковые модели обрабатывают текст по кусочкам, которые называются токенами.",
        "Үлкен тілдік модельдер мәтінді токендер деп аталатын бөліктермен өңдейді.",
    ),
]

LANGS = ["EN", "RU", "RU-lat", "KK", "KK-lat"]

# cl100k_base — GPT-4/3.5, o200k_base — GPT-4o
ENCODINGS = ["cl100k_base", "o200k_base"]


def variants(en, ru, kk):
    return {
        "EN": en,
        "RU": ru,
        "RU-lat": translit(ru, RU_TO_LAT),
        "KK": kk,
        "KK-lat": translit(kk, KK_TO_LAT),
    }


def piece(b):
    """Токен как текст; если он обрывает UTF-8 символ посередине — показать байты."""
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return "".join(f"\\x{x:02x}" for x in b)


def tokens(enc, text):
    ids = enc.encode(text)
    return [piece(enc.decode_single_token_bytes(i)) for i in ids]


def main():
    out = ["# Опыт 1: один текст на разных языках и письменностях\n"]
    out.append(
        "Варианты: EN — английский; RU — русский; RU-lat — русский латиницей; "
        "KK — казахский; KK-lat — казахский латиницей. "
        "Латиница получена транслитерацией (схемы в скрипте). "
        "`×EN` — во сколько раз токенов больше, чем в английском варианте.\n"
    )
    for enc_name in ENCODINGS:
        enc = tiktoken.get_encoding(enc_name)
        out.append(f"\n## Токенайзер `{enc_name}`\n")
        out.append("| Фраза | Вариант | Символов | Байт UTF-8 | Токенов | Симв./токен | ×EN |")
        out.append("|---|---|---|---|---|---|---|")
        details = []
        for title, en, ru, kk in PHRASES:
            vs = variants(en, ru, kk)
            n_en = len(enc.encode(vs["EN"]))
            for lang in LANGS:
                text = vs[lang]
                n = len(enc.encode(text))
                out.append(
                    f"| {title} | {lang} | {len(text)} | {len(text.encode('utf-8'))} | "
                    f"{n} | {len(text)/n:.2f} | {n/n_en:.2f} |"
                )
                details.append((title, lang, text, tokens(enc, text)))
        out.append("\n### Разбиение на токены\n")
        last = None
        for title, lang, text, pieces in details:
            if title != last:
                out.append(f"\n**{title}**\n")
                last = title
            shown = " ".join(f"[{p}]" for p in pieces)
            out.append(f"- **{lang}** — `{text}`  \n  {shown}")
    report = "\n".join(out) + "\n"
    print(report)
    Path(__file__).with_name("01_results.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
