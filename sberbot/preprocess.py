"""Предобработка пользовательских реплик.

Модуль отвечает за приведение произвольной фразы к нормализованному
списку лемм: токенизация средствами NLTK, отсев служебных слов,
приведение к начальной форме морфологическим анализатором pymorphy3.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pymorphy3 import MorphAnalyzer

_TOKEN_RE = re.compile(r"[а-яa-zё0-9]+", re.IGNORECASE)

# служебные слова, которые не несут смысла при определении темы вопроса
_EXTRA_STOPWORDS = {
    "пожалуйста", "скажите", "подскажите", "хотеть",
    "мочь", "это", "который", "такой",
}

# слова, без которых теряется тема обращения, поэтому из стоп-листа исключаются
_KEEP_WORDS = {"нет", "не", "куда", "где", "как", "какой", "сколько"}


def _ensure_nltk_data() -> None:
    """Загружает недостающие ресурсы NLTK при первом запуске."""
    for resource, path in (
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("stopwords", "corpora/stopwords"),
    ):
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


_ensure_nltk_data()
_MORPH = MorphAnalyzer()
_STOPWORDS = (set(stopwords.words("russian")) | _EXTRA_STOPWORDS) - _KEEP_WORDS


@lru_cache(maxsize=8192)
def lemmatize(word: str) -> str:
    """Возвращает начальную форму слова."""
    return _MORPH.parse(word)[0].normal_form


def tokenize(text: str) -> List[str]:
    """Разбивает строку на слова, отбрасывая знаки препинания."""
    raw = word_tokenize(text.lower(), language="russian")
    return [token for token in raw if _TOKEN_RE.fullmatch(token)]


def is_known(word: str) -> bool:
    """Проверяет, есть ли слово в морфологическом словаре."""
    return _MORPH.word_is_known(word)


def raw_tokens(text: str) -> List[str]:
    """Значимые слова исходной фразы без приведения к начальной форме."""
    result = []
    for token in tokenize(text):
        token = token.replace("ё", "е")
        if token in _STOPWORDS or len(token) < 2:
            continue
        result.append(token)
    return result


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """Приводит список слов к начальным формам с отсевом стоп-слов."""
    lemmas = []
    for token in tokens:
        lemma = lemmatize(token).replace("ё", "е")
        if lemma in _STOPWORDS or len(lemma) < 2:
            continue
        lemmas.append(lemma)
    return lemmas


def normalize(text: str) -> List[str]:
    """Полный цикл предобработки: токены без стоп-слов в начальной форме."""
    return lemmatize_tokens(raw_tokens(text))


def normalized_text(text: str) -> str:
    """Нормализованная строка для оценки посимвольного сходства."""
    return " ".join(normalize(text))
