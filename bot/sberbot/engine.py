"""Ядро чат-бота: распознавание намерения и подбор ответа."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .preprocess import (is_known, lemmatize, lemmatize_tokens, normalize,
                         normalized_text, raw_tokens)

DATA_FILE = Path(__file__).parent / "data" / "intents.json"

# пороги и веса подобраны по результатам прогонов на контрольной выборке
CONFIDENCE_THRESHOLD = 0.34
CLARIFY_THRESHOLD = 0.22
TYPO_RATIO = 0.82
KEYWORD_WEIGHT = 0.7
PATTERN_WEIGHT = 0.3


@dataclass
class Intent:
    """Тема обращения с набором ключевых слов, примеров и ответов."""

    id: str
    name: str
    keywords: List[str]
    patterns: List[str]
    responses: List[str]
    keyword_lemmas: set = field(default_factory=set)
    pattern_lemmas: List[set] = field(default_factory=list)
    pattern_texts: List[str] = field(default_factory=list)


@dataclass
class Recognition:
    """Результат разбора реплики пользователя."""

    intent_id: Optional[str]
    intent_name: Optional[str]
    score: float
    answer: str
    matched: List[str]
    alternatives: List[Tuple[str, float]]


class ChatBot:
    """Справочный бот на базе взвешенного сопоставления ключевых слов."""

    def __init__(self, data_file: Path = DATA_FILE, seed: Optional[int] = None) -> None:
        payload = json.loads(Path(data_file).read_text(encoding="utf-8"))
        self.meta: Dict = payload.get("meta", {})
        self.fallback: List[str] = payload["fallback"]
        self.clarification: str = payload["clarification"]
        self.intents: List[Intent] = [self._build(item) for item in payload["intents"]]
        self.weights: Dict[str, float] = self._build_weights()
        self.surface_vocabulary = sorted(
            {w.lower().replace("ё", "е") for i in self.intents for w in i.keywords}
            | set(self.weights)
        )
        self._random = random.Random(seed)

    # ---------- подготовка базы ----------

    @staticmethod
    def _build(item: Dict) -> Intent:
        intent = Intent(
            id=item["id"],
            name=item["name"],
            keywords=item["keywords"],
            patterns=item["patterns"],
            responses=item["responses"],
        )
        intent.keyword_lemmas = {
            lemmatize(w.lower()).replace("ё", "е") for w in item["keywords"]
        }
        for pattern in item["patterns"]:
            intent.pattern_lemmas.append(set(normalize(pattern)))
            intent.pattern_texts.append(normalized_text(pattern))
        return intent

    def _build_weights(self) -> Dict[str, float]:
        """Вес слова обратно пропорционален числу тем, где оно встречается.

        Слово, закрепленное за одной темой, получает максимальный вес,
        общеупотребительное - минимальный. Схема повторяет логику меры
        обратной документной частоты.
        """
        total = len(self.intents)
        document_frequency: Dict[str, int] = {}
        for intent in self.intents:
            for lemma in intent.keyword_lemmas:
                document_frequency[lemma] = document_frequency.get(lemma, 0) + 1
        return {
            lemma: 1.0 + math.log(total / count)
            for lemma, count in document_frequency.items()
        }

    # ---------- разбор реплики ----------

    def _correct(self, token: str) -> str:
        """Исправляет опечатку сопоставлением со словарем ключевых слов."""
        if token in self.surface_vocabulary:
            return token
        # словарное слово считается написанным верно и не правится
        if len(token) < 5 or is_known(token):
            return token
        best, best_ratio = token, 0.0
        for candidate in self.surface_vocabulary:
            if abs(len(candidate) - len(token)) > 2:
                continue
            ratio = SequenceMatcher(None, token, candidate).ratio()
            if ratio > best_ratio:
                best, best_ratio = candidate, ratio
        return best if best_ratio >= TYPO_RATIO else token

    def prepare(self, message: str) -> List[str]:
        """Токенизация, исправление опечаток, приведение к начальной форме."""
        corrected = [self._correct(token) for token in raw_tokens(message)]
        return lemmatize_tokens(corrected)

    def _keyword_score(self, intent: Intent, lemmas: List[str]) -> Tuple[float, List[str]]:
        tokens = set(lemmas)
        informative = {t for t in tokens if t in self.weights}
        if not informative:
            return 0.0, []
        matched = sorted(informative & intent.keyword_lemmas)
        if not matched:
            return 0.0, []
        matched_weight = sum(self.weights[t] for t in matched)
        total_weight = sum(self.weights[t] for t in informative)
        coverage = matched_weight / total_weight
        # доля покрытых ключевых слов темы удерживает короткие реплики
        density = min(len(matched) / max(len(intent.keyword_lemmas), 1) * 3, 1.0)
        return 0.8 * coverage + 0.2 * density, matched

    def _pattern_score(self, intent: Intent, lemmas: List[str], text: str) -> float:
        tokens = set(lemmas)
        best = 0.0
        for lemma_set, pattern_text in zip(intent.pattern_lemmas, intent.pattern_texts):
            union = tokens | lemma_set
            jaccard = len(tokens & lemma_set) / len(union) if union else 0.0
            ratio = SequenceMatcher(None, text, pattern_text).ratio()
            best = max(best, 0.65 * jaccard + 0.35 * ratio)
        return best

    def recognize(self, message: str) -> Recognition:
        lemmas = self.prepare(message)
        text = " ".join(lemmas)

        scored: List[Tuple[Intent, float, List[str]]] = []
        for intent in self.intents:
            keyword_score, matched = self._keyword_score(intent, lemmas)
            pattern_score = self._pattern_score(intent, lemmas, text)
            total = KEYWORD_WEIGHT * keyword_score + PATTERN_WEIGHT * pattern_score
            scored.append((intent, total, matched))
        scored.sort(key=lambda row: row[1], reverse=True)

        best, best_score, matched = scored[0]
        alternatives = [(item.name, round(value, 3)) for item, value, _ in scored[1:4]]

        if best_score >= CONFIDENCE_THRESHOLD:
            answer = self._random.choice(best.responses)
            return Recognition(best.id, best.name, round(best_score, 3), answer, matched, alternatives)

        if best_score >= CLARIFY_THRESHOLD:
            options = ", ".join(item.name.lower() for item, _, _ in scored[:3])
            answer = self.clarification.format(options=options)
            return Recognition(None, None, round(best_score, 3), answer, matched, alternatives)

        return Recognition(
            None, None, round(best_score, 3), self._random.choice(self.fallback), matched, alternatives
        )

    def reply(self, message: str) -> str:
        """Короткий доступ к тексту ответа."""
        return self.recognize(message).answer
