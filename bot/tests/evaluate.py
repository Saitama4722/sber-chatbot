"""Оценка качества распознавания и скорости ответа.

Запуск из корня проекта: python -m tests.evaluate
"""

from __future__ import annotations

import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sberbot import ChatBot  # noqa: E402
from tests.dataset import TEST_CASES  # noqa: E402


def main() -> int:
    bot = ChatBot(seed=1)
    bot.recognize("прогрев кэша лемматизатора")

    correct = 0
    latencies = []
    per_intent = defaultdict(lambda: {"total": 0, "hit": 0})
    errors = []

    for phrase, expected in TEST_CASES:
        started = time.perf_counter()
        result = bot.recognize(phrase)
        latencies.append((time.perf_counter() - started) * 1000)

        key = expected or "вне базы"
        per_intent[key]["total"] += 1
        if result.intent_id == expected:
            correct += 1
            per_intent[key]["hit"] += 1
        else:
            errors.append((phrase, expected, result.intent_id, result.score))

    total = len(TEST_CASES)
    accuracy = correct / total * 100

    print(f"Всего проверочных фраз: {total}")
    print(f"Верно распознано: {correct}")
    print(f"Точность распознавания: {accuracy:.1f}%")
    print()
    print("Точность по темам:")
    for key in sorted(per_intent):
        row = per_intent[key]
        print(f"  {key:<12} {row['hit']}/{row['total']} "
              f"({row['hit'] / row['total'] * 100:.0f}%)")
    print()
    print(f"Среднее время ответа: {statistics.mean(latencies):.2f} мс")
    print(f"Медиана: {statistics.median(latencies):.2f} мс")
    print(f"Минимум: {min(latencies):.2f} мс, максимум: {max(latencies):.2f} мс")

    if errors:
        print()
        print("Нераспознанные и ошибочные случаи:")
        for phrase, expected, got, score in errors:
            print(f"  {phrase!r}: ожидалось {expected}, получено {got} (оценка {score})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
