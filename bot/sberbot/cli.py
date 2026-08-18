"""Консольный интерфейс чат-бота."""

from __future__ import annotations

import argparse
import time

from .engine import ChatBot

BANNER = (
    "Справочный чат-бот ПАО Сбербанк\n"
    "Введите вопрос или команду: выход - завершить диалог, темы - список тем.\n"
)


def run(debug: bool = False) -> None:
    bot = ChatBot()
    print(BANNER)
    while True:
        try:
            message = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nБот: До свидания.")
            break
        if not message:
            continue
        if message.lower() in {"выход", "exit", "quit"}:
            print("Бот: До свидания.")
            break
        if message.lower() in {"темы", "help", "меню"}:
            print("Бот: " + ", ".join(intent.name for intent in bot.intents))
            continue
        started = time.perf_counter()
        result = bot.recognize(message)
        elapsed = (time.perf_counter() - started) * 1000
        print("Бот:", result.answer)
        if debug:
            print(f"  [намерение={result.intent_id} оценка={result.score} "
                  f"слова={result.matched} время={elapsed:.1f} мс]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Консольный режим чат-бота")
    parser.add_argument("--debug", action="store_true", help="показывать оценку распознавания")
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
