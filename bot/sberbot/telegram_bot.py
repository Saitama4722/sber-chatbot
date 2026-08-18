"""Интеграция чат-бота с мессенджером Telegram.

Токен передается через переменную окружения TELEGRAM_TOKEN.
Запуск: python -m sberbot.telegram_bot
"""

from __future__ import annotations

import logging
import os

try:
    import telebot
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Установите зависимость: pip install pyTelegramBotAPI") from exc

from .engine import ChatBot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

TOKEN = os.getenv("TELEGRAM_TOKEN", "")
if not TOKEN:
    raise SystemExit("Не задана переменная окружения TELEGRAM_TOKEN")

bot_engine = ChatBot()
telegram_bot = telebot.TeleBot(TOKEN)


@telegram_bot.message_handler(commands=["start"])
def handle_start(message) -> None:
    telegram_bot.reply_to(
        message,
        "Здравствуйте! Это справочный бот Сбербанка. Задайте вопрос по картам, "
        "вкладам, кредитам, ипотеке, переводам, бонусам или работе отделений.",
    )


@telegram_bot.message_handler(commands=["topics"])
def handle_topics(message) -> None:
    topics = ", ".join(intent.name for intent in bot_engine.intents)
    telegram_bot.reply_to(message, "Доступные темы: " + topics)


@telegram_bot.message_handler(func=lambda message: True)
def handle_message(message) -> None:
    result = bot_engine.recognize(message.text or "")
    logging.info("intent=%s score=%s", result.intent_id, result.score)
    telegram_bot.reply_to(message, result.answer)


if __name__ == "__main__":
    logging.info("Бот запущен")
    telegram_bot.infinity_polling(skip_pending=True)
