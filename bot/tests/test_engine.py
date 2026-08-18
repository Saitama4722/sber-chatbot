"""Модульные проверки ядра чат-бота."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sberbot import ChatBot  # noqa: E402

bot = ChatBot(seed=7)


@pytest.mark.parametrize(
    "message,intent_id",
    [
        ("Здравствуйте", "greeting"),
        ("Какая ставка по ипотеке", "mortgage"),
        ("Хочу открыть вклад", "deposit"),
        ("Как заблокировать карту", "card_block"),
        ("Комиссия за перевод в другой банк", "transfer"),
        ("Где ближайшее отделение", "branch"),
        ("Курс доллара сегодня", "exchange"),
        ("Как открыть расчетный счет для ИП", "business"),
    ],
)
def test_known_intents(message, intent_id):
    assert bot.recognize(message).intent_id == intent_id


@pytest.mark.parametrize("message", ["привт", "спасбо", "заблакировать карту"])
def test_typos_are_corrected(message):
    assert bot.recognize(message).intent_id is not None


@pytest.mark.parametrize("message", ["Как приготовить борщ", "Расскажи анекдот"])
def test_out_of_scope(message):
    assert bot.recognize(message).intent_id is None


def test_answer_is_not_empty():
    assert len(bot.reply("Как оформить кредитную карту")) > 40


def test_empty_message_goes_to_fallback():
    result = bot.recognize("   ")
    assert result.intent_id is None
