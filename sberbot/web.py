"""Веб-интерфейс чат-бота на Flask.

Запуск: python -m sberbot.web, интерфейс доступен по адресу http://127.0.0.1:5000
"""

from __future__ import annotations

from flask import Flask, jsonify, render_template_string, request

from .engine import ChatBot

app = Flask(__name__)
bot = ChatBot()

PAGE = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Справочный чат-бот Сбербанка</title>
<style>
  :root { --green: #21a038; --bg: #f4f6f8; --text: #1c1c1c; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: Arial, Helvetica, sans-serif; background: var(--bg); color: var(--text); }
  header { background: var(--green); color: #fff; padding: 16px 24px; font-size: 18px; }
  main { max-width: 720px; margin: 24px auto; background: #fff; border-radius: 10px;
         box-shadow: 0 2px 10px rgba(0,0,0,.08); display: flex; flex-direction: column; height: 70vh; }
  #log { flex: 1; overflow-y: auto; padding: 20px; }
  .row { display: flex; margin-bottom: 12px; }
  .row.user { justify-content: flex-end; }
  .bubble { max-width: 78%; padding: 10px 14px; border-radius: 12px; line-height: 1.45; font-size: 15px; }
  .user .bubble { background: var(--green); color: #fff; border-bottom-right-radius: 3px; }
  .bot .bubble { background: #eef1f4; border-bottom-left-radius: 3px; }
  form { display: flex; gap: 10px; padding: 16px; border-top: 1px solid #e3e6ea; }
  input { flex: 1; padding: 12px; border: 1px solid #ccd2d8; border-radius: 8px; font-size: 15px; }
  button { padding: 12px 22px; border: 0; border-radius: 8px; background: var(--green);
           color: #fff; font-size: 15px; cursor: pointer; }
  .hint { padding: 0 20px 12px; color: #6b7785; font-size: 13px; }
</style>
</head>
<body>
<header>Справочный чат-бот Сбербанка</header>
<main>
  <div id="log">
    <div class="row bot"><div class="bubble">Здравствуйте! Задайте вопрос по картам, вкладам,
    кредитам, ипотеке, переводам, бонусам, курсам валют или работе отделений.</div></div>
  </div>
  <div class="hint">Примеры: какая ставка по ипотеке, как заблокировать карту, комиссия за перевод.</div>
  <form id="form">
    <input id="message" autocomplete="off" placeholder="Введите вопрос">
    <button type="submit">Отправить</button>
  </form>
</main>
<script>
const log = document.getElementById('log');
function add(text, who) {
  const row = document.createElement('div');
  row.className = 'row ' + who;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  row.appendChild(bubble);
  log.appendChild(row);
  log.scrollTop = log.scrollHeight;
}
document.getElementById('form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const field = document.getElementById('message');
  const text = field.value.trim();
  if (!text) return;
  add(text, 'user');
  field.value = '';
  const response = await fetch('/api/message', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: text})
  });
  const data = await response.json();
  add(data.answer, 'bot');
});
</script>
</body>
</html>"""


@app.get("/")
def index():
    return render_template_string(PAGE)


@app.post("/api/message")
def api_message():
    payload = request.get_json(silent=True) or {}
    result = bot.recognize(payload.get("message", ""))
    return jsonify(
        {
            "answer": result.answer,
            "intent": result.intent_id,
            "score": result.score,
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
