# web/app.py
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
from utils.portfolio_manager import PortfolioManager
from utils.logger import logger
import json
from datetime import datetime

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

portfolio = PortfolioManager()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def status():
    with open("config/settings.json") as f:
        settings = json.load(f)
    now = datetime.now()
    start = datetime.strptime(settings["trading_session"]["start"], "%H:%M").replace(year=now.year, month=now.month, day=now.day)
    diff = start - now
    mins = int(diff.total_seconds() // 60) if diff.total_seconds() > 0 else 0
    return jsonify({
        "session": "Открыта" if (datetime.now().strftime('%H%M') >= '1000') else "Закрыта",
        "countdown": f"{mins} мин" if mins > 0 else "Сессия идёт",
        "step": "Торговля" if (datetime.now().strftime('%H%M') >= '1000') else "Ожидание",
        "portfolio": portfolio.to_dict(),
        "cash": portfolio.cash
    })

@socketio.on('connect')
def handle_connect():
    logger.log('INFO', 'Клиент подключён к веб-интерфейсу')