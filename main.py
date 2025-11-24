import schedule
import time
import threading
import traceback
from datetime import datetime, timedelta

from web.app import app, socketio
from models.smart_broker import SmartPortfolioBroker
from utils.logger import logger, init_socketio
import json

# =================================== КОНФИГУРАЦИЯ ===================================
with open("config/settings.json", "r", encoding="utf-8") as f:
    SETTINGS = json.load(f)

# Создаём единственный экземпляр брокера
broker = SmartPortfolioBroker(SETTINGS)


# =================================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===================================
#def is_trading_session() -> bool:
 #   """Текущий момент находится внутри торговой сессии МОЭКС (09:30–18:40 пн-пт)"""
 #   now = datetime.now()
 #   if now.weekday() >= 5:  # выходные
 #       return False
 #   start = SETTINGS["trading_session"]["start"]
 #   end = SETTINGS["trading_session"]["end"]
 #   return start <= now.strftime("%H:%M") <= end


#def is_pre_session() -> bool:
#    """За N минут до начала сессии (по настройке) — делаем предсессионный анализ"""
#    now = datetime.now()
#    session_start = datetime.strptime(SETTINGS["trading_session"]["start"], "%H:%M").replace(
#        year=now.year, month=now.month, day=now.day
#    )
#    return session_start - timedelta(minutes=SETTINGS["pre_session_analysis_minutes"]) <= now < session_start


def safe_cycle():
    try:
        broker.run_cycle()  # ← теперь внутри run_cycle() уже есть ВСЯ логика времени!
        # analyze_sentiment вызывается отдельно по schedule
    except Exception as e:
        logger.log('CRITICAL', 'Ошибка в основном цикле', traceback.format_exc())


# =================================== ПЛАНИРОВЩИК ===================================
schedule.every(1).minutes.do(lambda: safe_cycle())                    # основной торговый цикл
schedule.every(1).minutes.do(lambda: broker.analyze_sentiment())      # новости + sentiment каждую минуту
# schedule.every(10).minutes.do(lambda: train_model())                # можно включить, если будет дообучение руберта


# =================================== ЗАПУСК ВЕБ-СЕРВЕРА =================================
def start_web_server():
    socketio.run(
        app,
        host="0.0.0.0",
        port=SETTINGS["web_port"],
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=True
    )

threading.Thread(target=start_web_server, daemon=True).start()
time.sleep(2)  # даём серверу подняться
init_socketio(socketio)

logger.log('INFO', 'RUSSIAN PORTFOLIO BROKER v3.1 — ЗАПУСК УСПЕШЕН (ПРОДАКШЕН)')

# =================================== ОСНОВНОЙ БЕСКОНЕЧНЫЙ ЦИКЛ ===================================
try:
    while True:
        schedule.run_pending()
        time.sleep(1)
except KeyboardInterrupt:
    logger.log('INFO', 'Бот остановлен пользователем (Ctrl+C)')
except Exception as exc:
    logger.log('CRITICAL', 'Необрабатываемая ошибка', traceback.format_exc())
    # === СЕКРЕТНАЯ КОМАНДА ДЛЯ ОЧИСТКИ КЭША MOEX ===
    # Вставь это в конец main.py и запусти бота один раз
    if __name__ == "__main__":
        # ... весь твой код main.py ...

        # ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
        # Добавь это в самый конец файла main.py:
        import os

        if os.path.exists("CLEAR_MOEX_CACHE.txt"):
            try:
                os.remove("CLEAR_MOEX_CACHE.txt")
                broker.moex._securities_cache = None
                broker.moex._cache_time = 0
                logger.log('INFO', 'КЭШ MOEX ПРИНУДИТЕЛЬНО ОЧИЩЕН!')
            except:
                pass
        # ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←