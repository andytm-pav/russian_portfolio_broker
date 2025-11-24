# utils/logger.py
from datetime import datetime
import traceback

_socketio = None


def init_socketio(sio):
    global _socketio
    _socketio = sio


def log(level, msg, data=None):
    try:
        ts = datetime.now().strftime('%H:%M:%S')
        line = f"[{ts}] {msg}"
        if data:
            line += f" | {data}"
        print(line)

        if _socketio:
            # Исправленная строка - убираем broadcast
            _socketio.emit('log', {"level": level, "msg": msg, "data": data})
    except Exception as e:
        print(f"LOGGER ERROR: {e}")


class ProductionLogger:
    def log(self, level, msg, data=None):
        log(level, msg, data)

    def error_with_traceback(self, msg, e):
        self.log('ERROR', msg)
        self.log('DEBUG', f'Ошибка: {str(e)}')


logger = ProductionLogger()