# debug_moex.py
import requests
import json
from utils.logger import logger


def debug_moex():
    logger.log('INFO', 'Диагностика MOEX API...')

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })

    # Тест 1: Базовый запрос
    url = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json"
    params = {
        'iss.meta': 'off',
        'securities.columns': 'SECID,SECNAME,PREVADMITTEDQUOTE,PREVLEGALCLOSEPRICE',
        'marketdata.columns': 'LAST,OPEN,VOLUME'
    }

    try:
        r = session.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()

        print("=== СТРУКТУРА ОТВЕТА ===")
        for key in data.keys():
            print(f"{key}: {len(data[key].get('data', [])) if isinstance(data[key], dict) else data[key]}")

        print("\n=== SECURITIES DATA (первые 3 строки) ===")
        securities_data = data.get('securities', {}).get('data', [])
        for i, row in enumerate(securities_data[:3]):
            print(f"{i + 1}. {row}")
            if row:
                for j, val in enumerate(row):
                    print(f"   [{j}]: {val} (тип: {type(val).__name__})")

        print("\n=== MARKETDATA (первые 3 строки) ===")
        market_data = data.get('marketdata', {}).get('data', [])
        for i, row in enumerate(market_data[:3]):
            print(f"{i + 1}. {row}")
            if row:
                for j, val in enumerate(row):
                    print(f"   [{j}]: {val} (тип: {type(val).__name__})")

        print("\n=== COLUMNS INFO ===")
        print("securities.columns:", data.get('securities', {}).get('columns', []))
        print("marketdata.columns:", data.get('marketdata', {}).get('columns', []))

    except Exception as e:
        logger.log('ERROR', f'Ошибка диагностики: {e}')


if __name__ == "__main__":
    debug_moex()