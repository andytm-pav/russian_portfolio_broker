# debug_price.py
from fetchers.moex_fetcher import MoexFetcher
from utils.logger import logger


def debug_mrkv_price():
    logger.log('INFO', 'Диагностика цены MRKV...')
    moex = MoexFetcher()

    # Тест 1: Проверяем есть ли MRKV в списке бумаг
    securities = moex.get_all_securities()
    print(f"MRKV в списке бумаг: {'MRKV' in securities}")
    if 'MRKV' in securities:
        print(f"Данные MRKV: {securities['MRKV']}")

    # Тест 2: Пробуем получить цену разными способами
    print("\n=== ТЕСТ ПОЛУЧЕНИЯ ЦЕНЫ MRKV ===")

    # Способ 1: Наш метод
    price1 = moex.get_price("MRKV")
    print(f"Наш метод: {price1}")

    # Способ 2: Прямой запрос к API
    import requests
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })

    urls_to_try = [
        "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/gazp.json",
        "https://iss.moex.com/iss/engines/stock/markets/shares/securities/MRKV.json",
        "https://iss.moex.com/iss/engines/stock/markets/bonds/boards/TQCB/securities/MRKV.json",
        "https://iss.moex.com/iss/engines/stock/markets/index/securities/MRKV.json"
    ]

    for url in urls_to_try:
        try:
            params = {'iss.meta': 'off', 'marketdata.columns': 'LAST,OPEN,LOW,HIGH,CLOSE'}
            r = session.get(url, params=params, timeout=10)
            print(f"\nURL: {url}")
            print(f"Статус: {r.status_code}")
            if r.status_code == 200:
                data = r.json()
                market_data = data.get('marketdata', {}).get('data', [])
                if market_data:
                    print(f"Данные: {market_data[0]}")
                else:
                    print("Нет данных marketdata")
                    # Проверим другие секции
                    for key in data.keys():
                        if key != 'marketdata' and data[key].get('data'):
                            print(f"{key}: {data[key]['data'][:2]}")
        except Exception as e:
            print(f"Ошибка {url}: {e}")


if __name__ == "__main__":
    debug_mrkv_price()