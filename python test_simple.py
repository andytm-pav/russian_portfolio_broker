# test_simple.py
from fetchers.moex_fetcher_simple import MoexFetcher
from utils.logger import logger


def test_simple():
    logger.log('INFO', 'Тест упрощенной версии...')
    moex = MoexFetcher()

    # Очистим кэш
    import os
    cache_file = "data/moex_securities.json"
    if os.path.exists(cache_file):
        os.remove(cache_file)
        logger.log('INFO', 'Кэш очищен')

    # Тестируем
    securities = moex.get_all_securities()
    print(f"Получено бумаг: {len(securities)}")

    # Покажем первые 10 бумаг
    count = 0
    for ticker, data in securities.items():
        if count >= 10:
            break
        print(
            f"{ticker}: объем={data.get('volume', 0):,.0f}, тек.цена={data.get('current_price', 0):.2f}, имя={data.get('name', '')}")
        count += 1

    # Тест цен
    if securities:
        test_tickers = list(securities.keys())[:3]
        print(f"\nТест цен для первых 3 бумаг:")
        for ticker in test_tickers:
            price = moex.get_price(ticker)
            print(f"{ticker}: {price}")


if __name__ == "__main__":
    test_simple()