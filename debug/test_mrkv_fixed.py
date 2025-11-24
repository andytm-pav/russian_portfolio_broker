# test_mrkv_fixed.py
from fetchers.moex_fetcher_enhanced import MoexFetcher
from utils.logger import logger


def test_mrkv_fixed():
    logger.log('INFO', 'Тест MRKV с улучшенным фетчером...')
    moex = MoexFetcher()

    # Очистим кэш
    import os
    cache_file = "data/moex_securities.json"
    if os.path.exists(cache_file):
        os.remove(cache_file)

    # Тестируем
    securities = moex.get_all_securities()
    print(f"MRKV в списке: {'MRKV' in securities}")

    if 'MRKV' in securities:
        print(f"Данные MRKV: {securities['MRKV']}")

    # Тест цены
    price = moex.get_price("MRKV")
    print(f"Цена MRKV: {price}")

    # Тест других бумаг из портфеля
    portfolio_tickers = ["GAZP", "MTSS", "ROSN", "MRKV", "SBER"]
    print(f"\nТест цен портфеля:")
    for ticker in portfolio_tickers:
        price = moex.get_price(ticker)
        print(f"{ticker}: {price}")


if __name__ == "__main__":
    test_mrkv_fixed()