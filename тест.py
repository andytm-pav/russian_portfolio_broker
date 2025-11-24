# test_full_system.py
from models.smart_broker import SmartPortfolioBroker
from utils.logger import logger
import json


def test_full_system():
    logger.log('INFO', 'ТЕСТ ПОЛНОЙ СИСТЕМЫ...')

    # Загружаем настройки
    with open("config/settings.json") as f:
        settings = json.load(f)

    # Создаем брокера
    broker = SmartPortfolioBroker(settings)

    # Тест 1: Предсессионный анализ
    logger.log('INFO', '=== ТЕСТ 1: Предсессионный анализ ===')
    broker.pre_session_analysis()

    # Тест 2: Анализ настроений
    logger.log('INFO', '=== ТЕСТ 2: Анализ настроений ===')
    broker.analyze_sentiment()

    # Тест 3: Торговый цикл
    logger.log('INFO', '=== ТЕСТ 3: Торговый цикл ===')
    broker.run_cycle()

    # Тест 4: Портфель
    logger.log('INFO', '=== ТЕСТ 4: Состояние портфеля ===')
    portfolio = broker.portfolio.to_dict()
    print(f"Кэш: {portfolio.get('cash', 0):,.2f}₽")
    print(f"Позиций: {len(portfolio.get('positions', {}))}")

    for ticker, data in portfolio.get('positions', {}).items():
        current = data.get('current', 0)
        avg_price = data.get('avg_price', 0)
        change = ((current - avg_price) / avg_price * 100) if avg_price > 0 else 0
        print(f"{ticker}: {data.get('qty', 0)} шт @ {avg_price:.2f} → {current:.2f} ({change:+.1f}%)")


if __name__ == "__main__":
    test_full_system()