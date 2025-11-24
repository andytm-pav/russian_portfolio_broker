# fetchers/moex_fetcher_enhanced.py
import requests
import json
import time
from utils.logger import logger


class MoexFetcher:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.cache_file = "data/moex_securities.json"

    def get_all_securities(self):
        """Основной метод получения бумаг"""
        cached = self._load_from_cache()
        if cached:
            logger.log('INFO', f'Загружено из кэша: {len(cached)} бумаг')
            return cached

        # Фиксированный список с MRKV
        popular_stocks = {
            "SBER": {"volume": 150000000, "prev_price": 250.0, "current_price": 255.50, "name": "Сбербанк"},
            "GAZP": {"volume": 200000000, "prev_price": 150.0, "current_price": 152.30, "name": "Газпром"},
            "LKOH": {"volume": 80000000, "prev_price": 6000.0, "current_price": 6100.75, "name": "Лукойл"},
            "ROSN": {"volume": 120000000, "prev_price": 500.0, "current_price": 512.40, "name": "Роснефть"},
            "VTBR": {"volume": 300000000, "prev_price": 0.02, "current_price": 0.0215, "name": "ВТБ"},
            "MRKV": {"volume": 50000000, "prev_price": 0.12, "current_price": 0.127, "name": "МРСК Волги"},
            "ALRS": {"volume": 60000000, "prev_price": 145.0, "current_price": 148.90, "name": "АЛРОСА"},
            "MGNT": {"volume": 30000000, "prev_price": 3950.0, "current_price": 3980.45, "name": "Магнит"},
            "MOEX": {"volume": 40000000, "prev_price": 148.0, "current_price": 150.50, "name": "Московская Биржа"},
            "NLMK": {"volume": 50000000, "prev_price": 118.0, "current_price": 120.40, "name": "НЛМК"},
        }

        logger.log('INFO', f'Используем фиксированный список: {len(popular_stocks)} бумаг')
        self._save_to_cache(popular_stocks)
        return popular_stocks

    def _load_from_cache(self):
        """Загрузка данных из кэша"""
        try:
            import os
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                if time.time() - cache_data.get('timestamp', 0) < 86400:
                    return cache_data.get('securities', {})
        except Exception as e:
            logger.log('DEBUG', f'Ошибка загрузки кэша: {e}')
        return None

    def _save_to_cache(self, securities):
        """Сохранение данных в кэш"""
        try:
            import os
            os.makedirs('data', exist_ok=True)
            cache_data = {
                'timestamp': time.time(),
                'securities': securities
            }
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.log('DEBUG', f'Ошибка сохранения кэша: {e}')

    def get_price(self, ticker):
        """Улучшенный метод получения цены"""
        if not ticker:
            return None

        # Сначала проверяем кэш
        cached = self._load_from_cache()
        if cached and ticker in cached:
            price = cached[ticker].get('current_price')
            if price and price > 0:
                logger.log('DEBUG', f'Цена {ticker} из кэша: {price}')
                return price

        # Пробуем разные эндпоинты и рынки
        endpoints = [
            f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{ticker}.json",
            f"https://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}.json",
            f"https://iss.moex.com/iss/engines/stock/markets/bonds/boards/TQCB/securities/{ticker}.json",
            f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQTF/securities/{ticker}.json",
        ]

        for url in endpoints:
            try:
                params = {
                    'iss.meta': 'off',
                    'marketdata.columns': 'LAST,OPEN,LOW,HIGH,CLOSE,LCURRENTPRICE'
                }

                logger.log('DEBUG', f'Запрос цены {ticker} с {url}')
                r = self.session.get(url, params=params, timeout=10)

                if r.status_code == 200:
                    data = r.json()
                    market_data = data.get('marketdata', {}).get('data', [])

                    if market_data and market_data[0]:
                        # Пробуем разные поля с ценой
                        price_fields = [
                            0,  # LAST
                            1,  # OPEN
                            4,  # CLOSE
                            5  # LCURRENTPRICE
                        ]

                        for field in price_fields:
                            if (len(market_data[0]) > field and
                                    market_data[0][field] is not None and
                                    market_data[0][field] > 0):
                                price = float(market_data[0][field])
                                logger.log('INFO', f'Цена {ticker} найдена: {price} (поле {field})')
                                return price

                # Если в marketdata нет, проверяем другие секции
                securities_data = data.get('securities', {}).get('data', [])
                if securities_data and securities_data[0]:
                    # PREVADMITTEDQUOTE, PREVPRICE и т.д.
                    price_fields = [2, 3, 4, 5]  # Разные индексы для цен
                    for field in price_fields:
                        if (len(securities_data[0]) > field and
                                securities_data[0][field] is not None and
                                securities_data[0][field] > 0):
                            price = float(securities_data[0][field])
                            logger.log('INFO', f'Цена {ticker} из securities: {price}')
                            return price

            except Exception as e:
                logger.log('DEBUG', f'Ошибка {url}: {e}')
                continue

        # Если API не сработало, используем фиксированные цены
        fixed_prices = {
            "SBER": 0, "GAZP": 0, "LKOH": 0, "ROSN": 0,
            "VTBR": 0, "MRKV": 0, "ALRS": 0, "MGNT": 0,
            "MOEX": 0, "NLMK": 0, "GMKN": 20, "PLZL": 0,
            "TATN": 0, "AFKS": 0, "AFLT": 0, "PIKK": 60,
        }

        if ticker in fixed_prices:
            logger.log('INFO', f'Цена {ticker} из фиксированного списка: {fixed_prices[ticker]}')
            return fixed_prices[ticker]

        # Последняя попытка - цена из кэша
        if cached and ticker in cached:
            price = cached[ticker].get('prev_price')
            if price and price > 0:
                logger.log('INFO', f'Цена {ticker} из prev_price кэша: {price}')
                return price

        # Если ничего не сработало
        logger.log('WARNING', f'Не удалось получить цену для {ticker}')
        return None