# fetchers/moex_fetcher.py
import requests
import time
from utils.logger import logger

class MoexFetcher:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self._securities_cache = None
        self._cache_time = 0

    def get_all_securities(self) -> dict:
        """Получаем ликвидные акции — адаптировано под MOEX API 2025"""
        if self._securities_cache and time.time() - self._cache_time < 300:
            return self._securities_cache

        url = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json"
        params = {
            'iss.meta': 'off',
            'iss.only': 'securities,marketdata',
            'securities.columns': 'SECID,SECNAME,PREVADMITTEDQUOTE,PREVLEGALCLOSEPRICE',
            'marketdata.columns': 'LAST,OPEN,VOLUME',
            'limit': '100'
        }

        for attempt in range(3):
            try:
                r = self.session.get(url, params=params, timeout=15)
                r.raise_for_status()
                data = r.json()

                sec_data = data.get('securities', {}).get('data', [])
                if not sec_data:
                    raise ValueError("securities.data пустой")

                columns = data['securities']['columns']
                secid_idx = columns.index('SECID') if 'SECID' in columns else -1
                prev_idx = columns.index('PREVADMITTEDQUOTE') if 'PREVADMITTEDQUOTE' in columns else columns.index('PREVLEGALCLOSEPRICE') if 'PREVLEGALCLOSEPRICE' in columns else -1

                mkt_data = data.get('marketdata', {}).get('data', [])
                mkt_columns = data.get('marketdata', {}).get('columns', [])
                last_idx = mkt_columns.index('LAST') if 'LAST' in mkt_columns else -1
                volume_idx = mkt_columns.index('VOLUME') if 'VOLUME' in mkt_columns else -1

                result = {}
                for i in range(min(len(sec_data), len(mkt_data))):
                    sec_row = sec_data[i]
                    mkt_row = mkt_data[i]
                    if len(sec_row) <= max(secid_idx, prev_idx) or len(mkt_row) <= max(last_idx, volume_idx):
                        continue

                    ticker = sec_row[secid_idx]
                    prev_price = sec_row[prev_idx]
                    last_price = mkt_row[last_idx] if last_idx >= 0 else prev_price
                    volume_shares = mkt_row[volume_idx] if volume_idx >= 0 else 0

                    if not ticker or not prev_price:
                        continue

                    try:
                        prev_price = float(prev_price)
                        last_price = float(last_price)
                        volume_shares = float(volume_shares)
                        volume_rub = volume_shares * last_price
                    except (ValueError, TypeError):
                        continue

                    # Фильтр: объём в рублях или минимальная цена для старта сессии
                    if volume_rub > 30_000_000 or (volume_rub == 0 and prev_price > 10):
                        result[ticker] = {
                            "prev_price": prev_price,
                            "volume": int(volume_rub),
                            "current_price": last_price
                        }

                if result:
                    self._securities_cache = result
                    self._cache_time = time.time()
                    logger.log('INFO', f'Загружено {len(result)} ликвидных бумаг с MOEX')
                    return result
                else:
                    logger.log('WARNING', f'Данные получены ({len(sec_data)} строк), но после фильтра 0 бумаг')

            except Exception as e:
                logger.log('WARNING', f'Попытка {attempt+1}/3 получения списка MOEX: {e}')
                time.sleep(2)

        logger.log('ERROR', 'Не удалось загрузить список бумаг с MOEX за 3 попытки')
        return self._securities_cache or {}

    def get_price(self, ticker: str) -> float | None:
        """Получаем цену с повторными попытками и таймаутом"""
        url = f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{ticker}.json"
        params = {'iss.meta': 'off', 'marketdata.columns': 'LAST'}

        for attempt in range(3):
            try:
                r = self.session.get(url, params=params, timeout=12)
                if r.status_code == 200:
                    data = r.json()
                    if data.get('marketdata', {}).get('data'):
                        price = data['marketdata']['data'][0][0]
                        if price:
                            return float(price)
                time.sleep(1)
            except Exception as e:
                if attempt == 2:
                    logger.log('WARNING', f'Цена {ticker}: не удалось после 3 попыток')
                else:
                    time.sleep(2)
        return None