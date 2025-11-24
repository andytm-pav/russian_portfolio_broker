# models/smart_broker.py
import time
import json
import traceback
from datetime import datetime
from collections import defaultdict

from fetchers.moex_fetcher import MoexFetcher
from fetchers.news_fetcher import NewsFetcher
from analyzers.sentiment_analyzer import SentimentAnalyzer
from utils.portfolio_manager import PortfolioManager
from utils.logger import logger
from models.trader_model import TraderModel


class SmartPortfolioBroker:
    def __init__(self, settings):
        self.settings = settings
        self.moex = MoexFetcher()
        self.news = NewsFetcher()
        self.sentiment = SentimentAnalyzer()
        self.portfolio = PortfolioManager()
        self.model = TraderModel()
        self.last_news_ts = 0
        self.current_tickers = []

    def pre_session_analysis(self):
        logger.log('INFO', 'Запуск предсессионного анализа — сбор новостей')
        self.analyze_sentiment(force=True)

    def is_market_open(self) -> bool:
        if self.settings.get("force_trading_mode", False):
            logger.log('INFO', 'ПРИНУДИТЕЛЬНЫЙ РЕЖИМ ТОРГОВЛИ 24/7 — АКТИВИРОВАН')
            return True

        now = datetime.now()
        if now.weekday() >= 5:
            return False
        hm = now.strftime('%H%M')
        is_open = '1000' <= hm <= '1840'
        if is_open:
            logger.log('INFO', 'Рынок открыт — начинаем торговлю')
        return is_open

    def run_cycle(self):
        if not self.is_market_open():
            return

        try:
            # 1. Получаем список ликвидных бумаг
            securities = self.moex.get_all_securities()
            if not securities:
                logger.log('WARNING', 'Не удалось получить список бумаг с MOEX')
                return

            # Берём топ-120 по объёму
            tickers = sorted(securities.items(), key=lambda x: x[1]['volume'], reverse=True)[:120]
            tickers = [t[0] for t in tickers]

            # 2. Получаем текущие цены
            prices = {}
            for t in tickers:
                p = self.moex.get_price(t)
                if p:
                    prices[t] = p

            if not prices:
                return

            self.current_tickers = list(prices.keys())

            # 3. Анализ новостей и тональности по тикерам
            news_list = self.news.get_last_news()
            ticker_sentiment = defaultdict(float)
            mention_count = defaultdict(int)

            for item in news_list:
                item_ts = item['ts']  # ИСПРАВЛЕНО: было присваивание
                if item_ts <= self.last_news_ts:
                    continue
                title_upper = item['title'].upper()
                score = self.sentiment.predict(item['title'])

                for ticker in self.current_tickers:
                    if ticker in title_upper:
                        ticker_sentiment[ticker] += score
                        mention_count[ticker] += 1

            # Усредняем sentiment по упоминаниям
            for t in ticker_sentiment:
                if mention_count[t] > 0:
                    ticker_sentiment[t] /= mention_count[t]

            # Если тикер не упоминался — используем общее рыночное настроение
            for t in self.current_tickers:
                if t not in ticker_sentiment:
                    ticker_sentiment[t] = self.model.market_sentiment

            # 4. Ранжируем кандидатов
            candidates = self.model.rank_candidates(prices, securities, ticker_sentiment, mention_count)

            # 5. Стопы и тейк-профит
            self.check_stops_and_tp(prices)

            # 6. Ребалансировка портфеля
            self.rebalance_portfolio(prices, securities, candidates)

        except Exception as e:
            logger.log('CRITICAL', 'Критическая ошибка в run_cycle', traceback.format_exc())

    def check_stops_and_tp(self, prices: dict):
        cfg = json.load(open("config/broker.json", "r", encoding="utf-8"))
        sl = cfg["stop_loss"]
        tp = cfg["take_profit"]

        for ticker, pos in list(self.portfolio.positions.items()):
            price = prices.get(ticker)
            if not price:
                continue
            change = (price - pos['avg_price']) / pos['avg_price']

            if change <= -sl:
                self.portfolio.sell(ticker, pos['qty'], price)
                logger.log('SELL', f'СТОП-ЛОСС {ticker} {change:+.2%}')
                self.model.record_outcome(ticker, "STOP_LOSS", change)
            elif change >= tp:
                qty = pos['qty'] // 2
                if qty > 0:
                    self.portfolio.sell(ticker, qty, price)
                    logger.log('SELL', f'ТЕЙК-ПРОФИТ {ticker} {change:+.2%}')
                    self.model.record_outcome(ticker, "TAKE_PROFIT", change)

    def rebalance_portfolio(self, prices: dict, securities: dict, candidates: list):
        target = self.settings["target_positions"]
        current_cnt = len(self.portfolio.positions)
        total_value = self.portfolio.get_total_value(prices)

        logger.log('INFO', f'Портфель: {current_cnt}/{target} | Стоимость ≈ {total_value:,.0f}₽ | Кэш {self.portfolio.cash:,.0f}₽')

        # === ПОКУПКИ ===
        if current_cnt < target and self.portfolio.cash > self.settings["min_cash_per_trade"]:
            bought = 0
            for ticker, score in candidates:
                if bought >= 3:
                    break
                if ticker in self.portfolio.positions:
                    continue

                price = prices[ticker]
                qty = max(1, int(self.portfolio.cash * 0.11 / price))

                projected_weight = self.portfolio.calculate_projected_weight(ticker, qty, price, prices)
                if projected_weight > self.settings["max_position_weight"]:
                    continue

                if self.portfolio.buy(ticker, qty, price):
                    prev_price = securities.get(ticker, {}).get("prev_price", price)
                    momentum = (price / prev_price - 1) if prev_price else 0.0
                    self.model.record_outcome(ticker, "BUY", score, momentum)
                    bought += 1
                    logger.log('BUY', f'КУПЛЕНО {qty}×{ticker} @ {price:.2f} (вес ≈ {projected_weight:.1%})')

        # === ПРОДАЖИ при избытке позиций ===
        if current_cnt > target:
            worst_ticker, qty = self.model.get_worst_position(self.portfolio.positions, prices)
            if qty > 0 and worst_ticker in prices:
                self.portfolio.sell(worst_ticker, qty, prices[worst_ticker])
                logger.log('SELL', f'УМЕНЬШЕНИЕ {worst_ticker} (худшая позиция)')

    def analyze_sentiment(self, force: bool = False):
        news = self.news.get_last_news()
        if not news:
            return

        fresh_news = [n for n in news if n['ts'] > self.last_news_ts] if not force else news[-20:]
        if not fresh_news:
            return

        scores = [self.sentiment.predict(n['title']) for n in fresh_news]
        avg_sentiment = sum(scores) / len(scores) if scores else 0.0

        self.last_news_ts = max((n['ts'] for n in news), default=self.last_news_ts)

        logger.log('SENTIMENT', f'Рыночное настроение: {avg_sentiment:+.3f}', f'Новостей: {len(fresh_news)}')
        self.model.update_market_sentiment(avg_sentiment)