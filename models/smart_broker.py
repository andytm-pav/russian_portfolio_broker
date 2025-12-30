# models/smart_broker.py
import time
import schedule
import json
import traceback
from datetime import datetime
from collections import defaultdict

from fetchers.moex_fetcher import MoexFetcher
from fetchers.news_fetcher import NewsFetcher
from analyzers.sentiment_analyzer import SentimentAnalyzer
from utils.portfolio_manager import PortfolioManager
from utils.logger import logger
from models.trader_model import trader_model_instance


class SmartPortfolioBroker:
    def __init__(self, settings):
        self.settings = settings
        self.moex = MoexFetcher()
        self.news = NewsFetcher()
        self.sentiment = SentimentAnalyzer()
        self.portfolio = PortfolioManager()
        self.model = trader_model_instance
        self.last_news_ts = 0
        self.current_tickers = []
        self.current_sentiments = {}  # Словарь для хранения текущих сентиментов по тикерам

        print(f"[SmartBroker] Инициализирован. Загружена модель с sentiment={self.model.market_sentiment:.3f}")

    def pre_session_analysis(self):
        logger.log('INFO', 'Запуск предсессионного анализа — сбор новостей')
        self.analyze_sentiment(force=True)

    def is_market_open(self) -> bool:
        if self.settings.get("force_trading_mode", False):
            logger.log('INFO', 'ПРИНУДИТЕЛЬНЫЙ РЕЖИМ ТОРГОВЛИ 24/7 — АКТИВИРОВАН')
            print(f"[DEBUG is_market_open] force_trading_mode={self.settings.get('force_trading_mode')}")
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
            print(f"[DEBUG] Получено бумаг: {len(securities)}")
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

            print(f"[DEBUG] Получено цен: {len(prices)} из {len(tickers)} тикеров")
            if len(prices) < 10:
                print(f"[DEBUG] Тикеры без цены: {[t for t in tickers if t not in prices]}")

            if not prices:
                return

            self.current_tickers = list(prices.keys())
            print(f"[DEBUG 3] Начинаю сбор новостей")

            # 3. Анализ новостей и тональности по тикерам
            news_list = self.news.get_last_news()
            print(f"[DEBUG 3] Получено новостей: {len(news_list)}")

            ticker_sentiment = defaultdict(float)
            mention_count = defaultdict(int)
            news_by_ticker = defaultdict(list)  # Для новой модели

            if news_list:
                for item in news_list:
                    item_ts = item['ts']
                    if item_ts <= self.last_news_ts:
                        continue

                    title_upper = item['title'].upper()
                    score = self.sentiment.predict(item['title'])

                    for ticker in self.current_tickers:
                        if ticker in title_upper:
                            ticker_sentiment[ticker] += score
                            mention_count[ticker] += 1
                            news_by_ticker[ticker].append(item['title'])

                # Усредняем sentiment по упоминаниям
                for t in ticker_sentiment:
                    if mention_count[t] > 0:
                        ticker_sentiment[t] /= mention_count[t]

            # Сохраняем сентименты для использования в других методах
            self.current_sentiments = dict(ticker_sentiment)

            # Если тикер не упоминался — используем общее рыночное настроение
            for t in self.current_tickers:
                if t not in ticker_sentiment:
                    ticker_sentiment[t] = self.model.market_sentiment

            # 4. Ранжируем кандидатов через новую модель
            print(f"[DEBUG] Ранжирование кандидатов...")
            candidates = self.model.rank_candidates(
                prices=prices,
                securities=securities,
                ticker_sentiment=ticker_sentiment,
                news_by_ticker=news_by_ticker
            )

            print(f"[DEBUG] Получено кандидатов: {len(candidates)}")

            # 5. Стопы и тейк-профит
            self.check_stops_and_tp(prices)

            # 6. Ребалансировка портфеля
            self.rebalance_portfolio(prices, securities, candidates)

            # 7. Периодическое обучение модели
            try:
                loss = self.model.periodic_learning()
                if loss is not None:
                    logger.log('TRAINING', f'Обучение модели, Loss: {loss:.6f}')
            except Exception as e:
                logger.log('ERROR', f'Ошибка при обучении модели: {e}')

        except Exception as e:
            logger.log('CRITICAL', 'Критическая ошибка в run_cycle', traceback.format_exc())

    def get_current_sentiment(self, ticker: str) -> float:
        """Возвращает текущий сентимент для тикера"""
        if ticker in self.current_sentiments:
            return self.current_sentiments[ticker]

        # Если нет конкретного сентимента, возвращаем рыночный
        if hasattr(self.model, 'market_sentiment'):
            return self.model.market_sentiment

        return 0.0  # Значение по умолчанию

    def check_stops_and_tp(self, prices: dict):
        cfg = json.load(open("config/broker.json", "r", encoding="utf-8"))
        sl = cfg["stop_loss"]
        tp = cfg["take_profit"]

        for ticker, pos in list(self.portfolio.positions.items()):
            price = prices.get(ticker)
            if not price:
                continue

            change = (price - pos['avg_price']) / pos['avg_price']
            hold_time = time.time() - pos.get('buy_time', time.time())

            if change <= -sl:
                self.portfolio.sell(ticker, pos['qty'], price)
                logger.log('SELL', f'СТОП-ЛОСС {ticker} {change:+.2%}')

                # Записываем результат сделки для обучения
                try:
                    reward, pnl = self.model.record_trade_outcome(
                        ticker=ticker,
                        action="STOP_LOSS",
                        entry_price=pos['avg_price'],
                        exit_price=price,
                        hold_time=hold_time,
                        news_sentiment=self.get_current_sentiment(ticker),
                        market_conditions={
                            "reason": "stop_loss",
                            "change": change,
                            "volume": pos.get('qty', 0)
                        }
                    )
                    logger.log('DEBUG', f'STOP_LOSS записан: reward={reward:.3f}, pnl={pnl:.2%}')
                except Exception as e:
                    logger.log('ERROR', f'Ошибка записи STOP_LOSS: {e}')

            elif change >= tp:
                qty = pos['qty'] // 2
                if qty > 0:
                    self.portfolio.sell(ticker, qty, price)
                    logger.log('SELL', f'ТЕЙК-ПРОФИТ {ticker} {change:+.2%}')

                    # Записываем результат сделки для обучения
                    try:
                        reward, pnl = self.model.record_trade_outcome(
                            ticker=ticker,
                            action="TAKE_PROFIT",
                            entry_price=pos['avg_price'],
                            exit_price=price,
                            hold_time=hold_time,
                            news_sentiment=self.get_current_sentiment(ticker),
                            market_conditions={
                                "reason": "take_profit",
                                "change": change,
                                "volume": qty
                            }
                        )
                        logger.log('DEBUG', f'TAKE_PROFIT записан: reward={reward:.3f}, pnl={pnl:.2%}')
                    except Exception as e:
                        logger.log('ERROR', f'Ошибка записи TAKE_PROFIT: {e}')

    def rebalance_portfolio(self, prices: dict, securities: dict, candidates: list):
        target = self.settings["target_positions"]
        current_cnt = len(self.portfolio.positions)
        total_value = self.portfolio.get_total_value(prices)

        logger.log('INFO',
                   f'Портфель: {current_cnt}/{target} | Стоимость ≈ {total_value:,.0f}₽ | Кэш {self.portfolio.cash:,.0f}₽')

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

                    # Записываем покупку для обучения
                    try:
                        reward, _ = self.model.record_trade_outcome(
                            ticker=ticker,
                            action="BUY",
                            entry_price=price,
                            exit_price=price,  # При покупке exit = entry
                            hold_time=0.0,  # Сделка только открыта
                            news_sentiment=self.get_current_sentiment(ticker),
                            market_conditions={
                                "momentum": momentum,
                                "reason": "rebalance_buy",
                                "score": score,
                                "qty": qty
                            }
                        )
                        logger.log('DEBUG', f'BUY записан: reward={reward:.3f}, score={score:.3f}')
                    except Exception as e:
                        logger.log('ERROR', f'Ошибка записи BUY: {e}')

                    bought += 1
                    logger.log('BUY',
                               f'КУПЛЕНО {qty}×{ticker} @ {price:.2f} (вес ≈ {projected_weight:.1%}, score={score:.2f})')

                    # Обновляем время покупки в позиции
                    if ticker in self.portfolio.positions:
                        self.portfolio.positions[ticker]['buy_time'] = time.time()

        # === ПРОДАЖИ при избытке позиций ===
        if current_cnt > target:
            worst_ticker, qty = self.model.get_worst_position(self.portfolio.positions, prices)
            if qty > 0 and worst_ticker in prices:
                if worst_ticker in self.portfolio.positions:
                    pos = self.portfolio.positions[worst_ticker]
                    hold_time = time.time() - pos.get('buy_time', time.time())
                    change = (prices[worst_ticker] - pos['avg_price']) / pos['avg_price']

                    self.portfolio.sell(worst_ticker, qty, prices[worst_ticker])
                    logger.log('SELL', f'УМЕНЬШЕНИЕ {worst_ticker} (худшая позиция) {change:+.2%}')

                    # Записываем продажу для обучения
                    try:
                        reward, pnl = self.model.record_trade_outcome(
                            ticker=worst_ticker,
                            action="SELL",
                            entry_price=pos['avg_price'],
                            exit_price=prices[worst_ticker],
                            hold_time=hold_time,
                            news_sentiment=self.get_current_sentiment(worst_ticker),
                            market_conditions={
                                "reason": "worst_position",
                                "change": change,
                                "qty": qty
                            }
                        )
                        logger.log('DEBUG', f'SELL худшей позиции записан: reward={reward:.3f}, pnl={pnl:.2%}')
                    except Exception as e:
                        logger.log('ERROR', f'Ошибка записи SELL худшей позиции: {e}')

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