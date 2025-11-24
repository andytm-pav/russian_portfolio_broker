import json
import os
from collections import deque

class TraderModel:
    def __init__(self):
        self.market_sentiment = 0.0
        # История сделок для будущего дообучения (пока простая эвристика)
        self.trade_history = deque(maxlen=1000)  # (score, momentum, market_sent, outcome)

    def update_market_sentiment(self, score: float):
        # экспоненциальное сглаживание
        self.market_sentiment = 0.7 * self.market_sentiment + 0.3 * score

    def record_outcome(self, ticker: str, action: str, score: float, momentum: float = 0.0, pnl: float = 0.0):
        """Запоминаем сделку для будущего анализа эффективности стратегии"""
        self.trade_history.append({
            "ticker": ticker,
            "action": action,
            "score": score,
            "momentum": momentum,
            "market_sent": self.market_sentiment,
            "pnl": pnl
        })

    def rank_candidates(self, prices: dict, securities: dict, ticker_sent: dict, mentions: dict):
        candidates = []

        for ticker, price in prices.items():
            prev_price = securities.get(ticker, {}).get("prev_price", price)
            momentum = (price / prev_price - 1) if prev_price else 0.0

            sent = ticker_sent.get(ticker, self.market_sentiment)
            mention_bonus = min(mentions.get(ticker, 0), 10) * 0.04

            # Итоговый score — взвешенная формула (подобрана опытным путём)
            score = (
                momentum * 14.0 +
                sent * 5.0 +
                mention_bonus +
                self.market_sentiment * 2.5
            )

            candidates.append((ticker, score))

        return sorted(candidates, key=lambda x: x[1], reverse=True)[:40]

    def get_worst_position(self, positions: dict, prices: dict):
        if not positions:
            return None, 0
        worst = min(positions.items(), key=lambda item: prices.get(item[0], 0) / item[1]['avg_price'])
        ticker, data = worst
        qty = data['qty'] // 2
        return ticker, qty if qty > 0 else 0