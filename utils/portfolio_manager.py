# utils/portfolio_manager.py
import json
import os
from utils.logger import logger

class PortfolioManager:
    def __init__(self):
        self.file = "data/portfolio.json"
        self.load()

    def load(self):
        """Загрузка портфеля из файла"""
        if os.path.exists(self.file):
            try:
                with open(self.file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.cash = float(data.get("cash", 10000))
                self.positions = data.get("positions", {})
                self.trades = data.get("trades", [])
            except Exception as e:
                logger.log('ERROR', 'Ошибка загрузки портфеля, создаём новый', str(e))
                self.cash = 10000.0
                self.positions = {}
                self.trades = []
        else:
            self.cash = 10000.0
            self.positions = {}
            self.trades = []

    def save(self):
        """Сохранение портфеля в файл"""
        data = {
            "cash": round(self.cash, 2),
            "positions": {
                k: {"qty": v["qty"], "avg_price": round(v["avg_price"], 4)}
                for k, v in self.positions.items()
            },
            "trades": self.trades[-200:]  # храним последние 200 сделок
        }
        os.makedirs("data", exist_ok=True)
        with open(self.file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def buy(self, ticker: str, qty: int, price: float) -> bool:
        """Покупка акции с учётом комиссии и спреда"""
        with open("config/broker.json", "r", encoding="utf-8") as f:
            broker = json.load(f)

        cost = qty * price
        total_cost = cost * (1 + broker["spread_rate"]) + cost * broker["commission_rate"]

        if total_cost > self.cash:
            logger.log('WARNING', f'Недостаточно кэша для покупки {qty}×{ticker}')
            return False

        self.cash -= total_cost

        if ticker in self.positions:
            old = self.positions[ticker]
            new_qty = old["qty"] + qty
            new_avg = (old["qty"] * old["avg_price"] + qty * price) / new_qty
            self.positions[ticker] = {"qty": new_qty, "avg_price": new_avg}
        else:
            self.positions[ticker] = {"qty": qty, "avg_price": price}

        self.trades.append({
            "type": "BUY",
            "ticker": ticker,
            "qty": qty,
            "price": price,
            "total": round(total_cost, 2)
        })
        self.save()
        logger.log('BUY', f'КУПЛЕНО {qty}×{ticker} @ {price:.2f}', f'Потрачено {total_cost:,.0f}₽')
        return True

    def sell(self, ticker: str, qty: int, price: float) -> bool:
        """Продажа акции"""
        if ticker not in self.positions or self.positions[ticker]["qty"] < qty:
            return False

        with open("config/broker.json", "r", encoding="utf-8") as f:
            broker = json.load(f)

        income = qty * price
        total_income = income * (1 - broker["spread_rate"]) - income * broker["commission_rate"]
        self.cash += total_income

        self.positions[ticker]["qty"] -= qty
        if self.positions[ticker]["qty"] <= 0:
            del self.positions[ticker]

        self.trades.append({
            "type": "SELL",
            "ticker": ticker,
            "qty": qty,
            "price": price,
            "total": round(total_income, 2)
        })
        self.save()
        logger.log('SELL', f'ПРОДАНО {qty}×{ticker} @ {price:.2f}', f'Получено {total_income:,.0f}₽')
        return True

    def get_total_value(self, current_prices: dict) -> float:
        """Общая стоимость портфеля (кэш + позиции по текущим ценам)"""
        positions_value = sum(
            self.positions[t]["qty"] * current_prices.get(t, self.positions[t]["avg_price"])
            for t in self.positions
        )
        return self.cash + positions_value

    def get_position_weight(self, ticker: str, current_prices: dict) -> float:
        """Текущий вес позиции в портфеле"""
        if ticker not in self.positions:
            return 0.0
        price = current_prices.get(ticker)
        if not price:
            return 0.0
        value = self.positions[ticker]["qty"] * price
        total = self.get_total_value(current_prices)
        return value / total if total > 0 else 0.0

    def calculate_projected_weight(self, ticker: str, qty: int, price: float, current_prices: dict) -> float:
        """
        Расчёт веса позиции ПОСЛЕ покупки (чтобы не превысить max_position_weight)
        """
        current_total = self.get_total_value(current_prices)
        additional_value = qty * price

        # Учитываем будущую стоимость позиции
        future_position_value = additional_value
        if ticker in self.positions:
            current_price = current_prices.get(ticker, price)
            future_position_value += self.positions[ticker]["qty"] * current_price

        future_total = current_total + additional_value
        return future_position_value / future_total if future_total > 0 else 0.0

    def to_dict(self):
        """Для веб-интерфейса"""
        from fetchers.moex_fetcher import MoexFetcher
        moex = MoexFetcher()
        prices = {}
        for t in self.positions:
            p = moex.get_price(t)
            prices[t] = p if p else 0

        return {
            "cash": round(self.cash, 2),
            "positions": {
                t: {
                    "qty": p["qty"],
                    "avg_price": round(p["avg_price"], 4),
                    "current": prices.get(t, 0)
                } for t, p in self.positions.items()
            }
        }