# models/trader_model.py
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class NewsEncoder(nn.Module):
    """Нейросеть для анализа новостей и извлечения скрытых признаков"""

    def __init__(self, input_dim=312, hidden_dim=256):  # Изменено с 768 на 312
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, 128),
            nn.Tanh()  # Выход в диапазоне [-1, 1]
        )

    def forward(self, x):
        return self.encoder(x)


class TradingPolicyNetwork(nn.Module):
    """Политика трейдера: принимает решение на основе состояния рынка"""

    def __init__(self, state_dim=140, action_dim=3):  # Изменено с 150 на 140
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)  # Вероятности действий
        )

        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Оценка состояния
        )

    def forward(self, state):
        action_probs = self.policy_net(state)
        state_value = self.value_net(state)
        return action_probs, state_value


class AdvancedTraderModel:
    """Продвинутая модель трейдера с обучением и памятью ошибок"""

    def __init__(self,
                 model_dir: str = "models/saved_trader",
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,  # Коэффициент дисконтирования
                 memory_size: int = 10000):

        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        # Для вычисления градиентов
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Основные сети
        self.news_encoder = NewsEncoder()
        self.news_encoder.to(self.device)

        self.policy_net = TradingPolicyNetwork()
        self.policy_net.to(self.device)

        # Модель для анализа новостей (RuBERT)
        try:
            from transformers import AutoTokenizer, AutoModel
            self.news_tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
            self.bert_model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
            self.bert_model.to(self.device)
            self.bert_model.eval()
            print("[TraderModel] ✓ Загружена модель RuBERT для анализа новостей")
        except Exception as e:
            print(f"[TraderModel] ⚠ Не удалось загрузить RuBERT: {e}")
            print("[TraderModel] ⚠ Используется упрощенный анализ новостей")
            self.bert_model = None
            self.news_tokenizer = None

        # Оптимизаторы
        self.news_optimizer = optim.Adam(self.news_encoder.parameters(), lr=learning_rate)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Память для обучения с подкреплением
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma

        # Память ошибок (словарь тикер -> история неудач)
        self.error_memory = defaultdict(lambda: {
            'failed_trades': [],
            'avg_loss': 0.0,
            'last_failure': None,
            'failure_count': 0
        })

        # Статистика по тикерам
        self.ticker_stats = defaultdict(lambda: {
            'total_trades': 0,
            'profitable_trades': 0,
            'total_pnl': 0.0,
            'avg_hold_time': 0.0,
            'success_rate': 0.0
        })

        # Рыночное настроение
        self.market_sentiment = 0.0
        self.sentiment_history = deque(maxlen=100)

        # Загружаем сохранённые веса, если они есть
        self.load_model()

        print(f"[TraderModel] Инициализирована на {self.device}")
        print(f"[TraderModel] Загружено ошибок: {len(self.error_memory)} тикеров")

    def save_model(self):
        """Сохраняем все компоненты модели"""
        try:
            # Сохраняем веса нейросетей
            torch.save({
                'news_encoder': self.news_encoder.state_dict(),
                'policy_net': self.policy_net.state_dict(),
                'news_optimizer': self.news_optimizer.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict()
            }, os.path.join(self.model_dir, 'model_weights.pth'))

            # Сохраняем память ошибок и статистику
            state = {
                'error_memory': dict(self.error_memory),
                'ticker_stats': dict(self.ticker_stats),
                'market_sentiment': self.market_sentiment,
                'sentiment_history': list(self.sentiment_history),
                'memory_size': len(self.memory)
            }

            with open(os.path.join(self.model_dir, 'model_state.json'), 'w') as f:
                json.dump(state, f, indent=2, default=str)

            print(f"[TraderModel] Модель сохранена в {self.model_dir}")

        except Exception as e:
            print(f"[TraderModel] Ошибка сохранения: {e}")

    def load_model(self):
        """Загружаем сохранённые компоненты модели"""
        weights_path = os.path.join(self.model_dir, 'model_weights.pth')
        state_path = os.path.join(self.model_dir, 'model_state.json')

        try:
            if os.path.exists(weights_path):
                checkpoint = torch.load(weights_path, map_location=self.device)
                self.news_encoder.load_state_dict(checkpoint['news_encoder'])
                self.policy_net.load_state_dict(checkpoint['policy_net'])

                if 'news_optimizer' in checkpoint:
                    self.news_optimizer.load_state_dict(checkpoint['news_optimizer'])
                if 'policy_optimizer' in checkpoint:
                    self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])

                print(f"[TraderModel] ✓ Загружены веса нейросетей")

        except Exception as e:
            print(f"[TraderModel] Ошибка загрузки весов: {e}")

        try:
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)

                # Восстанавливаем память ошибок
                self.error_memory.clear()
                for ticker, data in state.get('error_memory', {}).items():
                    self.error_memory[ticker] = data

                # Восстанавливаем статистику
                self.ticker_stats.clear()
                for ticker, stats in state.get('ticker_stats', {}).items():
                    self.ticker_stats[ticker] = stats

                self.market_sentiment = state.get('market_sentiment', 0.0)
                self.sentiment_history = deque(state.get('sentiment_history', []), maxlen=100)

                print(f"[TraderModel] ✓ Загружено: {len(self.error_memory)} ошибок, "
                      f"{len(self.ticker_stats)} тикеров, sentiment={self.market_sentiment:.3f}")

        except Exception as e:
            print(f"[TraderModel] Ошибка загрузки состояния: {e}")

    def encode_news(self, news_texts: List[str]) -> torch.Tensor:
        """Кодируем новости в векторное представление через RuBERT"""
        if not news_texts:
            return torch.zeros(1, 128).to(self.device)

        try:
            if self.bert_model is not None and self.news_tokenizer is not None:
                # Токенизация
                inputs = self.news_tokenizer(
                    news_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(self.device)

                # Получение эмбеддингов
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)

                    # Проверяем размерность
                    embedding_dim = outputs.last_hidden_state.shape[2]
                    print(f"[TraderModel] BERT embedding dimension: {embedding_dim}")

                    # Автоматически определяем размерность
                    if embedding_dim != self.news_encoder.encoder[0].in_features:
                        print(
                            f"[TraderModel] ⚠ Размерность не совпадает: BERT={embedding_dim}, Encoder={self.news_encoder.encoder[0].in_features}")
                        print(f"[TraderModel] ⚠ Использую упрощенное кодирование")
                        raise ValueError("Dimension mismatch")

                    # Берем эмбеддинги [CLS] токена
                    embeddings = outputs.last_hidden_state[:, 0, :]

                # Пропускаем через энкодер
                self.news_encoder.eval()
                with torch.no_grad():
                    news_features = self.news_encoder(embeddings)

                print(f"[TraderModel] encode_news: {len(news_texts)} новостей → {news_features.shape}")
                return news_features

        except Exception as e:
            print(f"[TraderModel] Ошибка кодирования через BERT: {e}")
            print(f"[TraderModel] Использую упрощенное кодирование")

        # Упрощенное кодирование (фолбэк)
        return self.simple_encode_news(news_texts)

    def simple_encode_news(self, news_texts: List[str]) -> torch.Tensor:
        """Упрощенное кодирование новостей без BERT"""
        batch_size = len(news_texts)
        embeddings = torch.zeros(batch_size, 312).to(self.device)  # Размерность как у BERT

        for i, text in enumerate(news_texts):
            # Простые эвристики
            text = text.lower()
            words = text.split()

            # Признаки
            length_norm = min(len(text) / 500, 1.0)
            word_count = min(len(words) / 100, 1.0)
            unique_ratio = len(set(words)) / max(len(words), 1)

            # Ключевые слова для финансового анализа
            positive_words = ['рост', 'прибыль', 'увеличение', 'дивиденд', 'выше', 'улучшение']
            negative_words = ['падение', 'убыток', 'снижение', 'проблемы', 'ниже', 'сокращение']
            market_words = ['рынок', 'акция', 'фондовый', 'бирж', 'инвест', 'торг']

            # Вычисляем признаки
            pos_score = sum(1 for word in positive_words if word in text) / len(positive_words)
            neg_score = sum(1 for word in negative_words if word in text) / len(negative_words)
            market_score = sum(1 for word in market_words if word in text) / len(market_words)

            # Создаем эмбеддинг
            embedding = torch.zeros(312).to(self.device)
            embedding[0] = length_norm
            embedding[1] = word_count
            embedding[2] = unique_ratio
            embedding[3] = pos_score
            embedding[4] = neg_score
            embedding[5] = market_score
            embedding[6] = pos_score - neg_score  # Общий сентимент

            # Добавляем немного шума для разнообразия
            embedding[7:] = torch.randn(305).to(self.device) * 0.01

            embeddings[i] = embedding

        # Пропускаем через энкодер
        self.news_encoder.eval()
        with torch.no_grad():
            news_features = self.news_encoder(embeddings)

        return news_features

    def calculate_risk_score(self, ticker: str, price: float, sentiment: float) -> float:
        """Вычисляем риск-скор на основе истории ошибок"""
        error_data = self.error_memory[ticker]

        if error_data['failure_count'] == 0:
            base_risk = 0.5
        else:
            # Увеличиваем риск для тикеров с плохой историей
            failure_penalty = min(error_data['failure_count'] * 0.1, 1.0)
            loss_penalty = min(abs(error_data['avg_loss']) * 2, 1.0)
            base_risk = 0.5 + failure_penalty * 0.3 + loss_penalty * 0.2

        # Корректируем на основе сентимента
        sentiment_factor = 1.0 - abs(sentiment)  # Нейтральный сентимент = больше риска
        final_risk = base_risk * sentiment_factor

        return max(0.1, min(0.9, final_risk))  # Ограничиваем диапазон

    def build_state_vector(self,
                           ticker: str,
                           price: float,
                           momentum: float,
                           sentiment: float,
                           news_features: torch.Tensor,
                           market_data: Dict) -> torch.Tensor:
        """Строим вектор состояния для принятия решения"""

        # Извлекаем признаки из новостей
        if news_features.numel() > 0 and news_features.shape[1] == 128:
            news_vec = news_features.mean(dim=0).cpu().numpy()
        else:
            news_vec = np.zeros(128)

        # Статистика по тикеру
        stats = self.ticker_stats[ticker]
        success_rate = stats['success_rate'] if stats['total_trades'] > 0 else 0.5
        total_trades = stats['total_trades']

        # Оценка риска
        risk_score = self.calculate_risk_score(ticker, price, sentiment)

        # Базовые признаки (10 шт)
        features = [
            price / 1000.0,  # Нормализованная цена
            momentum * 10.0,  # Моментум (усиленный)
            sentiment,  # Сентимент
            risk_score,  # Риск
            success_rate,  # Историческая успешность
            min(total_trades / 100.0, 1.0),  # Опыт по тикеру
            self.market_sentiment,  # Рыночное настроение
            market_data.get('volume', 0) / 1e6 if market_data else 0,  # Объём (млн)
            market_data.get('spread', 0.01) * 100 if market_data else 1.0,  # Спред (%)
            market_data.get('liquidity', 0.5) if market_data else 0.5,  # Ликвидность
        ]

        # Новостные признаки (128)
        features.extend(news_vec.tolist())

        # Дополнительные рыночные признаки (2 шт)
        features.extend([
            market_data.get('rsi', 50) / 100.0 if market_data else 0.5,  # RSI нормализованный
            market_data.get('volatility', 0.1) if market_data else 0.1,  # Волатильность
        ])

        # ИТОГО: 10 + 128 + 2 = 140 признаков

        # Преобразуем в тензор
        state_vector = torch.FloatTensor(features).to(self.device)

        # Проверяем размерность
        if state_vector.shape[0] != 140:
            print(f"[TraderModel] ⚠ Предупреждение: размерность состояния {state_vector.shape[0]} != 140")
            # Корректируем размерность
            if state_vector.shape[0] < 140:
                padding = torch.zeros(140 - state_vector.shape[0]).to(self.device)
                state_vector = torch.cat([state_vector, padding])
            else:
                state_vector = state_vector[:140]

        return state_vector

    def choose_action(self,
                      state: torch.Tensor,
                      ticker: str,
                      current_price: float) -> Tuple[int, float, float]:
        """Выбираем действие на основе политики"""
        self.policy_net.eval()

        with torch.no_grad():
            action_probs, state_value = self.policy_net(state.unsqueeze(0))

        # Преобразуем в numpy
        probs = action_probs.cpu().numpy().flatten()

        # Корректируем вероятности на основе истории ошибок
        error_data = self.error_memory[ticker]
        if error_data['failure_count'] > 2:
            # Снижаем вероятность покупки для проблемных тикеров
            probs[0] *= 0.5  # BUY
            probs[2] *= 1.2  # SELL
            probs = probs / probs.sum()  # Нормализуем

        # Выбираем действие
        action = np.random.choice(len(probs), p=probs)
        confidence = probs[action]

        return action, confidence, state_value.item()

    def remember_experience(self,
                            state: torch.Tensor,
                            action: int,
                            reward: float,
                            next_state: torch.Tensor,
                            done: bool):
        """Сохраняем опыт для обучения"""
        self.memory.append({
            'state': state.cpu(),
            'action': action,
            'reward': reward,
            'next_state': next_state.cpu(),
            'done': done
        })

    def learn_from_experience(self, batch_size: int = 64):
        """Обучаемся на сохранённом опыте"""
        if len(self.memory) < batch_size:
            return None

        try:
            # Выбираем случайную выборку
            indices = np.random.choice(len(self.memory), batch_size, replace=False)
            batch = [self.memory[i] for i in indices]

            states = torch.stack([exp['state'] for exp in batch]).to(self.device)
            actions = torch.LongTensor([exp['action'] for exp in batch]).to(self.device)
            rewards = torch.FloatTensor([exp['reward'] for exp in batch]).to(self.device)
            next_states = torch.stack([exp['next_state'] for exp in batch]).to(self.device)
            dones = torch.FloatTensor([exp['done'] for exp in batch]).to(self.device)

            # Переключаем в режим обучения
            self.policy_net.train()

            # Вычисляем лосс
            _, current_values = self.policy_net(states)
            _, next_values = self.policy_net(next_states)

            # Целевые значения
            target_values = rewards + (1 - dones) * self.gamma * next_values

            # Лосс для value сети
            value_loss = nn.MSELoss()(current_values, target_values.detach())

            # Лосс для policy сети
            action_probs, _ = self.policy_net(states)
            dist = torch.distributions.Categorical(action_probs)
            log_probs = dist.log_prob(actions)
            policy_loss = -(log_probs * (target_values - current_values).detach()).mean()

            # Общий лосс
            total_loss = value_loss + policy_loss

            # Оптимизация
            self.policy_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.policy_optimizer.step()

            # Также обучаем энкодер новостей (если есть данные)
            if len(self.memory) > 1000 and hasattr(self, 'news_optimizer'):
                self.news_encoder.train()
                # Здесь можно добавить обучение энкодера на задаче реконструкции

            return total_loss.item()

        except Exception as e:
            print(f"[TraderModel] Ошибка обучения: {e}")
            return None

    def update_market_sentiment(self, sentiment_score: float):
        """Обновляем рыночное настроение с экспоненциальным сглаживанием"""
        self.market_sentiment = 0.8 * self.market_sentiment + 0.2 * sentiment_score
        self.sentiment_history.append(self.market_sentiment)

    def record_trade_outcome(self,
                             ticker: str,
                             action: str,
                             entry_price: float,
                             exit_price: float,
                             hold_time: float,
                             news_sentiment: float,
                             market_conditions: Dict) -> Tuple[float, float]:
        """Запоминаем результат сделки и извлекаем уроки"""

        # Рассчитываем PnL
        if action == 'BUY':
            pnl = (exit_price - entry_price) / entry_price
        elif action == 'SELL':
            pnl = (entry_price - exit_price) / entry_price
        else:
            pnl = 0.0  # Для HOLD или других действий

        # Обновляем статистику
        stats = self.ticker_stats[ticker]
        stats['total_trades'] += 1
        stats['total_pnl'] += pnl

        if pnl > 0:
            stats['profitable_trades'] += 1

        # Обновляем среднее время удержания
        if stats['total_trades'] == 1:
            stats['avg_hold_time'] = hold_time
        else:
            stats['avg_hold_time'] = (stats['avg_hold_time'] * (stats['total_trades'] - 1) + hold_time) / stats[
                'total_trades']

        stats['success_rate'] = stats['profitable_trades'] / stats['total_trades'] if stats['total_trades'] > 0 else 0.0

        # Запоминаем ошибки (существенные убытки)
        if pnl < -0.03:  # Убыток более 3%
            error_data = self.error_memory[ticker]
            error_data['failed_trades'].append({
                'date': datetime.now().isoformat(),
                'pnl': pnl,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'sentiment': news_sentiment,
                'action': action,
                'hold_time': hold_time
            })

            error_data['failure_count'] += 1
            error_data['last_failure'] = datetime.now().isoformat()

            # Обновляем средний убыток
            if error_data['failed_trades']:
                losses = [t['pnl'] for t in error_data['failed_trades']]
                error_data['avg_loss'] = sum(losses) / len(losses)

            print(f"[TraderModel] Запомнена ошибка по {ticker}: {action}, убыток {pnl:.2%}")

        # Награда для обучения
        reward = pnl * 10.0  # Масштабируем

        # Дополнительные бонусы/штрафы
        if hold_time < 0.01 and abs(pnl) < 0.01:  # Слишком быстро закрыли без результата
            reward -= 0.3
        elif abs(pnl) > 0.1:  # Большая прибыль/убыток
            reward += np.sign(pnl) * 0.5
        elif pnl > 0:  # Небольшая прибыль
            reward += 0.1

        # Запоминаем опыт для обучения
        try:
            # Здесь можно создать состояние для запоминания в память
            # Для простоты пока пропускаем, но в будущем можно добавить
            pass
        except:
            pass

        return reward, pnl

    def rank_candidates(self,
                        prices: Dict[str, float],
                        securities: Dict[str, Dict],
                        ticker_sentiment: Dict[str, float],
                        news_by_ticker: Dict[str, List[str]]) -> List[Tuple[str, float]]:
        """Ранжируем кандидатов для торговли"""

        candidates = []

        for ticker, price in prices.items():
            if ticker not in securities:
                continue

            market_data = securities[ticker]
            momentum = market_data.get('momentum', 0.0)
            sentiment = ticker_sentiment.get(ticker, self.market_sentiment)

            # Кодируем новости по тикеру
            ticker_news = news_by_ticker.get(ticker, [])
            if ticker_news:
                news_features = self.encode_news(ticker_news[:3])  # Берём 3 последние новости
            else:
                news_features = torch.zeros(1, 128).to(self.device)

            # Строим вектор состояния
            state = self.build_state_vector(
                ticker=ticker,
                price=price,
                momentum=momentum,
                sentiment=sentiment,
                news_features=news_features,
                market_data=market_data
            )

            # Выбираем действие
            try:
                action, confidence, _ = self.choose_action(state, ticker, price)
            except Exception as e:
                print(f"[TraderModel] Ошибка choose_action для {ticker}: {e}")
                action, confidence = 1, 0.5  # HOLD по умолчанию

            # Оценка кандидата (чем выше, тем лучше)
            if action == 0:  # BUY
                stats = self.ticker_stats[ticker]
                success_bonus = stats['success_rate'] * 0.3 if stats['total_trades'] > 0 else 0.0

                risk_score = self.calculate_risk_score(ticker, price, sentiment)
                risk_penalty = risk_score * 0.2

                score = (confidence * 0.6 +
                         sentiment * 0.3 +
                         success_bonus -
                         risk_penalty +
                         self.market_sentiment * 0.1 +
                         momentum * 0.2)

            elif action == 2:  # SELL
                error_data = self.error_memory[ticker]
                failure_penalty = min(error_data['failure_count'] * 0.1, 0.5)

                score = (confidence * 0.4 -
                         sentiment * 0.3 +
                         failure_penalty +
                         (-momentum * 0.2))  # Отрицательный моментум для продажи

            else:  # HOLD
                score = 0.0

            candidates.append((ticker, score))

        # Сортируем по убыванию score
        candidates.sort(key=lambda x: x[1], reverse=True)

        print(f"[TraderModel] Отранжировано {len(candidates)} кандидатов")

        return candidates[:40]  # Возвращаем топ-40

    def get_worst_position(self,
                           positions: Dict[str, Dict],
                           prices: Dict[str, float]) -> Tuple[Optional[str], int]:
        """Определяем худшую позицию для продажи"""
        if not positions:
            return None, 0

        worst_score = float('inf')
        worst_ticker = None

        for ticker, pos_data in positions.items():
            if ticker not in prices:
                continue

            current_price = prices[ticker]
            avg_price = pos_data['avg_price']

            if avg_price > 0:
                pnl = (current_price - avg_price) / avg_price
            else:
                pnl = 0.0

            # Учитываем историю ошибок
            error_data = self.error_memory[ticker]
            failure_penalty = error_data['failure_count'] * 0.15

            # Время удержания (чем дольше, тем выше приоритет на продажу)
            hold_time_penalty = min(pos_data.get('hold_time_days', 0) / 30.0, 1.0) * 0.2

            # Итоговый score (чем ниже, тем хуже)
            score = pnl - failure_penalty - hold_time_penalty

            if score < worst_score:
                worst_score = score
                worst_ticker = ticker

        if worst_ticker:
            qty = positions[worst_ticker]['qty'] // 2
            return worst_ticker, qty if qty > 0 else 0

        return None, 0

    def periodic_learning(self):
        """Периодическое обучение на накопленном опыте"""
        if len(self.memory) > 100:
            loss = self.learn_from_experience(batch_size=64)

            # Сохраняем модель каждые 100 шагов обучения
            if hasattr(self, '_learn_steps'):
                self._learn_steps += 1
                if self._learn_steps % 100 == 0:
                    self.save_model()
            else:
                self._learn_steps = 1

            return loss
        return None


# Создаём глобальный экземпляр для использования в проекте
trader_model_instance = AdvancedTraderModel()