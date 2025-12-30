# test_trader_model.py
from models.trader_model import trader_model_instance

# Тест кодирования новостей
test_news = ["Сбербанк объявляет о росте прибыли на 20%"]
features = trader_model_instance.encode_news(test_news)
print(f"Новостные фичи: {features.shape}")

# Тест вектора состояния
state = trader_model_instance.build_state_vector(
    ticker="SBER",
    price=300.0,
    momentum=0.02,
    sentiment=0.5,
    news_features=features,
    market_data={'volume': 1000000, 'spread': 0.01, 'rsi': 55, 'volatility': 0.15}
)
print(f"Размер вектора состояния: {state.shape}")