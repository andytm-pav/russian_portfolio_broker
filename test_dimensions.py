# test_dimensions.py
import torch
from models.trader_model import trader_model_instance

# Тестируем создание вектора состояния
test_ticker = "SBER"
test_price = 300.0
test_momentum = 0.02
test_sentiment = 0.5
test_news_features = torch.randn(1, 128).to(trader_model_instance.device)
test_market_data = {
    'volume': 1000000,
    'spread': 0.01,
    'rsi': 55,
    'volatility': 0.15,
    'trend': 0.1
}

state_vector = trader_model_instance.build_state_vector(
    ticker=test_ticker,
    price=test_price,
    momentum=test_momentum,
    sentiment=test_sentiment,
    news_features=test_news_features,
    market_data=test_market_data
)

print(f"Размер вектора состояния: {state_vector.shape}")
print(f"Ожидаемый нейросетью: {trader_model_instance.policy_net.policy_net[0].in_features}")

# Проверяем прохождение через нейросеть
try:
    trader_model_instance.policy_net.eval()
    with torch.no_grad():
        action_probs, state_value = trader_model_instance.policy_net(state_vector.unsqueeze(0))
    print(f"✓ Успешно! Action probs: {action_probs}")
except Exception as e:
    print(f"✗ Ошибка: {e}")