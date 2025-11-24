import torch
from transformers import pipeline
from utils.logger import logger

class SentimentAnalyzer:
    """
    Быстрая и очень точная модель для русского финансового текста
    cointegrated/rubert-tiny-sentiment-balanced — всего 40 МБ, работает даже на CPU за миллисекунды
    """
    def __init__(self):
        try:
            self.pipe = pipeline(
                "sentiment-analysis",
                model="cointegrated/rubert-tiny-sentiment-balanced",
                return_all_scores=True,
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.log('ERROR', 'Не удалось загрузить модель sentiment', str(e))
            raise

    def predict(self, text: str) -> float:
        """
        Возвращает значение от -1.0 (очень негативно) до +1.0 (очень позитивно)
        """
        try:
            result = self.pipe(text[:512])[0]
            scores = {item['label']: item['score'] for item in result}
            positive = scores.get('positive', 0.0)
            negative = scores.get('negative', 0.0)
            neutral = scores.get('neutral', 0.0)
            # взвешенная формула, чтобы нейтраль немного тянула к нулю
            return positive - negative + neutral * 0.1
        except Exception as e:
            logger.log('ERROR', 'Ошибка predict sentiment', str(e))
            return 0.0