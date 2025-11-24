# models/trainer.py (исправленная версия)
import os
from utils.logger import logger


def train_model():
    try:
        if not os.path.exists("data/labeled_news.json"):
            logger.log('INFO', 'Нет данных для дообучения модели')
            return

        # Заглушка для обучения
        logger.log('INFO', 'Проверка данных для обучения...')

    except Exception as e:
        logger.log('ERROR', f'Ошибка обучения: {e}')