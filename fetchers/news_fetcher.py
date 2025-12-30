# fetchers/news_fetcher.py
import json
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from io import BytesIO
from typing import List, Dict, Any

import feedparser
import requests
from utils.logger import logger

# Загружаем конфигурацию источников
with open("config/news_sources.json", "r", encoding="utf-8") as f:
    RSS_SOURCES = json.load(f)


class NewsFetcher:
    def __init__(self):
        self.cache_file = "data/news_cache.json"
        self.last_seen = set()
        self.timeout = 5  # Таймаут на каждый источник (секунды)
        self.max_workers = 3  # Максимальное количество параллельных загрузок
        self.max_entries_per_source = 300  # Максимальное количество новостей с одного источника

    def _fetch_single_feed(self, url: str) -> Dict[str, Any]:
        """Загружает и парсит один RSS-фид с таймаутом"""
        try:
            # Загружаем RSS с использованием requests (поддерживает таймаут)
            response = requests.get(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'},
                timeout=self.timeout
            )
            response.raise_for_status()

            # Парсим содержимое через feedparser
            feed = feedparser.parse(BytesIO(response.content))

            return {
                'success': True,
                'url': url,
                'feed': feed,
                'entries_count': len(feed.entries) if hasattr(feed, 'entries') else 0,
                'error': None
            }

        except requests.exceptions.Timeout:
            return {
                'success': False,
                'url': url,
                'feed': None,
                'entries_count': 0,
                'error': f'Таймаут ({self.timeout}с)'
            }
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'url': url,
                'feed': None,
                'entries_count': 0,
                'error': f'Ошибка сети: {str(e)}'
            }
        except Exception as e:
            return {
                'success': False,
                'url': url,
                'feed': None,
                'entries_count': 0,
                'error': f'Ошибка парсинга: {str(e)}'
            }

    def _process_feed_entries(self, feed_data: Dict[str, Any], seen: set) -> List[Dict]:
        """Обрабатывает записи из фида и возвращает список новостей"""
        if not feed_data['success'] or not feed_data['feed']:
            return []

        news_items = []
        feed = feed_data['feed']

        # Берем только свежие записи (ограниченное количество)
        for entry in feed.entries[:self.max_entries_per_source]:
            link = entry.get('link', '')

            # Пропускаем уже обработанные ссылки
            if link in self.last_seen or link in seen:
                continue

            seen.add(link)

            # Извлекаем заголовок
            title = entry.get('title', 'Без заголовка').strip()
            if not title:
                continue

            # Пытаемся определить дату публикации
            published = entry.get('published') or entry.get('updated')
            try:
                if published:
                    # Используем встроенный парсинг дат feedparser
                    ts = time.mktime(feedparser._parse_date(published))
                else:
                    ts = time.time()
            except:
                ts = time.time()

            news_items.append({
                "title": title,
                "link": link,
                "ts": int(ts)
            })

        return news_items

    def _load_from_cache(self) -> List[Dict]:
        """Загружает новости из кэша, если они актуальны"""
        if not os.path.exists(self.cache_file):
            return None

        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)

            # Проверяем, не устарел ли кэш (1 минута)
            if time.time() - cache.get("ts", 0) < 60:
                cached_news = cache.get("news", [])
                logger.log('INFO', f'Загружено из кэша: {len(cached_news)} новостей')
                return cached_news

        except Exception as e:
            logger.log('WARNING', f'Ошибка загрузки кэша: {e}')

        return None

    def _save_to_cache(self, news_items: List[Dict]):
        """Сохраняет новости в кэш"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            data = {
                "ts": int(time.time()),
                "news": news_items
            }
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.log('ERROR', f'Ошибка сохранения кэша: {e}')

    def get_last_news(self) -> List[Dict]:
        """Основной метод получения новостей"""
        # 1. Проверяем кэш
        cached_news = self._load_from_cache()
        if cached_news is not None:
            return cached_news

        logger.log('INFO', f'Начинаю сбор новостей из {len(RSS_SOURCES)} источников')
        start_time = time.time()

        all_news = []
        seen = set()
        successful_sources = 0
        failed_sources = 0

        # 2. Параллельная загрузка всех источников
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Запускаем задачи для всех источников
            future_to_url = {
                executor.submit(self._fetch_single_feed, url): url
                for url in RSS_SOURCES
            }

            # Обрабатываем результаты по мере готовности
            for future in as_completed(future_to_url, timeout=self.timeout * 2):
                url = future_to_url[future]

                try:
                    # Получаем результат с небольшим дополнительным таймаутом
                    feed_data = future.result(timeout=1)

                    if feed_data['success']:
                        # Обрабатываем успешно загруженный фид
                        source_news = self._process_feed_entries(feed_data, seen)
                        all_news.extend(source_news)

                        status_msg = f"✓ {url[:50]}... ({feed_data['entries_count']} записей → {len(source_news)} новых)"
                        if source_news:
                            print(f"[NewsFetcher] {status_msg}")

                        successful_sources += 1
                    else:
                        # Логируем ошибку
                        print(f"[NewsFetcher] ✗ {url[:50]}... — {feed_data['error']}")
                        failed_sources += 1

                except Exception as e:
                    print(f"[NewsFetcher] ✗ Ошибка обработки {url[:50]}... — {type(e).__name__}")
                    failed_sources += 1

        # 3. Обновляем множество уже виденных ссылок
        self.last_seen.update(seen)

        # 4. Сохраняем в кэш
        if all_news:
            self._save_to_cache(all_news)

        # 5. Логируем результат
        total_time = time.time() - start_time
        logger.log('INFO',
                   f'Сбор новостей завершён: {len(all_news)} новостей '
                   f'({successful_sources}/{len(RSS_SOURCES)} источников, '
                   f'ошибок: {failed_sources}, время: {total_time:.1f}с)'
                   )

        return all_news


# Для отладки можно добавить
if __name__ == "__main__":
    fetcher = NewsFetcher()
    news = fetcher.get_last_news()
    print(f"Получено новостей: {len(news)}")
    if news:
        print(f"Пример новости: {news[0]['title'][:50]}...")