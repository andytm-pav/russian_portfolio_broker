# fetchers/news_fetcher.py
import feedparser, json, time, os
from utils.logger import logger
import json

with open("config/news_sources.json") as f:
    RSS = json.load(f)

class NewsFetcher:
    def __init__(self):
        self.cache_file = "data/news_cache.json"
        self.last_seen = set()

    def get_last_news(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                if time.time() - cache.get("ts", 0) < 60:
                    return cache["news"]
            except:
                pass

        all_news = []
        seen = set()
        for url in RSS:
            try:
                feed = feedparser.parse(url, request_headers={'User-Agent': 'Mozilla/5.0'})
                for e in feed.entries[:10]:
                    link = e.link
                    if link in self.last_seen or link in seen:
                        continue
                    seen.add(link)
                    title = e.title
                    published = e.get('published') or e.get('updated') or str(int(time.time()))
                    try:
                        ts = int(feedparser._parse_date(published).timestamp())
                    except:
                        ts = int(time.time())
                    all_news.append({
                        "title": title,
                        "link": link,
                        "ts": ts
                    })
            except:
                continue

        self.last_seen.update(seen)
        data = {"ts": int(time.time()), "news": all_news}
        os.makedirs("data", exist_ok=True)
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        logger.log('INFO', f'Новости обновлены: {len(all_news)}')
        return all_news