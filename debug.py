# debug.py — ЗАПУСТИ ЭТО ОТДЕЛЬНО
from fetchers.news_fetcher import NewsFetcher
from fetchers.moex_fetcher import MoexFetcher

print("=== НОВОСТИ ===")
news = NewsFetcher().get_last_news()
for n in news[:3]:
    print(f"{n['ts']} | {n['title']}")

print("\n=== ЦЕНЫ ===")
moex = MoexFetcher()
print("SBER:", moex.get_price("SBER"))
print("GAZP:", moex.get_price("GAZP"))