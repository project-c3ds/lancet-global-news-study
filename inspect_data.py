import gzip
import json

path = "data/world_news_premium/ft_com/crawls/crawl_09032026.jsonl.gz"

with gzip.open(path, "rt", encoding="utf-8") as f:
    line = f.readline()
    print(json.loads(line))
