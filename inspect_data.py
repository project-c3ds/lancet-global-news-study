import json

path = "data/newsapi_articles/theguardian.com.jsonl"

with open(path, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

print(f"Loaded {len(records)} records")
print(records[0])
