import json

QUERIES_PATH = '/data/ephemeral/home/pro-recsys-finalproject-recsys-07/backend/ml/data/processed/generated_queries_train.jsonl'
ITEMS_PATH = '/data/ephemeral/home/pro-recsys-finalproject-recsys-07/backend/ml/data/processed/items_for_embedding.jsonl'

# 쿼리 parent_asin 전체 수집
query_asins = set()
with open(QUERIES_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        pa = data.get('parent_asin')
        if pa:
            query_asins.add(pa)

# 아이템 asin 전체 수집
item_asins = set()
with open(ITEMS_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        a = data.get('asin')
        if a:
            item_asins.add(a)

# 교집합
intersection = query_asins & item_asins
print(f"쿼리 parent_asin 전체: {len(query_asins)}개, 아이템 asin 전체: {len(item_asins)}개")
print(f"교집합 개수: {len(intersection)}")
if intersection:
    print(f"매칭되는 값: {list(intersection)[:10]}")
else:
    print("매칭되는 값 없음")
