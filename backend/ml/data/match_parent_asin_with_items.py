import json

query_path = 'backend/ml/data/processed/generated_queries_train_non_insufficient.jsonl'
item_path = 'backend/ml/data/processed/items_for_embedding.jsonl'
out_path = 'backend/ml/data/processed/matched_query_item.jsonl'

dict_items = {}
with open(item_path, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        asin = item.get('asin')
        if asin:
            dict_items[asin] = item

count = 0
with open(query_path, 'r', encoding='utf-8') as fin, open(out_path, 'w', encoding='utf-8') as fout:
    for line in fin:
        data = json.loads(line)
        parent_asin = data.get('parent_asin')
        if parent_asin and parent_asin in dict_items:
            query = data.get('generated_query')
            item = dict_items[parent_asin]
            fout.write(json.dumps({'query': query, 'item': item}, ensure_ascii=False) + '\n')
            count += 1
print(f'완료: {out_path}, {count}개 매칭')