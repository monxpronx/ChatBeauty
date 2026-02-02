import json
import random
import os
from tqdm import tqdm

# 파일 경로
QUERIES_PATH = '/data/ephemeral/home/pro-recsys-finalproject-recsys-07/backend/ml/data/processed/generated_queries_train.jsonl'
ITEMS_PATH = '/data/ephemeral/home/pro-recsys-finalproject-recsys-07/backend/ml/data/processed/items_for_embedding.jsonl'
OUT_PATH = '/data/ephemeral/home/pro-recsys-finalproject-recsys-07/backend/ml/data/processed/training_pairs.jsonl'
NEGATIVE_PER_QUERY = 100
NEGATIVE_PER_QUERY = 100

# 1. 아이템 임베딩 로드 (parent_asin별)

print('Loading item embeddings...')
item_embeddings = {}
item_asins_set = set()
with open(ITEMS_PATH, 'r', encoding='utf-8') as f:
    for line in tqdm(f, desc='Items'):
        item = json.loads(line)
        asin = item.get('asin')
        embedding = item.get('embedding')
        if asin and embedding:
            item_embeddings[asin] = embedding
            item_asins_set.add(asin)
all_asins = list(item_embeddings.keys())

# 2. 쿼리별 매칭 쌍 생성
print('Processing queries...')
num_pairs = 0
with open(QUERIES_PATH, 'r', encoding='utf-8') as fin, open(OUT_PATH, 'w', encoding='utf-8') as fout:
    for line in tqdm(fin, desc='Queries'):
        data = json.loads(line)
        query = data.get('generated_query')
        parent_asin = data.get('parent_asin')
        if not query or not parent_asin or not str(query).strip():
            continue
        if parent_asin in item_asins_set:
            pos_emb = item_embeddings[parent_asin]
            try:
                neg_asins = random.sample([a for a in all_asins if a != parent_asin], min(NEGATIVE_PER_QUERY, len(all_asins)-1))
                neg_embs = [item_embeddings[a] for a in neg_asins]
            except Exception as e:
                print(f"[NEGATIVE ERROR] parent_asin={parent_asin}, error={e}")
                continue
            out = {
                'query': query,
                'parent_asin': parent_asin,
                'positive': pos_emb,
                'negatives': neg_embs
            }
            fout.write(json.dumps(out, ensure_ascii=False) + '\n')
            num_pairs += 1
print(f'Created {num_pairs} training pairs. Saved to {OUT_PATH}')
