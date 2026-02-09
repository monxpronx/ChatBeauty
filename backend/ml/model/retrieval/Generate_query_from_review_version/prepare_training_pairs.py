import json
import random
import numpy as np
from typing import List, Tuple
from embed_queries_from_matched_jsonl import embed_queries_from_matched_jsonl
from load_item_embeddings import load_item_embeddings_from_chromadb

def prepare_training_pairs(
    matched_jsonl_path: str,
    chromadb_dir: str,
    collection_name: str = "items",
    n_neg_per_query: int = 100,
    seed: int = 42,
    cache_dir: str = None
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    (query 임베딩, item 임베딩, label) 쌍으로 positive/negative 샘플 생성
    Args:
        matched_jsonl_path (str): matched_query_item.jsonl 경로
        chromadb_dir (str): ChromaDB 디렉토리 경로
        collection_name (str): ChromaDB 컬렉션 이름
        negative_ratio (float): negative 샘플 비율 (1:1)
        seed (int): 랜덤 시드
    Returns:
        List[Tuple[np.ndarray, np.ndarray, float]]: (query_emb, item_emb, label)
    """
    random.seed(seed)
    # 1. 쿼리 임베딩
    print("1. Loading query embeddings...", flush=True)
    
    # 캐시 파일 경로 설정
    cache_file = None
    if cache_dir:
        from pathlib import Path
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        cache_file = str(cache_path / "query_embeddings.pkl")
    
    query_embeddings = embed_queries_from_matched_jsonl(matched_jsonl_path, cache_file=cache_file)
    print(f"   Loaded {len(query_embeddings)} queries", flush=True)
    # 2. item 임베딩
    print("2. Loading item embeddings from ChromaDB...", flush=True)
    item_embeddings = load_item_embeddings_from_chromadb(collection_name, chromadb_dir)
    print(f"   Loaded {len(item_embeddings)} items", flush=True)
    # 3. positive 샘플 및 매핑 저장
    print("3. Creating positive training pairs...", flush=True)
    pairs = []
    query_to_pos_asin = {}
    all_items = set()
    error_count = 0
    with open(matched_jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                query = data.get('query')
                item = data.get('item', {})
                asin = item.get('asin')
                if query in query_embeddings and asin in item_embeddings:
                    pairs.append((query_embeddings[query], item_embeddings[asin], 1.0))
                    query_to_pos_asin[query] = asin
                    all_items.add(asin)
            except json.JSONDecodeError:
                error_count += 1
                continue
    
    if error_count > 0:
        print(f"   Skipped {error_count} lines due to JSON errors", flush=True)
    print(f"   Created {len(pairs)} positive pairs", flush=True)
    
    # 4. negative 샘플 (각 query마다 n개, query 고정)
    print("4. Creating negative training pairs...", flush=True)
    all_items = list(all_items)
    for query, pos_asin in query_to_pos_asin.items():
        neg_candidates = [asin for asin in all_items if asin != pos_asin]
        for _ in range(n_neg_per_query):
            if not neg_candidates:
                break
            neg_asin = random.choice(neg_candidates)
            if query in query_embeddings and neg_asin in item_embeddings:
                pairs.append((query_embeddings[query], item_embeddings[neg_asin], 0.0))
    return pairs
