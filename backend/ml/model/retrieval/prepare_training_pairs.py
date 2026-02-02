import json
import random
import numpy as np
from typing import List, Tuple
from .embed_queries_from_matched_jsonl import embed_queries_from_matched_jsonl
from .load_item_embeddings import load_item_embeddings_from_chromadb

def prepare_training_pairs(
    matched_jsonl_path: str,
    chromadb_dir: str,
    collection_name: str = "items",
    n_neg_per_query: int = 100,
    seed: int = 42
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
    query_embeddings = embed_queries_from_matched_jsonl(matched_jsonl_path)
    # 2. item 임베딩
    item_embeddings = load_item_embeddings_from_chromadb(collection_name, chromadb_dir)
    # 3. positive 샘플 및 매핑 저장
    pairs = []
    query_to_pos_asin = {}
    all_items = set()
    with open(matched_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            query = data.get('query')
            item = data.get('item', {})
            asin = item.get('asin')
            if query in query_embeddings and asin in item_embeddings:
                pairs.append((query_embeddings[query], item_embeddings[asin], 1.0))
                query_to_pos_asin[query] = asin
                all_items.add(asin)
    # 4. negative 샘플 (각 query마다 n개, query 고정)
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
