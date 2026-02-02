import json
import os
from typing import Dict
import numpy as np
from .query_embedding import get_query_embedding

def embed_queries_from_matched_jsonl(jsonl_path: str) -> Dict[str, np.ndarray]:
    """
    matched_query_item.jsonl에서 쿼리별 임베딩을 생성합니다.
    Args:
        jsonl_path (str): matched_query_item.jsonl 경로
    Returns:
        Dict[str, np.ndarray]: {query: embedding}
    """
    query_embeddings = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            query = data.get('query')
            if query:
                emb = get_query_embedding(query)
                query_embeddings[query] = emb
    return query_embeddings
