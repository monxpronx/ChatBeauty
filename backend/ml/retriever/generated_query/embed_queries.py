import json
import os
from typing import Dict
import numpy as np
import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from query_embedding import get_query_embedding

def embed_queries_from_matched_jsonl(jsonl_path: str, cache_file: str = None) -> Dict[str, np.ndarray]:
    """
    matched_query_item.jsonl에서 쿼리별 임베딩을 생성합니다.
    Args:
        jsonl_path (str): matched_query_item.jsonl 경로
        cache_file (str): 캐시 파일 경로 (선택사항)
    Returns:
        Dict[str, np.ndarray]: {query: embedding}
    """
    # 캐시 파일이 있으면 로드
    if cache_file and os.path.exists(cache_file):
        print(f"   Loading cached embeddings from {cache_file}...", flush=True)
        with open(cache_file, 'rb') as f:
            query_embeddings = pickle.load(f)
        print(f"   Loaded {len(query_embeddings)} cached queries", flush=True)
        return query_embeddings
    
    print(f"   Reading queries from {jsonl_path}...", flush=True)
    query_embeddings = {}
    error_count = 0
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                query = data.get('query')
                if query:
                    if i % 100 == 0:
                        print(f"   Processing query {i}... ({len(query_embeddings)} unique queries)", flush=True)
                    emb = get_query_embedding(query)
                    query_embeddings[query] = emb
            except json.JSONDecodeError as e:
                error_count += 1
                if error_count <= 5:  # 처음 5개만 출력
                    print(f"   WARNING: Skipping line {i+1} due to JSON error: {str(e)}", flush=True)
                continue
    
    if error_count > 5:
        print(f"   Total {error_count} lines skipped due to JSON errors", flush=True)
    
    # 캐시 파일에 저장
    if cache_file:
        cache_path = Path(cache_file)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"   Saving embeddings to cache: {cache_file}...", flush=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(query_embeddings, f)
        print(f"   Cache saved successfully", flush=True)
    
    return query_embeddings
