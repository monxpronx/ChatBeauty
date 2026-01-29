import json
import numpy as np
import sys
import os
# embedding_bge_m3.py의 절대경로 import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from query_embedding import get_query_embedding

def embed_queries_from_jsonl(jsonl_path):
    """
    jsonl 파일에서 generate_query 컬럼을 읽어와 embedding합니다.
    Args:
        jsonl_path (str): jsonl 파일 경로
    Returns:
        list: (query, embedding) 튜플 리스트
    """
    results = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            query = data.get('generate_query')
            if query:
                emb = get_query_embedding(query)
                results.append((query, emb))
    return results

# 사용 예시 (테스트용)
if __name__ == "__main__":
    # 절대경로로 파일 지정 예시
    # 실제 파일 위치: workspace 최상위 폴더
    jsonl_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../generated_queries_train.jsonl"))
    # 10개 샘플링해서 임베딩
    sample_queries = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            query = data.get('generated_query')
            # 'INSUFFICIENT_INFO'가 아닌 쿼리만 샘플링 (5개)
            if query and query != "INSUFFICIENT_INFO":
                sample_queries.append(query)
                if len(sample_queries) >= 5:
                    break
    print(f"샘플 쿼리 {len(sample_queries)}개 임베딩 진행...")
    embeddings = [(q, get_query_embedding(q)) for q in sample_queries]
    for idx, (q, emb) in enumerate(embeddings):
        print(f"[{idx+1}] 쿼리: {q}")
        print(f"    임베딩 shape: {emb.shape}")
        print(f"    임베딩 벡터(앞 5개): {emb[:5]}")

    # 임베딩 벡터만 numpy 배열로 저장
    emb_array = np.stack([emb for _, emb in embeddings])
    npy_path = os.path.join(os.path.dirname(jsonl_file), "sample_query_embeddings.npy")
    np.save(npy_path, emb_array)
    print(f"임베딩 벡터 {emb_array.shape}를 {npy_path}에 저장 완료.")

    # 쿼리-임베딩 매핑을 jsonl로 저장 (임베딩은 list로 변환)
    out_jsonl = os.path.join(os.path.dirname(jsonl_file), "sample_query_embeddings.jsonl")
    with open(out_jsonl, 'w', encoding='utf-8') as f:
        for q, emb in embeddings:
            json.dump({"query": q, "embedding": emb.tolist()}, f, ensure_ascii=False)
            f.write("\n")
    print(f"쿼리-임베딩 매핑을 {out_jsonl}에 저장 완료.")