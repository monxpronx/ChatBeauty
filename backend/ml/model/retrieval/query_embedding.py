

# FlagEmbedding 라이브러리 사용
import torch
from FlagEmbedding import BGEM3FlagModel

# BAAI의 bge-m3 모델명 (FlagEmbedding)
MODEL_NAME = "BAAI/bge-m3"

# 모델 로드 (최초 1회만 실행되도록 전역에서 로드)
model = BGEM3FlagModel(MODEL_NAME, use_fp16=True, device="cuda" if torch.cuda.is_available() else "cpu")

def get_query_embedding(query: str):
    """
    BAAI의 bge-m3 FlagEmbedding 모델을 이용해 query를 embedding합니다.
    Args:
        query (str): 입력 쿼리
    Returns:
        numpy.ndarray: 임베딩 벡터
    """
    # encode 함수는 dict 반환, 'dense_vecs' 키에 임베딩 벡터가 있음
    emb = model.encode([query])["dense_vecs"][0]
    return emb

# 사용 예시 (테스트용)
if __name__ == "__main__":
    test_query = "I want a laptop with long battery life."
    emb = get_query_embedding(test_query)
    print(emb)