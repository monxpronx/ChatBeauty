from transformers import AutoTokenizer, AutoModel
import torch

# BAAI의 bge-m3 모델명
MODEL_NAME = "BAAI/bge-m3"

# 토크나이저와 모델 로드 (최초 1회만 실행되도록 전역에서 로드)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def get_query_embedding(query: str):
    """
    BAAI의 bge-m3 모델을 이용해 query를 embedding합니다.
    Args:
        query (str): 입력 쿼리
    Returns:
        numpy.ndarray: 임베딩 벡터
    """
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        # [CLS] 토큰의 임베딩 사용 (첫 번째 토큰)
        embedding = outputs.last_hidden_state[:, 0, :]
        return embedding.squeeze().cpu().numpy()

# 사용 예시 (테스트용)
if __name__ == "__main__":
    test_query = "I want a laptop with long battery life."
    emb = get_query_embedding(test_query)
    print(emb)