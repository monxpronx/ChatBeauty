import requests
import os
from typing import List

def clova_embed_v2(texts: List[str], api_key: str, api_url: str = "https://clovastudio.naver.com/api/explorer/embedding/v2") -> list:
    """
    네이버 Clova Studio Explorer 임베딩 v2 API로 임베딩 벡터를 반환합니다.
    Args:
        texts (List[str]): 임베딩할 텍스트 리스트
        api_key (str): Clova Studio API 키
        api_url (str): API 엔드포인트 URL
    Returns:
        list: 임베딩 벡터 리스트 (각 텍스트별)
    """
    headers = {
        "X-NCP-APIGW-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    payload = {"texts": texts}
    response = requests.post(api_url, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()
    # API 응답 구조에 따라 수정 필요
    return result["embeddings"]

# 사용 예시
if __name__ == "__main__":
    api_key = os.getenv("CLOVA_API_KEY", "YOUR_API_KEY")
    test_texts = ["이 상품 정말 좋아요!", "배터리 오래가는 노트북 추천해줘"]
    embeddings = clova_embed_v2(test_texts, api_key)
    print(embeddings)
