import os
import json
import re
import requests
from dotenv import load_dotenv

load_dotenv()

def require_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"환경 변수 {key} 가 설정되지 않았습니다.")
    return value

CLOVA_URL = os.getenv("CLOVA_URL")

HEADERS = {
    "X-NCP-CLOVASTUDIO-REQUEST-ID": os.getenv("CLOVA_REQUEST_ID"),
    "Authorization": f"Bearer {os.getenv('CLOVA_API_KEY')}",
    "Content-Type": "application/json",
    "Accept": "application/json"
}
SYSTEM_PROMPT = """
너는 쇼핑 추천 전문가야. 리랭킹된 상품 리스트와 사용자의 검색어를 분석하여, 사용자에게 이 상품이 왜 추천되었는지 '데이터에 기반해' 설명하는 역할을 수행한다.

[중요 규칙]
- 제공된 정보(상품명, 가격, 평점, 카테고리 등) 내에서만 설명할 것. 추측 금지.
- '알고리즘', '리랭킹', '점수', '임베딩' 등 기술적 용어 절대 사용 금지.
- 마케팅 수식어(최고의, 환상적인 등)를 배제하고 객관적인 사실 위주로 작성할 것.
- 실제 구매자 리뷰(top_reviews)를 인용하여 신뢰도를 높일 것
- 각 상품의 설명은 1~2문장으로 간결하게 작성할 것.
- 5개에 상품에 대한 설명을 요청하니 explanation은 5개가 꼭 응답되어야함

[설명 가이드라인]
1. 사용자의 검색 키워드가 상품명의 어느 부분과 일치하는지 언급.
2. 실제 리뷰 내용을 인용하여 근거 제시

[출력 형식]
- 반드시 아래 JSON 구조로만 응답하고, 다른 텍스트는 포함하지 말 것.
- 모든 설명(explanation)은 반드시 한국어로 작성할 것

{
  "explanations": [
    {
      "item_id": "상품 ID",
      "explanation": "추천 이유 설명"
    }
  ]
}
"""


def generate_explanation(explanation_input: dict) -> dict:
    request_data = {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": json.dumps(explanation_input, ensure_ascii=False)}]
            }
        ],
        "temperature": 0.2,
        "maxTokens": 1500,
        "topP": 0.8,
        "topK": 0,
        "repetitionPenalty": 1.1,
        "stop": [],
        "seed": 0
    }

    try:
        response = requests.post(
            CLOVA_URL,
            headers=HEADERS,
            json=request_data,
            timeout=30
        )

        if response.status_code == 200:
            content = response.json()
            raw_llm_text = content.get("result", {}).get("message", {}).get("content", "")
            
            clean_json_str = re.sub(r'```json|```', '', raw_llm_text).strip()
            
            try:
                parsed_json = json.loads(clean_json_str)
                
                return parsed_json
            except Exception as e:
                print(f"[JSON 파싱 실패] 에러: {e}")
                return {"explanations": [{"item_id": "all", "explanation": clean_json_str}]}
        
        return {"explanations": []}

    except Exception as e:
        print(f"Error: {e}")
        return {"explanations": []}