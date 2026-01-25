from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# llama 3.1 8B 모델명 (환경에 맞게 수정 필요)
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

# 모델과 토크나이저 로드 (최초 1회만 실행되도록 전역에서 로드)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
llama_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128)

def extract_aspects(query: str) -> list:
    """
    Llama 3.1 8B 모델을 이용해 query에서 aspect(속성, 특징 등)를 추출합니다.
    Args:
        query (str): 입력 쿼리
    Returns:
        list: 추출된 aspect 리스트
    """
    prompt = (
        f"Given the following query, extract all main aspects mentioned, including both the item's features/attributes and the situations, contexts, or purposes in which the item is used. "
        f"List all aspects (features, attributes, usage situations, contexts, or purposes) as comma-separated values.\n\nQuery: {query}\nAspects:"
    )
    result = llama_pipe(prompt)[0]['generated_text']
    aspects_line = result.split("Aspects:")[-1].strip().split('\n')[0]
    aspects = [a.strip() for a in aspects_line.split(',') if a.strip()]
    return aspects

# 사용 예시 (테스트용)
if __name__ == "__main__":
    test_query = "I want a laptop with long battery life, lightweight design, and a high-resolution display."
    print(extract_aspects(test_query))