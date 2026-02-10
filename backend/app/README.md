# App - FastAPI Application

뷰티 제품 추천 API 서버입니다.

## Directory Structure

```
app/
├── main.py              # FastAPI 앱 진입점
├── api/
│   └── routes/
│       └── recommend.py # 추천 API 엔드포인트
├── services/
│   ├── retrieval.py     # ChromaDB에서 후보 검색
│   ├── reranking.py     # 후보 재정렬 (LightGBM)
│   └── explanation.py   # LLM 기반 설명 생성
├── models/
│   └── schemas.py       # Pydantic 요청/응답 스키마
└── core/                # 설정 및 공통 유틸리티
```

## Quick Start

### 서버 실행

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

### API 문서

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### `POST /recommend`

사용자 시나리오를 입력받아 맞춤형 제품을 추천합니다.

**Request**

```json
{
  "user_input": "I have thin hair and looking for volumizing shampoo",
  "top_k": 5
}
```

**Response**

```json
{
  "recommendations": [
    {"item_id": "B001234567", "score": 0.95},
    {"item_id": "B002345678", "score": 0.89},
    ...
  ],
  "explanation": "추천 이유에 대한 설명..."
}
```

### `GET /`

Health check 엔드포인트

**Response**

```json
{"message": "hi"}
```

## Request/Response Schemas

### RecommendRequest

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `user_input` | string | (required) | 사용자의 상황/니즈 설명 |
| `top_k` | int | 5 | 반환할 추천 개수 |

### RecommendResponse

| Field | Type | Description |
|-------|------|-------------|
| `recommendations` | List[ItemScore] | 추천된 제품 목록 |
| `explanation` | string | 추천 이유 설명 |

### ItemScore

| Field | Type | Description |
|-------|------|-------------|
| `item_id` | string | 제품 ASIN |
| `score` | float | 추천 점수 (0~1) |

## Service Layer

### 1. Retrieval (`services/retrieval.py`)

ChromaDB에서 사용자 쿼리와 유사한 제품 후보를 검색합니다.

- Fine-tuned BGE-M3로 쿼리 인코딩
- Cosine similarity 기반 Top-K 검색
- 현재 Top-100 후보 반환

### 2. Reranking (`services/reranking.py`)

검색된 후보를 재정렬하여 최종 Top-K를 선정합니다.

- LightGBM LambdaRank 모델 (TODO)
- Features: retrieval score, price, rating, review count 등

### 3. Explanation (`services/explanation.py`)

최종 추천 제품에 대한 설명을 생성합니다.

- LLM을 사용한 RAG 기반 설명 생성
- 제품 메타데이터 + 리뷰 요약 활용

## Configuration

### CORS 설정

현재 `localhost:5173` (Vite dev server) 허용

```python
# main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    ...
)
```

프로덕션 배포 시 적절한 origin으로 변경 필요

## Dependencies

```
fastapi
uvicorn
pydantic
chromadb
sentence-transformers
```
