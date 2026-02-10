# Backend

LLM & RAG 기반 뷰티 제품 추천 시스템의 백엔드입니다.

## Architecture Overview

```
backend/
├── app/                    # FastAPI 애플리케이션 (API 서버)
│   ├── api/routes/         # API 엔드포인트
│   ├── services/           # 비즈니스 로직 (retrieval, reranking, explanation)
│   ├── models/             # Pydantic 스키마
│   └── main.py             # FastAPI 앱 진입점
│
├── ml/                     # ML 파이프라인 (오프라인 학습/평가)
│   ├── data/               # 데이터셋 (raw, processed, evaluation)
│   ├── model/              # 학습된 모델 저장소
│   ├── retriever/          # Retrieval 관련 스크립트
│   ├── features/           # Feature engineering
│   ├── evaluation/         # 평가 스크립트
│   └── utils/              # 공통 유틸리티
│
└── notebooks/              # 실험용 Jupyter 노트북
    ├── eda/
    ├── retrieval_test/
    ├── reranking_test/
    └── explanation_test/
```

## System Flow

```
[User Scenario]
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│                    FastAPI (app/)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │  Retrieval  │→ │  Reranking  │→ │   Explanation   │  │
│  │  (Top-100)  │  │   (Top-5)   │  │  (LLM 생성)     │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────┘
       │                    │                    │
       ▼                    ▼                    ▼
   ChromaDB            LightGBM              LLM API
   (ml/data/)        (ml/model/)           (외부 서비스)
```

## Quick Start

### 1. 환경 설정

```bash
cd backend

# 가상환경 생성
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. API 서버 실행

```bash
uvicorn app.main:app --reload --port 8000
```

API 문서: http://localhost:8000/docs

### 3. ML 파이프라인 실행

```bash
cd ml

# 전체 파이프라인 (순서대로 실행)
python retriever/extract_keywords_with_llama.py BACKEND=vllm
python features/merge_metadata.py BACKEND=vllm
python retriever/save_to_chromadb.py
python retriever/create_training_pairs.py QUERY_TYPE=both
python retriever/finetune_bge_m3.py EPOCHS=2 BATCH_SIZE=32
python evaluation/retrieve_candidates.py MODEL_PATH=./model/retriever/bge-m3-finetuned-xxx SPLIT=valid
python evaluation/evaluate_recall.py SPLIT=valid
```

## Documentation

| 문서 | 설명 |
|------|------|
| [app/README.md](app/README.md) | API 서버 설정 및 엔드포인트 문서 |
| [ml/README.md](ml/README.md) | ML 파이프라인 상세 가이드 |

## Tech Stack

- **API Framework**: FastAPI
- **Embedding Model**: BAAI/bge-m3 (fine-tuned)
- **Vector Database**: ChromaDB
- **LLM Backend**: vLLM, Ollama (Llama 3.1:8B)
- **Reranker**: LightGBM (TODO)

## Dataset

- **Amazon Reviews 2023 (All_Beauty)**
- 632k users / 112k items / 701k ratings
- Source: https://amazon-reviews-2023.github.io/
