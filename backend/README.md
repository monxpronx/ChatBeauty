# Backend

LLM & RAG 기반 뷰티 제품 추천 시스템의 백엔드입니다.

## Architecture Overview

```
backend/
├── app/                          # FastAPI 애플리케이션 (API 서버)
│   ├── api/routes/               # API 엔드포인트
│   ├── services/                 # 비즈니스 로직 (retrieval, reranking, explanation)
│   ├── models/                   # Pydantic 스키마
│   └── main.py                   # FastAPI 앱 진입점
│
├── ml/                           # ML 파이프라인 (오프라인 학습/평가)
│   ├── data/                     # 데이터셋 (raw, processed, evaluation, chromadb)
│   ├── model/                    # 학습된 모델 저장소
│   ├── features/                 # Feature engineering (키워드 추출, 메타데이터 병합)
│   │   ├── extract_keywords.py   #   리뷰에서 키워드 추출 (vLLM)
│   │   └── merge_metadata.py     #   아이템 메타데이터 + 키워드 병합
│   ├── retriever/                # Retrieval 관련 스크립트
│   │   ├── keyword_based/        #   방법 A: 키워드/리뷰 텍스트 기반 fine-tuning
│   │   │   ├── create_training_pairs.py
│   │   │   └── finetune_bge_m3.py
│   │   ├── generated_query/      #   방법 B: LLM 생성 쿼리 기반 fine-tuning
│   │   │   ├── generate_query.py
│   │   │   ├── generate_query_vllm.py
│   │   │   ├── prepare_training_pairs.py
│   │   │   ├── finetune_bge-m3.py
│   │   │   ├── embed_queries.py
│   │   │   └── query_embedding.py
│   │   ├── save_to_chromadb.py   #   ChromaDB 임베딩 저장 (공통)
│   │   └── update_chromadb_metadata.py  # ChromaDB 메타데이터 업데이트 (공통)
│   ├── item_ranker/              # Re-ranking 모델 (LightGBM / XGBoost)
│   │   ├── dataset_lgbm.py       #   LightGBM 데이터셋
│   │   ├── dataset_xgb.py        #   XGBoost 데이터셋
│   │   ├── features_lgbm.py      #   LightGBM 피처 빌더
│   │   ├── features_xgb.py       #   XGBoost 피처 빌더
│   │   └── modeling/             #   학습/예측/추론
│   │       ├── train_lgbm.py
│   │       ├── train_xgb.py
│   │       ├── predict_lgbm.py
│   │       ├── predict_xgb.py
│   │       └── tree_reranker.py
│   ├── evaluation/               # 평가 스크립트
│   │   ├── retrieve_candidates.py
│   │   └── evaluate_recall.py
│   ├── scripts/                  # 실행 스크립트
│   │   ├── train_lgbm.py
│   │   ├── train_xgb.py
│   │   ├── evaluate_lgbm.py
│   │   ├── evaluate_xgb.py
│   │   └── inspect_feature_importance.py
│   └── utils/                    # 공통 유틸리티
│       └── llm_client.py
│
└── notebooks/                    # 실험용 Jupyter 노트북
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
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │  Retrieval  │→ │  Reranking  │→ │   Explanation   │ │
│  │  (Top-100)  │  │   (Top-5)   │  │  (LLM 생성)     │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────┘
       │                    │                    │
       ▼                    ▼                    ▼
   ChromaDB         LightGBM/XGBoost         LLM API
   (ml/data/)        (ml/model/)            (vLLM)
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

### 2. 서버 실행

```bash
# Frontend(React) + Backend(FastAPI) 동시 실행
./dev.sh
# Frontend: http://localhost:5173
# Backend:  http://localhost:8000

# Backend만 실행
cd backend
uvicorn app.main:app --reload
```

API 문서: http://localhost:8000/docs

### 3. ML 파이프라인 실행

#### Item Representation Pipeline (ml/features/)

```bash
# Step 1: 리뷰에서 키워드 추출 (vLLM, GPU 필요)
python ml/features/extract_keywords.py

# Step 2: 메타데이터 병합
python ml/features/merge_metadata.py BACKEND=vllm
```

#### Indexing (ml/retriever/)

```bash
# Step 3: ChromaDB에 임베딩 저장
python ml/retriever/save_to_chromadb.py

# Step 3-1: ChromaDB 메타데이터만 업데이트 (재임베딩 없이, GPU 불필요)
python ml/retriever/update_chromadb_metadata.py
```

#### Fine-tuning - 방법 A: Keyword-based (ml/retriever/keyword_based/)

```bash
# Step 4A: 학습 페어 생성
python ml/retriever/keyword_based/create_training_pairs.py QUERY_TYPE=both

# Step 5A: BGE-M3 fine-tuning
python ml/retriever/keyword_based/finetune_bge_m3.py EPOCHS=2 BATCH_SIZE=32
```

#### Fine-tuning - 방법 B: Generated Query (ml/retriever/generated_query/)

```bash
# Step 4B: LLM으로 쿼리 생성
python ml/retriever/generated_query/generate_query_vllm.py

# Step 5B: BGE-M3 fine-tuning
python ml/retriever/generated_query/finetune_bge-m3.py
```

#### ChromaDB 재구축 & 평가 (ml/evaluation/)

```bash
# Step 6: Fine-tuned 모델로 ChromaDB 재구축
python ml/retriever/save_to_chromadb.py MODEL_PATH=./ml/model/retrieval/bge-m3-finetuned-YYYYMMDD

# Step 7: 후보 추출
python ml/evaluation/retrieve_candidates.py MODEL_PATH=./ml/model/retrieval/bge-m3-finetuned-YYYYMMDD SPLIT=valid

# Step 8: Recall 평가
python ml/evaluation/evaluate_recall.py SPLIT=valid
```

#### Re-ranking (ml/item_ranker/)

```bash
# LightGBM
python ml/scripts/train_lgbm.py
python ml/scripts/evaluate_lgbm.py

# XGBoost
python ml/scripts/train_xgb.py
python ml/scripts/evaluate_xgb.py
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
- **LLM Backend**: vLLM (Llama 3.1:8B)
- **Reranker**: LightGBM / XGBoost

## Dataset

- **Amazon Reviews 2023 (All_Beauty)**
- 632k users / 112k items / 701k ratings
- Source: https://amazon-reviews-2023.github.io/
