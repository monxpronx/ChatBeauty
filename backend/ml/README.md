# ML Pipeline

뷰티 제품 추천을 위한 ML 파이프라인입니다.

## Directory Structure

```
ml/
├── data/
│   ├── raw/                    # 원본 데이터
│   │   ├── All_Beauty.jsonl    # 리뷰 데이터 (701k)
│   │   └── meta_All_Beauty.jsonl # 메타데이터 (112k)
│   ├── processed/              # 전처리된 데이터
│   │   ├── keywords_{split}.jsonl
│   │   ├── items_for_embedding.jsonl
│   │   └── training_pairs.jsonl
│   ├── evaluation/             # 평가 결과
│   │   └── retrieval_candidates_{split}.jsonl
│   └── chromadb/               # 벡터 DB
│
├── model/
│   └── retriever/              # Fine-tuned 모델
│       └── bge-m3-finetuned-YYYYMMDD/
│
├── retriever/                  # Retrieval 파이프라인
│   ├── extract_keywords_with_llama.py
│   ├── create_training_pairs.py
│   ├── finetune_bge_m3.py
│   └── save_to_chromadb.py
│
├── features/                   # Feature Engineering
│   └── merge_metadata.py
│
├── evaluation/                 # 평가 스크립트
│   ├── retrieve_candidates.py
│   └── evaluate_recall.py
│
└── utils/                      # 공통 유틸리티
    └── llm_client.py           # Ollama/vLLM 클라이언트
```

## Pipeline Overview

```
[Raw Data]
     │
     ▼
┌─────────────────────────────────────────┐
│ Step 1: Extract Keywords (LLM)          │
│ extract_keywords_with_llama.py          │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ Step 2: Merge Metadata                  │
│ merge_metadata.py                       │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ Step 3: Save to ChromaDB                │
│ save_to_chromadb.py                     │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ Step 4: Create Training Pairs           │
│ create_training_pairs.py                │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ Step 5: Fine-tune BGE-M3                │
│ finetune_bge_m3.py                      │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ Step 6: Rebuild ChromaDB (Fine-tuned)   │
│ save_to_chromadb.py MODEL_PATH=...      │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ Step 7: Evaluate                        │
│ retrieve_candidates.py + evaluate_recall│
└─────────────────────────────────────────┘
```

## Commands

모든 명령어는 `backend/ml/` 디렉토리에서 실행합니다.

```bash
cd backend/ml
```

### Step 1: Extract Keywords

리뷰에서 키워드를 추출합니다 (WHO/WHEN/WHY).

```bash
# vLLM 사용 (권장)
python retriever/extract_keywords_with_llama.py BACKEND=vllm BATCH_SIZE=64

# Ollama 사용
python retriever/extract_keywords_with_llama.py BACKEND=ollama

# 옵션
#   MAX_ITEMS=1000      # 처리할 최대 리뷰 수
#   GPU=0,1             # 사용할 GPU 지정
```

**Output**: `data/processed/keywords_output.jsonl`

### Step 2: Merge Metadata

키워드와 제품 메타데이터를 병합하여 임베딩용 텍스트를 생성합니다.

```bash
python features/merge_metadata.py BACKEND=vllm BATCH_SIZE=64

# LLM 없이 실행 (description 요약 스킵)
python features/merge_metadata.py USE_LLM=false
```

**Output**: `data/processed/items_for_embedding.jsonl`

**Item Text Format**:
```
[Title] Product Name [Review Keywords] keyword1, keyword2 [Description Summary] summary [Features] feature1, feature2
```

### Step 3: Save to ChromaDB

아이템 임베딩을 생성하고 ChromaDB에 저장합니다.

```bash
# Base model 사용
python retriever/save_to_chromadb.py

# Fine-tuned model 사용
python retriever/save_to_chromadb.py MODEL_PATH=./model/retriever/bge-m3-finetuned-xxx

# 옵션
#   BATCH_SIZE=32       # 임베딩 배치 크기
#   USE_GPU=true        # GPU 사용 여부
```

**Output**: `data/chromadb/` (112k items)

### Step 4: Create Training Pairs

BGE-M3 fine-tuning을 위한 학습 데이터를 생성합니다.

```bash
python retriever/create_training_pairs.py QUERY_TYPE=both

# Query type 옵션
#   QUERY_TYPE=keywords     # 키워드만 사용
#   QUERY_TYPE=review_text  # 리뷰 텍스트 사용
#   QUERY_TYPE=both         # 둘 다 사용 (권장)
```

**Output**: `data/processed/training_pairs.jsonl` (~1M pairs)

### Step 5: Fine-tune BGE-M3

Contrastive learning으로 BGE-M3를 fine-tune합니다.

```bash
python retriever/finetune_bge_m3.py EPOCHS=2 BATCH_SIZE=32

# 옵션
#   EPOCHS=2            # 학습 에폭
#   BATCH_SIZE=32       # 배치 크기 (GPU 메모리에 따라 조정)
#   LR=2e-5             # Learning rate
#   USE_FP16=true       # Mixed precision training
```

**Output**: `model/retriever/bge-m3-finetuned-YYYYMMDD-HHMMSS/`

### Step 6: Rebuild ChromaDB

Fine-tuned 모델로 ChromaDB를 재구축합니다.

```bash
python retriever/save_to_chromadb.py MODEL_PATH=./model/retriever/bge-m3-finetuned-YYYYMMDD
```

### Step 7: Evaluate

Retrieval 성능을 평가합니다.

```bash
# 후보 검색
python evaluation/retrieve_candidates.py \
    MODEL_PATH=./model/retriever/bge-m3-finetuned-xxx \
    SPLIT=valid \
    QUERY_TYPE=review_text

# Recall 평가
python evaluation/evaluate_recall.py SPLIT=valid
```

**Metrics Output**:
```
========================================
Split: valid | Queries: 68,000
========================================
  Recall@1    0.0523
  Recall@5    0.1342
  Recall@10   0.1876
  Recall@20   0.2456
  Recall@50   0.3127
  Recall@100  0.3574
  MRR         0.1024
========================================
```

## Data Files

| File | Location | Records | Description |
|------|----------|---------|-------------|
| `All_Beauty.jsonl` | `data/raw/` | 701k | 원본 리뷰 데이터 |
| `meta_All_Beauty.jsonl` | `data/raw/` | 112k | 제품 메타데이터 |
| `keywords_{split}.jsonl` | `data/processed/` | 538k/68k/68k | LLM 추출 키워드 |
| `items_for_embedding.jsonl` | `data/processed/` | 112k | 임베딩용 아이템 텍스트 |
| `training_pairs.jsonl` | `data/processed/` | ~1M | Fine-tuning 학습 데이터 |
| `retrieval_candidates_{split}.jsonl` | `data/evaluation/` | varies | Top-100 검색 결과 |

## Data Splitting

시간 기반 분할 (sort_timestamp 기준):

| Split | Ratio | Usage |
|-------|-------|-------|
| Train | 80% | 모델 학습 |
| Valid | 10% | 하이퍼파라미터 튜닝 |
| Test | 10% | 최종 성능 평가 |

## Model Architecture

### Retrieval (Two-Tower)

- **Query Encoder**: Fine-tuned BGE-M3
- **Item Encoder**: Fine-tuned BGE-M3 (shared)
- **Loss**: MultipleNegativesRankingLoss (in-batch negatives)
- **Vector Dimension**: 1024

### Current Performance

| Metric | Value |
|--------|-------|
| Recall@100 | 0.3574 |
| MRR | 0.1024 |

## LLM Backend Configuration

### vLLM (권장 - GPU 환경)

```bash
python script.py BACKEND=vllm GPU=0,1 GPU_MEM=0.8
```

### Ollama (로컬 개발용)

```bash
# Ollama 서버 시작
ollama serve

# 모델 다운로드
ollama pull llama3.1:8b

# 스크립트 실행
python script.py BACKEND=ollama
```

## Dependencies

```
sentence-transformers>=2.2.0
chromadb
torch
tqdm
vllm  # GPU 환경
```
