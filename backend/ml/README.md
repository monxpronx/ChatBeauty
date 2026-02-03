# ML Pipeline

This directory contains scripts for the 2-stage recommendation pipeline:
1. **Stage 1 (Retrieval)**: Fine-tuned BGE-M3 + ChromaDB for candidate extraction
2. **Stage 2 (Re-ranking)**: LightGBM/XGBoost for final ranking (TODO)

## Pipeline Overview

```
Raw Data                          Processed Data                              Model Training & Inference
────────                          ──────────────                              ──────────────────────────

All_Beauty.jsonl ──[Split]──→ reviews_train.jsonl ──[Step 1]──→ keywords_train.jsonl ─┐
    (701k reviews)             reviews_valid.jsonl              keywords_valid.jsonl  │
                               reviews_test.jsonl               keywords_test.jsonl   │
                                                                                       │
                                                                                       ├──[Step 2]──→ items_for_embedding.jsonl
meta_All_Beauty.jsonl ─────────────────────────────────────────────────────────────────┘               (112k items)
    (112k items)                                                                                              │
                                                                                                              ↓
                                                                                    [Step 3: Embed] ──→ ChromaDB (base model)
                                                                                                              │
                                                                       ┌──────────────────────────────────────┘
                                                                       │
training_pairs.jsonl ←──[Step 4]── keywords_train.jsonl + items_for_embedding.jsonl
    (~1M pairs with QUERY_TYPE=both)               │
                                                    ↓
                                    [Step 5: Fine-tune BGE-M3] ──→ models/bge-m3-finetuned-YYYYMMDD-HHMMSS/
                                                                              │
                                                                              ↓
                                                    [Step 6: Rebuild ChromaDB] ──→ ChromaDB (fine-tuned)
                                                                                          │
                                                                                          ↓
                                                    [Step 7: Retrieve Candidates] ──→ retrieval_candidates_{split}.jsonl
                                                                                          │
                                                    ┌─────────────────────────────────────┴─────────────────────────────────────┐
                                                    │                                                                           │
                                          SPLIT=valid/test                                                            SPLIT=train
                                                    │                                                                           │
                                                    ↓                                                                           ↓
                                    [Step 8: Evaluate Recall] ──→ Recall@K, MRR                         [Step 9: Train Re-ranker] (TODO)
                                                                                                                     │
                                                                                                                     ↓
                                                                                                        LightGBM/XGBoost model
```

## Key Concepts

### Data Splitting (Time-based)

Reviews are split by `timestamp` to simulate real-world deployment:
- **Train (80%)**: timestamp ≤ 1621867671638 (~Jun 2021) — used for item embeddings
- **Valid (10%)**: timestamp ≤ 1643212573029 (~Jan 2022) — for hyperparameter tuning
- **Test (10%)**: newest reviews — for final evaluation

### parent_asin vs asin

- `asin`: Variant-specific ID (e.g., red vs blue version)
- `parent_asin`: Product-level ID that groups all variants

The pipeline uses `parent_asin` to:
1. Aggregate keywords from all variants' reviews
2. Match reviews to metadata (metadata is keyed by `parent_asin`)
3. Create one embedding per product (not per variant)

## Data Files

| File | Location | Records | Description |
|------|----------|---------|-------------|
| `All_Beauty.jsonl` | `data/raw/` | 701,528 | Raw user reviews |
| `meta_All_Beauty.jsonl` | `data/raw/` | 112,590 | Product metadata |
| `reviews_train.jsonl` | `data/processed/` | 561,222 | Train split reviews |
| `reviews_valid.jsonl` | `data/processed/` | 70,152 | Validation split reviews |
| `reviews_test.jsonl` | `data/processed/` | 70,154 | Test split reviews |
| `keywords_train.jsonl` | `data/processed/` | 538,013 | Keywords from train reviews (with `parent_asin`) |
| `keywords_valid.jsonl` | `data/processed/` | 68,034 | Keywords from valid reviews |
| `keywords_test.jsonl` | `data/processed/` | 68,108 | Keywords from test reviews |
| `description_summaries_cache.jsonl` | `data/processed/` | 17,443 | LLM-summarized descriptions |
| `items_for_embedding.jsonl` | `data/processed/` | 112,589 | Final merged items for embedding |
| `training_pairs.jsonl` | `data/processed/` | ~1M (with QUERY_TYPE=both) | (query, positive) pairs for fine-tuning |
| ChromaDB | `data/chromadb/` | 112k items | Vector database with item embeddings |
| `retrieval_candidates_{split}.jsonl` | `data/evaluation/` | varies | Top-100 candidates per query for evaluation |

## Running the Pipeline

### Prerequisites

```bash
cd backend

# For LLM processing (Ollama or vLLM)
pip install requests tqdm vllm

# For embeddings and fine-tuning
pip install chromadb FlagEmbedding sentence-transformers torch
```

### Step 0: Split Data by Timestamp

```bash
# Already done - creates reviews_*.jsonl and keywords_*.jsonl files
# Timestamp cutoffs: train_end=1621867671638, valid_end=1643212573029
```

### Step 1: Extract Keywords from Reviews

Extract contextual keywords using LLaMA 3.1 (WHO/WHEN/WHY the product is useful).

```bash
# Ollama (default)
python ml/03_retriever/extract_keywords_with_llama.py

# vLLM (faster, requires GPU)
python ml/03_retriever/extract_keywords_with_llama.py BACKEND=vllm BATCH_SIZE=64

# Test with limited items
python ml/03_retriever/extract_keywords_with_llama.py BACKEND=vllm MAX_ITEMS=100
```

**Note**: After extraction, add `parent_asin` by joining with reviews file (see data preparation scripts).

### Step 2: Merge Metadata

Combines keywords + metadata + description summaries into final item representations.

```bash
# With description summarization (uses cached summaries if available)
python ml/02_features/merge_metadata.py BACKEND=vllm

# Skip LLM (use cached summaries only)
python ml/02_features/merge_metadata.py USE_LLM=false
```

**Output format** (items_for_embedding.jsonl):
```
[Title] Product Name [Review Keywords] kw1, kw2, ... [Description Summary] summary1, summary2 [Features] feat1, feat2
```

### Step 3: Save to ChromaDB (Base Model)

Generate embeddings with BGE-M3 and store in ChromaDB.

```bash
# Base BGE-M3 model
python ml/03_retriever/save_to_chromadb.py

# With options
python ml/03_retriever/save_to_chromadb.py BATCH_SIZE=64 USE_GPU=true
```

### Step 4: Create Training Pairs

Generate (query, positive_document) pairs for fine-tuning. The query represents user language (keywords or raw review text), and the positive is the item's embedding_text.

```bash
# Keywords only (default)
python ml/03_retriever/create_training_pairs.py

# Raw review text as query (recommended for better recall)
python ml/03_retriever/create_training_pairs.py QUERY_TYPE=review_text

# Both keywords and review text (~2x training pairs, best results)
python ml/03_retriever/create_training_pairs.py QUERY_TYPE=both
```

**Query Types**:
- `keywords`: LLM-extracted keywords from reviews (e.g., "hair, spray, texture, beachy waves")
- `review_text`: Raw review text as query (closer to real user input)
- `both`: Creates pairs for both, doubling training data

**Output** (training_pairs.jsonl):
```json
{
  "query": "I bought this for my daughter who has thin hair...",
  "positive": "[Title] Herbivore Sea Mist... [Review Keywords] ...",
  "parent_asin": "B00YQ6X8EO"
}
```

### Step 5: Fine-tune BGE-M3

Fine-tune using sentence-transformers with MultipleNegativesRankingLoss. In-batch negatives are used for contrastive learning.

```bash
# Recommended settings for V100 32GB with QUERY_TYPE=both training data
nohup python ml/03_retriever/finetune_bge_m3.py EPOCHS=2 BATCH_SIZE=32 EVAL_STEPS=10000 > finetune_bge_m3.log 2>&1 &

# Basic training (smaller GPU)
python ml/03_retriever/finetune_bge_m3.py BATCH_SIZE=16

# Lower batch size if GPU OOM (review text queries are longer than keywords)
python ml/03_retriever/finetune_bge_m3.py BATCH_SIZE=8
```

**GPU Memory Notes**:
- BATCH_SIZE=64 may OOM on V100-32GB when using review text queries (longer sequences)
- BATCH_SIZE=32 is safe for most setups
- Larger batch = more in-batch negatives = better contrastive learning

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EPOCHS` | 1 | Number of training epochs |
| `BATCH_SIZE` | 16 | Batch size (larger = more in-batch negatives) |
| `LR` | 2e-5 | Learning rate |
| `USE_FP16` | true | Mixed precision training |
| `MAX_QUERY_LENGTH` | 256 | Max query tokens |
| `MAX_DOC_LENGTH` | 512 | Max document tokens |
| `EVAL_STEPS` | 1000 | Evaluate every N steps |

### Step 6: Rebuild ChromaDB with Fine-tuned Model

After fine-tuning, rebuild ChromaDB with the new model to update item embeddings.

```bash
python ml/03_retriever/save_to_chromadb.py MODEL_PATH=./models/bge-m3-finetuned-YYYYMMDD-HHMMSS
```

## Evaluation

### Metrics

- **Stage 1 (Retrieval)**: Recall@K, MRR — Did the correct item appear in top K candidates?
- **Stage 2 (Re-ranking)**: NDCG@5 — How well ranked are the final 5 recommendations?

### Current Results (Fine-tuned BGE-M3, QUERY_TYPE=review_text, 2 epochs)

| Metric | Valid Set |
|--------|-----------|
| Recall@1 | 0.0217 |
| Recall@5 | 0.0730 |
| Recall@10 | 0.1115 |
| Recall@20 | 0.1642 |
| Recall@50 | 0.2576 |
| Recall@100 | 0.3574 |
| MRR | 0.0543 |

**Note**: Recall@100=0.36 means the correct item appears in top 100 for 36% of queries. This is the input for the re-ranker stage.

### Evaluation Protocol

1. Take a review from valid/test set as a simulated user query
2. Use raw review text as query (or keywords)
3. Encode query with fine-tuned BGE-M3
4. Retrieve top-100 items from ChromaDB
5. Check if the actual purchased item (`parent_asin`) is in the results

### Step 8: Evaluate Retrieval Quality

```bash
# First, retrieve candidates
python ml/04_evaluation/retrieve_candidates.py MODEL_PATH=./models/bge-m3-finetuned-YYYYMMDD SPLIT=valid QUERY_TYPE=review_text

# Then evaluate
python ml/04_evaluation/evaluate_recall.py SPLIT=valid
```

## Command Reference

### extract_keywords_with_llama.py

| Option | Default (Ollama) | Default (vLLM) | Description |
|--------|------------------|----------------|-------------|
| `BACKEND` | `ollama` | - | `ollama` or `vllm` |
| `MODEL` | `llama3.1:8b` | `meta-llama/Llama-3.1-8B-Instruct` | Model name |
| `GPU` | - | all available | GPU IDs (e.g., `GPU=0,1`) |
| `BATCH_SIZE` | 1 | 64 | Batch size |
| `MAX_ITEMS` | all | all | Limit items to process |

### merge_metadata.py

nohup python ml/02_features/merge_metadata.py BACKEND=vllm > merge_metadata.log 2>&1 &

# Skip LLM summarization (just merge existing data)
python ml/02_features/merge_metadata.py USE_LLM=false
```
| Option | Default | Description |
|--------|---------|-------------|
| `BACKEND` | `ollama` | `ollama` or `vllm` |
| `USE_LLM` | `true` | Set `false` to skip description summarization |
| `BATCH_SIZE` | 1 (ollama) / 64 (vllm) | Batch size |

### save_to_chromadb.py

| Option | Default | Description |
|--------|---------|-------------|
| `MODEL_PATH` | `BAAI/bge-m3` | Path to model (base or fine-tuned) |
| `BATCH_SIZE` | 32 | Embedding batch size |
| `USE_GPU` | `true` | Use GPU for encoding |

### create_training_pairs.py

| Option | Default | Description |
|--------|---------|-------------|
| `OUTPUT_FORMAT` | `pair` | `pair` (query, positive) or `triplet` (+ negative) |
| `QUERY_TYPE` | `keywords` | `keywords`, `review_text`, or `both` |
| `MAX_KEYWORDS` | 20 | Max keywords in query |
| `INCLUDE_REVIEW_TEXT` | `false` | Also use raw review text as query variant (overridden by QUERY_TYPE) |

### finetune_bge_m3.py

| Option | Default | Description |
|--------|---------|-------------|
| `EPOCHS` | 1 | Training epochs |
| `BATCH_SIZE` | 16 | Batch size |
| `LR` | 2e-5 | Learning rate |
| `USE_FP16` | `true` | Mixed precision |
| `EVAL_STEPS` | 1000 | Evaluation frequency |

### retrieve_candidates.py

Retrieve top-K candidates from ChromaDB for a given data split using the fine-tuned BGE-M3 model.

```bash
# Retrieve candidates for test split (default)
python ml/04_evaluation/retrieve_candidates.py MODEL_PATH=./models/bge-m3-finetuned-20260128-174129

# Retrieve candidates for train split (for re-ranker training data)
python ml/04_evaluation/retrieve_candidates.py MODEL_PATH=./models/bge-m3-finetuned-20260128-174129 SPLIT=train

# With options
python ml/04_evaluation/retrieve_candidates.py MODEL_PATH=./models/bge-m3-finetuned-20260128-174129 SPLIT=test TOP_K=100 BATCH_SIZE=256
```

| Option | Default | Description |
|--------|---------|-------------|
| `MODEL_PATH` | (required) | Path to fine-tuned BGE-M3 model |
| `SPLIT` | `test` | Data split: `train`, `valid`, or `test` |
| `QUERY_TYPE` | `keywords` | `keywords` or `review_text` |
| `TOP_K` | 100 | Number of candidates to retrieve per query |
| `BATCH_SIZE` | 256 | Query encoding batch size |
| `MAX_KEYWORDS` | 20 | Max keywords per query |
| `COLLECTION` | `beauty_products` | ChromaDB collection name |

**Output**: `data/evaluation/retrieval_candidates_{SPLIT}.jsonl`

### evaluate_recall.py

Compute Recall@K and MRR from retrieval candidates.

```bash
python ml/04_evaluation/evaluate_recall.py SPLIT=valid
python ml/04_evaluation/evaluate_recall.py SPLIT=test
```

| Option | Default | Description |
|--------|---------|-------------|
| `SPLIT` | `test` | Data split to evaluate |

**Output**: Prints Recall@1,5,10,20,50,100 and MRR to console.

## Directory Structure

```
backend/ml/
├── 02_features/
│   └── merge_metadata.py          # Merge keywords + metadata → items_for_embedding
├── 03_retriever/
│   ├── extract_keywords_with_llama.py  # Extract keywords from reviews
│   ├── create_training_pairs.py        # Generate fine-tuning pairs
│   ├── finetune_bge_m3.py              # Fine-tune BGE-M3
│   └── save_to_chromadb.py             # Generate embeddings → ChromaDB
├── 04_evaluation/
│   ├── retrieve_candidates.py         # Retrieve top-K candidates from ChromaDB
│   └── evaluate_recall.py             # Compute Recall@K and MRR
├── 04_item_ranker/
│   ├── dataset.py                     # Data classes for re-ranking (TODO)
│   └── features.py                    # Feature builder for re-ranking (TODO)
├── utils/
│   └── llm_client.py              # LLM backend abstraction (Ollama/vLLM)
└── README.md
```
========================================
Split: valid | Queries: 65,062
========================================
  Recall@1    0.0207  (1,344/65,062)
  Recall@5    0.0526  (3,421/65,062)
  Recall@10   0.0762  (4,957/65,062)
  Recall@20   0.1079  (7,022/65,062)
  Recall@50   0.1649  (10,726/65,062)
  Recall@100  0.2224  (14,470/65,062)
  MRR        0.0395
========================================

========================================
Split: test | Queries: 65,079
========================================
  Recall@1    0.0269  (1,749/65,079)
  Recall@5    0.0668  (4,346/65,079)
  Recall@10   0.0944  (6,146/65,079)
  Recall@20   0.1292  (8,410/65,079)
  Recall@50   0.1878  (12,225/65,079)
  Recall@100  0.2433  (15,836/65,079)
  MRR        0.0495
========================================