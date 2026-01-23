# ML Pipeline

This directory contains scripts for the retrieval pipeline: keyword extraction, metadata merging, and embedding generation.

## Pipeline Overview

```
All_Beauty.jsonl → [Step 1: Extract Keywords] → keywords_output.jsonl
                                                        ↓
meta_All_Beauty.jsonl → [Step 2: Merge Metadata] → items_for_embedding.jsonl
                                                        ↓
                                    [Step 3: Save to ChromaDB] → ChromaDB
```

## Backend Options

The pipeline supports two LLM backends:

| Backend | Description | GPU Required |
|---------|-------------|--------------|
| `ollama` | HTTP API to local Ollama server | No (Ollama handles) |
| `vllm` | Direct vLLM with auto GPU detection | Yes |

## Running the Pipeline

### Option 1: Ollama (Default)

Requires Ollama running locally with `llama3.1:8b` model.

```bash
# Start Ollama server first
ollama serve

# In another terminal, run the pipeline
cd backend

# Step 1: Extract keywords from reviews
python ml/03_retriever/extract_keywords_with_llama.py

# Step 2: Merge metadata and summarize descriptions
python ml/02_features/merge_metadata.py

# Step 3: Generate embeddings and save to ChromaDB
python ml/03_retriever/save_to_chromadb.py
```

### Option 2: vLLM (GPU)

Automatically detects available GPUs and loads the model directly.

```bash
cd backend

# Step 1: Extract keywords (batched for speed)
python ml/03_retriever/extract_keywords_with_llama.py BACKEND=vllm

# Step 2: Merge metadata (batched for speed)
python ml/02_features/merge_metadata.py BACKEND=vllm

# Step 3: Generate embeddings and save to ChromaDB
python ml/03_retriever/save_to_chromadb.py
```

## Command Options

### extract_keywords_with_llama.py

| Option | Default (Ollama) | Default (vLLM) | Description |
|--------|------------------|----------------|-------------|
| `BACKEND` | `ollama` | - | Backend to use: `ollama` or `vllm` |
| `MODEL` | `llama3.1:8b` | `meta-llama/Llama-3.1-8B-Instruct` | Model name |
| `GPU` | - | all available | Specific GPU IDs (e.g., `GPU=0,1`) |
| `BATCH_SIZE` | 1 | 32 | Batch size for processing |
| `DELAY` | 0.5 | 0.0 | Delay between requests (seconds) |
| `MAX_ITEMS` | all | all | Limit number of items to process |

**Examples:**
```bash
# vLLM with specific GPUs
python ml/03_retriever/extract_keywords_with_llama.py BACKEND=vllm GPU=0,1

# vLLM with custom model and batch size
python ml/03_retriever/extract_keywords_with_llama.py BACKEND=vllm MODEL=meta-llama/Llama-3.1-8B-Instruct BATCH_SIZE=64

# Test with limited items
python ml/03_retriever/extract_keywords_with_llama.py BACKEND=vllm MAX_ITEMS=100
```

### merge_metadata.py

| Option | Default (Ollama) | Default (vLLM) | Description |
|--------|------------------|----------------|-------------|
| `BACKEND` | `ollama` | - | Backend to use: `ollama` or `vllm` |
| `MODEL` | `llama3.1:8b` | `meta-llama/Llama-3.1-8B-Instruct` | Model name |
| `GPU` | - | all available | Specific GPU IDs (e.g., `GPU=0,1`) |
| `BATCH_SIZE` | 1 | 32 | Batch size for processing |
| `DELAY` | 0.3 | 0.0 | Delay between requests (seconds) |
| `USE_LLM` | `true` | `true` | Set to `false` to skip description summarization |

**Examples:**
```bash
# vLLM with specific GPUs
python ml/02_features/merge_metadata.py BACKEND=vllm GPU=0,1

# Skip LLM summarization (just merge existing data)
python ml/02_features/merge_metadata.py USE_LLM=false
```

### save_to_chromadb.py

| Option | Default | Description |
|--------|---------|-------------|
| (none) | - | Uses BGE-M3 model, auto-detects GPU |

**Examples:**
```bash
# Run with default settings
python ml/03_retriever/save_to_chromadb.py
```

## Output Files

| File | Location | Description |
|------|----------|-------------|
| `keywords_output.jsonl` | `data/processed/` | Extracted keywords from reviews |
| `description_summaries_cache.jsonl` | `data/processed/` | Cached description summaries |
| `items_for_embedding.jsonl` | `data/processed/` | Final merged data for embedding |
| ChromaDB | `data/chromadb/` | Vector database with embeddings |

## Dependencies

### For Ollama backend:
```bash
pip install requests tqdm
```

### For vLLM backend:
```bash
pip install vllm torch tqdm
```

### For ChromaDB:
```bash
pip install chromadb FlagEmbedding
```
