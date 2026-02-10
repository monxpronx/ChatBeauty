"""
Retrieve top-100 candidates for each test query using fine-tuned BGE-M3 + ChromaDB.

Reads keywords_test.jsonl, encodes queries with the fine-tuned model,
queries ChromaDB for top-100 similar items, and writes results with metadata.

Usage:
    python retrieve_candidates.py MODEL_PATH=./model/retriever/bge-m3-finetuned-xxx
    python retrieve_candidates.py MODEL_PATH=./model/retriever/bge-m3-finetuned-xxx TOP_K=100 BATCH_SIZE=256
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

import chromadb
from sentence_transformers import SentenceTransformer


def parse_args() -> Dict[str, str]:
    args = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            args[key.upper()] = value
    return args


def create_query_text(keywords: List, max_keywords: int = 20) -> str:
    """Create query text from keywords (same logic as create_training_pairs.py)."""
    if not keywords:
        return ""
    selected = keywords[:max_keywords]
    return ", ".join(str(kw) for kw in selected if kw is not None)


def load_test_queries(test_file_path: str) -> List[dict]:
    """Load queries from generated_queries_test_filtered.jsonl.
    
    Uses generated_query field directly as query text.
    """
    queries = []
    skipped = 0
    with open(test_file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading queries"):
            try:
                item = json.loads(line)
                query_text = item.get('generated_query', '')
                parent_asin = item.get('parent_asin', '')
                
                if not query_text or not parent_asin:
                    skipped += 1
                    continue
                
                queries.append({
                    'parent_asin': parent_asin,
                    'query_text': query_text,
                })
            except json.JSONDecodeError:
                skipped += 1
    print(f"Loaded {len(queries)} queries (skipped {skipped})")
    return queries


def main():
    cli_args = parse_args()

    MODEL_PATH = cli_args.get('MODEL_PATH')
    if not MODEL_PATH:
        print("Error: MODEL_PATH is required")
        print("Usage: python retrieve_candidates.py MODEL_PATH=./models/bge-m3-finetuned-xxx")
        return

    TOP_K = int(cli_args.get('TOP_K', 100))
    BATCH_SIZE = int(cli_args.get('BATCH_SIZE', 256))
    COLLECTION_NAME = cli_args.get('COLLECTION', 'beauty_products')

    SPLIT = cli_args.get('SPLIT', 'test')

    # File is at: backend/ml/model/evaluation/retrieval_candidates.py
    # Need to go up to backend/ml/
    base_dir = Path(__file__).parent.parent.parent  # backend/ml/
    test_file_path = base_dir / f'data/processed/generated_queries_{SPLIT}_filtered.jsonl'
    chromadb_path = base_dir / 'data/chromadb'
    output_dir = base_dir / 'data/evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'retrieval_candidates_{SPLIT}.jsonl'

    print("=" * 60)
    print("Retrieve Top-K Candidates from ChromaDB")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Top-K: {TOP_K}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Input file: {test_file_path.name}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Output: {output_path}")
    print()

    # Check inputs
    if not test_file_path.exists():
        print(f"Error: {test_file_path} not found")
        return
    if not chromadb_path.exists():
        print(f"Error: ChromaDB not found at {chromadb_path}")
        return

    # Step 1: Load model
    print("=== Step 1: Loading model ===")
    model = SentenceTransformer(MODEL_PATH)
    print(f"Model loaded. Max seq length: {model.max_seq_length}")

    # Step 2: Connect to ChromaDB
    print("\n=== Step 2: Connecting to ChromaDB ===")
    client = chromadb.PersistentClient(path=str(chromadb_path))
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' has {collection.count()} items")

    # Step 3: Load test queries
    print("\n=== Step 3: Loading test queries ===")
    queries = load_test_queries(str(test_file_path))

    # Step 4: Encode all queries
    print("\n=== Step 4: Encoding queries ===")
    query_texts = [q['query_text'] for q in queries]
    query_embeddings = model.encode(
        query_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).tolist()
    print(f"Encoded {len(query_embeddings)} queries")

    # Step 5: Query ChromaDB and write results
    print(f"\n=== Step 5: Retrieving top-{TOP_K} candidates ===")
    # ChromaDB supports batch queries
    chroma_batch_size = 200  # ChromaDB query batch limit
    written = 0

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for i in tqdm(range(0, len(queries), chroma_batch_size), desc="Querying ChromaDB"):
            batch_end = min(i + chroma_batch_size, len(queries))
            batch_embeddings = query_embeddings[i:batch_end]
            batch_queries = queries[i:batch_end]

            results = collection.query(
                query_embeddings=batch_embeddings,
                n_results=TOP_K,
                include=['distances', 'metadatas'],
            )

            for j, query in enumerate(batch_queries):
                candidates = []
                for k in range(len(results['ids'][j])):
                    item_asin = results['ids'][j][k]
                    # ChromaDB cosine distance: distance = 1 - similarity
                    distance = results['distances'][j][k]
                    score = 1.0 - distance
                    meta = results['metadatas'][j][k]

                    candidates.append({
                        'item_asin': item_asin,
                        'score': round(score, 6),
                        'title': meta.get('title', ''),
                        'price': meta.get('price', 0.0),
                        'average_rating': meta.get('average_rating', 0.0),
                        'store': meta.get('store', ''),
                        'categories': meta.get('categories', ''),
                    })

                output = {
                    'parent_asin': query['parent_asin'],
                    'query_text': query['query_text'],
                    'candidates': candidates,
                }
                f_out.write(json.dumps(output, ensure_ascii=False) + '\n')
                written += 1

    print(f"\n=== Complete ===")
    print(f"  Queries processed: {written}")
    print(f"  Output saved to: {output_path}")


if __name__ == '__main__':
    main()