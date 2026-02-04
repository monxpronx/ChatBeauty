"""
Retrieve top-K candidates for evaluation and re-ranking.

Reads directly from All_Beauty.jsonl, splits by timestamp (80/10/10),
encodes review text with fine-tuned BGE-M3, and queries ChromaDB.

Output format is lean (re-ranking features only, no large text fields).
Text fields for explanation can be fetched from ChromaDB after re-ranking.

Usage:
    python retrieve_candidates.py MODEL_PATH=./model/retriever/bge-m3-finetuned-xxx SPLIT=valid
    python retrieve_candidates.py MODEL_PATH=./model/retriever/bge-m3-finetuned-xxx SPLIT=train TOP_K=100
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
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


def load_and_split_reviews(reviews_path: str, split: str) -> List[dict]:
    """
    Load reviews from All_Beauty.jsonl and split by timestamp.

    Split ratios (by sort_timestamp):
        - train: first 80%
        - valid: middle 10%
        - test: last 10%

    Returns list of reviews for the requested split.
    """
    print(f"Loading reviews from {reviews_path}...")

    # Load all reviews with timestamp
    reviews = []
    skipped = 0

    with open(reviews_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading reviews"):
            try:
                review = json.loads(line)

                # Skip reviews without required fields
                text = review.get('text', '')
                if not text or len(text.strip()) < 20:
                    skipped += 1
                    continue

                parent_asin = review.get('parent_asin') or review.get('asin')
                if not parent_asin:
                    skipped += 1
                    continue

                timestamp = review.get('timestamp', 0)

                reviews.append({
                    'parent_asin': parent_asin,
                    'text': text,
                    'timestamp': timestamp,
                })
            except json.JSONDecodeError:
                skipped += 1

    print(f"Loaded {len(reviews)} reviews (skipped {skipped})")

    # Sort by timestamp
    print("Sorting by timestamp...")
    reviews.sort(key=lambda x: x['timestamp'])

    # Split 80/10/10
    total = len(reviews)
    train_end = int(total * 0.8)
    valid_end = int(total * 0.9)

    if split == 'train':
        selected = reviews[:train_end]
    elif split == 'valid':
        selected = reviews[train_end:valid_end]
    elif split == 'test':
        selected = reviews[valid_end:]
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'valid', or 'test'")

    print(f"Split '{split}': {len(selected)} reviews")
    print(f"  (train: {train_end}, valid: {valid_end - train_end}, test: {total - valid_end})")

    return selected


def main():
    cli_args = parse_args()

    MODEL_PATH = cli_args.get('MODEL_PATH')
    if not MODEL_PATH:
        print("Error: MODEL_PATH is required")
        print("Usage: python retrieve_candidates.py MODEL_PATH=./models/bge-m3-finetuned-xxx SPLIT=valid")
        return

    TOP_K = int(cli_args.get('TOP_K', 100))
    BATCH_SIZE = int(cli_args.get('BATCH_SIZE', 256))
    COLLECTION_NAME = cli_args.get('COLLECTION', 'beauty_products')
    SPLIT = cli_args.get('SPLIT', 'valid').lower()

    base_dir = Path(__file__).parent.parent  # backend/ml/
    reviews_path = base_dir / 'data/raw/All_Beauty.jsonl'
    chromadb_path = base_dir / 'data/chromadb'
    output_dir = base_dir / 'data/evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'retrieval_candidates_{SPLIT}.jsonl'

    print("=" * 60)
    print("Retrieve Top-K Candidates from ChromaDB")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Split: {SPLIT}")
    print(f"Top-K: {TOP_K}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Output: {output_path}")
    print()

    # Check inputs
    if not reviews_path.exists():
        print(f"Error: {reviews_path} not found")
        return
    if not chromadb_path.exists():
        print(f"Error: ChromaDB not found at {chromadb_path}")
        return

    # Step 1: Load and split reviews
    print("=== Step 1: Loading and splitting reviews ===")
    reviews = load_and_split_reviews(str(reviews_path), SPLIT)

    # Step 2: Load model
    print("\n=== Step 2: Loading model ===")
    model = SentenceTransformer(MODEL_PATH)
    print(f"Model loaded. Max seq length: {model.max_seq_length}")

    # Step 3: Connect to ChromaDB
    print("\n=== Step 3: Connecting to ChromaDB ===")
    client = chromadb.PersistentClient(path=str(chromadb_path))
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' has {collection.count()} items")

    # Step 4: Encode all queries
    print("\n=== Step 4: Encoding queries ===")
    query_texts = [r['text'] for r in reviews]
    query_embeddings = model.encode(
        query_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).tolist()
    print(f"Encoded {len(query_embeddings)} queries")

    # Step 5: Query ChromaDB and write results (lean format)
    print(f"\n=== Step 5: Retrieving top-{TOP_K} candidates ===")
    chroma_batch_size = 200  # ChromaDB query batch limit
    written = 0

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for i in tqdm(range(0, len(reviews), chroma_batch_size), desc="Querying ChromaDB"):
            batch_end = min(i + chroma_batch_size, len(reviews))
            batch_embeddings = query_embeddings[i:batch_end]
            batch_reviews = reviews[i:batch_end]

            results = collection.query(
                query_embeddings=batch_embeddings,
                n_results=TOP_K,
                include=['distances', 'metadatas'],
            )

            for j, review in enumerate(batch_reviews):
                candidates = []
                for k in range(len(results['ids'][j])):
                    item_asin = results['ids'][j][k]
                    distance = results['distances'][j][k]
                    score = 1.0 - distance  # cosine similarity
                    meta = results['metadatas'][j][k]

                    # Lean format: only re-ranking features (no large text fields)
                    candidates.append({
                        'item_asin': item_asin,
                        'score': round(score, 6),
                        # Re-ranking features only
                        'price': meta.get('price', 0.0),
                        'average_rating': meta.get('average_rating', 0.0),
                        'rating_number': meta.get('rating_number', 0),
                        'total_helpful_votes': meta.get('total_helpful_votes', 0),
                        'store': meta.get('store', ''),
                    })

                output = {
                    'parent_asin': review['parent_asin'],  # ground truth
                    'query_text': review['text'],  # for re-ranking text features
                    'candidates': candidates,
                }
                f_out.write(json.dumps(output, ensure_ascii=False) + '\n')
                written += 1

    print(f"\n=== Complete ===")
    print(f"  Queries processed: {written}")
    print(f"  Output saved to: {output_path}")
    print()
    print("Note: Text fields (title, description, features, top_reviews, details, image)")
    print("      can be fetched from ChromaDB after re-ranking for top-5 explanation.")


if __name__ == '__main__':
    main()
