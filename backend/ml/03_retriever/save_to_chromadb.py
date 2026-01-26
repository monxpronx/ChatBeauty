"""
Save item embeddings to ChromaDB using BGE-M3.

This script:
1. Loads preprocessed items from items_for_embedding.jsonl
2. Generates embeddings using BAAI/bge-m3
3. Saves to ChromaDB in backend/data/chromadb/

Usage:
    python save_to_chromadb.py
"""

import json
from pathlib import Path
from tqdm import tqdm
import chromadb
from FlagEmbedding import BGEM3FlagModel


def load_items(items_path: str) -> list:
    """Load items from JSONL file"""
    items = []
    with open(items_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading items"):
            items.append(json.loads(line))
    return items


def create_embeddings(model: BGEM3FlagModel, texts: list, batch_size: int = 32) -> list:
    """Generate embeddings for texts in batches"""
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch = texts[i:i + batch_size]
        embeddings = model.encode(batch)['dense_vecs']
        all_embeddings.extend(embeddings.tolist())

    return all_embeddings


def save_to_chromadb(
    items_path: str,
    chromadb_path: str,
    collection_name: str = "beauty_products",
    batch_size: int = 32,
    use_gpu: bool = True
):
    """Main function to save embeddings to ChromaDB"""

    # Step 1: Load items
    print("\n=== Step 1: Loading items ===")
    items = load_items(items_path)
    print(f"Loaded {len(items)} items")

    # Step 2: Initialize BGE-M3 model
    print("\n=== Step 2: Initializing BGE-M3 model ===")
    model = BGEM3FlagModel(
        'BAAI/bge-m3',
        use_fp16=use_gpu,
        device='cuda' if use_gpu else 'cpu'
    )
    print("Model loaded")

    # Step 3: Prepare data
    print("\n=== Step 3: Preparing data ===")
    documents = []
    ids = []
    metadatas = []

    for item in items:
        # Skip items without embedding text
        if not item.get('embedding_text'):
            continue

        documents.append(item['embedding_text'])
        ids.append(item['asin'])
        metadatas.append({
            'title': item.get('title', ''),
            'review_keywords': ', '.join(str(k) for k in item.get('review_keywords', [])),
            'description_summary': ', '.join(str(s) for s in item.get('description_summary', [])),
            'features': ', '.join(str(f) for f in item.get('features', []))[:500],  # Truncate long features
            # Additional metadata
            'price': item.get('price'),
            'average_rating': item.get('average_rating'),
            'store': item.get('store', ''),
            'categories': ', '.join(str(c) for c in item.get('categories', [])),
            'main_category': item.get('main_category', ''),
        })

    print(f"Prepared {len(documents)} documents for embedding")

    # Step 4: Generate embeddings
    print("\n=== Step 4: Generating embeddings ===")
    embeddings = create_embeddings(model, documents, batch_size)
    print(f"Generated {len(embeddings)} embeddings")

    # Step 5: Save to ChromaDB
    print("\n=== Step 5: Saving to ChromaDB ===")

    # Create ChromaDB directory
    Path(chromadb_path).mkdir(parents=True, exist_ok=True)

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=chromadb_path)

    # Delete existing collection if exists
    try:
        client.delete_collection(name=collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except Exception:
        pass

    # Create new collection
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # Add in batches (ChromaDB has limits)
    chroma_batch_size = 5000
    for i in tqdm(range(0, len(documents), chroma_batch_size), desc="Saving to ChromaDB"):
        end_idx = min(i + chroma_batch_size, len(documents))
        collection.add(
            embeddings=embeddings[i:end_idx],
            documents=documents[i:end_idx],
            ids=ids[i:end_idx],
            metadatas=metadatas[i:end_idx]
        )

    print(f"\n=== Complete ===")
    print(f"  Collection: {collection_name}")
    print(f"  Total items: {collection.count()}")
    print(f"  Saved to: {chromadb_path}")


def main():
    # Paths
    base_dir = Path(__file__).parent.parent.parent  # backend/

    items_path = base_dir / 'data/processed/items_for_embedding.jsonl'
    chromadb_path = base_dir / 'data/chromadb'

    # Check if input file exists
    if not items_path.exists():
        print(f"Error: Input file not found: {items_path}")
        print("Run merge_metadata.py first to create items_for_embedding.jsonl")
        return

    save_to_chromadb(
        items_path=str(items_path),
        chromadb_path=str(chromadb_path),
        collection_name="beauty_products",
        batch_size=32,
        use_gpu=True  # Set to False if no GPU available
    )


if __name__ == '__main__':
    main()
