"""
Save item embeddings to ChromaDB using BGE-M3.

This script:
1. Loads preprocessed items from items_for_embedding.jsonl
2. Generates embeddings using BAAI/bge-m3 (or fine-tuned model)
3. Saves to ChromaDB in backend/data/chromadb/

Usage:
    python save_to_chromadb.py
    python save_to_chromadb.py MODEL_PATH=./models/bge-m3-finetuned-xxx  # Use fine-tuned model
"""

import json
import sys
from pathlib import Path
from tqdm import tqdm
import chromadb


def parse_args():
    """Parse KEY=VALUE arguments from command line"""
    args = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            args[key.upper()] = value
    return args


def load_items(items_path: str) -> list:
    """Load items from JSONL file"""
    items = []
    with open(items_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading items"):
            items.append(json.loads(line))
    return items


def create_embeddings_flagembedding(model, texts: list, batch_size: int = 32) -> list:
    """Generate embeddings using FlagEmbedding BGEM3FlagModel"""
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch = texts[i:i + batch_size]
        embeddings = model.encode(batch)['dense_vecs']
        all_embeddings.extend(embeddings.tolist())

    return all_embeddings


def create_embeddings_sentence_transformers(model, texts: list, batch_size: int = 32) -> list:
    """Generate embeddings using SentenceTransformer"""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    return embeddings.tolist()


def load_model(model_path: str, use_gpu: bool = True):
    """
    Load embedding model - either FlagEmbedding or SentenceTransformer.

    Returns: (model, model_type)
    """
    model_path = Path(model_path)

    # Check if it's a fine-tuned sentence-transformers model
    if model_path.exists() and (model_path / 'config.json').exists():
        print(f"Loading fine-tuned SentenceTransformer model from {model_path}")
        from sentence_transformers import SentenceTransformer
        device = 'cuda' if use_gpu else 'cpu'
        model = SentenceTransformer(str(model_path), device=device)
        return model, 'sentence_transformers'
    else:
        # Use FlagEmbedding for base BGE-M3
        print(f"Loading FlagEmbedding model: {model_path}")
        from FlagEmbedding import BGEM3FlagModel
        model = BGEM3FlagModel(
            str(model_path),
            use_fp16=use_gpu,
            device='cuda' if use_gpu else 'cpu'
        )
        return model, 'flagembedding'


def save_to_chromadb(
    items_path: str,
    chromadb_path: str,
    model_path: str = 'BAAI/bge-m3',
    collection_name: str = "beauty_products",
    batch_size: int = 32,
    use_gpu: bool = True
):
    """Main function to save embeddings to ChromaDB"""

    # Step 1: Load items
    print("\n=== Step 1: Loading items ===")
    items = load_items(items_path)
    print(f"Loaded {len(items)} items")

    # Step 2: Initialize model
    print("\n=== Step 2: Initializing embedding model ===")
    model, model_type = load_model(model_path, use_gpu)
    print(f"Model loaded (type: {model_type})")

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
        # ChromaDB does not accept None — guard every field
        meta = {
            'title': item.get('title') or '',
            'review_keywords': ', '.join(str(k) for k in (item.get('review_keywords') or [])),
            'description_summary': ', '.join(str(s) for s in (item.get('description_summary') or [])),
            'features': ', '.join(str(f) for f in (item.get('features') or []))[:500],
            'price': float(item.get('price') or 0.0),
            'average_rating': float(item.get('average_rating') or 0.0),
            'store': item.get('store') or '',
            'categories': ', '.join(str(c) for c in (item.get('categories') or [])),
            'main_category': item.get('main_category') or '',
        }
        metadatas.append(meta)

    print(f"Prepared {len(documents)} documents for embedding")

    # Step 4: Generate embeddings
    print("\n=== Step 4: Generating embeddings ===")
    if model_type == 'sentence_transformers':
        embeddings = create_embeddings_sentence_transformers(model, documents, batch_size)
    else:
        embeddings = create_embeddings_flagembedding(model, documents, batch_size)
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
    cli_args = parse_args()

    # Paths
    base_dir = Path(__file__).parent.parent.parent  # backend/

    items_path = base_dir / 'data/processed/items_for_embedding.jsonl'
    chromadb_path = base_dir / 'data/chromadb'

    # Model configuration
    model_path = cli_args.get('MODEL_PATH', 'BAAI/bge-m3')
    batch_size = int(cli_args.get('BATCH_SIZE', 32))
    use_gpu = cli_args.get('USE_GPU', 'true').lower() == 'true'

    print("=" * 60)
    print("ChromaDB Embedding Generator")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Batch size: {batch_size}")
    print(f"Use GPU: {use_gpu}")
    print()

    # Check if input file exists
    if not items_path.exists():
        print(f"Error: Input file not found: {items_path}")
        print("Run merge_metadata.py first to create items_for_embedding.jsonl")
        return

    save_to_chromadb(
        items_path=str(items_path),
        chromadb_path=str(chromadb_path),
        model_path=model_path,
        collection_name="beauty_products",
        batch_size=batch_size,
        use_gpu=use_gpu
    )


if __name__ == '__main__':
    main()
