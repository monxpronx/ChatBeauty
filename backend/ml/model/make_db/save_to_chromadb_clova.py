"""
Clova Explorer 임베딩 v2 API로 item tower 임베딩 후 ChromaDB에 저장
"""
import json
import os
from pathlib import Path
from tqdm import tqdm
import chromadb
from ..retrieval.clova_embed_v2 import clova_embed_v2

def save_items_to_chromadb_clova(
    items_path: str,
    chromadb_path: str,
    api_key: str,
    collection_name: str = "beauty_products_clova",
    batch_size: int = 32
):
    # Step 1: Load items
    print("\n=== Step 1: Loading items ===")
    items = []
    with open(items_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading items"):
            items.append(json.loads(line))
    print(f"Loaded {len(items)} items")

    # Step 2: Prepare data
    print("\n=== Step 2: Preparing data ===")
    documents = []
    ids = []
    metadatas = []
    for item in items:
        if not item.get('embedding_text'):
            continue
        documents.append(item['embedding_text'])
        ids.append(item['asin'])
        meta = {
            'title': item.get('title') or '',
            'review_keywords': ', '.join(str(k) for k in (item.get('review_keywords') or [])),
            'description_summary': ', '.join(str(s) for s in (item.get('description_summary') or [])),
            'features': ', '.join(str(f) for f in (item.get('features') or []))[:500],
            'price': float(item.get('price') or 0.0),
            'average_rating': float(item.get('average_rating') or 0.0),
            'store': item.get('store') or '',
            'categories': ', '.join(str(c) for c in (item.get('categories') or [])),
        }
        metadatas.append(meta)
    print(f"Prepared {len(documents)} documents for embedding")

    # Step 3: Generate embeddings (Clova)
    print("\n=== Step 3: Generating embeddings (Clova) ===")
    embeddings = []
    for i in tqdm(range(0, len(documents), batch_size), desc="Clova embedding"):
        batch = documents[i:i+batch_size]
        batch_emb = clova_embed_v2(batch, api_key)
        embeddings.extend(batch_emb)
    print(f"Generated {len(embeddings)} embeddings")

    # Step 4: Save to ChromaDB
    print("\n=== Step 4: Saving to ChromaDB ===")
    Path(chromadb_path).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=chromadb_path)
    try:
        client.delete_collection(name=collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
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

if __name__ == "__main__":
    # 환경변수 또는 직접 입력
    api_key = os.getenv("CLOVA_API_KEY", "YOUR_API_KEY")
    base_dir = Path(__file__).parent.parent.parent
    items_path = base_dir / 'data/processed/items_for_embedding.jsonl'
    chromadb_path = base_dir / 'data/chromadb'
    save_items_to_chromadb_clova(
        items_path=str(items_path),
        chromadb_path=str(chromadb_path),
        api_key=api_key,
        collection_name="beauty_products_clova",
        batch_size=32
    )
