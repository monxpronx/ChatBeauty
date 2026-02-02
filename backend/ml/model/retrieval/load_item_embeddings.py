import chromadb
from chromadb.config import Settings
import numpy as np
from typing import Dict

def load_item_embeddings_from_chromadb(collection_name: str = "items", persist_directory: str = None) -> Dict[str, np.ndarray]:
    """
    ChromaDB에서 asin별 임베딩을 dict로 불러옵니다.
    Args:
        collection_name (str): ChromaDB 컬렉션 이름
        persist_directory (str): ChromaDB 디렉토리 경로
    Returns:
        Dict[str, np.ndarray]: {asin: embedding(np.ndarray)}
    """
    client = chromadb.PersistentClient(persist_directory=persist_directory) if persist_directory else chromadb.Client()
    collection = client.get_collection(collection_name)
    # 모든 item 불러오기
    results = collection.get()
    embeddings = {}
    for asin, emb in zip(results['ids'], results['embeddings']):
        embeddings[asin] = np.array(emb, dtype=np.float32)
    return embeddings
