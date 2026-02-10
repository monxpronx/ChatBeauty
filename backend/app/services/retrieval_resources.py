import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models/bge-m3-finetuned-20260202-120852"
CHROMA_PATH = BASE_DIR / "data/chromadb"
COLLECTION_NAME = "beauty_products"

model = SentenceTransformer(str(MODEL_PATH))

client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_collection(name=COLLECTION_NAME)
