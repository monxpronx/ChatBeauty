import chromadb

# ChromaDB 데이터베이스가 있는 폴더 경로를 지정하세요
chroma_path = "/data/ephemeral/home/pro-recsys-finalproject-recsys-07/backend/data/chromadb"
collection_name = "beauty_products"  # 실제 컬렉션명으로 변경

# ChromaDB 클라이언트 연결
client = chromadb.PersistentClient(path=chroma_path)
collection = client.get_collection(name=collection_name)

# 전체 데이터 개수 확인
print("Total items:", collection.count())

# 일부 데이터 샘플 조회 (예: 5개)
results = collection.get(limit=5)
for i, (id, meta) in enumerate(zip(results["ids"], results["metadatas"])):
    print(f"ID: {id}")
    print(f"Metadata: {meta}")
    print("-" * 40)
