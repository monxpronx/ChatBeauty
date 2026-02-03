import chromadb
from chromadb.config import Settings
import requests
from bs4 import BeautifulSoup
import time

def get_amazon_image_url(title):
    # 아마존 검색 URL 생성
    search_url = f"https://www.amazon.com/s?k={requests.utils.quote(title)}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    resp = requests.get(search_url, headers=headers)
    if resp.status_code != 200:
        return None
    soup = BeautifulSoup(resp.text, "html.parser")
    # 아마존 검색 결과에서 첫 번째 이미지 추출
    img_tag = soup.select_one("img.s-image")
    if img_tag and img_tag.get("src"):
        return img_tag["src"]
    return None

def add_image_urls_to_chromadb(collection_name="items"):
    # chroma db 연결
    client = chromadb.Client(Settings(
        persist_directory="backend/data/chromadb",  # 경로는 환경에 맞게 수정
        chroma_db_impl="duckdb+parquet"
    ))
    collection = client.get_collection(collection_name)
    # 모든 item 불러오기
    items = collection.get()
    for i, item in enumerate(items["metadatas"]):
        title = item.get("title")
        if not title:
            continue
        print(f"[{i+1}/{len(items['metadatas'])}] '{title}' 이미지 검색 중...")
        image_url = get_amazon_image_url(title)
        if image_url:
            print(f"이미지 URL: {image_url}")
            # item에 image_url 추가
            item["image_url"] = image_url
            # chroma db에 업데이트 (id 기준)
            collection.update(
                ids=[items["ids"][i]],
                metadatas=[item]
            )
        else:
            print("이미지 찾지 못함.")
        time.sleep(2)  # 아마존 차단 방지용 딜레이

if __name__ == "__main__":
    add_image_urls_to_chromadb()
