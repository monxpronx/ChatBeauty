"""
Clova Explorer 임베딩 v2 API로 쿼리/아이템 임베딩을 불러와 코사인 유사도 기반 파인튜닝
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, losses
from pathlib import Path
from .clova_embed_v2 import clova_embed_v2
import json
import os
from tqdm import tqdm

def load_matched_query_item(jsonl_path):
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

class ClovaEmbeddingPairDataset(Dataset):
    def __init__(self, query_embs, item_embs, labels):
        self.query_embs = query_embs
        self.item_embs = item_embs
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {
            'features_a': np.array(self.query_embs[idx], dtype=np.float32),
            'features_b': np.array(self.item_embs[idx], dtype=np.float32),
            'label': float(self.labels[idx])
        }

def prepare_clova_training_pairs(matched_jsonl_path, api_key, n_neg_per_query=100):
    data = load_matched_query_item(matched_jsonl_path)
    queries = [d['query'] for d in data]
    items = [d['item']['embedding_text'] for d in data]
    asins = [d['item']['asin'] for d in data]
    # Positive 임베딩
    query_embs = clova_embed_v2(queries, api_key)
    item_embs = clova_embed_v2(items, api_key)
    labels = [1.0] * len(queries)
    # Negative 샘플링 (query 고정, item만 랜덤)
    for idx, q in enumerate(queries):
        pos_asin = asins[idx]
        neg_candidates = [i for i, a in enumerate(asins) if a != pos_asin]
        for _ in range(n_neg_per_query):
            if not neg_candidates:
                break
            neg_idx = np.random.choice(neg_candidates)
            query_embs.append(query_embs[idx])
            item_embs.append(item_embs[neg_idx])
            labels.append(0.0)
    return query_embs, item_embs, labels

def finetune_clova(
    matched_jsonl_path,
    api_key,
    output_dir="./clova-finetuned-model",
    n_neg_per_query=100,
    batch_size=16,
    epochs=1,
    lr=2e-5
):
    # 데이터 준비
    print("Preparing training pairs (Clova embeddings)...")
    query_embs, item_embs, labels = prepare_clova_training_pairs(matched_jsonl_path, api_key, n_neg_per_query)
    dataset = ClovaEmbeddingPairDataset(query_embs, item_embs, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # 임베딩 입력용 SentenceTransformer dummy 모델
    model = SentenceTransformer('BAAI/bge-m3')
    train_loss = losses.CosineSimilarityLoss(model)
    # 학습
    print("Start fine-tuning...")
    model.fit(
        train_objectives=[(dataloader, train_loss)],
        epochs=epochs,
        optimizer_params={'lr': lr},
        output_path=output_dir,
        show_progress_bar=True
    )
    print(f"Model saved to: {output_dir}")

if __name__ == "__main__":
    api_key = os.getenv("CLOVA_API_KEY", "YOUR_API_KEY")
    base_dir = Path(__file__).parent.parent.parent
    matched_jsonl_path = base_dir / 'ml/data/processed/matched_query_item.jsonl'
    finetune_clova(
        matched_jsonl_path=str(matched_jsonl_path),
        api_key=api_key,
        output_dir="./clova-finetuned-model",
        n_neg_per_query=100,
        batch_size=16,
        epochs=1,
        lr=2e-5
    )
