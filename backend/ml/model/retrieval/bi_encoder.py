import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def negative_sampling(query_emb, item_embs, pos_idx, num_neg=10):
    """
    query_emb: (embedding_dim,) - 쿼리 임베딩
    item_embs: (num_items, embedding_dim) - 전체 아이템 임베딩
    pos_idx: int - 정답 아이템 인덱스
    num_neg: int - negative 샘플 개수
    Returns:
        query_batch: (1+num_neg, embedding_dim)
        item_batch: (1+num_neg, embedding_dim)
        labels: (1+num_neg,) (positive=1, negative=0)
    """
    all_indices = np.arange(len(item_embs))
    neg_indices = np.setdiff1d(all_indices, [pos_idx])
    neg_sample_idx = np.random.choice(neg_indices, size=num_neg, replace=False)
    # positive
    pos_item = item_embs[pos_idx]
    # negative
    neg_items = item_embs[neg_sample_idx]
    # batch
    query_batch = np.vstack([query_emb] * (1 + num_neg))
    item_batch = np.vstack([pos_item, neg_items])
    labels = np.array([1] + [0] * num_neg, dtype=np.float32)
    # torch tensor 변환
    return torch.tensor(query_batch, dtype=torch.float32), torch.tensor(item_batch, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

class BiEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        # 쿼리/아이템 임베딩을 입력받아 추가로 학습할 수 있는 projection layer (선택)
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.item_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, query_emb, item_emb):
        # (batch, dim)
        q = F.normalize(self.query_proj(query_emb), dim=-1)
        i = F.normalize(self.item_proj(item_emb), dim=-1)
        # 쿼리와 아이템의 내적 (유사도)
        score = (q * i).sum(dim=-1)  # (batch,)
        return score

if __name__ == "__main__":
    embedding_dim = 1024
    model = BiEncoder(embedding_dim)
    num_items = 112000
    # 쿼리 임베딩은 고정, 아이템 임베딩은 전체
    query_emb = np.random.randn(embedding_dim)
    item_embs = np.random.randn(num_items, embedding_dim)
    num_neg = 10
    # pos_idx를 1부터 5까지 반복 (예시)
    for pos_idx in range(1, 6):
        q_batch, i_batch, labels = negative_sampling(query_emb, item_embs, pos_idx, num_neg=num_neg)
        print(f"pos_idx={pos_idx} | q_batch: {q_batch.shape}, i_batch: {i_batch.shape}, labels: {labels}")
        scores = model(q_batch, i_batch)
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        print(f"loss: {loss.item()}")
        loss.backward()
        print(f"pos_idx={pos_idx} 역전파 완료")
