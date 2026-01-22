from typing import List
from item_ranker.dataset import RerankSample


def train_reranker(samples: List[RerankSample]):
    """
    reranking 모델을 학습한다.

    입력:
    - samples: label이 포함된 RerankSample 리스트

    출력:
    - 학습이 완료된 모델 객체 (예: LightGBM, XGBoost 등)
    """
    raise NotImplementedError
