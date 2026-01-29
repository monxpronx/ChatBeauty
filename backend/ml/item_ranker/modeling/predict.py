from abc import ABC, abstractmethod
from typing import List, Tuple

from item_ranker.dataset import RerankSample, Candidate


class Reranker(ABC):
    
    """
    모든 reranker 구현체가 반드시 따라야 하는 추상 클래스
    """
    
    @abstractmethod
    def score(self, sample: RerankSample) -> List[float]:
        """
        하나의 RerankSample에 대해
        candidate 개수와 동일한 score 리스트를 반환해야 한다.

        규칙:
        - sample.candidates 길이 == score 개수
        - score 값이 클수록 더 상위 랭크
        """
        pass


def rerank(
    sample: RerankSample,
    reranker: Reranker,
) -> List[Tuple[Candidate, float]]:
    scores = reranker.score(sample)

    ranked = sorted(
        zip(sample.candidates, scores),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked
