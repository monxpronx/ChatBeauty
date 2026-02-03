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
    """
    reranker를 사용해 candidate들을 점수 기준으로 정렬한다.

    반환값:
    - (Candidate, score) 튜플의 리스트
    - score 내림차순 정렬 결과
    """
    raise NotImplementedError
