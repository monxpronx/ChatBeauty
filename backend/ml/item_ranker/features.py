from item_ranker.dataset import RerankSample


class FeatureBuilder:
    """
    RerankSample 안에 있는 각 candidate에 대해
    모델 입력용 feature 벡터를 생성하는 클래스
    """

    def build(self, sample: RerankSample):
        """
        반환값:
        - List[List[float]]
        - candidate 하나당 하나의 feature 벡터

        예:
        - sample.candidates 길이 == feature 벡터 개수
        """
        raise NotImplementedError
