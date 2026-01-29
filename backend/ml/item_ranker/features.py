from item_ranker.dataset import RerankSample


class FeatureBuilder:
    """
    RerankSample 안에 있는 각 candidate에 대해
    모델 입력용 feature 벡터를 생성하는 클래스
    """

    def build(self, sample: RerankSample):
        
        return [
            [c.retrieval_score]
            for c in sample.candidates
        ]
