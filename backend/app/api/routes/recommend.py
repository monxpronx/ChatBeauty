from fastapi import APIRouter
from app.models.schemas import RecommendRequest, RecommendResponse, ItemScore
from app.services.retrieval import retrieve_candidates
from app.services.reranking import rerank_items
from app.services.explanation import generate_explanation

router = APIRouter(prefix="/recommend", tags=["recommend"])

@router.post("", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    candidates = retrieve_candidates(request.user_input)
    ranked_items = rerank_items(
        query=request.user_input,
        candidates=candidates,
        top_k=5,
    )
    explanation = generate_explanation(ranked_items)

    return RecommendResponse(
        recommendations=ranked_items
    )