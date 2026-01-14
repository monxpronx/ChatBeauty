from fastapi import APIRouter
from app.models.schemas import RecommendRequest, RecommendResponse, ItemScore
from app.services.retrieval import retrieve_candidates
from app.services.reranking import rerank_items
from app.services.explanation import generate_explanation

router = APIRouter(prefix="/recommend", tags=["recommend"])

@router.post("/", response_model=RecommendResponse)
def recommed(request: RecommendRequest):
    candidates = retrieve_candidates()
    ranked_items = rerank_items(candidates,request.top_k)
    explanation = generate_explanation(ranked_items)
    
    return RecommendResponse(
        recommendations=[
            ItemScore(**item) for item in ranked_items
        ],
        explanation=explanation
    )