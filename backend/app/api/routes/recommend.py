import logging

from fastapi import APIRouter
from app.models.schemas import RecommendRequest, RecommendResponse, ItemScore
from app.services.retrieval import retrieve_candidates
from app.services.reranking import rerank_items
from app.services.explanation import generate_explanation

router = APIRouter(prefix="/recommend", tags=["recommend"])
logger = logging.getLogger(__name__)

@router.post("", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    candidates = retrieve_candidates(request.user_input)
    print("Candidates returned:", candidates)
    ranked_items = rerank_items(
        query=request.user_input,
        candidates=candidates,
        top_k=5,
    )
    explanation_input = build_explanation_input(
        user_query=request.user_input,
        ranked_items=ranked_items,
    )

    logger.info("LLM explanation input", extra=explanation_input)

    return RecommendResponse(
        recommendations=ranked_items
    )
    
def build_explanation_input(user_query: str, ranked_items: list[dict]):
    items = []

    for item in ranked_items:
        items.append({
            "title": item["item_name"],
            "price": item.get("price"),
            "average_rating": item.get("average_rating"),
            "categories": item.get("categories"),
            "features": item.get("features"),
            "description_summary": item.get("description_summary"),
            "review_keywords": item.get("review_keywords"),
        })

    return {
        "user_query": user_query,
        "items": items
    }