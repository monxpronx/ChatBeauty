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

    ranked_items = rerank_items(
        query=request.user_input,
        candidates=candidates,
        top_k=5,
    )

    explanation_input = build_explanation_input(
        user_query=request.user_input,
        ranked_items=ranked_items,
    )

    explanations_data = generate_explanation(explanation_input)

    formatted_recommendations = [
        {
            "item_id": item["item_id"],
            "item_name": item.get("title", "Unknown Item"),
            "score": item.get("score", 0.0)
        }
        for item in ranked_items
    ]

    return RecommendResponse(
        recommendations=formatted_recommendations,
        explanation=explanations_data.get("explanations", [])
    )
    
def build_explanation_input(user_query: str, ranked_items: list[dict]):
    return {
        "user_query": user_query,
        "items": [
            {
                "item_id": item["item_id"],
                "title": item.get("title", ""),
                "price": item.get("price", 0.0),
                "average_rating": item.get("average_rating", 0.0),
                "categories": item.get("categories", ""),
                "features": item.get("features", ""),
                "description_summary": item.get("description_summary", ""),
                "review_keywords": item.get("review_keywords", ""),
            }
            for item in ranked_items
        ]
    }