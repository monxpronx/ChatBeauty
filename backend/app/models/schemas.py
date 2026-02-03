from pydantic import BaseModel
from typing import List, Optional

class RecommendRequest(BaseModel):
    user_input: str
    top_k: int = 5
    
class ItemScore(BaseModel):
    item_id: str
    score: float
    
class RecommendResponse(BaseModel):
    recommendations: List[ItemScore]
    explanation: Optional[str] = None
