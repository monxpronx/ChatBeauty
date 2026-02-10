from pydantic import BaseModel
from typing import List, Optional, Any

class RecommendRequest(BaseModel):
    user_input: str
    
class ItemScore(BaseModel):
    item_id: str
    item_name: str
    score: float
    explanation: Optional[str] = None
    
class RecommendResponse(BaseModel):
    recommendations: List[ItemScore]
