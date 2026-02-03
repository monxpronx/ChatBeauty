from pydantic import BaseModel
from typing import List, Optional

class RecommendRequest(BaseModel):
    user_input: str
    
class ItemScore(BaseModel):
    item_id: str
    score: float
    item_name: str
    
class RecommendResponse(BaseModel):
    recommendations: List[ItemScore]
    explanation: Optional[str] = None
