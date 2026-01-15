export interface RecommendRequest {
    user_input: string;
    top_k: number;
}

export interface ItemScore {
    item_id: string;
    score: number;
}

export interface RecommendResponse {
    recommendations: ItemScore[];
    explanation: string
}