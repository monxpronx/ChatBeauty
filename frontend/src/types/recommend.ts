export interface RecommendRequest {
    user_input: string;
    top_k: number;
}

export interface ItemScore {
    item_id: string;
    score: number;
    item_name: string;
}

export interface RecommendResponse {
    recommendations: ItemScore[];
    explanation: string
}