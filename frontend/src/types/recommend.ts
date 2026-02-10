export interface RecommendRequest {
    user_input: string;
    top_k: number;
}

export interface ItemScore {
    item_id: string;
    score: number;
    item_name: string;
    explanation?: string;
    image?: string;
    price?: number;
    average_rating?: number;
    rating_number?: number;
    store?: string;
}

export interface RecommendResponse {
    recommendations: ItemScore[];
}