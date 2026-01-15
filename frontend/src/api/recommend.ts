import type { RecommendRequest, RecommendResponse } from "../types/recommend";

export async function fetchRecommend(
  payload: RecommendRequest
): Promise<RecommendResponse> {
  const res = await fetch("http://localhost:8000/recommend", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    throw new Error("Recommendation API failed");
  }

  return res.json();
}