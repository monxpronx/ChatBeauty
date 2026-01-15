import { useState } from "react";
import { fetchRecommend } from "./api/recommend";
import type { RecommendResponse } from "./types/recommend";
import "./App.css";

function App() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<RecommendResponse | null>(null);

  const handleClick = async () => {
    const data = await fetchRecommend({
      user_input: query,
      top_k: 5,
    });
    setResult(data);
  };

  return (
    <>
      <header className="header">
        <h1 className="title">뷰티 상품 추천 시스템</h1>
      </header>

      <main className="main">
        <div className="input-wrapper">
          <div className="input-inner">
            <textarea
              value={query}
              placeholder="추천 요청을 입력하세요"
              rows={1}
              onChange={(e) => {
                setQuery(e.target.value);
                e.target.style.height = "auto";
                e.target.style.height = e.target.scrollHeight + "px";
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleClick();
                }
              }}
            />

            <button className="submit-btn" onClick={handleClick}>
              ▶
            </button>
          </div>

          {result && (
            <div className="result-box">
              <ul className="item-list">
                {result.recommendations.map((item) => (
                  <li key={item.item_id}>
                    <span className="item-id">{item.item_id}</span>
                    <span className="item-score">
                      {item.score.toFixed(2)}
                    </span>
                  </li>
                ))}
              </ul>

              <div className="explanation">
                <h3>추천 이유 설명</h3>
                <p>{result.explanation}</p>
              </div>
            </div>
          )}
        </div>
      </main>
    </>
  );
}

export default App;
