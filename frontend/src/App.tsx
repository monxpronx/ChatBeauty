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
                  <li key={item.item_id} className="item-card">
                    <div className="item-name-container">
                      <strong className="item-name">{item.item_name}</strong>
                    </div>

                    <div className="item-info">
                      <span className="item-id">ID: {item.item_id}</span>
                      <span className="item-score">
                        연관 점수: <strong>{item.score.toFixed(2)}</strong>
                      </span>
                    </div>
                  </li>
                ))}
              </ul>

              {result.explanation && (
                <div className="explanation">
                  <p>{result.explanation}</p>
                </div>
              )}
            </div>
          )}
        </div>
      </main>
    </>
  );
}

export default App;
