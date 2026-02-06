import { useState, useRef, useEffect } from "react";
import { fetchRecommend } from "./api/recommend";
import type { RecommendResponse } from "./types/recommend";
import "./App.css";

function App() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<RecommendResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleClick = async () => {
    if (!query.trim() || isLoading) return;

    setIsLoading(true);
    try {
      const data = await fetchRecommend({
        user_input: query,
        top_k: 5,
      });
      setResult(data);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1 className="title">ChatBeauty</h1>
      </header>

      <main className="chat-content">
        {result && (
          <div className="result-box">
            <ul className="item-list">
              {result.recommendations.map((item, index) => (
                <li key={item.item_id} className="item-card">
                  <div className="item-info-group">
                    <div className="item-name-container">
                      <span className="item-number">{index + 1}.</span>
                      <strong className="item-name">{item.item_name}</strong>
                    </div>

                    <p className="item-description">
                      {item.explanation
                        ? item.explanation
                        : "답변을 생성하는데 실패했어요."}
                    </p>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        )}
        {isLoading && <div className="loading-dots">답변을 생각 중이에요...</div>}
      </main>

      <footer className="footer-input">
        <div className="input-wrapper">
          <div className="input-inner">
            <textarea
              value={query}
              placeholder="어떤 뷰티 상품을 찾으시나요?"
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
            <button
              className={`submit-btn ${isLoading ? 'loading' : ''}`}
              onClick={handleClick}
              disabled={isLoading}
            >
              {isLoading ? "●" : "▲"}
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;