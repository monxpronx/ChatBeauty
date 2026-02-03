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
            placeholder="어떤 뷰티 상품을 찾으시나요? (예: 건성 피부용 수분 크림)"
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
          <button className="submit-btn" onClick={handleClick}>▶</button>
        </div>

        {result && (
          <div className="result-box">
            {/* 1. 추천 상품 리스트 (이름만 깔끔하게) */}
            <ul className="item-list">
              {result.recommendations.map((item, index) => (
                <li key={item.item_id} className="item-card">
                  <div className="item-name-container">
                    <span className="item-number">{index + 1}.</span>
                    <strong className="item-name">{item.item_name}</strong>
                  </div>
                </li>
              ))}
            </ul>

            {/* 2. 전체 추천 사유 (설명) */}
            {result.explanation && (
              <div className="explanation-section">
                <h3 className="explanation-title">✨ AI 추천 분석</h3>
                <div className="explanation-content">
                  {/* 설명이 리스트로 올 경우를 대비해 처리 */}
                  {Array.isArray(result.explanation) ? (
                    result.explanation.map((exp, i) => (
                      <p key={i} className="exp-text">{exp.explanation || exp}</p>
                    ))
                  ) : (
                    <p className="exp-text">{result.explanation}</p>
                  )}
                </div>
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
