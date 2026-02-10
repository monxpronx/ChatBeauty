# 💄 ChatBeauty - LLM & RAG 기반 아마존 뷰티 상품 추천 시스템

ChatBeauty는 사용자의 피부 타입, 피부 고민, 선호 조건을 기반으로  
**개인 맞춤형 화장품을 추천하고 추천 이유까지 설명해주는 서비스**입니다.

> “어떤 화장품이 나한테 맞을까?”  
> “성분은 괜찮을까?”  
> “종류는 많은데 뭘 골라야 할지 모르겠어…”

이런 고민을 해결하기 위해 만들어졌습니다.

---

## 📌 Project Overview

### 🔹 What is ChatBeauty?

ChatBeauty는 자연어로 입력한 사용자 질의와 피부 정보를 기반으로  
가장 적합한 화장품을 추천하는 AI 추천 시스템입니다.

단순 인기 제품 추천이 아니라,

- 왜 이 제품이 나에게 맞는지  
- 어떤 성분이 도움이 되는지  

를 함께 설명하는 것이 핵심 목표입니다.

---

### 🔹 Expected Impact

ChatBeauty는 다음을 제공합니다.

- 사용자 **피부 타입, 고민, 선호 조건 반영**
- 대규모 화장품 데이터 기반 추천
- 추천 결과에 대한 **설명 제공**
- 선택에 대한 불확실성 감소

즉,  
> "많이 팔린 제품"이 아니라  
> **"나에게 맞는 제품"을 추천합니다.**

---

## 🏗 Service Pipeline

서비스 전체 흐름은 다음과 같습니다.

- 사용자 질의 입력
- User Encoder를 통한 Query Embedding 생성
- Item Encoder로 생성된 Item Embedding과 Vector DB 검색
- 후보 아이템 Retrieval
- LightGBM Ranker로 재정렬
- Top-N 추천 + 설명 생성

---

## 📊 Data

### 🔹 Data Structure

- Product 정보
- Ingredient 정보
- Skin type / concern 정보
- Metadata (category, brand, etc.)

---

### 🔹 Data Preprocessing

- 결측치 처리
- 텍스트 정제
- 성분 벡터화
- 사용자 질의 전처리
- 추천에 필요한 Feature 생성

---

### 🔹 Database Schema

- User
- Product
- Ingredient
- Review
- Metadata 테이블 구성

---

## 🤖 Recommendation Model

### 🔹 Architecture

ChatBeauty는 Two-Tower 기반 구조를 사용합니다.

- **User Tower**
  - 사용자 질의 인코딩
  - 피부 타입 및 선호 조건 반영

- **Item Tower**
  - 제품 설명, 성분, 메타데이터 인코딩

- **Vector DB**
  - Embedding 기반 Retrieval

- **Ranker**
  - LightGBM으로 Top-K 재정렬

- **Explainability**
  - 추천 이유 생성

---

## 🚀 Project Process

- 데이터 수집 및 전처리
- 추천 파이프라인 설계
- Two-Tower 모델 구축
- Vector DB 연동
- Ranking 모델 학습
- 추천 결과 설명 생성

---

## 👥 Team

ChatBeauty Project Team - RecSys-07

---

## 🎥 Demo Video

👉 https://youtu.be/g0UO8cHWX9I

---

## 🛠 Tech Stack

- Python
- PyTorch
- LightGBM
- Vector DB (FAISS / Chroma)
- FastAPI
- Pandas / Numpy
- HuggingFace Embedding Models

---

## 📂 Repository Structure (Example)

```bash
.
├── data
├── preprocessing
├── model
├── retrieval
├── ranking
├── api
├── notebooks
└── README.md
