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

![Service Pipeline](images/service_pipeline.png)

---

## 📊 Data

### 🔹 Data Structure

![data structure](images/Amazon_data.png)

---

### 🔹 EDA

![eda](images/EDA.png)

---

### 🔹 Data Preprocessing

### 문제 상황
 사용자 리뷰 데이터에 대한 신뢰성 확보
### 해결 방법
- 활동 시간 대비 리뷰 과다
  : 1시간 이내 리뷰를 10개 이상 작성한 유저 
- 평점 분산 기반
  : 리뷰 수가 5개 이상인 유저 중 모든 평점을 동일하게 작성한 유저

→ 위 조건 중 하나라도 만족할 경우 비정상 의심 유저로 분류

적용 결과 
약 0.3% 유저 데이터 제거

---

### 🔹 Database Schema

![database schema](images/data_schema.png)

---

## 🤖 Recommendation Model

### 🔹 Architecture

ChatBeauty는 Two-Tower 기반 구조를 사용합니다.

![model architecture](images/model_architecture.png)

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
![team members](images/team_members.png)

---

## 🎥 Demo Video

👉 https://youtu.be/g0UO8cHWX9I

---

## 🛠 Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Chroma](https://img.shields.io/badge/ChromaDB-5A67D8?style=flat)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow&logoColor=white)



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
