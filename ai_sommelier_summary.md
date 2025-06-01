# AI 소믈리에 RAG 서비스 프로젝트

## 🍷 프로젝트 개요

**AI 소믈리에 RAG 서비스**는 음식 사진과 간단한 설명을 입력받아 AI가 최적의 와인을 추천하는 웹 서비스입니다.

### 핵심 기술 스택
- **Frontend**: Streamlit
- **Backend**: Python 3.12+
- **AI 모델**: GPT-4o-mini (Vision & LLM)
- **임베딩**: text-embedding-3-small
- **벡터DB**: Pinecone

## 🔄 서비스 워크플로우

### 1단계: 이미지 분석
- 사용자가 음식 사진 업로드 + 한 줄 설명 입력
- GPT-4o-mini Vision이 음식의 맛과 특징을 분석

### 2단계: 벡터 검색
- 음식 설명을 임베딩으로 변환
- Pinecone DB에서 유사한 와인 리뷰를 Top-K 검색
- 유사도 점수와 함께 관련 리뷰 추출

### 3단계: 추천 생성
- 검색된 리뷰와 음식 정보를 LLM에 전달
- GPT-4o-mini가 한글로 와인 추천 및 이유 설명
- 결과를 Streamlit UI에 표시

## 🏗️ 시스템 구조

### 핵심 모듈
```
├── app.py              # Streamlit 웹 UI
├── sommelier.py        # 핵심 비즈니스 로직
│   ├── describe_dish_flavor()    # 이미지→맛 분석
│   ├── search_wine()            # 벡터DB 검색
│   └── recommand_wine()         # LLM 추천 생성
└── .env               # 환경 변수 설정
```

### 환경 변수 예시
```env
OPENAI_API_KEY=sk-...
OPENAI_LLM_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=wine-reviews
PINECONE_INDEX_DIMENSION=1536
PINECONE_INDEX_METRIC=cosine
```

## 💡 주요 특징

### RAG (Retrieval-Augmented Generation)
- 대규모 와인 리뷰 데이터베이스 활용
- 벡터 유사도 기반 정확한 검색
- 실시간 컨텍스트 기반 추천

### 멀티모달 AI
- 이미지 + 텍스트 입력 처리
- Vision LLM을 통한 시각적 음식 분석
- 자연어 기반 직관적인 추천 결과

### 개인화된 경험
- 음식별 맞춤형 와인 페어링
- 상세한 추천 이유 제공
- 유사도 점수를 통한 투명성

## 🚀 기대 효과

1. **초개인화 와인 추천**: 개별 음식에 최적화된 와인 선택
2. **기술 실습**: LLM+RAG+벡터DB 전체 파이프라인 경험
3. **실전 AI 응용**: 최신 생성형 AI 기술의 실용적 활용

## 🔮 확장 가능성

- 와인 라벨 이미지 추가 표시
- 사용자 피드백 학습 시스템
- 다양한 음식/와인 카테고리 확장
- 선호도 기반 개인화 알고리즘

## ⚠️ 주의사항

- API 키 보안 관리 필수
- 모델 사용량/비용 모니터링
- .env 파일 공개 저장소 업로드 금지
- 추천 품질은 데이터 품질에 의존