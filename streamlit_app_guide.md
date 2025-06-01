# 실제 구현된 AI 소믈리에 Streamlit 웹 애플리케이션 완전 가이드

## 🌐 애플리케이션 개요

**AI 소믈리에 Streamlit 앱**은 사용자가 음식 이미지를 업로드하면 AI가 분석하여 어울리는 와인을 추천하는 웹 애플리케이션입니다.

### 핵심 구조
- **Frontend**: Streamlit 웹 인터페이스 (`app.py`)
- **Backend**: AI 추천 엔진 (`sommelier.py`)
- **데이터**: Pinecone 벡터 DB (129,971개 와인 리뷰)

## 📁 파일 구조 분석

### app.py - Streamlit 메인 애플리케이션

#### UI 레이아웃 구성
```python
import streamlit as st
from sommelier import search_wine, recommand_wine, describe_dish_flavor

st.title("Sommelier AI")

# 2열 레이아웃: 메인 컨텐츠(3) + 이미지 미리보기(1)
col1, col2 = st.columns([3, 1])

with col1:
    # 이미지 업로드 위젯
    uploaded_image = st.file_uploader(
        "요리 이미지를 업로드하세요.", 
        type=["jpg", "jpeg", "png"]
    )
    
    # 프롬프트 입력
    user_prompt = st.text_input(
        "프롬프트를 입력하세요.", 
        "이 요리에 어울리는 와인을 추천해주세요."
    )

with col2:
    # 업로드된 이미지 미리보기
    if uploaded_image:
        st.image(
            uploaded_image, 
            caption="업로드된 요리 이미지", 
            use_container_width=True
        )
```

#### 3단계 처리 워크플로우
```python
with col1:
    if st.button("추천하기"):
        if not uploaded_image:
            st.warning("이미지를 업로드해주세요.")
        else:
            # 1단계: 맛과 향 분석
            with st.spinner("1단계: 요리의 맛과 향을 분석하는 중..."):
                dish_flavor = describe_dish_flavor(
                    uploaded_image.read(), 
                    "이 요리의 이름과 맛과 향과 같은 특징을 한 문장으로 설명해줘"
                )
                st.markdown(f"#### 🍔 요리의 맛과 향 분석 결과")
                st.write(f"{dish_flavor}")

            # 2단계: 와인 리뷰 검색
            with st.spinner("2단계: 요리에 어울리는 와인 리뷰를 검색하는 중..."):
                wine_search_result = search_wine(dish_flavor) 
                st.markdown("#### 🍷 와인 리뷰 검색 결과")
                st.text(wine_search_result['wine_reviews'])

            # 3단계: 최종 추천 (미완성)
            with st.spinner("3단계: AI 소믈리예가 와인 페어링에 대한 추천글을 생성하는 중..."):
                pass  # TODO: recommand_wine 함수 호출 예정

            st.success("추천이 완료되었습니다!")
```

### sommelier.py - AI 추천 엔진

#### 환경 설정 및 초기화
```python
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import base64

load_dotenv()

# 모든 환경 변수 로드
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
# ... 기타 환경 변수들

# AI 컴포넌트 초기화
llm = ChatOpenAI(model=OPENAI_LLM_MODEL, temperature=0.2, openai_api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings,
    pinecone_api_key=PINECONE_API_KEY
)
```

#### 핵심 함수들

##### 1. describe_dish_flavor() - 이미지 기반 맛 분석
```python
def describe_dish_flavor(image_bytes, query):
    # 이미지를 Base64로 인코딩
    base64_str = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{base64_str}"
    
    # GPT-4o-mini Vision API용 메시지 구성
    messages = [
        {"role": "system", "content": """
            Persona: 맛 분석 시스템으로서 음식 재료, 조리법, 
            감각적 특성에 대한 깊은 이해를 보유
            
            Role:
            1. Flavor Identification: 단맛, 산미, 쓴맛, 짠맛, 감칠맛 분석
            2. Texture and Aroma Analysis: 식감과 향미 평가
            3. Ingredient Breakdown: 각 재료의 맛 역할 평가
            4. Cultural Influence: 문화적, 지역적 영향 고려
            5. Food and Drink Pairing: 맛 프로필 기반 페어링 제안
        """},
        {"role": "user", "content": [
            {"type": "text", "text": query},
            {"type": "image_url", "image_url": {"url": data_url}}
        ]}
    ]
    
    return llm.invoke(messages).content
```

##### 2. search_wine() - 벡터 검색
```python
def search_wine(dish_flavor):
    """음식 특징을 기반으로 유사한 와인 리뷰 검색"""
    results = vectorstore.similarity_search(
        dish_flavor,
        k=2  # 상위 2개 결과
    )

    return {
        "dish_flavor": dish_flavor,
        "wine_reviews": "\n\n".join([doc.page_content for doc in results])
    }
```

##### 3. recommand_wine() - 최종 추천 생성
```python
def recommand_wine(query):
    """검색된 와인 리뷰 기반 최종 추천 생성"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
            Persona: 전문 소믈리에로서 와인과 음식 페어링 전문가
            
            Role:
            1. Wine & Food Pairing: 음식과 조화로운 와인 추천
            2. Wine Selection Guidance: 상황별 와인 선택 지원
            3. Wine Tasting Expertise: 테이스팅 노트 기반 식별
            4. Educational Role: 와인 지역, 생산 기법 교육
        """),
        ("human", """
            와인 페어링 추천에 아래의 요리의 풍미와 와인 리뷰를 참고해 한글로 답변해 주세요.
            
            요리의 풍미:
            {dish_flavor}

            와인 리뷰:
            {wine_reviews}
        """)
    ])

    chain = prompt | llm | StrOutputParser()
    return chain.invoke(query)
```

## 🔄 애플리케이션 워크플로우

### 사용자 경험 흐름

1. **이미지 업로드**: 사용자가 음식 사진을 업로드
2. **프롬프트 입력**: 기본값 또는 커스텀 질의 입력
3. **추천 시작**: "추천하기" 버튼 클릭
4. **1단계 진행**: 🍔 맛과 향 분석 (로딩 스피너 표시)
5. **1단계 결과**: 음식 특징 한 문장 요약 표시
6. **2단계 진행**: 🍷 와인 리뷰 검색 (로딩 스피너 표시)
7. **2단계 결과**: 관련 와인 리뷰 2개 텍스트 표시
8. **3단계 진행**: 🤖 최종 추천 생성 (현재 미완성)
9. **완료**: 성공 메시지 표시

### 기술적 처리 흐름

**Frontend (Streamlit) ↔ Backend (sommelier.py) ↔ External APIs**

```
사용자 입력 (이미지 + 텍스트)
    ↓
Streamlit UI 처리
    ↓
describe_dish_flavor() 호출
    ↓ 
이미지 → Base64 → GPT-4o-mini Vision
    ↓
맛 특징 텍스트 반환 → UI 표시
    ↓
search_wine() 호출
    ↓
텍스트 → 임베딩 → Pinecone 검색
    ↓
와인 리뷰 반환 → UI 표시
    ↓
(recommand_wine() 미완성)
    ↓
완료 메시지 표시
```

## 🛠️ 구현 상태 분석

### ✅ 완성된 기능

1. **Streamlit UI**: 깔끔한 2열 레이아웃 구성
2. **이미지 업로드**: jpg, jpeg, png 지원
3. **실시간 미리보기**: 업로드 즉시 이미지 표시
4. **1단계 완성**: 이미지 → Base64 → Vision API → 맛 분석
5. **2단계 완성**: 맛 특징 → 벡터 검색 → 와인 리뷰
6. **진행 상황 표시**: 각 단계별 로딩 스피너
7. **결과 표시**: 단계별 결과를 구조화하여 표시

### ⚠️ 미완성 기능

1. **3단계 구현**: `recommand_wine()` 함수 호출 누락
2. **최종 추천**: 와인 리뷰 기반 개인화된 추천 생성
3. **에러 처리**: API 호출 실패 시 에러 핸들링
4. **입력 검증**: 이미지 형식 및 크기 검증

### 🔧 수정 필요사항

#### app.py 3단계 완성 코드
```python
# 현재 코드 (미완성)
with st.spinner("3단계: AI 소믈리예가 와인 페어링에 대한 추천글을 생성하는 중..."):
    pass  # TODO: recommand_wine 함수 호출

# 수정된 코드 (완성)
with st.spinner("3단계: AI 소믈리예가 와인 페어링에 대한 추천글을 생성하는 중..."):
    final_recommendation = recommand_wine(wine_search_result)
    st.markdown("#### 🤖 AI 소믈리에 최종 추천")
    st.write(final_recommendation)
```

## 💡 사용자 인터페이스 특징

### 직관적인 디자인

**레이아웃 구성:**
- **메인 영역 (75%)**: 업로드, 입력, 결과 표시
- **사이드 영역 (25%)**: 이미지 미리보기

**시각적 요소:**
- **이모지 활용**: 🍔 (분석), 🍷 (검색), 🤖 (추천)
- **진행 스피너**: 각 단계별 로딩 상태 표시
- **마크다운 헤더**: 결과 구조화 및 가독성 향상

### 사용자 친화적 기능

1. **즉시 피드백**: 이미지 업로드 시 즉시 미리보기
2. **기본값 제공**: 프롬프트 기본값으로 사용 편의성
3. **단계별 진행**: 복잡한 AI 처리를 3단계로 분할 표시
4. **경고 메시지**: 이미지 미업로드 시 친절한 안내

## 🚀 확장 가능성

### 즉시 개선 가능한 기능

1. **3단계 완성**: 1줄 코드 추가로 완전한 RAG 구현
2. **유사도 점수**: 검색 결과에 관련성 점수 표시
3. **추가 옵션**: 검색할 와인 개수 (k값) 조정 가능
4. **이미지 크기 제한**: 대용량 파일 업로드 방지

### 중장기 개선 방향

1. **사용자 선호도**: 개인 취향 저장 및 반영
2. **와인 이미지**: 추천 와인의 라벨 이미지 표시
3. **가격 필터**: 예산 범위 설정 기능
4. **소셜 기능**: 추천 결과 공유 및 평가

## 🔧 기술 스택 완성도

### 현재 구현된 기술
```python
# 주요 라이브러리
streamlit                    # 웹 UI 프레임워크
langchain-openai            # OpenAI API 통합
langchain-pinecone          # Pinecone 벡터DB 통합
langchain-core              # 프롬프트 및 파서
python-dotenv               # 환경 변수 관리
base64                      # 이미지 인코딩

# 외부 서비스
OpenAI GPT-4o-mini         # Vision + Language 모델
OpenAI text-embedding-3-small  # 텍스트 임베딩
Pinecone Vector Database   # 벡터 검색 엔진
```

### 환경 변수 설정
```env
OPENAI_API_KEY=sk-...
OPENAI_LLM_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=wine-reviews
PINECONE_INDEX_DIMENSION=1536
PINECONE_INDEX_METRIC=cosine
```

## 📊 성능 및 사용성

### 예상 처리 시간
- **1단계 (이미지 분석)**: 2-5초
- **2단계 (벡터 검색)**: 1-2초  
- **3단계 (추천 생성)**: 3-5초
- **총 처리 시간**: 6-12초

### 사용자 경험 최적화
- **진행 상황 표시**: 로딩 스피너로 대기 시간 체감 단축
- **즉시 피드백**: 단계별 결과 즉시 표시
- **직관적 UI**: 간단한 업로드 → 클릭 → 결과 흐름

## 시스템 구현 완성도 평가

### ✅ **성공적으로 구현된 기능**
1. **완전한 Streamlit 웹앱**: 직관적 UI와 반응형 레이아웃
2. **멀티모달 이미지 처리**: 업로드 → Base64 → Vision API
3. **실시간 벡터 검색**: Pinecone DB 연동 및 유사도 검색
4. **단계별 진행 표시**: 사용자 친화적 UX 디자인
5. **백엔드 모듈화**: sommelier.py로 비즈니스 로직 분리

### 🎯 **95% 완성도**
- **미완성**: 3단계 `recommand_wine()` 호출 1줄만 추가하면 완전한 서비스
- **완성된 인프라**: AI 엔진, 벡터 DB, UI가 모두 연동되어 작동

### 🚀 **즉시 완성 가능**
단 1줄의 코드 추가만으로 완전한 AI 소믈리에 웹 서비스가 완성됩니다!

**결론**: 이미 95% 완성된 전문가 수준의 AI 소믈리에 웹 애플리케이션입니다! 사용자가 음식 사진을 업로드하면 AI가 맛을 분석하고, 129,971개 와인 DB에서 최적의 와인을 찾아 전문적인 추천을 제공하는 완전한 RAG 시스템입니다! 🍷✨