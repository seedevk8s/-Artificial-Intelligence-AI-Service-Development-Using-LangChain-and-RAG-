# 완성된 영화 추천 시스템 구조도

## 1. 전체 시스템 발전 과정

```mermaid
flowchart TD
    subgraph Phase1["1단계: 기초 학습"]
        DEMO[langchain-demo<br/>3개 기본 문서]
        BASIC[벡터 검색 원리 학습]
        DEMO --> BASIC
    end
    
    subgraph Phase2["2단계: 프로토타입"]
        INDEX1[movie-index<br/>2개 영화]
        PROTO[영화 추천 프로토타입]
        INDEX1 --> PROTO
    end
    
    subgraph Phase3["3단계: 완성 시스템"]
        INDEX2[movie-index2<br/>7개 영화]
        COMPLETE[실용적 추천 시스템]
        INDEX2 --> COMPLETE
    end
    
    Phase1 --> Phase2
    Phase2 --> Phase3
    
    classDef phaseStyle1 fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef phaseStyle2 fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef phaseStyle3 fill:#e8f5e8,stroke:#4caf50,stroke-width:2px
    
    class DEMO,BASIC phaseStyle1
    class INDEX1,PROTO phaseStyle2
    class INDEX2,COMPLETE phaseStyle3
```

## 2. 최종 시스템 아키텍처

```mermaid
graph TD
    subgraph DataPrep["데이터 준비"]
        MOVIES[7개 한국 영화<br/>드라마 4편 + 범죄 3편]
        EXTRACT[설명 텍스트 추출]
        BATCH[배치 임베딩 처리]
        
        MOVIES --> EXTRACT
        EXTRACT --> BATCH
    end
    
    subgraph Storage["저장 시스템"]
        EMBED[OpenAI 임베딩<br/>1536차원 벡터]
        META[메타데이터 구성<br/>제목, 연도, 장르]
        PINECONE[Pinecone movie-index2<br/>7개 벡터 저장]
        
        BATCH --> EMBED
        EMBED --> META
        META --> PINECONE
    end
    
    subgraph Query["검색 시스템"]
        USER[사용자 질의<br/>감동적인 영화 추천해줘]
        SEARCH[유사도 검색<br/>코사인 유사도]
        RESULT[추천 결과<br/>7번방의 선물, 미나리]
        
        USER --> SEARCH
        PINECONE --> SEARCH
        SEARCH --> RESULT
    end
    
    classDef dataStyle fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef storeStyle fill:#e8f5e8,stroke:#4caf50,stroke-width:2px
    classDef queryStyle fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    
    class MOVIES,EXTRACT,BATCH dataStyle
    class EMBED,META,PINECONE storeStyle
    class USER,SEARCH,RESULT queryStyle
```

## 3. 영화 데이터베이스 구성

```mermaid
graph LR
    subgraph Drama["드라마 장르 (4편)"]
        D1[7번방의 선물<br/>2013년, 감동 가족]
        D2[미나리<br/>2020년, 이민 가족]
        D3[기생충<br/>2019년, 사회 풍자]
        D4[헤어질 결심<br/>2022년, 미스터리 멜로]
    end
    
    subgraph Crime["범죄 장르 (3편)"]
        C1[범죄도시<br/>2017년, 액션]
        C2[범죄도시 2<br/>2022년, 액션 속편]
        C3[다만 악에서 구하소서<br/>2020년, 범죄 액션]
    end
    
    DB[(movie-index2<br/>Pinecone DB)]
    
    Drama --> DB
    Crime --> DB
    
    classDef dramaStyle fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef crimeStyle fill:#ffebee,stroke:#c62828,stroke-width:2px
    classDef dbStyle fill:#e8f5e8,stroke:#4caf50,stroke-width:2px
    
    class D1,D2,D3,D4 dramaStyle
    class C1,C2,C3 crimeStyle
    class DB dbStyle
```

## 4. 추천 시나리오별 검색 결과

```mermaid
graph TD
    subgraph Scenarios["추천 시나리오"]
        Q1[감동적인 가족 영화 추천해줘]
        Q2[액션 영화 보고 싶어]
        Q3[사회적 메시지가 있는 영화]
        Q4[최신 영화 추천]
    end
    
    subgraph Results["예상 검색 결과"]
        R1[7번방의 선물<br/>미나리]
        R2[범죄도시<br/>범죄도시 2]
        R3[기생충<br/>미나리]
        R4[헤어질 결심<br/>범죄도시 2]
    end
    
    Q1 --> R1
    Q2 --> R2
    Q3 --> R3
    Q4 --> R4
    
    classDef queryStyle fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef resultStyle fill:#e8f5e8,stroke:#4caf50,stroke-width:2px
    
    class Q1,Q2,Q3,Q4 queryStyle
    class R1,R2,R3,R4 resultStyle
```

## 5. 기술 스택 및 구성 요소

```mermaid
graph TD
    subgraph Frontend["사용자 인터페이스"]
        USER_INPUT[자연어 질의 입력]
        RESULT_DISPLAY[추천 결과 표시]
    end
    
    subgraph Backend["백엔드 시스템"]
        LANGCHAIN[LangChain Framework]
        OPENAI[OpenAI Embeddings<br/>text-embedding-3-small]
        PINECONE[Pinecone Vector DB<br/>AWS 서버리스]
    end
    
    subgraph Data["데이터 레이어"]
        MOVIE_DB[영화 메타데이터]
        VECTOR_DB[벡터 임베딩]
        METADATA[검색 메타데이터]
    end
    
    USER_INPUT --> LANGCHAIN
    LANGCHAIN --> OPENAI
    LANGCHAIN --> PINECONE
    PINECONE --> VECTOR_DB
    PINECONE --> METADATA
    MOVIE_DB --> METADATA
    LANGCHAIN --> RESULT_DISPLAY
    
    classDef frontStyle fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef backStyle fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef dataStyle fill:#e8f5e8,stroke:#4caf50,stroke-width:2px
    
    class USER_INPUT,RESULT_DISPLAY frontStyle
    class LANGCHAIN,OPENAI,PINECONE backStyle
    class MOVIE_DB,VECTOR_DB,METADATA dataStyle
```

## 6. 코드 실행 완전 워크플로우

```mermaid
sequenceDiagram
    participant Dev as 개발자
    participant Env as 환경설정
    participant OpenAI as OpenAI API
    participant Pinecone as Pinecone DB
    participant System as 추천시스템
    
    Note over Dev,System: 시스템 구축 단계
    Dev->>Env: 라이브러리 설치 및 API 키 설정
    Dev->>OpenAI: 임베딩 모델 초기화
    Dev->>Pinecone: 인덱스 생성 (movie-index2)
    
    Note over Dev,System: 데이터 준비 단계
    Dev->>Dev: 7개 영화 데이터 정의
    Dev->>OpenAI: 영화 설명 배치 임베딩
    OpenAI-->>Dev: 1536차원 벡터 7개 반환
    Dev->>Dev: 메타데이터 구조화
    Dev->>Pinecone: 벡터 업서트 (7개)
    Pinecone-->>Dev: 업로드 완료 확인
    
    Note over Dev,System: 검색 테스트 단계
    Dev->>System: 추천 질의 테스트
    System->>OpenAI: 질의 임베딩
    System->>Pinecone: 유사도 검색
    System-->>Dev: 추천 결과 반환
```

## 시스템 완성도 및 특징

### 데이터 풍부성
- **총 7개 영화**: 드라마 4편 + 범죄 3편
- **시대적 다양성**: 2013년~2022년 작품
- **장르별 균형**: 감동, 액션, 사회 풍자, 미스터리

### 기술적 완성도
- **효율적 임베딩**: 배치 처리로 API 비용 절약
- **확장 가능한 구조**: 새로운 영화 쉽게 추가 가능
- **메타데이터 활용**: 제목, 연도, 장르 정보 제공
- **실시간 검색**: 밀리초 단위 응답 속도

### 실용적 활용도
- **다양한 질의 지원**: 장르, 감정, 주제별 추천
- **자연어 이해**: "감동적인", "신나는", "최신" 등 이해
- **맞춤형 추천**: 사용자 의도에 맞는 정확한 결과
- **확장성**: 수천 개 영화로 확장 가능한 아키텍처

## 다음 단계 발전 방향

### 기능 확장
1. **사용자 선호도 학습**: 개인화된 추천
2. **다중 필터링**: 연도, 평점, 배우별 검색
3. **유사 영화 추천**: "기생충과 비슷한 영화"
4. **리뷰 기반 추천**: 사용자 리뷰 데이터 활용

### 기술적 개선
1. **하이브리드 검색**: 벡터 + 키워드 검색 결합
2. **A/B 테스트**: 추천 정확도 지속적 개선
3. **실시간 업데이트**: 새 영화 자동 추가
4. **성능 최적화**: 검색 속도 및 정확도 튜닝

**결론**: 이 시스템은 실제 영화 추천 서비스의 핵심 엔진으로 사용할 수 있는 완전한 RAG 기반 벡터 검색 시스템입니다! 🎬✨