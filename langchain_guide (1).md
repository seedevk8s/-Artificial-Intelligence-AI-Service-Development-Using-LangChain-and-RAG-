# LangChain v0.3.x 종합 가이드

## 📋 목차
1. [LangChain 개요](#langchain-개요)
2. [아키텍처 및 모듈 구조](#아키텍처-및-모듈-구조)
3. [환경 설정](#환경-설정)
4. [핵심 컴포넌트](#핵심-컴포넌트)
5. [LCEL 기반 체인 구성](#lcel-기반-체인-구성)
6. [메모리 관리](#메모리-관리)
7. [도구(Tools) 활용](#도구tools-활용)
8. [비용 모니터링 및 캐싱](#비용-모니터링-및-캐싱)
9. [실습 예제](#실습-예제)

## LangChain 개요

LangChain은 대형 언어 모델(LLM)을 활용한 애플리케이션 개발을 돕는 파이썬 프레임워크입니다. 

### 주요 특징
- **체인(Chain) 기반 처리**: 프롬프트 작성 → LLM 호출 → 응답 파싱의 일련 과정을 체계적으로 관리
- **LCEL (LangChain Expression Language)**: 파이프라인 구축을 위한 선언적 문법
- **모듈화된 아키텍처**: 필요한 기능만 선택적으로 설치 가능

## 아키텍처 및 모듈 구조

### 핵심 패키지
- **`langchain-core`**: 기본 추상 클래스와 LCEL 실행 기능
- **`langchain`**: 메인 패키지 (langchain-core 포함)
- **`langchain-openai`**: OpenAI API 연동
- **`langchain-anthropic`**: Anthropic API 연동
- **`langchain-community`**: 미분리 통합 기능들
- **`langchain-experimental`**: 실험적 기능

### 장점
- 경량 설치 가능
- 독립적인 업데이트 관리
- 필요한 기능만 선택적 사용

## 환경 설정

### 1. 패키지 설치
```bash
pip install langchain langchain-openai python-dotenv
```

### 2. API 키 설정
`.env` 파일 생성:
```
OPENAI_API_KEY=sk-********************
REDIS_URI=redis://localhost:6379
```

### 3. 환경 변수 로드
```python
from dotenv import load_dotenv
import os

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
```

## 핵심 컴포넌트

### 1. PromptTemplate
프롬프트 문자열에 변수를 삽입하여 동적 프롬프트 생성

```python
from langchain_core.prompts import PromptTemplate

template = "{product}를 만드는 회사 이름은?"
prompt = PromptTemplate.from_template(template)
formatted = prompt.format(product="커피")
```

### 2. ChatOpenAI
OpenAI GPT 모델 래퍼

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

### 3. Output Parsers
LLM 출력을 구조화된 형태로 변환

#### StrOutputParser
```python
from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()
```

#### JsonOutputParser
```python
from langchain_core.output_parsers import JsonOutputParser
parser = JsonOutputParser()
```

#### CommaSeparatedListOutputParser
```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser
parser = CommaSeparatedListOutputParser()
```

## LCEL 기반 체인 구성

### 기본 체인
```python
chain = prompt | llm | parser
result = chain.invoke({"product": "커피"})
```

### 다중 체인 연결
```python
# 1단계: 제목 생성
subject_prompt = PromptTemplate.from_template("이메일 제목: {content}")
title_chain = subject_prompt | llm

# 2단계: 본문 생성  
body_prompt = PromptTemplate.from_template("제목: {subject}\n본문:")
body_chain = body_prompt | llm

# 연결된 체인
email_chain = (
    subject_prompt | llm | 
    {"subject": RunnablePassthrough()} |
    body_prompt | llm
)
```

### 조건 분기 체인
```python
from langchain_core.runnables import RunnableBranch, RunnableLambda

def is_summary(input_text):
    return input_text.startswith("요약:")

branch_chain = RunnableBranch(
    (RunnableLambda(is_summary), summary_chain),
    email_chain  # 기본값
)
```

## 메모리 관리

### ChatMessageHistory
```python
from langchain_core.chat_history import InMemoryChatMessageHistory

history = InMemoryChatMessageHistory()
history.add_user_message("안녕하세요")
history.add_ai_message("안녕하세요!")
```

### RunnableWithMessageHistory
```python
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)
```

### Redis 기반 영속 메모리
```python
from langchain_redis import RedisChatMessageHistory

history = RedisChatMessageHistory(
    session_id="user_123",
    redis_url=os.getenv("REDIS_URI")
)
```

### 대화 요약 메모리
```python
# 요약 체인
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "대화 내용을 간결하게 요약하세요."),
    ("human", "{conversation}")
])
summary_chain = summary_prompt | llm

# 요약 적용
def summarize_conversation(messages):
    dialog = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
    summary = summary_chain.invoke({"conversation": dialog})
    
    new_history = InMemoryChatMessageHistory()
    new_history.messages.append(SystemMessage(content=f"요약: {summary}"))
    return new_history
```

## 도구(Tools) 활용

### 커스텀 도구 생성
```python
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    city: str = Field(description="날씨를 조회할 도시")

def get_weather(city: str) -> str:
    # 날씨 API 호출 로직
    return f"{city}의 현재 날씨는 맑음입니다."

weather_tool = RunnableLambda(
    lambda input: get_weather(input["city"])
).as_tool(
    name="get_weather",
    description="도시 날씨 조회",
    args_schema=WeatherInput
)
```

### 에이전트 구성
```python
from langchain.agents import initialize_agent, AgentType

tools = [weather_tool, news_tool]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

result = agent.run("서울 날씨 알려줘")
```

## 비용 모니터링 및 캐싱

### 토큰 사용량 추적
```python
from langchain_community.callbacks import get_openai_callback

with get_openai_callback() as cb:
    result = llm.invoke("질문")
    print(f"총 비용: ${cb.total_cost:.6f}")
    print(f"토큰 수: {cb.total_tokens}")
```

### 응답 캐싱
```python
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache

# 메모리 캐시
set_llm_cache(InMemoryCache())

# SQLite 캐시
from langchain_community.cache import SQLiteCache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))
```

## 실습 예제

### 1. 간단한 질의응답
```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate.from_template("Q: {question}\n한 단어로 답하세요.")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = prompt | llm | StrOutputParser()

answer = chain.invoke({"question": "대한민국의 수도는?"})
print(answer)  # "서울"
```

### 2. 메모리가 있는 챗봇
```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절한 상담봇입니다."),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{question}")
])

chain = prompt | llm | StrOutputParser()

# 메모리 통합
chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

# 사용
response = chatbot.invoke(
    {"question": "안녕하세요, 저는 김철수입니다."},
    config={"configurable": {"session_id": "user_1"}}
)
```

### 3. JSON 출력 파싱
```python
from langchain_core.output_parsers import JsonOutputParser

movie_prompt = PromptTemplate.from_template(
    "다음 영화를 JSON 형식으로 추천해주세요:\n"
    "취향: {preference}\n"
    '형식: {{"title": "제목", "year": 연도, "genre": "장르"}}'
)

json_chain = movie_prompt | llm | JsonOutputParser()

result = json_chain.invoke({"preference": "SF"})
print(result)  # {"title": "인터스텔라", "year": 2014, "genre": "SF"}
```

## 🔗 주요 참고사항

### LCEL 장점
- 간결한 파이프라인 표현
- 자동 병렬화 및 스트리밍 지원
- 일관된 실행 인터페이스

### 메모리 전략
- 단기: InMemoryCache (빠름, 휘발성)
- 장기: Redis/SQLite (영속적, 약간 느림)
- 긴 대화: 요약 메모리 활용

### 비용 최적화
- 모델 선택: GPT-4o-mini > GPT-4 (성능 대비 비용)
- 캐싱 활용: 동일 요청 반복 방지
- 토큰 모니터링: 사용량 추적 및 예산 관리

이 가이드를 통해 LangChain v0.3.x의 핵심 기능들을 체계적으로 활용할 수 있습니다.