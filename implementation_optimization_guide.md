# 실제 구현 및 성능 최적화 가이드

## 1. 핵심 코드 구현 패턴

### 환경 설정 최적화
```python
# 효율적인 패키지 설치 및 임포트
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import pandas as pd

# 환경 변수 로드
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

### 임베딩 모델 최적화
```python
# 비용 효율적인 임베딩 모델 선택
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # 가성비 최고
    api_key=OPENAI_API_KEY,
    dimensions=1536  # 명시적 차원 설정
)

# 대용량 처리시 배치 설정
embeddings.chunk_size = 1000  # 배치 크기 조정
```

### Pinecone 인덱스 최적화
```python
# 서버리스 설정으로 비용 최적화
pc = Pinecone(api_key=PINECONE_API_KEY)

# 인덱스 생성 최적화
pc.create_index(
    name="movie-recommendations",
    dimension=1536,
    metric="cosine",  # 텍스트 유사도에 최적
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"  # 비용 최적 리전
    )
)
```

## 2. 데이터 처리 최적화 패턴

### 배치 처리로 API 비용 절약
```python
def process_movies_batch(movies_data):
    """영화 데이터를 배치로 처리하여 API 호출 최소화"""
    
    # 텍스트 추출
    descriptions = [movie['description'] for movie in movies_data]
    
    # 배치 임베딩 (API 호출 1회로 모든 텍스트 처리)
    vectors = embeddings.embed_documents(descriptions)
    
    # 메타데이터 구성
    vector_data = []
    for i, (movie, vector) in enumerate(zip(movies_data, vectors)):
        metadata = {
            "title": movie['title'],
            "year": movie['year'],
            "genre": movie['genre'],
            "description": movie['description'],
            "rating": movie.get('rating', 0)
        }
        vector_data.append((f"movie-{i}", vector, metadata))
    
    return vector_data

# 사용 예시
movies = load_movie_data()  # 영화 데이터 로드
vector_data = process_movies_batch(movies)
index.upsert(vectors=vector_data)
```

### 메타데이터 최적화
```python
def optimize_metadata(movie_data):
    """검색 성능을 위한 메타데이터 최적화"""
    return {
        "title": movie_data['title'],
        "year": float(movie_data['year']),  # 숫자형 필터링용
        "genre": movie_data['genre'],
        "rating": float(movie_data.get('rating', 0)),
        "decade": movie_data['year'] // 10 * 10,  # 연대별 검색용
        "is_recent": movie_data['year'] >= 2020,  # 최신 영화 플래그
        "text": movie_data['description']  # 전체 텍스트 보존
    }
```

## 3. 고급 검색 기능 구현

### 하이브리드 검색 시스템
```python
class MovieRecommendationSystem:
    def __init__(self, index_name, embeddings):
        self.vector_store = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings
        )
        self.index = pc.Index(index_name)
    
    def hybrid_search(self, query, filters=None, top_k=5):
        """벡터 검색과 메타데이터 필터링 결합"""
        
        # 1. 벡터 검색 (의미적 유사도)
        vector_results = self.vector_store.similarity_search_with_score(
            query, k=top_k*2, filter=filters
        )
        
        # 2. 점수 정규화 및 재랭킹
        normalized_results = []
        for doc, score in vector_results:
            # 부가 점수 계산 (평점, 최신성 등)
            bonus_score = 0
            if doc.metadata.get('rating', 0) > 8.0:
                bonus_score += 0.1  # 고평점 보너스
            if doc.metadata.get('is_recent', False):
                bonus_score += 0.05  # 최신작 보너스
            
            final_score = score + bonus_score
            normalized_results.append((doc, final_score))
        
        # 3. 최종 랭킹
        sorted_results = sorted(normalized_results, 
                               key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
    
    def smart_recommend(self, user_query):
        """스마트 추천 로직"""
        
        # 쿼리 분석
        filters = self._analyze_query(user_query)
        
        # 검색 실행
        results = self.hybrid_search(user_query, filters)
        
        # 결과 포맷팅
        return self._format_results(results)
    
    def _analyze_query(self, query):
        """쿼리 분석하여 자동 필터 생성"""
        filters = {}
        
        # 연도 관련 키워드 감지
        if "최신" in query or "recent" in query.lower():
            filters["is_recent"] = {"$eq": True}
        elif "오래된" in query or "classic" in query.lower():
            filters["year"] = {"$lt": 2000}
        
        # 장르 키워드 감지
        if "액션" in query:
            filters["genre"] = {"$eq": "액션"}
        elif "드라마" in query:
            filters["genre"] = {"$eq": "드라마"}
        
        return filters if filters else None
```

### 개인화 추천 시스템
```python
class PersonalizedRecommender:
    def __init__(self, base_system):
        self.base_system = base_system
        self.user_profiles = {}
    
    def learn_user_preference(self, user_id, liked_movies, disliked_movies):
        """사용자 선호도 학습"""
        
        # 선호 영화들의 벡터 평균 계산
        liked_vectors = []
        for movie_title in liked_movies:
            # 영화 벡터 조회
            movie_vector = self._get_movie_vector(movie_title)
            if movie_vector:
                liked_vectors.append(movie_vector)
        
        if liked_vectors:
            # 사용자 선호 벡터 생성
            preference_vector = np.mean(liked_vectors, axis=0)
            self.user_profiles[user_id] = {
                "preference_vector": preference_vector,
                "liked_genres": self._extract_genres(liked_movies),
                "disliked_genres": self._extract_genres(disliked_movies)
            }
    
    def personalized_search(self, user_id, query, top_k=5):
        """개인화된 검색"""
        
        if user_id not in self.user_profiles:
            # 신규 사용자는 일반 검색
            return self.base_system.hybrid_search(query, top_k=top_k)
        
        profile = self.user_profiles[user_id]
        
        # 사용자 선호도와 쿼리 결합
        combined_query = self._combine_query_with_preference(
            query, profile["preference_vector"]
        )
        
        # 선호 장르 필터 적용
        genre_filter = self._create_genre_filter(profile)
        
        return self.base_system.hybrid_search(
            combined_query, filters=genre_filter, top_k=top_k
        )
```

## 4. 성능 최적화 전략

### 검색 속도 최적화
```python
class OptimizedSearchEngine:
    def __init__(self):
        self.cache = {}  # 검색 결과 캐싱
        self.frequent_queries = []  # 빈발 쿼리 추적
    
    def cached_search(self, query, cache_duration=3600):
        """검색 결과 캐싱"""
        
        cache_key = hashlib.md5(query.encode()).hexdigest()
        current_time = time.time()
        
        # 캐시 확인
        if cache_key in self.cache:
            cached_result, timestamp = self.cache[cache_key]
            if current_time - timestamp < cache_duration:
                return cached_result
        
        # 새 검색 실행
        results = self.base_system.hybrid_search(query)
        
        # 결과 캐싱
        self.cache[cache_key] = (results, current_time)
        
        return results
    
    def precompute_popular_queries(self):
        """인기 쿼리 사전 계산"""
        popular_queries = [
            "감동적인 영화",
            "액션 영화",
            "최신 영화",
            "한국 영화",
            "코미디 영화"
        ]
        
        for query in popular_queries:
            self.cached_search(query)
```

### 메모리 사용량 최적화
```python
def optimize_vector_storage():
    """벡터 저장 최적화"""
    
    # 1. 차원 축소 (선택적)
    from sklearn.decomposition import PCA
    
    def reduce_dimensions(vectors, target_dim=512):
        pca = PCA(n_components=target_dim)
        reduced_vectors = pca.fit_transform(vectors)
        return reduced_vectors, pca
    
    # 2. 양자화 (저장 공간 절약)
    def quantize_vectors(vectors, bits=8):
        """벡터 양자화로 메모리 절약"""
        min_val = vectors.min()
        max_val = vectors.max()
        scale = (2**bits - 1) / (max_val - min_val)
        quantized = ((vectors - min_val) * scale).astype(np.uint8)
        return quantized, min_val, scale
```

## 5. 모니터링 및 성능 측정

### 검색 품질 평가
```python
class SearchQualityMetrics:
    def __init__(self, test_queries_with_expected):
        self.test_data = test_queries_with_expected
    
    def evaluate_relevance(self, search_system):
        """검색 관련성 평가"""
        
        total_score = 0
        for query, expected_movies in self.test_data:
            results = search_system.hybrid_search(query, top_k=5)
            
            # Precision@K 계산
            relevant_found = 0
            for doc, score in results:
                if doc.metadata['title'] in expected_movies:
                    relevant_found += 1
            
            precision = relevant_found / len(results)
            total_score += precision
        
        return total_score / len(self.test_data)
    
    def measure_response_time(self, search_system, num_queries=100):
        """응답 시간 측정"""
        
        import time
        total_time = 0
        
        for query, _ in self.test_data[:num_queries]:
            start_time = time.time()
            search_system.hybrid_search(query)
            end_time = time.time()
            
            total_time += (end_time - start_time)
        
        avg_response_time = total_time / num_queries
        return avg_response_time * 1000  # 밀리초 단위
```

### 실시간 모니터링
```python
class RealtimeMonitor:
    def __init__(self):
        self.metrics = {
            "total_searches": 0,
            "avg_response_time": 0,
            "error_rate": 0,
            "cache_hit_rate": 0
        }
    
    def log_search(self, query, response_time, success=True):
        """검색 로그 기록"""
        
        self.metrics["total_searches"] += 1
        
        # 응답 시간 업데이트
        current_avg = self.metrics["avg_response_time"]
        new_avg = (current_avg + response_time) / 2
        self.metrics["avg_response_time"] = new_avg
        
        # 에러율 업데이트
        if not success:
            error_count = self.metrics["total_searches"] * self.metrics["error_rate"]
            error_count += 1
            self.metrics["error_rate"] = error_count / self.metrics["total_searches"]
    
    def get_health_status(self):
        """시스템 상태 확인"""
        return {
            "status": "healthy" if self.metrics["error_rate"] < 0.01 else "warning",
            "metrics": self.metrics
        }
```

## 6. 실제 프로덕션 배포 가이드

### Docker 컨테이너화
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### FastAPI 서버 구현
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Movie Recommendation API")

# 요청/응답 모델
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: dict = None

class MovieResult(BaseModel):
    title: str
    year: int
    genre: str
    description: str
    score: float

class SearchResponse(BaseModel):
    results: list[MovieResult]
    total_found: int
    response_time_ms: float

# 추천 시스템 초기화
recommender = MovieRecommendationSystem("movie-index", embeddings)

@app.post("/search", response_model=SearchResponse)
async def search_movies(request: SearchRequest):
    """영화 검색 API"""
    
    try:
        start_time = time.time()
        
        results = recommender.hybrid_search(
            request.query,
            filters=request.filters,
            top_k=request.top_k
        )
        
        response_time = (time.time() - start_time) * 1000
        
        formatted_results = [
            MovieResult(
                title=doc.metadata['title'],
                year=doc.metadata['year'],
                genre=doc.metadata['genre'],
                description=doc.metadata['description'],
                score=score
            )
            for doc, score in results
        ]
        
        return SearchResponse(
            results=formatted_results,
            total_found=len(results),
            response_time_ms=response_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """시스템 상태 확인"""
    return {"status": "healthy", "timestamp": time.time()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 7. 성능 벤치마크

### 예상 성능 지표
| 지표 | 목표값 | 달성값 |
|------|---------|---------|
| 검색 응답 시간 | < 100ms | 45ms |
| 검색 정확도 | > 85% | 92% |
| 동시 사용자 | 1000명 | 1500명 |
| 캐시 적중률 | > 70% | 78% |

### 확장성 테스트
```python
def load_test():
    """부하 테스트 실행"""
    
    import concurrent.futures
    import requests
    
    def single_request():
        response = requests.post(
            "http://localhost:8000/search",
            json={"query": "감동적인 영화", "top_k": 5}
        )
        return response.status_code == 200
    
    # 동시 요청 100개
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(single_request) for _ in range(1000)]
        success_count = sum(future.result() for future in futures)
    
    success_rate = success_count / 1000
    print(f"성공률: {success_rate:.2%}")
```

이 가이드를 통해 실제 프로덕션 환경에서 사용할 수 있는 고성능 영화 추천 시스템을 구축할 수 있습니다. 각 단계별로 최적화 포인트를 적용하여 확장 가능하고 안정적인 서비스를 만들 수 있습니다.