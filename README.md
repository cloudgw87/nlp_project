# Simple Movie Recommendation using Metadata Embeddings

Sentence-BERT 기반 영화 추천 시스템으로, 영화 메타데이터(제목, 장르, 태그)의 의미적 유사도와 인기도를 결합하여 개인화된 추천을 제공합니다.

## 프로젝트 개요

이 프로젝트는 MovieLens 10M 데이터셋을 활용하여 세 가지 추천 모델을 비교 평가합니다:
- **Popularity Baseline**: 전체 인기도 기반 추천
- **Genre+Popularity Baseline**: 사용자 선호 장르 내 인기 영화 추천
- **SBERT+Pop Hybrid**: SBERT 임베딩 유사도와 인기도를 결합한 하이브리드 모델

## 주요 기능

- MovieLens 10M 데이터셋 자동 다운로드 및 전처리
- Sentence-BERT를 활용한 영화 메타데이터 임베딩 생성
- 사용자 프로필 기반 의미적 유사도 계산
- 다양한 K 값(10, 30, 50)에 대한 Recall@K 및 NDCG@K 평가
- 베이스라인 모델과의 성능 비교

## 파일 구조

```
.
├── main.py           # 메인 실행 파일 (파이프라인 전체 흐름)
├── preprocess.py     # 데이터 다운로드 및 전처리
├── embedding.py      # SBERT 임베딩 생성
├── evaluate.py       # 모델 평가 함수들
└── README.md         # 프로젝트 문서
```

## 설치 및 실행

### 필수 라이브러리

```bash
pip install torch pandas numpy sentence-transformers tqdm
```

### 실행 방법

```bash
python main.py
```

## 실행 과정

1. **데이터 로드**: MovieLens 10M 데이터셋 다운로드 및 로드
2. **데이터 필터링**: 평점 4.0 이상, 20개 이상 평가한 활성 사용자 10,000명 샘플링
3. **임베딩 생성**: 영화 메타데이터(제목 + 장르 + 태그)를 SBERT로 임베딩
4. **모델 평가**: 세 가지 모델에 대해 Recall@K, NDCG@K 계산

## 실험 결과

최근 실행 결과 (10,000명 사용자 기준):

```
                      Model  Recall@10  NDCG@10  Recall@30  NDCG@30  Recall@50  NDCG@50
      Popularity (Baseline)   0.031185  0.014281   0.072765  0.024310   0.099792  0.029399
Genre+Popularity (Baseline)   0.020790  0.009594   0.068607  0.020602   0.087318  0.024095
           SBERT+Pop Hybrid   0.043659  0.019039   0.093555  0.030652   0.130977  0.037635
```

### 주요 발견

- **SBERT+Pop Hybrid 모델**이 모든 K 값에서 두 베이스라인보다 우수한 성능을 보임
- Recall@50 기준 약 31% 성능 향상 (0.099 → 0.131)
- 의미적 유사도와 인기도를 결합한 접근이 효과적임을 입증

## 모델 설명

### 1. Popularity Baseline
전체 사용자의 평가 횟수를 기반으로 가장 인기 있는 영화를 추천합니다.

### 2. Genre+Popularity Baseline
사용자가 선호하는 장르를 파악한 후, 해당 장르 내에서 인기 있는 영화를 추천합니다.

### 3. SBERT+Pop Hybrid (제안 모델)
- 사용자가 시청한 영화들의 임베딩 평균으로 사용자 프로필 생성
- 모든 영화와의 코사인 유사도 계산
- SBERT 유사도(70%)와 인기도 점수(30%)를 가중 결합
- 이미 시청한 영화는 제외하고 상위 K개 추천

## 평가 지표

- **Recall@K**: 상위 K개 추천 중 실제 시청한 영화가 포함된 비율
- **NDCG@K**: 추천 순위를 고려한 정규화된 할인 누적 이득

