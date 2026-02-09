from preprocess import load_and_preprocess
from embedding import generate_embeddings
from evaluate import evaluate_hybrid, evaluate_popularity_baseline, evaluate_genre_popularity_baseline
import torch
import pandas as pd
import random

random.seed(42)

def main():
    print("\n" + "="*50)
    print("SBERT Movie Recommendation System Pipeline")
    print("="*50)
    
    # 1. 데이터 로드 및 다운로드
    print("\n[Step 1/4] Loading and verifying dataset...")
    df, ratings = load_and_preprocess()

    # 2. 데이터 분할 및 필터링
    print("\n[Step 2/4] Filtering data and sampling active users...")
    user_counts = ratings[ratings['rating'] >= 4.0]['userId'].value_counts()
    active_users = user_counts[user_counts >= 20]
    sampled_users = random.sample(list(active_users), 10000)
    
    ratings = ratings[ratings['userId'].isin(sampled_users)].sort_values(['userId', 'timestamp'])
    test_ratings = ratings.groupby('userId').tail(1)
    train_ratings = ratings.drop(test_ratings.index)
    train_ratings = train_ratings[train_ratings['rating'] >= 4.0]

    # 3. SBERT 임베딩 생성
    print("\n[Step 3/4] Generating SBERT embeddings for movie metadata...")
    movie_embeddings = generate_embeddings(df['combined_text'].tolist())

    # 4. 성능 비교 실험
    print("\n[Step 4/4] Starting multi-K evaluation...")
    k_values = [10, 30, 50]

    # Baseline: Simple Popularity
    print("Running Popularity Baseline...")
    pop_results = evaluate_popularity_baseline(train_ratings, test_ratings, df, k_list=k_values)
    pop_results['Model'] = 'Popularity (Baseline)'
    
    # Baseline: Genre+Popularity
    print("Running Genre+Popularity Baseline...")
    genre_pop_results = evaluate_genre_popularity_baseline(train_ratings, test_ratings, df, k_list=k_values)
    genre_pop_results['Model'] = 'Genre+Popularity (Baseline)'

    # Proposed: SBERT+Pop Hybrid
    print("Running SBERT+Popularity Hybrid Model (Weight=0.7)...")
    hybrid_results = evaluate_hybrid(movie_embeddings, train_ratings, test_ratings, df, k_list=k_values, sbert_weight=0.7)
    hybrid_results['Model'] = 'SBERT+Pop Hybrid'

    # 5. Display Final Table
    all_results = [pop_results, genre_pop_results, hybrid_results]
    results_df = pd.DataFrame(all_results)
    
    # 컬럼 순서 조정 (Model이 가장 먼저 오도록)
    cols = ['Model'] + [c for c in results_df.columns if c != 'Model']
    results_df = results_df[cols]

    print("\n" + "="*80)
    print("📊 Final Benchmarking Report")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    main()