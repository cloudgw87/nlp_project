import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

def evaluate_popularity_baseline(train_df, test_df, final_df, k_list=[10, 20, 30, 40, 50]):
    """모든 유저에게 가장 인기 있는 영화 상위 K개를 추천"""
    # 이미 본 영화 제외를 위해 영화별 전체 순위 리스트 생성
    pop_list = train_df[train_df['movieId'].isin(final_df['movieId'])]['movieId'].value_counts().index.tolist()

    common_users = set(test_df['userId']) & set(train_df['userId'])
    results = {k: {'recalls': [], 'ndcgs': []} for k in k_list}
    max_k = max(k_list)

    for user_id in tqdm(common_users, desc="Pop Baseline"):
        user_train_set = set(train_df[train_df['userId'] == user_id]['movieId'])
        target_movie_id = test_df[test_df['userId'] == user_id]['movieId'].iloc[0]
        
        # 미시청 인기 영화 상위 K개
        recs = [m for m in pop_list if m not in user_train_set][:max_k]
        
        for k in k_list:
            current_top_k = recs[:k]
            hit = 1 if target_movie_id in current_top_k else 0
            results[k]['recalls'].append(hit)
            results[k]['ndcgs'].append(1 / np.log2(current_top_k.index(target_movie_id) + 2) if hit else 0)

    final_metrics = {}
    for k in k_list:
        final_metrics[f'Recall@{k}'] = np.mean(results[k]['recalls'])
        final_metrics[f'NDCG@{k}'] = np.mean(results[k]['ndcgs'])

    return final_metrics

def evaluate_genre_popularity_baseline(train_df, test_df, final_df, k_list=[10, 20, 30, 40, 50]):
    """유저의 최선호 장르 내에서 인기 있는 영화를 추천"""
    movie_to_genres = final_df.set_index('movieId')['genres'].to_dict()
    pop_counts = train_df['movieId'].value_counts()
    
    # 장르별 인기 영화 사전 계산
    genre_movie_counts = {}
    for mid, genres in movie_to_genres.items():
        for g in genres.split('|'):
            genre_movie_counts.setdefault(g, []).append((mid, pop_counts.get(mid, 0)))
    
    for g in genre_movie_counts:
        genre_movie_counts[g] = [m[0] for m in sorted(genre_movie_counts[g], key=lambda x: x[1], reverse=True)]

    common_users = set(test_df['userId']) & set(train_df['userId'])
    results = {k: {'recalls': [], 'ndcgs': []} for k in k_list}
    max_k = max(k_list)

    for user_id in tqdm(common_users, desc="Genre-Pop Baseline"):
        user_train_movies = train_df[train_df['userId'] == user_id]['movieId'].tolist()
        user_train_set = set(user_train_movies)
        
        # 유저 선호 장르 1위 추출
        user_genres = []
        for m in user_train_movies:
            if m in movie_to_genres:
                user_genres.extend(movie_to_genres[m].split('|'))
        top_genre = pd.Series(user_genres).value_counts().index[0] if user_genres else 'Drama'
        
        # 해당 장르 내 미시청 상위 K개
        recs = [m for m in genre_movie_counts.get(top_genre, []) if m not in user_train_set][:max_k]

        target_movie_id = test_df[test_df['userId'] == user_id]['movieId'].iloc[0]

        for k in k_list:
            current_top_k = recs[:k]
            hit = 1 if target_movie_id in current_top_k else 0
            results[k]['recalls'].append(hit)
            results[k]['ndcgs'].append(1 / np.log2(current_top_k.index(target_movie_id) + 2) if hit else 0)

    final_metrics = {}
    for k in k_list:
        final_metrics[f'Recall@{k}'] = np.mean(results[k]['recalls'])
        final_metrics[f'NDCG@{k}'] = np.mean(results[k]['ndcgs'])

    return final_metrics

def evaluate_hybrid(model_embeddings, train_df, test_df, final_df, k_list=[10, 20, 30, 40, 50], sbert_weight=0.7):
    movie_id_to_idx = {id: i for i, id in enumerate(final_df['movieId'])}
    pop_scores = torch.tensor(final_df['pop_score'].values, dtype=torch.float32).to(model_embeddings.device)
    
    common_users = set(test_df['userId']) & set(train_df['userId'])
    results = {k: {'recalls': [], 'ndcgs': []} for k in k_list}

    for user_id in tqdm(common_users, desc="Evaluating"):
        user_train_movies = train_df[train_df['userId'] == user_id]['movieId'].tolist()
        user_train_indices = [movie_id_to_idx[m] for m in user_train_movies if m in movie_id_to_idx]
        
        target_movie_id = test_df[test_df['userId'] == user_id]['movieId'].iloc[0]
        if not user_train_indices or target_movie_id not in movie_id_to_idx: continue
        target_idx = movie_id_to_idx[target_movie_id]
        
        # SBERT 유사도 계산
        user_profile = model_embeddings[user_train_indices].mean(dim=0, keepdim=True)
        sbert_scores = torch.mm(user_profile, model_embeddings.T).squeeze(0)
        
        # score 결합
        final_scores = (sbert_weight * sbert_scores) + ((1.0-sbert_weight) * pop_scores)
        
        # 이미 본 영화는 제외
        final_scores[user_train_indices] = -1.0
        
        max_k = max(k_list)
        _, top_indices = torch.topk(final_scores, k=max_k)
        top_indices = top_indices.tolist()
        
        for k in k_list:
            current_top_k = top_indices[:k]
            hit = 1 if target_idx in current_top_k else 0
            results[k]['recalls'].append(hit)
            results[k]['ndcgs'].append(1 / np.log2(current_top_k.index(target_idx) + 2) if hit else 0)

    final_metrics = {}
    for k in k_list:
        final_metrics[f'Recall@{k}'] = np.mean(results[k]['recalls'])
        final_metrics[f'NDCG@{k}'] = np.mean(results[k]['ndcgs'])

    return final_metrics