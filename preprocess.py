import pandas as pd
import numpy as np
import os
import urllib.request
import zipfile

def load_and_preprocess(path='ml-10M100K/'):
    # 데이터 다운로드
    if not os.path.exists(path):
        url = "https://files.grouplens.org/datasets/movielens/ml-10m.zip"
        zip_path = "ml-10m.zip"
        urllib.request.urlretrieve(url, zip_path)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove(zip_path)
        print("✅ 다운로드 및 압축 해제 완료.")

    # 데이터 로드
    movies = pd.read_csv(path + 'movies.dat', sep='::', names=['movieId', 'title', 'genres'], engine='python')
    tags_raw = pd.read_csv(path + 'tags.dat', sep='::', names=['userId', 'movieId', 'tag', 'timestamp'], engine='python')
    ratings = pd.read_csv(path + 'ratings.dat', sep='::', names=['userId', 'movieId', 'rating', 'timestamp'], engine='python')

    # 태그 정제
    movie_tags = tags_raw.groupby('movieId')['tag'].apply(lambda x: ' '.join(set(x.astype(str).str.lower()))).reset_index()
    
    # 데이터 결합
    final_df = pd.merge(movies, movie_tags, on='movieId', how='left').fillna('')
    final_df['clean_genres'] = final_df['genres'].str.replace('|', ' ')
    
    # 임베딩용 텍스트 생성
    final_df['combined_text'] = (final_df['title'] + " " + final_df['clean_genres'] + " " + final_df['tag']).str.strip()
    
    # 인기도(Pop Score) 계산
    pop_counts = ratings['movieId'].value_counts()
    final_df['pop_score'] = final_df['movieId'].map(pop_counts).fillna(0)
    final_df['pop_score'] = np.log1p(final_df['pop_score'])
    final_df['pop_score'] = (final_df['pop_score'] - final_df['pop_score'].min()) / (final_df['pop_score'].max() - final_df['pop_score'].min())
    
    return final_df, ratings