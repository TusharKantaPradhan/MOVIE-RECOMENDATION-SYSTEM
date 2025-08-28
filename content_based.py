from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    """
    Build content vectors from genres + tags text, then compute user profile and recommend.
    """
    def __init__(self, max_features: int = 5000):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.genre_binarizer = MultiLabelBinarizer()
        self.item_matrix = None
        self.movie_ids = None

    @staticmethod
    def _split_genres(s: str):
        if pd.isna(s):
            return []
        return s.split('|')

    def fit(self, movies: pd.DataFrame, tags: Optional[pd.DataFrame] = None):
        movies = movies.copy()
        # Build a text field: genres words + tag text
        movies['genres_list'] = movies['genres'].apply(self._split_genres)
        genre_matrix = self.genre_binarizer.fit_transform(movies['genres_list'])

        tag_text = None
        if tags is not None and len(tags):
            # aggregate tags per movie
            tag_text = tags.groupby('movieId')['tag'].apply(lambda s: ' '.join(map(str, s))).reindex(movies['movieId']).fillna('')
        else:
            tag_text = pd.Series(['']*len(movies), index=movies['movieId'])

        text_corpus = (movies['genres_list'].apply(lambda xs: ' '.join(xs)).astype(str) + ' ' + tag_text.values)
        tfidf = self.vectorizer.fit_transform(text_corpus.values)
        # concatenate sparse tfidf with dense genre matrix
        from scipy.sparse import hstack
        self.item_matrix = hstack([tfidf, genre_matrix]).tocsr()
        self.movie_ids = movies['movieId'].astype(int).tolist()
        self.movies_df = movies[['movieId','title','genres']]
        return self

    def recommend_for_user(self, user_ratings: pd.DataFrame, top_k: int = 10, like_threshold: float = 4.0):
        """
        user_ratings: df with columns [movieId, rating]
        """
        if user_ratings.empty:
            # cold start: just return top popular by genres similarity (fallback: first K)
            scores = self.item_matrix.sum(axis=1).A1
            idx = np.argsort(scores)[::-1][:top_k]
            return [(int(self.movie_ids[i]), float(scores[i])) for i in idx]

        # user profile as weighted sum of liked items
        liked = user_ratings[user_ratings['rating'] >= like_threshold]
        if liked.empty:
            liked = user_ratings.nlargest(min(5, len(user_ratings)), 'rating')

        # build profile
        item_index = {mid:i for i, mid in enumerate(self.movie_ids)}
        rows = [item_index.get(int(mid)) for mid in liked['movieId'] if int(mid) in item_index]
        from scipy.sparse import vstack
        user_vec = self.item_matrix[rows].multiply(liked['rating'].values[:,None]).mean(axis=0)
        sims = cosine_similarity(user_vec, self.item_matrix).ravel()
        # exclude already rated
        rated_set = set(int(m) for m in user_ratings['movieId'])
        for i, mid in enumerate(self.movie_ids):
            if int(mid) in rated_set:
                sims[i] = -1e9
        idx = np.argsort(sims)[::-1][:top_k]
        return [(int(self.movie_ids[i]), float(sims[i])) for i in idx]
