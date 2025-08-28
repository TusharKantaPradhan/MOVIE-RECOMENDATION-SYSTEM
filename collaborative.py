from typing import Optional, Literal
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .utils import top_k_indices, to_user_item_matrix

class CollaborativeFiltering:
    """
    Memory-based Collaborative Filtering (user-based & item-based).
    """
    def __init__(self, mode: Literal["user","item"]="user", k: int = 20, min_common: int = 1):
        self.mode = mode
        self.k = k
        self.min_common = min_common
        self.user_item = None
        self.user_sim = None
        self.item_sim = None
        self.user_index = None
        self.item_index = None
        self.movie_ids = None

    def fit(self, ratings: pd.DataFrame):
        self.user_item, uidx, iidx = to_user_item_matrix(ratings)
        # keep the mapping to original ids
        self.user_index = pd.Series(np.arange(self.user_item.shape[0]), index=ratings['userId'].astype('category').cat.categories)
        self.item_index = pd.Series(np.arange(self.user_item.shape[1]), index=ratings['movieId'].astype('category').cat.categories)
        self.movie_ids = list(self.item_index.index.values)

        if self.mode == "user":
            self.user_sim = cosine_similarity(self.user_item)
        else:
            self.item_sim = cosine_similarity(self.user_item.T)
        return self

    def _predict_user(self, user_idx: int) -> np.ndarray:
        # neighbors
        sims = self.user_sim[user_idx]
        # zero out self-sim
        sims[user_idx] = 0.0
        # weighted sum
        num = sims @ self.user_item
        den = np.abs(sims).sum() + 1e-8
        return num / den

    def _predict_item(self, user_idx: int) -> np.ndarray:
        ratings_u = self.user_item[user_idx]
        num = self.item_sim @ ratings_u
        den = np.abs(self.item_sim).sum(axis=1) + 1e-8
        return num / den

    def recommend_for_user(self, user_original_id: int, top_k: int = 10, exclude_rated: bool = True):
        user_idx = int(self.user_index.loc[user_original_id])
        if self.mode == "user":
            preds = self._predict_user(user_idx)
        else:
            preds = self._predict_item(user_idx)

        rated_mask = self.user_item[user_idx] > 0
        if exclude_rated:
            preds = preds.copy()
            preds[rated_mask] = -1e9

        idx = top_k_indices(preds, top_k)
        rec_movie_ids = [int(self.movie_ids[i]) for i in idx]
        rec_scores = preds[idx].tolist()
        return list(zip(rec_movie_ids, rec_scores))
