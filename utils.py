import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def top_k_indices(a: np.ndarray, k: int):
    if k >= len(a):
        return np.argsort(a)[::-1]
    idx = np.argpartition(a, -k)[-k:]
    return idx[np.argsort(a[idx])[::-1]]

def normalize_scores(scores: np.ndarray):
    s = np.array(scores, dtype=float)
    if len(s)==0:
        return s
    mn, mx = np.min(s), np.max(s)
    if mx - mn < 1e-12:
        return np.zeros_like(s)
    return (s - mn) / (mx - mn)

def to_user_item_matrix(ratings: pd.DataFrame, n_users=None, n_items=None):
    uid = ratings['userId'].astype('category').cat.codes
    iid = ratings['movieId'].astype('category').cat.codes
    nU = n_users or (uid.max()+1)
    nI = n_items or (iid.max()+1)
    mat = np.zeros((nU, nI), dtype=float)
    mat[uid, iid] = ratings['rating'].values
    return mat, uid, iid
