"""
Minimal evaluation script for RMSE and Precision@K/Recall@K using MovieLens.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from .data_loader import load_movielens
from .collaborative import CollaborativeFiltering
from .content_based import ContentBasedRecommender
from .hybrid import HybridRecommender

def precision_recall_at_k(recommended, relevant_set, k=10):
    rec_k = [m for m,_ in recommended[:k]]
    hit = sum(1 for m in rec_k if m in relevant_set)
    precision = hit / k
    recall = hit / max(1, len(relevant_set))
    return precision, recall

def main():
    ratings, movies, tags = load_movielens("data/ml-latest-small")
    # Split by user
    train, test = train_test_split(ratings, test_size=0.2, random_state=42)
    # CF
    cf = CollaborativeFiltering(mode="item", k=30).fit(train)
    # CBF
    cbf = ContentBasedRecommender().fit(movies, tags)

    users = test['userId'].unique()[:50]
    precs, recs = [], []
    for u in users:
        test_u = test[test['userId']==u]
        # CF recs
        try:
            cf_recs = cf.recommend_for_user(u, top_k=50)
        except Exception:
            continue
        # CBF recs (build from train history)
        train_u = train[train['userId']==u][['movieId','rating']]
        cbf_recs = cbf.recommend_for_user(train_u, top_k=50)
        # Hybrid
        hyb = HybridRecommender(alpha=0.6)
        recs_final = hyb.combine(cf_recs, cbf_recs, top_k=10)
        relevant = set(test_u[test_u['rating']>=4.0]['movieId'].astype(int).tolist())
        p, r = precision_recall_at_k(recs_final, relevant, k=10)
        precs.append(p); recs.append(r)
    print(f"Mean Precision@10: {np.mean(precs):.3f}, Mean Recall@10: {np.mean(recs):.3f}")

if __name__ == "__main__":
    main()
