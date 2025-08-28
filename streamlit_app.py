import streamlit as st
import pandas as pd
from pathlib import Path
from src.data_loader import load_movielens
from src.collaborative import CollaborativeFiltering
from src.content_based import ContentBasedRecommender
from src.hybrid import HybridRecommender

DATA_DIR = Path("data/ml-latest-small")

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")

if not DATA_DIR.exists():
    st.warning("MovieLens data not found. Run: `python data/get_movielens.py --variant small --dest data/`")
    st.stop()

ratings, movies, tags = load_movielens(str(DATA_DIR))

st.sidebar.header("Choose Model")
model_type = st.sidebar.selectbox("Model", ["Item-based CF", "User-based CF", "Content-Based", "Hybrid (CF + CBF)"])
top_k = st.sidebar.slider("Top K", 5, 30, 10)

user_ids = ratings['userId'].unique()
user = st.selectbox("Select a User ID", sorted(user_ids))

# Fit models (for demo; in production cache these)
if model_type in ["Item-based CF", "User-based CF", "Hybrid (CF + CBF)"]:
    mode = "item" if model_type=="Item-based CF" else "user"
    cf = CollaborativeFiltering(mode=mode, k=30).fit(ratings)
if model_type in ["Content-Based", "Hybrid (CF + CBF)"]:
    cbf = ContentBasedRecommender().fit(movies, tags)

def render_recs(recs, title):
    st.subheader(title)
    df = pd.DataFrame(recs, columns=["movieId", "score"])
    df = df.merge(movies[['movieId','title','genres']], on='movieId', how='left')
    st.dataframe(df)

if model_type == "Item-based CF":
    recs = cf.recommend_for_user(int(user), top_k=top_k)
    render_recs(recs, "Recommendations (Item CF)")
elif model_type == "User-based CF":
    recs = cf.recommend_for_user(int(user), top_k=top_k)
    render_recs(recs, "Recommendations (User CF)")
elif model_type == "Content-Based":
    user_hist = ratings[ratings['userId']==user][['movieId','rating']]
    recs = cbf.recommend_for_user(user_hist, top_k=top_k)
    render_recs(recs, "Recommendations (CBF)")
else:
    cf_recs = cf.recommend_for_user(int(user), top_k=50)
    user_hist = ratings[ratings['userId']==user][['movieId','rating']]
    cbf_recs = cbf.recommend_for_user(user_hist, top_k=50)
    hyb = HybridRecommender(alpha=0.6)
    recs = hyb.combine(cf_recs, cbf_recs, top_k=top_k)
    render_recs(recs, "Recommendations (Hybrid)")
